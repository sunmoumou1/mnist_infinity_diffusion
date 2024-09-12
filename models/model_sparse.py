# 重新复习

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat # type: ignore
from pytorch3d.ops import knn_points, knn_gather # type: ignore
import math
import warnings

from .conv_uno import UNO, UNOEncoder
from .sparse_conv_block import (
    SparseConvResBlock,
    convert_to_backend_form,
    convert_to_backend_form_like,
    calculate_norm,
    get_features_from_backend_form,
    get_normalising_conv,
)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SparseUNet(nn.Module):
    def __init__(
        self,
        channels=3,
        nf=64,
        time_emb_dim=256,
        img_size=128, # 或者是一个元组
        num_conv_blocks=3,
        knn_neighbours=3,
        uno_res=64, # 或者是一个元组
        uno_mults=(1, 2, 4, 8),
        z_dim=None,
        out_channels=None, # 这个代表输出的通道数
        conv_type="conv", # 作用于UNO
        depthwise_sparse=True, # 表示是否使用稀疏深度卷积，作用于SparseConvResBlock
        kernel_size=7,
        backend="torchsparse",
        optimise_dense=True, # 这个参数表示是否优化当假如x是四维的输入
        blocks_per_level=(2, 2, 2, 2), # 作用于UNO
        attn_res=[16, 8],
        dropout_res=16,
        dropout=0.1,
        uno_base_nf=64,
    ):
        super().__init__()
        self.backend = backend
        self.img_size = img_size
        self.uno_res = uno_res
        self.knn_neighbours = knn_neighbours
        self.kernel_size = kernel_size
        self.optimise_dense = optimise_dense
        
        # Input projection
        self.linear_in = nn.Linear(channels, nf)
        # Output projection
        self.linear_out = nn.Linear(
            nf, out_channels if out_channels is not None else channels
        )

        # TODO: Better to have more features here? 64 by default isn't many
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )


        # 检查 uno_res 的类型并相应处理
        if isinstance(uno_res, int):
            # 如果 uno_res 是整数，使用相同的分辨率生成网格
            uno_coords = torch.stack(
                torch.meshgrid(*[torch.linspace(0, 1, steps=uno_res) for _ in range(2)])
            )
        elif isinstance(uno_res, tuple) and len(uno_res) == 2:
            # 如果 uno_res 是元组，分别使用元组中的两个值
            uno_coords = torch.stack(
                torch.meshgrid(torch.linspace(0, 1, steps=uno_res[0]), torch.linspace(0, 1, steps=uno_res[1]))
            )
        else:
            raise ValueError("uno_res must be an integer or a tuple of two integers.")

        # 对坐标进行排列
        uno_coords = rearrange(uno_coords, "c h w -> () (h w) c")  # 注意这里的 c 实际上就是 2
        self.register_buffer("uno_coords", uno_coords)

        self.normalising_conv = get_normalising_conv(
            kernel_size=kernel_size, backend=backend
        )

        self.down_blocks = nn.ModuleList([])
        for _ in range(num_conv_blocks):
            self.down_blocks.append(
                SparseConvResBlock(
                    img_size,
                    nf,
                    kernel_size=kernel_size,
                    mult=2,
                    time_emb_dim=time_emb_dim,
                    z_dim=z_dim,
                    depthwise=depthwise_sparse,
                    backend=backend,
                )
            )
            
        self.uno_linear_in = nn.Linear(nf, uno_base_nf)
        self.uno_linear_out = nn.Linear(uno_base_nf, nf)
        
        self.up_blocks = nn.ModuleList([])
        for _ in range(num_conv_blocks):
            self.up_blocks.append(
                SparseConvResBlock(
                    img_size,
                    nf,
                    kernel_size=kernel_size,
                    mult=2,
                    skip_dim=nf,
                    time_emb_dim=time_emb_dim,
                    z_dim=z_dim,
                    depthwise=depthwise_sparse,
                    backend=backend,
                )
            )

        self.uno = UNO(
            uno_base_nf,
            uno_base_nf,
            width=uno_base_nf,
            mults=uno_mults,
            blocks_per_level=blocks_per_level,
            time_emb_dim=time_emb_dim,
            z_dim=z_dim,
            conv_type=conv_type,
            res=uno_res,
            attn_res=attn_res,
            dropout_res=dropout_res,
            dropout=dropout,
        )

    def knn_interpolate_to_grid(self, x, coords):
        with torch.no_grad():
            _, assign_index, neighbour_coords = knn_points(
                self.uno_coords.repeat(x.size(0), 1, 1),
                coords, # coords 的形状为 (B,L,2)
                K=self.knn_neighbours,
                return_nn=True,
            )

            # neighbour_coords: (B, y_length, K, 2)
            diff = neighbour_coords - self.uno_coords.unsqueeze(
                2
            )
            
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(
                squared_distance, min=1e-16
            )  # (B, y_length, K, 1)

        # Inverse square distance weighted mean
        neighbours = knn_gather(x, assign_index)  # (B, y_length, K, C)
        
        out = (neighbours * weights).sum(2) / weights.sum(2)

        return out.to(x.dtype)

    def forward(self, x, t, z=None, sample_lst=None, coords=None):
        batch_size = x.size(0)

        # If x is image shaped (4D) then treat it as a dense tensor for better optimisation
        if len(x.shape) == 4 and self.optimise_dense:
            # 注意self.optimise_dense默认就是True
            if sample_lst is not None:
                warnings.warn(
                    "Ignoring sample_lst: Recieved 4D x and sample_list != None so treating x as a dense Image."
                )
            if coords is not None:
                warnings.warn(
                    "Ignoring coords: Recieved 4D x and coords != None so treating x as a dense Image."
                )
            return self.dense_forward(x, t, z=z)


        assert sample_lst is not None, "In sparse mode sample_lst must be provided"
        if coords is None:
            # 判断 self.img_size 是整数还是元组
            if isinstance(self.img_size, int):
                img_height, img_width = self.img_size, self.img_size
            elif isinstance(self.img_size, tuple) and len(self.img_size) == 2:
                img_height, img_width = self.img_size
            else:
                raise ValueError("self.img_size must be either an integer or a tuple of two integers.")

            # 根据图像的高度和宽度生成网格坐标
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, 1, steps=img_height),
                    torch.linspace(0, 1, steps=img_width)
                )
            ).to(x.device)
            coords = rearrange(coords, "c h w -> () (h w) c")
            coords = repeat(coords, "() ... -> b ...", b=x.size(0))
            coords = torch.gather(
                coords, 1, sample_lst.unsqueeze(2).repeat(1, 1, coords.size(2))
            ).contiguous()
            # 根据 sample_lst 进行采样，第二个维度变为 L，其中 L 是 sample_lst 的长度（即采样索引的数量）。最终结果的形状为 (b, L, 2)。

        x = self.linear_in(x)
        t = self.time_mlp(t)

        # 1. Down conv blocks
        x = convert_to_backend_form(x, sample_lst, self.img_size, backend=self.backend)
        # 注意这里返回的x来源于下面的代码：
        # sparse_indices = sample_lst_to_sparse_indices(sample_lst, img_size, ndims=3)
        # x = torchsparse.SparseTensor(
        #     coords=sparse_indices, feats=rearrange(x, "b l c -> (b l) c")
        # )
        backend_tensor = x
        norm = calculate_norm(
            self.normalising_conv,
            backend_tensor,
            backend=self.backend,
        )

        downs = []
        for block in self.down_blocks:
            x = block(x, t=t, z=z, norm=norm)
            downs.append(x)

        # 2. Interpolate to regular grid
        x = get_features_from_backend_form(x, sample_lst, backend=self.backend)
        # 上面这行代码返回        return rearrange(x.feats, "(b l) c -> b l c", b=sample_lst.size(0))
        x = self.uno_linear_in(x)
        x = self.knn_interpolate_to_grid(x, coords)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.uno_res[0])

        # 3. UNO
        x = self.uno(x, t, z=z)

        # 4. Interpolate back to sparse coordinates
        x = F.grid_sample(x, coords.unsqueeze(2), mode="bilinear")
        # 注意上面这行代码中，X是b c h w, coords.unsqueeze(2)是(b, L, 1, 2)
        # 如果输入形状为 (N, C, H, W)，grid 形状为 (N, H_out, W_out, 2)，则输出形状为 (N, C, H_out, W_out)。
        x = rearrange(x, "b c l () -> b l c")
        x = self.uno_linear_out(x)
        x = convert_to_backend_form_like(
            x,
            backend_tensor,
            backend=self.backend,
        )

        # 5. Up conv blocks
        for block in self.up_blocks:
            skip = downs.pop()
            x = block(x, t=t, z=z, skip=skip, norm=norm)

        x = get_features_from_backend_form(x, sample_lst, backend=self.backend)
        x = self.linear_out(x)

        return x


    def get_torch_norm_kernel_size(self, img_size, round_down=True):
        # 判断 img_size 和 self.img_size 是否为整数或元组，并分别处理
        if isinstance(img_size, int):
            img_height, img_width = img_size, img_size
        elif isinstance(img_size, tuple) and len(img_size) == 2:
            img_height, img_width = img_size
        else:
            raise ValueError("img_size must be either an integer or a tuple of two integers.")

        if isinstance(self.img_size, int):
            self_height, self_width = self.img_size, self.img_size
        elif isinstance(self.img_size, tuple) and len(self.img_size) == 2:
            self_height, self_width = self.img_size
        else:
            raise ValueError("self.img_size must be either an integer or a tuple of two integers.")

        # 检查当前尺寸和目标尺寸是否一致
        if (img_height, img_width) != (self_height, self_width):
            # 分别计算高度和宽度的比例
            ratio_height = img_height / self_height
            ratio_width = img_width / self_width
            # new kernel_size becomes:
            # 1 -> 1, 1.5 -> 1, 2 -> 1 or 3, 2.5 -> 3, 3 -> 3, 3.5 -> 3, 4 -> 3 or 5, 4.5 -> 5, ...
            # where there are multiple options this is determined by round_down
            
            # 选择较小的比例来计算新的 kernel size
            ratio = min(ratio_height, ratio_width)
            
            # 根据比例计算新的 kernel size
            new_kernel_size = self.kernel_size * ratio
            if round_down:
                new_kernel_size = 2 * round((new_kernel_size - 1) / 2) + 1
            else:
                new_kernel_size = math.floor(new_kernel_size / 2) * 2 + 1
            return max(int(new_kernel_size), 3)  # 确保返回值为整数
        else:
            return self.kernel_size

    def dense_forward(self, x, t, z=None):
        # 获取输入图像的高度和宽度
        height, width = x.size(2), x.size(3)

        # 生成坐标网格并转移到输入张量的设备上
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, steps=height), 
                torch.linspace(0, 1, steps=width)
            )
        ).to(x.device)
        coords = rearrange(coords, "c h w -> () (h w) c")
        coords = repeat(coords, "() ... -> b ...", b=x.size(0))

        # 应用训练好的 self.linear_in 的参数进行卷积操作
        x = F.conv2d(
            x, self.linear_in.weight[:, :, None, None], bias=self.linear_in.bias
        )
        # 上面这行代码巧妙地化用了训练好的 self.linear_in 的参数，实现了从 B, C, H, W 到 B, nf, H, W
        t = self.time_mlp(t)

        # NOTE: 归一化以避免边缘伪影
        mask = torch.ones(
            x.size(0), 1, x.size(2), x.size(3), dtype=x.dtype, device=x.device
        )
        kernel_size = self.get_torch_norm_kernel_size((height,width))
        weight = torch.ones(
            1, 1, kernel_size, kernel_size, dtype=x.dtype, device=x.device
        ) / (self.kernel_size**2)
        norm = F.conv2d(mask, weight, padding=kernel_size // 2)

        # 1. 下采样卷积块
        downs = []
        for block in self.down_blocks:
            x = block(x, t=t, z=z, norm=norm)
            downs.append(x)

        # 2. 插值到常规网格
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.uno_linear_in(x)
        x = self.knn_interpolate_to_grid(x, coords)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.uno_res[0], w=self.uno_res[1])

        # 3. UNO 处理
        x = self.uno(x, t, z=z)

        # 4. 插值回稀疏坐标
        x = F.grid_sample(x, coords.unsqueeze(2), mode="bilinear")
        # 注意上面这行代码中，X是b c h w, coords.unsqueeze(2)是(b, L, 1, 2)
        # 如果输入形状为 (N, C, H, W)，grid 形状为 (N, H_out, W_out, 2)，则输出形状为 (N, C, H_out, W_out)。
        x = rearrange(x, "b c (h w) () -> b c h w", h=height, w=width)
        x = F.conv2d(
            x,
            self.uno_linear_out.weight[:, :, None, None],
            bias=self.uno_linear_out.bias,
        )

        # 5. 上采样卷积块
        for block in self.up_blocks:
            skip = downs.pop()
            x = block(x, t=t, z=z, skip=skip, norm=norm)

        x = F.conv2d(
            x, self.linear_out.weight[:, :, None, None], bias=self.linear_out.bias
        )

        return x


class SparseEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        channels=3,
        nf=64,
        img_size=128, # (img_height, img_width)
        num_conv_blocks=3,
        knn_neighbours=3,
        uno_res=64, # 也可能是元组，比如类似这种形式 (128, 64)
        uno_mults=(1, 2, 4, 8),
        z_dim=None,
        conv_type="conv",
        depthwise_sparse=True,
        kernel_size=7,
        backend="torch_dense",
        optimise_dense=True,
        blocks_per_level=(2, 2, 2, 2),
        attn_res=[16, 8],
        dropout_res=16,
        dropout=0.1,
        uno_base_nf=64,
        stochastic=False,
    ):
        super().__init__()
        self.backend = backend
        self.img_size = img_size
        if isinstance(img_size, tuple) and len(img_size) == 2:
            self.img_height, self.img_width = img_size
            
        self.uno_res = uno_res
        self.knn_neighbours = knn_neighbours
        self.kernel_size = kernel_size
        self.optimise_dense = optimise_dense
        self.stochastic = stochastic
        # Input projection
        self.linear_in = nn.Linear(channels, nf)
        # Output projection
        self.linear_out = nn.Linear(out_channels, out_channels)


        if isinstance(uno_res, int):
            # 如果 uno_res 是整数，使用相同的分辨率生成网格
            uno_coords = torch.stack(
                torch.meshgrid(*[torch.linspace(0, 1, steps=uno_res) for _ in range(2)])
            )
        elif isinstance(uno_res, tuple) and len(uno_res) == 2:
            # 如果 uno_res 是元组，分别使用元组中的两个值
            uno_coords = torch.stack(
                torch.meshgrid(torch.linspace(0, 1, steps=uno_res[0]), torch.linspace(0, 1, steps=uno_res[1]))
            )
        else:
            raise ValueError("uno_res must be an integer or a tuple of two integers.")
        
        uno_coords = rearrange(uno_coords, "c h w -> () (h w) c")
        self.register_buffer("uno_coords", uno_coords)

        self.normalising_conv = get_normalising_conv(
            kernel_size=kernel_size, backend=backend
        )

        self.down_blocks = nn.ModuleList([])
        for _ in range(num_conv_blocks):
            self.down_blocks.append(
                SparseConvResBlock(
                    img_size,
                    nf,
                    kernel_size=kernel_size,
                    mult=2,
                    time_emb_dim=nf,
                    z_dim=z_dim,
                    depthwise=depthwise_sparse,
                    backend=backend,
                )
            )
        self.uno_linear_in = nn.Linear(nf, uno_base_nf)

        self.uno = UNOEncoder(
            uno_base_nf,
            out_channels,
            width=uno_base_nf,
            mults=uno_mults,
            blocks_per_level=blocks_per_level,
            time_emb_dim=nf,
            z_dim=z_dim,
            conv_type=conv_type,
            res=uno_res,
            attn_res=attn_res,
            dropout_res=dropout_res,
            dropout=dropout,
        ) # 要特别注意这里使用的其实是UNOEncoder

        if stochastic:
            self.mu = nn.Linear(out_channels, out_channels)
            self.logvar = nn.Linear(out_channels, out_channels)

    def knn_interpolate_to_grid(self, x, coords):

        with torch.no_grad():
            _, assign_index, neighbour_coords = knn_points(
                self.uno_coords.repeat(x.size(0), 1, 1),
                coords,
                K=self.knn_neighbours,
                return_nn=True,
            )
            
            # neighbour_coords: (B, y_length, K, 2)
            diff = neighbour_coords - self.uno_coords.unsqueeze(2)
            
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(
                squared_distance, min=1e-16
            )  # (B, y_length, K, 1)

        # Inverse square distance weighted mean
        neighbours = knn_gather(x, assign_index)  # (B, y_length, K, C)
        
        out = (neighbours * weights).sum(2) / weights.sum(2)

        return out


    def forward(self, x, sample_lst=None, coords=None):
        # 注意此时希望x的形状是B,L,C；当然x传入的形状也可能是4维的，不过最后还是要把它变成B,L,C
        # 此时sample_lst的形状是B,L
        
        batch_size = x.size(0)
        if len(x.shape) == 4:
            x = rearrange(x, "b c h w -> b (h w) c")

        if coords is None:
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, 1, steps=self.img_height, device=x.device),
                    torch.linspace(0, 1, steps=self.img_width, device=x.device)
                )
            )
            coords = rearrange(coords, "c h w -> () (h w) c")
            coords = repeat(coords, "() ... -> b ...", b=x.size(0))
            if sample_lst is not None:
                coords = torch.gather(
                    coords, 1, sample_lst.unsqueeze(2).repeat(1, 1, coords.size(2))
                ).contiguous() # 此时coords的形状变为B,L,2

        if sample_lst is None:
            # 如果没有传入sample_lst，那么表示使用全部的图像格点来当做假装随机采样过的点
            sample_lst = torch.arange(self.img_size[0] * self.img_size[1], device=x.device)
            sample_lst = repeat(sample_lst, "s -> b s", b=x.size(0))

        x = self.linear_in(x)

        # 1. Down conv blocks
        x = convert_to_backend_form(x, sample_lst, self.img_size, backend=self.backend)
        backend_tensor = x
        norm = calculate_norm(
            self.normalising_conv,
            backend_tensor,
            backend=self.backend,
        )

        downs = []
        for block in self.down_blocks:
            x = block(x, norm=norm)
            downs.append(x)

        # 2. Interpolate to regular grid
        x = get_features_from_backend_form(x, sample_lst, backend=self.backend)
        x = self.uno_linear_in(x)
        x = self.knn_interpolate_to_grid(x, coords)
        # print("self.uno_res[0]", self.uno_res[0])
        # print("self.uno_res[1]", self.uno_res[1])
        
        x = rearrange(x, "b (h w) c -> b c h w", h=self.uno_res[0])

        # 3. UNO
        x = self.uno(x)
        x = x.mean(dim=(2, 3))

        x = self.linear_out(x)

        if self.stochastic:
            mu = self.mu(x)
            logvar = self.logvar(x)
            x = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            return x, mu, logvar

        return x

