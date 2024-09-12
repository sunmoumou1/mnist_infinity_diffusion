# 重新复习

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import calculate_gain
from einops import rearrange, repeat # type: ignore
import math

# torchsparse
try:
    import torchsparse # type: ignore
    from torchsparse import nn as spnn # type: ignore
    from torchsparse import SparseTensor # type: ignore
    TORCHSPARSE_AVAILABLE = True
except Exception:
    TORCHSPARSE_AVAILABLE = False

# todo 删除了 TorchDenseConvResBlock, MinkowskiConvResBlock


class SparseConvResBlock(nn.Module):
    '''
        这段代码用于创建一个稀疏卷积残差块（Residual Block）。这个类的设计目的是在不同的稀疏卷积库（如 spconv, torchsparse, MinkowskiEngine）或传统的 torch_dense 背景下实现残差卷积操作。
    '''
    def __init__(
        self,
        img_size, # 可能是元组
        embed_dim,
        kernel_size=7,
        mult=2, # 这个用于扩展全连接网络中间隐藏层神经元的数量
        skip_dim=None,
        time_emb_dim=None,
        epsilon=1e-5,
        z_dim=None,
        depthwise=True,
        backend="torchsparse", # 或者选择torch_dense
    ):
        super().__init__()
        self.backend = backend

        if self.backend == "torchsparse":
            assert TORCHSPARSE_AVAILABLE, "torchsparse backend is not detected."
            block = TorchsparseResBlock
        elif self.backend == "torch_dense":
            block = TorchDenseConvResBlock
        else:
            raise Exception("Unrecognised backend.")

        self.block = block(
            img_size,
            embed_dim,
            kernel_size=kernel_size,
            mult=mult,
            skip_dim=skip_dim,
            time_emb_dim=time_emb_dim,
            epsilon=epsilon,
            z_dim=z_dim,
            depthwise=depthwise,
        )

    def forward(self, x, t=None, skip=None, z=None, norm=None):

        if isinstance(x, torch.Tensor) and len(x.shape) == 4 and self.backend != "torch_dense":
            # If image shape passed in 4, then use more efficient dense convolution
            return self.block.dense_forward(x, t=t, skip=skip, z=z, norm=norm)
        else:
            return self.block(x, t=t, skip=skip, z=z, norm=norm)
         

def ts_add(a, b):
    if isinstance(b, SparseTensor):
        feats = a.feats + b.feats
    else:
        feats = a.feats + b
    out = SparseTensor(coords=a.coords, feats=feats, stride=a.stride)
    out.cmaps = a.cmaps
    out.kmaps = a.kmaps
    return out


def ts_div(a, b):
    if isinstance(b, SparseTensor):
        feats = a.feats / b.feats
    else:
        feats = a.feats / b
    out = SparseTensor(coords=a.coords, feats=feats, stride=a.stride)
    out.cmaps = a.cmaps
    out.kmaps = a.kmaps
    return out


class TorchsparseResBlock(nn.Module):
    def __init__(
        self,
        img_size, # 可能是一个元组
        embed_dim,
        kernel_size=7,
        mult=2, # 这个用于扩展全连接网络中间隐藏层神经元的数量
        skip_dim=None,
        time_emb_dim=None,
        epsilon=1e-5,
        z_dim=None,
        depthwise=True, # 表示是否使用深度卷积
    ):
        super().__init__()
   
        # 判断 img_size 是整数还是元组
        if isinstance(img_size, int):
            self.img_height, self.img_width = img_size, img_size
        elif isinstance(img_size, tuple) and len(img_size) == 2:
            self.img_height, self.img_width = img_size
        else:
            raise ValueError("img_size must be either an integer or a tuple of two integers.")

        self.img_size = (self.img_height, self.img_width)
        self.spatial_size = self.img_height * self.img_width  # 计算空间大小
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        self.embed_dim = embed_dim
        self.groups = embed_dim if depthwise else 1

        if skip_dim is not None:
            self.skip_linear = nn.Linear(embed_dim + skip_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.conv = spnn.Conv3d(
            embed_dim,
            embed_dim,
            kernel_size=(1, kernel_size, kernel_size),
            depthwise=depthwise,
            bias=False,
        ) # 这样代码特别重要，别看这个self.conv平平无奇，它却完美体现出什么是稀疏的深度卷积  
        
        self._custom_kaiming_uniform_(self.conv.kernel, a=math.sqrt(5))

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mult),
            nn.GELU(),
            nn.Linear(embed_dim * mult, embed_dim),
        )

        self.time_mlp1, self.time_mlp2, self.z_mlp1, self.z_mlp2 = (
            None,
            None,
            None,
            None,
        )
        if time_emb_dim is not None:
            self.time_mlp1 = nn.Sequential(
                nn.GELU(), nn.Linear(time_emb_dim, embed_dim * 2)
            )
            self.time_mlp2 = nn.Sequential(
                nn.GELU(), nn.Linear(time_emb_dim, embed_dim * 2)
            )
        if z_dim is not None:
            self.z_mlp1 = nn.Sequential(
                nn.Linear(z_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
            )
            self.z_mlp2 = nn.Sequential(
                nn.Linear(z_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
            )

    def forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, torchsparse.SparseTensor)

        # Skip connection
        if skip is not None:
            feats = torch.cat((x.feats, skip.feats), dim=-1)
            feats = self.skip_linear(feats)
            x = convert_to_backend_form_like(
                feats, x, backend="torchsparse", rearrange_x=False
            )

        h = x
        if t is not None or z is not None:
            h = self.modulate(
                h, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1
            )

        h = self.conv(h)
        h = ts_div(h, norm)
        x = ts_add(x, h)

        if t is not None or z is not None:
            h = self.modulate(
                x, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2
            )
        x = ts_add(x, self.mlp(h.feats))

        return x

    def _custom_kaiming_uniform_(self, tensor, a=0, nonlinearity="leaky_relu"):
        fan = self.embed_dim * (self.kernel_size**2)
        # fan 表示权重张量中参与输入的元素数量（即 fan-in）
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    def modulate(self, h, t=None, z=None, norm=None, t_mlp=None, z_mlp=None):
        if isinstance(h, torchsparse.SparseTensor):
            feats = h.feats
        else:
            feats = h
        feats = norm(feats)

        q_sample = feats.size(0) // t.size(0)
        if t is not None:
            t = t_mlp(t)
            t = repeat(t, "b c -> (b l) c", l=q_sample)
            # 重复后的形状是 (b * l, c)，即 (b * q_sample, c)，与 feats 的第一个维度匹配
            t_scale, t_shift = t.chunk(2, dim=-1)
            feats = feats * (1 + t_scale) + t_shift
        if z is not None:
            z_scale = z_mlp(z)
            z_scale = repeat(z_scale, "b c -> (b l) c", l=q_sample)
            feats = feats * (1 + z_scale)
        if isinstance(h, torchsparse.SparseTensor):
            h = convert_to_backend_form_like(
                feats, h, backend="torchsparse", rearrange_x=False
            )
        else:
            h = feats
        return h


    def get_torch_kernel(self, img_size, round_down=True):
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

        # 如果尺寸不同，则调整核的大小
        if (img_height, img_width) != (self_height, self_width):
            # 分别计算高度和宽度的比例
            ratio_height = img_height / self_height
            ratio_width = img_width / self_width
            
            # 计算新的 kernel size
            new_kernel_size_height = self.kernel_size * ratio_height
            new_kernel_size_width = self.kernel_size * ratio_width
            
            # 调整 kernel size
            if round_down:
                new_kernel_size_height = 2 * round((new_kernel_size_height - 1) / 2) + 1
                new_kernel_size_width = 2 * round((new_kernel_size_width - 1) / 2) + 1
            else:
                new_kernel_size_height = math.floor(new_kernel_size_height / 2) * 2 + 1
                new_kernel_size_width = math.floor(new_kernel_size_width / 2) * 2 + 1

            new_kernel_size = max(new_kernel_size_height, new_kernel_size_width, 3)
            
            # 调整卷积核的尺寸
            kernel = rearrange(self.conv.kernel, "(h w) i o -> o i w h", h=self.kernel_size)
            kernel = F.interpolate(kernel, size=new_kernel_size, mode="bilinear")
            return kernel
        else:
            # 如果尺寸相同，直接调整卷积核的形状
            return rearrange(self.conv.kernel, "(h w) i o -> o i w h", h=self.kernel_size)


    def dense_forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, torch.Tensor), "Dense forward expects x to be a torch Tensor"
        assert len(x.shape) == 4, "Dense forward expects x to be 4D: (b, c, h, w)"

        # Skip connection
        batch_size, height, width = x.size(0), x.size(2), x.size(3)
        h = rearrange(x, "b c h w -> (b h w) c")
        if skip is not None:
            skip = rearrange(skip, "b c h w -> (b h w) c")
            h = torch.cat((h, skip), dim=-1)
            h = self.skip_linear(h)
        x = rearrange(h, "(b h w) c -> b c h w", b=batch_size, h=height, w=width)
        
        if t is not None or z is not None:
            h = self.modulate(h, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1)
        h = rearrange(h, "(b h w) c -> b c h w", b=batch_size, h=height, w=width)
        
        # Conv and norm
        kernel = self.get_torch_kernel(height)
        h = F.conv2d(h, kernel, padding=kernel.size(-1)//2, groups=self.groups)
        h = h / norm
        x = x + h 

        # elementwise MLP
        h = rearrange(x, "b c h w -> (b h w) c")
        if t is not None or z is not None:
            h = self.modulate(h, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2)
        h = self.mlp(h)
        h = rearrange(h, "(b h w) c -> b c h w", b=batch_size, h=height, w=width)

        x = x + h

        return x


class TorchDenseConvResBlock(nn.Module):
    def __init__(
        self,
        img_size,
        embed_dim,
        kernel_size=7,
        mult=2,
        skip_dim=None,
        time_emb_dim=None,
        epsilon=1e-5,
        z_dim=None,
        depthwise=True,
    ):
        super().__init__()
        pass

    def forward(self, x, t=None, skip=None, z=None, norm=None):
        pass
        return x




##############################
###### HELPER FUNCTIONS ######
##############################


def convert_to_backend_form(x, sample_lst, img_size, backend="torch_dense"):

    if backend == "torchsparse":
        sparse_indices = sample_lst_to_sparse_indices(sample_lst, img_size, ndims=3)
        x = torchsparse.SparseTensor(
            coords=sparse_indices, feats=rearrange(x, "b l c -> (b l) c")
        )
    elif backend == "torch_dense":
        pass
    else:
        raise Exception("Unrecognised backend.")

    return x


def convert_to_backend_form_like(
    x,
    backend_tensor,
    backend="torch_dense",
    rearrange_x=True,
):
    '''
        这个函数 convert_to_backend_form_like 的主要目的是根据给定的后端类型（backend）将输入的张量 x 转换为与一个已有的 backend_tensor 相似的格式或结构。这个函数与之前介绍的 convert_to_backend_form 函数类似，但它使用现有的 backend_tensor 作为参考，以确保新生成的张量与现有张量具有相似的结构或属性。
    '''
    if backend == "torchsparse":
        x = torchsparse.SparseTensor(
            coords=backend_tensor.coords,
            feats=rearrange(x, "b l c -> (b l) c") if rearrange_x else x,
            stride=backend_tensor.stride,
        )
        x.cmaps = backend_tensor.cmaps
        x.kmaps = backend_tensor.kmaps
    elif backend == "torch_dense":
        pass
    else:
        raise Exception("Unrecognised backend.")

    return x


def get_features_from_backend_form(x, sample_lst, backend="torch_dense"):
    '''
        这个函数 get_features_from_backend_form 的主要作用是从不同后端（backend）格式的张量 x 中提取特征，并将其重新排列成形状为 (batch_size, num_samples, num_channels) 的形式。
    '''

    if backend == "torchsparse":
        return rearrange(x.feats, "(b l) c -> b l c", b=sample_lst.size(0))
    elif backend == "torch_dense":
        pass
    else:
        raise Exception("Unrecognised backend.")


def get_normalising_conv(kernel_size, backend="torch_dense"):
    if backend == "torchsparse":
        assert TORCHSPARSE_AVAILABLE, "torchsparse backend is not detected."
        weight = torch.ones(kernel_size**2, 1, 1) / (kernel_size**2)
        conv = spnn.Conv3d(1, 1, kernel_size=(1, kernel_size, kernel_size), bias=False)
        conv.kernel.data = weight
        conv.kernel.requires_grad_(False)
    elif backend == "torch_dense":
        pass
    else:
        raise Exception("Unrecognised backend.")

    return conv


"""
sample_lst is a tensor of shape (B, L)
which can be used to index flattened 2D images.
This functions converts it to a tensor of shape (BxL, 3)
    indices[:,0] is the number of the item in the batch
    indices[:,1] is the number of the item in the y direction
    indices[:,2] is the number of the item in the x direction
"""
def sample_lst_to_sparse_indices(sample_lst, img_size, ndims=2, dtype=torch.int32):
    """
    函数的目的是将一个包含格点索引的列表转换为稀疏索引格式
    """
    # 判断 img_size 是整数还是元组
    if isinstance(img_size, int):
        img_height, img_width = img_size, img_size
    elif isinstance(img_size, tuple) and len(img_size) == 2:
        img_height, img_width = img_size
    else:
        raise ValueError("img_size must be either an integer or a tuple of two integers.")

    # number of the item in the batch - (B,)
    batch_idx = torch.arange(
        sample_lst.size(0), device=sample_lst.device, dtype=dtype
    )
    batch_idx = repeat(batch_idx, "b -> b l", l=sample_lst.size(1))

    # pixel number in vertical direction - (B,L)
    sample_lst_h = sample_lst.div(img_width, rounding_mode="trunc").to(dtype)  # 纵向索引
    # pixel number in horizontal direction - (B,L)
    sample_lst_w = (sample_lst % img_width).to(dtype)  # 横向索引

    if ndims == 2:
        indices = torch.stack([batch_idx, sample_lst_h, sample_lst_w], dim=2)
        indices = rearrange(indices, "b l three -> (b l) three")
    else:
        zeros = torch.zeros_like(sample_lst_h)
        indices = torch.stack([zeros, sample_lst_h, sample_lst_w, batch_idx], dim=2)
        indices = rearrange(indices, "b l four -> (b l) four")

    return indices


def calculate_norm(conv, backend_tensor, backend="torchsparse"):
    if backend == "torchsparse":
        device, dtype = backend_tensor.feats.device, backend_tensor.feats.dtype
        ones = torch.ones(backend_tensor.feats.size(0), 1, device=device, dtype=dtype)
        mask = torchsparse.SparseTensor(
                coords=backend_tensor.coords,
                feats=ones,
                stride=backend_tensor.stride
            )
        mask.cmaps = backend_tensor.cmaps
        mask.kmaps = backend_tensor.kmaps
        norm = conv(mask) 
    elif backend == "torch_dense":
        pass
    else:
        raise Exception("Unrecognised backend.")
    
    return norm
