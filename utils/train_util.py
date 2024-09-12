# 重新复习

import torch
from ml_collections import ConfigDict  # type: ignore
import wandb # type: ignore
import os
import sys
from tqdm import tqdm

# 将 utils 目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data_util import get_data_loader


def plot_images(H, x, title=""):
    x = wandb.Image(x, caption=title)
    return {title: x}


def flatten_collection(d, parent_key="", sep="_"):
    '''
        flatten_collection 函数的作用是将一个嵌套的字典（例如具有多层嵌套结构的配置字典）转换为一个扁平化的字典，其中嵌套结构的键通过指定的分隔符（sep）连接成一个新的键。

        d: 输入的嵌套字典。
        parent_key: 用于递归时传递的父键（初始为空字符串）。
        sep: 用于连接嵌套键的分隔符，默认是下划线 _。
        函数会递归地遍历字典 d 的所有键值对。如果值 v 是一个嵌套的字典（ConfigDict 类型），则会递归调用 flatten_collection 并将结果添加到 items 列表中。否则，将当前键值对直接添加到 items 中。

        最终，函数返回一个扁平化的字典，其中所有嵌套的键都被展开并通过分隔符连接。
    '''
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, ConfigDict):
            items.extend(flatten_collection(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
# 注意区别items.extend()和items.append()
# 作用: extend() 方法用于将一个可迭代对象（如列表、元组、集合等）的所有元素逐一添加到列表的末尾。
# 行为: 当你使用 extend() 时，传入的可迭代对象的每个元素都会被依次添加到列表中，而不是作为一个单独的元素添加。




def load_latents(H, Encoder=None):

    latent_path = f"checkpoints/{H.run.experiment}/latents.pkl"
    if os.path.exists(latent_path):
        return torch.load(latent_path)
    else:
        print("Latent file not found so generating...")
        os.makedirs(latent_path.replace("latents.pkl", ""), exist_ok=True)
        decoder_checkpoint_path = (
            f"checkpoints/{H.decoder.run.experiment}/checkpoint.pkl"
        )
        decoder_state_dict = torch.load(decoder_checkpoint_path, map_location="cpu")
        encoder = Encoder(
            out_channels=H.decoder.model.z_dim,
            channels=H.decoder.data.channels,
            nf=H.decoder.model.nf,
            img_size=H.decoder.data.img_size,
            num_conv_blocks=H.decoder.model.num_conv_blocks,
            knn_neighbours=H.decoder.model.knn_neighbours,
            uno_res=H.decoder.model.uno_res,
            uno_mults=H.decoder.model.uno_mults,
            conv_type=H.decoder.model.uno_conv_type,
            depthwise_sparse=H.decoder.model.depthwise_sparse,
            kernel_size=H.decoder.model.kernel_size,
            backend=H.decoder.model.backend,
            blocks_per_level=H.decoder.model.uno_blocks_per_level,
            attn_res=H.decoder.model.uno_attn_resolutions,
            dropout_res=H.decoder.model.uno_dropout_from_resolution,
            dropout=H.decoder.model.uno_dropout,
            uno_base_nf=H.decoder.model.uno_base_channels,
            stochastic=H.decoder.model.stochastic_encoding,
        )
        encoder.load_state_dict(decoder_state_dict["encoder_state_dict"])
        encoder = encoder.to("cuda")

        dataloader, val_dataloader = get_data_loader(
            H.decoder, flip_p=0.0, drop_last=False, shuffle=False
        )
        flipped_dataloader, val_flipped_dataloader = get_data_loader(
            H.decoder, flip_p=1.0, drop_last=False, shuffle=False
        )

        latents = []
        mus = []
        logvars = []
        for x, x_flip in tqdm(
            zip(dataloader, flipped_dataloader), total=len(dataloader)
        ):
            if isinstance(x, tuple) or isinstance(x, list):
                x = x[0]
            if isinstance(x_flip, tuple) or isinstance(x_flip, list):
                x_flip = x_flip[0]
            x, x_flip = x.to("cuda"), x_flip.to("cuda")

            x = torch.cat((x, x_flip), dim=0)
            x = x * 2 - 1
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=H.decoder.train.amp):
                    if H.decoder.model.stochastic_encoding:
                        z, mu, logvar = encoder(x)
                        latents.append(z.cpu())
                        mus.append(mu.cpu())
                        logvars.append(logvar.cpu())
                    else:
                        z = encoder(x)
                        latents.append(z.cpu())
        latents = torch.cat(latents, dim=0)
        if H.decoder.model.stochastic_encoding:
            mus = torch.cat(mus, dim=0)
            logvars = torch.cat(logvars, dim=0)

        mean = latents.mean(
            dim=0, keepdim=True
        )  # BUG: Results in NaNs even though they should be 0.
        mean = torch.zeros_like(mean) # 注意此时mean是一个一维的数组，长度为out_channels=H.decoder.model.z_dim
        std = latents.std(dim=0, keepdim=True) # 注意此时std是一个一维的数组，长度为out_channels=H.decoder.model.z_dim
        print(f"Mean: {mean} ({mean.mean()}), std: {std} ({std.mean()})")
        latents = (latents - mean) / std

        # Validaton set
        val_latents = []
        val_mus = []
        val_logvars = []
        for x, x_flip in tqdm(
            zip(val_dataloader, val_flipped_dataloader), total=len(val_dataloader)
        ):
            if isinstance(x, tuple) or isinstance(x, list):
                x = x[0]
            if isinstance(x_flip, tuple) or isinstance(x_flip, list):
                x_flip = x_flip[0]
            x, x_flip = x.to("cuda"), x_flip.to("cuda")

            x = torch.cat((x, x_flip), dim=0)
            x = x * 2 - 1
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=H.decoder.train.amp):
                    if H.decoder.model.stochastic_encoding:
                        z, mu, logvar = encoder(x)
                        val_latents.append(z.cpu())
                        val_mus.append(mu.cpu())
                        val_logvars.append(logvar.cpu())
                    else:
                        z = encoder(x)
                        val_latents.append(z.cpu())
        val_latents = torch.cat(val_latents, dim=0)
        if H.decoder.model.stochastic_encoding:
            val_mus = torch.cat(val_mus, dim=0)
            val_logvars = torch.cat(val_logvars, dim=0)
        val_latents = (val_latents - mean) / std

        if H.decoder.model.stochastic_encoding:
            torch.save((mus, logvars, val_mus, val_logvars, mean, std), latent_path)
            return (mus, logvars, val_mus, val_logvars, mean, std)
        else:
            torch.save((latents, val_latents, mean, std), latent_path)
            return (latents, val_latents, mean, std)
        # 注意此时返回的latents是样本个数×H.decoder.model.z_dim
