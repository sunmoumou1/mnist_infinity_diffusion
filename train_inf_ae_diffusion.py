# 重新复习

import torch
import numpy as np
from scipy.stats.qmc import Halton
# scipy.stats.qmc 是 SciPy 库中的一个模块，提供了一些用于准蒙特卡罗（Quasi-Monte Carlo）方法的工具。
# Halton 是其中的一种序列生成方法，用于生成多维均匀分布的准随机数。这些数通常用于数值积分、优化等问题，尤其是在高维空间中进行均匀采样时。

import wandb # type: ignore
from absl import app
from absl import flags
# absl 是 Abseil-Py 的缩写，是 Google 开源的 Python 库，提供了一些增强的应用程序开发工具。
# app 模块用于管理应用程序的主循环，通常用来定义 main 函数并处理命令行参数。
# flags 模块用于定义命令行标志（flags），这些标志可以让你在运行时配置程序的行为，类似于解析命令行参数的功能。

from ml_collections.config_flags import config_flags # type: ignore
import time
import os

from models import SparseUNet, SparseEncoder
from utils import (
    get_data_loader,
    flatten_collection,
    optim_warmup,
    plot_images,
    update_ema,
    create_named_schedule_sampler,
    LossAwareSampler,
    get_named_beta_schedule
)
import diffusion as gd
from diffusion import GaussianDiffusion

# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
# "config" 是参数的名字，你在命令行中会用 --config 来指定配置文件。
# None 是参数的默认值，意味着如果没有指定，config 默认是 None。
# "Training configuration." 是对这个参数的描述，当用户请求帮助信息时会显示出来。
# lock_config=True 表示在加载配置文件后不允许修改配置的内容，这可以防止在运行过程中意外地改变配置

flags.mark_flags_as_required(["config"])
# 这行代码将 config 参数标记为必需的。意味着如果在命令行中没有指定 --config 参数，程序会抛出错误并停止运行。
# 这通常用于确保用户在运行程序时提供必要的配置文件。

# Torch options
torch.backends.cuda.matmul.allow_tf32 = True
# TF32 是一种在 Ampere 架构的 NVIDIA GPU 上引入的浮点格式，能在计算速度和精度之间取得一个平衡。启用 TF32 可以显著提升深度学习模型的训练速度，同时保持足够的数值精度。
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device = torch.device("cuda")


def train(
    H,
    model,
    ema_model,
    train_loader,
    optim,
    diffusion,
    schedule_sampler,
    checkpoint_path="",
    global_step=0,
):    
    halton = Halton(2)
    scaler = torch.cuda.amp.GradScaler()

    mean_loss = 0
    mean_step_time = 0
    mean_total_norm = 0
    skip = 0

    img_height, img_width = H.data.img_size[0]._value, H.data.img_size[1]._value  # 获取新的图像尺寸

    while True:
        for x in train_loader:
            if isinstance(x, tuple) or isinstance(x, list):
                x = x[0]
            start_time = time.time()
            # print("data loader中x的形状为：", x.shape) # data loader中x的形状为： torch.Size([30, 1, 240, 121])
            
            if global_step < H.optimizer.warmup_steps:
                optim_warmup(
                    global_step,
                    optim,
                    H.optimizer.learning_rate,
                    H.optimizer.warmup_steps,
                )

            global_step += 1
            x = x.to(device, non_blocking=True)
            # non_blocking=True: 允许数据传输操作（从 CPU 到 GPU 或从 GPU 到 GPU）在可能的情况下异步进行，而不是等待传输完成后再继续执行后续的代码。这意味着数据传输可以与其他计算任务并行，从而可能提高整体训练效率。
            x = x * 2 - 1

            t, weights = schedule_sampler.sample(x.size(0), device)
            # schedule_sampler 是一个用于在训练过程中采样时间步的工具，目的是选择在每个批次中需要计算损失的时间步 t。采样的时间步数是 x.size(0)，即当前批次中的样本数。
            # weights 则对应于每个样本在该时间步上的权重，可能用于加权损失。

            if H.mc_integral.type == "uniform":
                # 对于非正方形图像，使用宽高的乘积生成随机样本
                sample_lst = torch.stack(
                    [
                        torch.from_numpy(
                            np.random.choice(
                                img_height * img_width,
                                H.mc_integral.q_sample,
                                replace=False,
                            )
                        )
                        for _ in range(H.train.batch_size)
                    ]
                ).to(device)
            elif H.mc_integral.type == "halton":
                # 生成 Halton 样本并按新的尺寸映射
                sample_lst = torch.stack(
                    [
                        torch.from_numpy(
                            (
                                halton.random(H.mc_integral.q_sample)
                                * [img_height, img_width]
                            ).astype(np.int64)
                        )
                        for _ in range(H.train.batch_size)
                    ]
                ).to(device)
                sample_lst = (
                    sample_lst[:, :, 0] * img_width + sample_lst[:, :, 1]
                )  # 使用行列信息生成索引
            else:
                raise Exception("Unknown Monte Carlo Integral type")

            with torch.cuda.amp.autocast(enabled=H.train.amp):
                losses = diffusion.training_losses(
                    model, x, t, sample_lst=sample_lst
                )

                if H.diffusion.multiscale_loss:
                    loss = (losses["multiscale_loss"] * weights).mean()
                else:
                    loss = (losses["loss"] * weights).mean()

            optim.zero_grad()
            if H.train.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                # 在梯度计算完成后，调用 scaler.unscale_(optim) 对梯度进行反缩放，使其恢复到正常范围。这是为了在进行梯度裁剪之前，确保梯度的数值在合适的范围内。
                model_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                ) # model_total_norm 是裁剪前所有参数梯度的总范数。

                if (
                    H.optimizer.gradient_skip
                    and model_total_norm >= H.optimizer.gradient_skip_threshold
                ):
                    # 通过检查 model_total_norm 和 encoder_total_norm 是否超过设定的阈值 H.optimizer.gradient_skip_threshold 来决定是否跳过当前的参数更新。如果超过阈值，认为梯度异常大，可能会导致不稳定的更新，因此跳过该步更新，并将 skip 计数器加 1。
                    skip += 1
                    
                    scaler.update()
                else:
                    scaler.step(optim)
                    # 调用 scaler.step(optim) 来更新模型的参数。此时 scaler 会根据缩放后的梯度来更新参数。
                    scaler.update()
                    # 更新 scaler 的缩放因子。scaler.update() 会根据梯度是否发生溢出或下溢来动态调整缩放因子，以确保接下来的训练步骤稳定进行。
            else:
                loss.backward()
                model_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                if (
                    H.optimizer.gradient_skip
                    and model_total_norm >= H.optimizer.gradient_skip_threshold
                ):
                    skip += 1
                else:
                    optim.step()

            if isinstance(schedule_sampler, LossAwareSampler):
                schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            if global_step % H.train.ema_update_every == 0:
                update_ema(model, ema_model, H.train.ema_decay)

            mean_loss += loss.item()
            mean_step_time += time.time() - start_time
            mean_total_norm += model_total_norm.item()

            wandb_dict = dict()
            # wandb_dict 是一个用于记录训练过程中各种指标的字典，将其初始化为空字典。
            if global_step % H.train.plot_graph_steps == 0 and global_step > 0:
                norm = H.train.plot_graph_steps
                print(
                    f"Step: {global_step}, Loss {mean_loss / norm:.5f}, Step Time: {mean_step_time / norm:.5f}, Skip: {skip / norm:.5f}, Gradient Norm: {mean_total_norm / norm:.5f}"
                )
                wandb_dict |= {
                    "Step Time": mean_step_time / norm,
                    "Loss": mean_loss / norm,
                    "Skip": skip / norm,
                    "Gradient Norm": mean_total_norm / norm,
                }
                mean_loss = 0
                mean_step_time = 0
                skip = 0
                mean_total_norm = 0


            # 创建保存图片的子文件夹
            output_folder = "sampling_image"
            os.makedirs(output_folder, exist_ok=True)

            if global_step % H.train.plot_samples_steps == 0 and global_step > 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=H.train.amp):
                        # 定义图像尺寸的不同缩放比例
                        sizes = [
                            (img_height, img_width),           # 原始尺寸
                            # (2 * img_height, 2 * img_width),   # 双倍尺寸
                            # (img_height // 2, img_width // 2)  # 一半尺寸
                        ]

                        # 依次生成不同尺寸的图像
                        for size in sizes:
                            h, w = size

                            samples, _ = diffusion.p_sample_loop(
                                ema_model,
                                (
                                    H.train.sample_size,
                                    H.data.channels,
                                    h,
                                    w,
                                ),
                                clip_denoised = True,
                                progress=True,
                                # model_kwargs=dict(z=encoding),
                                return_all=False,  # 表示只返回最终生成的样本图像，而不返回扩散过程中的所有中间状态
                            ) # 要特别记得自己最开始实验的时候并没有设置clip_denoised = False
                            
                            samples = (samples + 1) / 2

                            # 将张量的值剪裁到 0 到 1 之间, 不确定这行代码到底需要不需要
                            samples = torch.clamp(samples, min=0.0, max=1.0)

                            # 使用生成的图像尺寸作为标题后缀，以区分不同的图像
                            title_suffix = f"{h}x{w}"
                            wandb_dict |= plot_images(H, samples, title=f"samples_{title_suffix}")

                            if H.diffusion.model_mean_type == "mollified_epsilon":
                                deblurred_samples = diffusion.mollifier.undo_wiener(samples)
                                wandb_dict |= plot_images(
                                    H,
                                    deblurred_samples,
                                    title=f"deblurred_samples_{title_suffix}",
                                )

                            # 定义保存为 .npy 文件的函数
                            def save_to_npy(array, filename):
                                # 将 tensor 转换为 numpy 数组并保存为 .npy 文件
                                np.save(os.path.join(output_folder, filename), array.cpu().numpy())

                            save_to_npy(
                                samples,
                                filename=f"samples_{global_step}_{h}x{w}.npy"
                            )

                            if H.diffusion.model_mean_type == "mollified_epsilon":
                                save_to_npy(
                                    deblurred_samples,
                                    filename=f"deblurred_samples_{global_step}_{h}x{w}.npy"
                                )
                                
    
            if wandb_dict:
                wandb.log(wandb_dict, step=global_step)

            if global_step % H.train.checkpoint_steps == 0 and global_step > 0:
                # 定义包含 global_step 的 checkpoint 文件路径
                checkpoint_path_new = os.path.join(checkpoint_path, f"checkpoint_step_{global_step}.pkl")
                
                # 保存模型和相关状态
                torch.save(
                    {
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "model_ema_state_dict": ema_model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                    },
                    checkpoint_path_new,
                )

def main(argv):
    H = FLAGS.config
    # 从命令行参数或配置文件中读取配置，并将其赋值给 H，它包含了训练过程中所有需要的参数和设置。
    train_kwargs = {}

    # wandb can be disabled by passing in --config.run.wandb_mode=disabled
    wandb.init(
        project=H.run.name,
        config=flatten_collection(H), # 将配置 H 扁平化后传递给 W&B，以便记录实验配置。
        save_code=True,
        # 在这段代码中，save_code=True 是用来告诉 Weights & Biases (W&B) 平台自动保存当前运行代码的副本。这是 W&B 的一个功能，能够帮助你在每次运行实验时，将对应的代码保存到 W&B 平台上，这样你可以在以后查看或分享时，确保你有准确记录的代码版本。
        dir=H.run.wandb_dir,
        mode=H.run.wandb_mode, # online
    )
    
    img_height, img_width = H.data.img_size[0]._value, H.data.img_size[1]._value

    model = SparseUNet(
        channels=H.data.channels,
        nf=H.model.nf,
        time_emb_dim=H.model.time_emb_dim,
        img_size=(img_height, img_width),
        num_conv_blocks=H.model.num_conv_blocks,
        knn_neighbours=H.model.knn_neighbours,
        uno_res=H.model.uno_res,
        uno_mults=H.model.uno_mults,
        z_dim=H.model.z_dim,
        conv_type=H.model.uno_conv_type,
        depthwise_sparse=H.model.depthwise_sparse,
        kernel_size=H.model.kernel_size,
        backend=H.model.backend,
        blocks_per_level=H.model.uno_blocks_per_level,
        attn_res=H.model.uno_attn_resolutions,
        dropout_res=H.model.uno_dropout_from_resolution,
        dropout=H.model.uno_dropout,
        uno_base_nf=H.model.uno_base_channels,
    )

    ema_model = SparseUNet(
        channels=H.data.channels,
        nf=H.model.nf,
        time_emb_dim=H.model.time_emb_dim,
        img_size=(img_height, img_width),
        num_conv_blocks=H.model.num_conv_blocks,
        knn_neighbours=H.model.knn_neighbours,
        uno_res=H.model.uno_res,
        uno_mults=H.model.uno_mults,
        z_dim=H.model.z_dim,
        conv_type=H.model.uno_conv_type,
        depthwise_sparse=H.model.depthwise_sparse,
        kernel_size=H.model.kernel_size,
        backend=H.model.backend,
        blocks_per_level=H.model.uno_blocks_per_level,
        attn_res=H.model.uno_attn_resolutions,
        dropout_res=H.model.uno_dropout_from_resolution,
        dropout=H.model.uno_dropout,
        uno_base_nf=H.model.uno_base_channels,
    )

    print(
        f"Number of SparseUNet parameters: {sum(p.numel() for p in model.parameters())}"
    )
    
    if H.run.experiment != "":
        checkpoint_path = f"checkpoints/{H.run.experiment}/"
    else:
        checkpoint_path = "checkpoints/"
        
    os.makedirs(checkpoint_path, exist_ok=True)
    train_kwargs["checkpoint_path"] = checkpoint_path

    model = model.to(device)
    ema_model = ema_model.to(device)
    train_loader, _ = get_data_loader(H)
    optim = torch.optim.Adam(
        list(model.parameters()),
        lr=H.optimizer.learning_rate,
        betas=(H.optimizer.adam_beta1, H.optimizer.adam_beta2),
    )

    if H.train.load_checkpoint and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        print(f"Loading Model from step {state_dict['global_step']}")
        train_kwargs["global_step"] = state_dict["global_step"]
        model.load_state_dict(state_dict["model_state_dict"], strict=False)
        # strict=False: 允许在加载状态时略过模型中不存在的部分（可能是在调整模型结构时引入的变化）。
        ema_model.load_state_dict(state_dict["model_ema_state_dict"], strict=False)
        try:
            optim.load_state_dict(state_dict["optimizer_state_dict"])
        except ValueError:
            print("Failed to load optim params.")

    betas = get_named_beta_schedule(
        H.diffusion.noise_schedule, H.diffusion.steps
    )

    if H.diffusion.model_mean_type == "epsilon":
        model_mean_type = gd.ModelMeanType.EPSILON
    elif H.diffusion.model_mean_type == "xstart":
        model_mean_type = gd.ModelMeanType.START_X
    elif H.diffusion.model_mean_type == "mollified_epsilon":
        assert (
            H.diffusion.gaussian_filter_std > 0
        ), "Error: Predicting mollified_epsilon but gaussian_filter_std == 0."
        model_mean_type = gd.ModelMeanType.MOLLIFIED_EPSILON
    else:
        raise Exception(
            "Unknown model mean type. Expected value in [epsilon, mollified_epsilon, xstart]"
        )
        

    model_var_type = (
        gd.ModelVarType.FIXED_LARGE
        if not H.model.sigma_small
        else gd.ModelVarType.FIXED_SMALL
    )
    
    if H.diffusion.loss_type == "mse":
        loss_type = gd.LossType.MSE 
    elif H.diffusion.loss_type == "L1":
        loss_type = gd.LossType.L1
    else:
        raise Exception(
            "H.diffusion.loss_type must be in [mse, L1]"
        )
        
    
    diffusion = GaussianDiffusion(
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        H.diffusion.gaussian_filter_std,
        (img_height, img_width),
        rescale_timesteps=False,
        multiscale_loss=H.diffusion.multiscale_loss, # False
        multiscale_max_img_size=H.diffusion.multiscale_max_img_size,
        mollifier_type=H.diffusion.mollifier_type, # dct
        stochastic_encoding=H.model.stochastic_encoding, # False
    ).to(device)

    schedule_sampler = create_named_schedule_sampler(
        H.diffusion.schedule_sampler, diffusion
    ) # uniform

    train(
        H,
        model,
        ema_model,
        train_loader,
        optim,
        diffusion,
        schedule_sampler,
        **train_kwargs,
    )


if __name__ == "__main__":
    app.run(main)
