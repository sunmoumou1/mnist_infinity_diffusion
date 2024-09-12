# 重新复习

import enum
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm  # type: ignore 导入这个库主要就是用tgm.image.get_gaussian_kernel2d(kernel_size, std)
import functools 
# import torch_dct # type: ignore
import os
import sys

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前目录添加到 Python 解释器的搜索路径
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dct_util import *



def mean_flat(x):
    """
        该函数计算张量 x 在除了第一个维度（通常是批次维度）之外的所有维度上的平均值。
    """
    return x.mean(dim=list(range(1, len(x.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
        Compute the KL divergence between two gaussians.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
        A fast approximation of the cumulative distribution function of the standard normal.
    """
    return 0.5 * (
        1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a given image.
    :param x: the target images. It is assumed that this was uint8 values, rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities.
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
        ),
    )
    assert log_probs.shape == x.shape
    return log_probs


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
        Get a pre-defined beta schedule for the given name.
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "const0.008":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.008] * num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")



class ModelMeanType(enum.Enum):
    """
        Which type of output the model predicts.
    """

    # enum.auto() 是 Python enum 模块中的一个辅助函数，用来自动为枚举成员赋值。使用 enum.auto() 时，枚举成员的值会自动递增，从 1 开始，依次为 2, 3, 4，等等。
    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    MOLLIFIED_EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    L1 = enum.auto()
    RESCALED_MSE = enum.auto()  # use raw MSE loss (with RESCALED_KL when learning variances)


def blur(kernel_size, std):
    # 注意这个dims指的是高斯卷积核的尺寸
    return tgm.image.get_gaussian_kernel2d(kernel_size, std)


def get_conv(kernel_size, std, mode="reflect", channels=3):
    '''
        这个函数就是用来mollifier的, 但是我们的实验没有采用这种方式, 而且这种方式其实也没有办法来真正的使用
    '''
    kernel = blur(kernel_size, std)
    conv = nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        padding=int((kernel_size[0] - 1) / 2),
        padding_mode=mode,
        bias=False,
        groups=channels,
    )
    with torch.no_grad():
        kernel = torch.unsqueeze(kernel, 0)
        kernel = torch.unsqueeze(kernel, 0)
        kernel = kernel.repeat(channels, 1, 1, 1)
        conv.weight = nn.Parameter(kernel)

    return conv


class DCTGaussianBlur(nn.Module):
    def __init__(self, img_size, std, inv_snr=0.05):
        super().__init__()
        self.inv_snr = inv_snr
        H, W = img_size  # img_size 应为一个元组 (H, W)
        
        # 分别创建高度和宽度方向的 DCT 和 IDCT 层
        self.dct_h = LinearDCT(H, 'dct')
        self.dct_w = LinearDCT(W, 'dct')
        self.idct_h = LinearDCT(H, 'idct')
        self.idct_w = LinearDCT(W, 'idct')
        
        # 调整高斯核的生成方式，支持 (H, W)
        gaussian = self.gaussian_quadrant(
            [H, W], [H / (np.pi * std), W / (np.pi * std)]
        ).float()
        gaussian_conj = torch.conj(gaussian)  # conj 表示共轭
        
        # 将高斯核注册为模型的缓冲区
        self.register_buffer("gaussian", gaussian)
        self.register_buffer("gaussian_conj", gaussian_conj)

    def gaussian_quadrant(self, shape, standards):
        # 生成高斯核，对应 (H, W)
        return torch.from_numpy(
            functools.reduce(
                np.multiply,
                (
                    np.exp(-(dx**2) / (2 * sd**2))
                    for sd, dx in zip(standards, np.indices(shape))
                ),
            )
        )

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        # 对输入进行 DCT 和高斯滤波
        x = apply_linear_2d(x, self.dct_h, self.dct_w)  # 分别在 H 和 W 方向上进行 DCT
        x = x * self.gaussian.to(x.dtype)  # 应用高斯核
        x = apply_linear_2d(x, self.idct_h, self.idct_w)  # 分别在 H 和 W 方向上进行 IDCT
        return x

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def undo_wiener(self, x):
        # 逆 Wiener 去卷积
        x = apply_linear_2d(x, self.dct_h, self.dct_w)  # 分别在 H 和 W 方向上进行 DCT
        x = (
            x
            * self.gaussian_conj.to(x.dtype)
            / (
                self.gaussian.to(x.dtype) * self.gaussian_conj.to(x.dtype)
                + self.inv_snr**2
            )
        )
        x = apply_linear_2d(x, self.idct_h, self.idct_w)  # 分别在 H 和 W 方向上进行 IDCT
        return x

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)