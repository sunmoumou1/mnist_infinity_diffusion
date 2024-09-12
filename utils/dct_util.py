# 重新复习

import torch
import torch.nn as nn
import numpy as np


class LinearDCT(nn.Linear):
    """实现DCT作为线性层，可以对不等高宽的二维场进行处理。
    :param in_features: 输入的特征维度
    :param type: 使用的DCT类型，'dct', 'idct' 等
    :param norm: 归一化参数
    """
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # 初始化权重矩阵为DCT或IDCT矩阵
        I = torch.eye(self.N)
        if self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False  # 不更新权重


def apply_linear_2d(x, linear_layer_h, linear_layer_w):
    """
    使用LinearDCT层对最后两个维度进行2D DCT。
    :param x: 输入信号，形状为 (B, C, H, W)
    :param linear_layer_h: 高度方向的LinearDCT层
    :param linear_layer_w: 宽度方向的LinearDCT层
    :return: 经过DCT的结果
    """

    X1 = linear_layer_w(x)  
    X2 = linear_layer_h(X1.transpose(-1, -2)) 
    return X2.transpose(-1, -2)


def dct(x, norm=None):
    """
    一维DCT，类型II
    :param x: 输入信号
    :param norm: 归一化选项
    :return: DCT-II 变换结果
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    一维逆DCT，类型III
    :param X: 输入信号
    :param norm: 归一化选项
    :return: 逆DCT-II 变换结果
    """
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    # 注意这里的t表示temporal
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


