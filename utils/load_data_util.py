# 重新复习
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataset import Subset
import torchvision
from torchvision import transforms
import numpy as np
import os
import xarray as xr # type: ignore

class EnumerateDataset(Dataset):
    '''
        EnumerateDataset 是一个简单的 PyTorch 数据集类，用于包装另一个数据集。
        它的主要作用是在获取数据时返回数据的索引以及对应的数据条目。
    '''
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return idx, self.dataset[idx]


def train_val_split(dataset, train_val_ratio):
    indices = list(range(len(dataset)))
    split_index = int(len(dataset) * train_val_ratio)
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
    
    return train_dataset, val_dataset


class MNISTImageDataset(Dataset):
    def __init__(self, root_dir):
        """
        初始化自定义数据集
        :param root_dir: 包含图像的根目录
        """
        self.root_dir = root_dir
        # 获取文件夹中所有图片文件路径
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.jpg')]
        # 定义图像处理转换，调整为24x24大小并转换为张量
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),  # 调整图像大小为24x24
            transforms.ToTensor(),        # 将图像转换为张量，并自动将值归一化到[0, 1]
        ])

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取指定索引的图像及其标签
        :param idx: 图片索引
        :return: 归一化的24x24矩阵
        """
        # 获取当前索引的图片路径
        img_path = self.image_paths[idx]
        # 打开图像
        image = Image.open(img_path).convert('L')  # 转换为灰度图像
        # 应用转换
        image = self.transform(image)
        return image
    
def get_data_loader(
    H,
    enumerate_data=False,
    drop_last=True,
    shuffle=True,
    train_val_split_ratio=0.95,
):
    """
    创建 DataLoader 用于加载指定年份范围内的湿度数据，并划分为训练集和验证集。

    :param H: 包含配置信息的对象或字典
    :param enumerate_data: 如果为 True，则使用 EnumerateDataset 类包装数据集，以便在获取数据时也返回数据的索引
    :param drop_last: 决定是否在批处理中丢弃最后一个不完整的批次
    :param shuffle: 是否对数据进行洗牌
    :param train_val_split_ratio: 训练集和验证集的划分比例
    :return: 训练集和验证集的 DataLoader 对象
    """
    # 获取湿度数据的目录和年份范围
    data_dir = H.data.root_dir
    
    # 创建完整数据集
    dataset = MNISTImageDataset(data_dir)

    # 划分训练集和验证集
    train_dataset, val_dataset = train_val_split(dataset, train_val_split_ratio)

    # 创建训练集和验证集的 DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=H.train.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=6,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=H.train.batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=2,
    )

    return train_dataloader, val_dataloader


