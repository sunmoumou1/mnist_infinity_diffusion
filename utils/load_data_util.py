# 重新复习

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
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(
        dataset, val_indices
    )
    return train_dataset, val_dataset


class SpecificHumidityDataset(Dataset):
    def __init__(self, data_file):
        # Load preprocessed data from the .npy file
        self.data = np.load(data_file)
        
        # Check for values less than 0, count them, and replace with 0
        negative_count = np.sum(self.data < 0)
        if negative_count > 0:
            print(f"Found {negative_count} values less than 0, replacing them with 0.")
            self.data[self.data < 0] = 0
            
        self.data = 100 * np.expand_dims(self.data, axis=1)  # Insert a new dimension at axis 1 to get (B, 1, H, W)
        
        # Calculate mean and standard deviation for standardization
        self.mean = self.data.mean()
        self.std = self.data.std()
        print(f"SpecificHumidityDataset - Mean: {self.mean}, Std: {self.std}")

        # Avoid division by zero if std is extremely small
        if self.std < 1e-6:
            self.std = 1e-6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Fetches the idx-th humidity field and standardizes it"""
        humidity_field = self.data[idx]  # Shape: (1, longitude, latitude)
        # Standardize the data
        humidity_field = (humidity_field - self.mean) / self.std
        return torch.tensor(humidity_field, dtype=torch.float32)

    
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
    start_year = H.data.start_year
    end_year = H.data.end_year

    # 创建完整数据集
    dataset = SpecificHumidityDataset(data_dir)
    mean = dataset.mean
    std = dataset.std

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

    return train_dataloader, val_dataloader, mean, std


