import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
import requests

# 测试 PyTorch 和 torchvision 的版本
print(f"PyTorch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")

# 测试是否可以使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
print("flash-attn success!")










