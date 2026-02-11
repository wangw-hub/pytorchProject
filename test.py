# -*- coding: utf-8 -*-
"""
@File    : test.py
@Time    : 2026/2/11 下午3:05 星期三
@Author  : wangw
@Email   : wangw_heart@163.com
@Description: 
"""

import torch
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
import torch
print(f"PyTorch Version: {torch.__version__}")  # 应该是 2.x.x (没有 dev 字样)
print(f"CUDA Version: {torch.version.cuda}")    # 必须显示 12.8
print(f"GPU Support: {torch.cuda.is_available()}")

# 终极测试：创建张量
try:
    x = torch.rand(5, 3).cuda()
    print("Success! RTX 5060 is working on Stable PyTorch.")
    print(x)
except Exception as e:
    print(f"Error: {e}")

import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

print(f"Pytorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")
device = "cuda" if torch.cuda.is_available() else 'cpu'