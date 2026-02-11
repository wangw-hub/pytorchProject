# -*- coding: utf-8 -*-
"""
@File    : ResNet-101.py
@Time    : 2026/2/11 下午4:36 星期三
@Author  : wangw
@Email   : wangw_heart@163.com
@Description: 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import sys
# 解决你之前的 OMP Error #15 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # ================= ⚙️ 狂暴模式配置区 =================
    # 针对 8GB 显存，128 batch + 224分辨率通常能压榨 90% 以上显存
    BATCH_SIZE = 64
    EPOCHS = 5  # 根据速度可调，ResNet101 较慢，5轮通常足够 10 分钟
    NUM_WORKERS = 8 # 充分利用你蛟龙 16 Pro 的多核 CPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 正在启动显卡: {torch.cuda.get_device_name(0)}")
    print(f"📦 环境检查: PyTorch {torch.__version__} | CUDA {torch.version.cuda}")

    # 1. 数据增强（增加 CPU/GPU 交互负担，抹平波动）
    transform = transforms.Compose([
        transforms.Resize(224), # 增加到 ImageNet 标准尺寸，计算量剧增
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.446), (0.202, 0.199, 0.201)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # 关键：开启 persistent_workers 和 pin_memory 解决阶段性停顿
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             persistent_workers=True)

    # 2. 使用更深层的 ResNet-101（计算密度远超 ResNet50）
    model = torchvision.models.resnet101(weights=None, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # 开启混合精度训练，充分发挥 RTX 5060 的 Tensor Core 性能
    scaler = torch.amp.GradScaler('cuda')

    print("\n[INFO] 压力测试开始，请观察任务管理器 GPU 利用率...")
    print("-" * 50)

    model.train()
    start_time = time.time()

    for epoch in range(EPOCHS):
        e_start = time.time()
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 20 == 0:
                # 打印显存占用，监控是否达到满载
                mem = torch.cuda.memory_reserved(0) / 1024**3
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Step [{i}/{len(trainloader)}] | 显存占用: {mem:.2f}GB", end='\r',flush=True)

        print(f"\n✅ Epoch {epoch+1} 完成，耗时: {time.time() - e_start:.1f}s",flush=True)

    total_time = time.time() - start_time
    print("-" * 50)
    print(f"🏁 测试结束！总耗时: {total_time/60:.2f} 分钟",flush=True)

if __name__ == '__main__':
    # Windows 下多进程必须加这个
    torch.multiprocessing.freeze_support()
    main()