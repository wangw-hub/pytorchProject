# -*- coding: utf-8 -*-
"""
@File    : ResNet50.py
@Time    : 2026/2/10 11:51 Tuesday
@Author  : wangw
@Email   : wangw_heart@163.com
@Description: 
"""
# ==========================================
# 🛑 必须放在最前面，防止 PyCharm 远程绘图卡死
import matplotlib

matplotlib.use('Agg')
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import time
import os

# ================= ⚙️ 高负载配置区 =================
# 显存压力测试：如果显存报错，请将 BATCH_SIZE 调小 (例如 64)
BATCH_SIZE = 128
# 计算压力测试：ResNet50 + 64x64分辨率
EPOCHS = 20
LEARNING_RATE = 0.01
# WSL2 建议设为 2 或 4，设太大可能会导致 CPU 内存交换卡顿
NUM_WORKERS = 4


# ===================================================

def get_device_info():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 计算设备: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
        print(f"   显存总量: {props.total_memory / 1024 ** 3:.2f} GB")
        print(f"   多处理器(SM)数量: {props.multi_processor_count}")
    return device


device = get_device_info()

# 1. 数据准备 (高负载版)
print("\n📦 正在加载并预处理数据 (Resize -> 64x64)...")

# 强行放大图片，增加 GPU 吞吐压力
transform_train = transforms.Compose([
    transforms.Resize(64),  # <--- 关键点：分辨率翻倍，计算量 x4
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# 2. 定义重型模型 (ResNet50)
print("🏗️ 正在加载 ResNet50 模型 (参数量巨大)...")
# 不使用预训练权重，强迫显卡从零计算梯度
model = torchvision.models.resnet50(weights=None, num_classes=10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
scaler = torch.amp.GradScaler('cuda')  # 混合精度

# 3. 训练循环
print(f"\n🔥 开始高负载训练 (共 {EPOCHS} 轮)...")
print(f"   预计耗时: 8 ~ 12 分钟")
print("-" * 60)

train_losses = []
train_accs = []
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 进度条
    pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=False)

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 混合精度前向传播
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 统计数据
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 实时更新进度条后缀
        pbar.set_postfix({'Loss': f"{running_loss / (pbar.n + 1):.4f}", 'Acc': f"{100. * correct / total:.2f}%"})

    # Epoch 结束统计
    epoch_duration = time.time() - epoch_start
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # 打印该轮简报（避免每一步都打印导致卡顿）
    tqdm.write(f"✅ Epoch {epoch + 1} | Time: {epoch_duration:.1f}s | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

total_time = time.time() - start_time
print("-" * 60)
print(f"🏁 训练完成! 总耗时: {total_time / 60:.2f} 分钟")

# 4. 保存并绘制结果
print("📊 正在生成结果图表...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='red')
plt.title('Loss Curve (ResNet50)')
plt.xlabel('Epoch')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Training Accuracy', color='blue')
plt.title('Accuracy Curve (ResNet50)')
plt.xlabel('Epoch')
plt.grid(True)

save_path = './training_result.png'
plt.savefig(save_path)
print(f"🎉 图表已保存为: {os.path.abspath(save_path)}")
plt.close()  # 关闭画布，释放内存