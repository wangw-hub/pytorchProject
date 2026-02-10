import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm  # 使用 notebook 专用的进度条，减少卡顿风险
import matplotlib.pyplot as plt
import time
import os

# =================配置区域=================
# 你的 RTX 5060 应该能轻松处理 128 或 256 的 Batch Size
BATCH_SIZE = 128
EPOCHS = 5  # 测试运行 5 轮即可，想跑完整训练可以改成 50
LEARNING_RATE = 0.01
NUM_WORKERS = 2  # WSL2 下建议不要设置太高，2-4 之间为宜


# =========================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


device = get_device()
print(f"🚀 正在使用的计算设备: {device}")
if device.type == 'cuda':
    print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
    print(f"   显存总量的: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

# 1. 数据准备 (使用 CIFAR-10)
# 第一次运行会自动下载约 160MB 的数据
print("\n📦 正在准备数据...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 建议将 root 改为你 D 盘的挂载路径，或者直接用相对路径
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# 2. 定义模型 (ResNet18)
print("🏗️ 正在加载 ResNet18 模型...")
# 修改 num_classes=10 适配 CIFAR-10
model = torchvision.models.resnet18(weights=None, num_classes=10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
scaler = torch.amp.GradScaler('cuda')  # 混合精度训练，利用 RTX 显卡的 Tensor Cores

# 3. 训练循环
print(f"\n🔥 开始训练 (共 {EPOCHS} 轮)...")
train_losses = []
train_accs = []

start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 显示进度条，desc 用于显示当前 Epoch 信息
    pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='batch')

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 开启混合精度上下文
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条后缀，显示实时 Loss 和 Accuracy
        pbar.set_postfix({'Loss': running_loss / (pbar.n + 1), 'Acc': 100. * correct / total})

    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

end_time = time.time()
print(f"\n✅ 训练完成! 总耗时: {end_time - start_time:.2f} 秒")

# 4. 绘制结果 (测试绘图是否卡顿)
print("📊 正在绘制结果...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Training Accuracy', color='orange')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

print("🎉 全部任务结束")