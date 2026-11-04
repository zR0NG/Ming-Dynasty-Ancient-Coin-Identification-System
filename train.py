import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # 进度条库，提升训练可视化体验

# ====================== 1. 超参数配置（可根据需求调整） ======================
# 训练轮次
EPOCHS = 20
# 批次大小
BATCH_SIZE = 32
# 学习率
LEARNING_RATE = 1e-4
# 分类数（根据你的明朝钱币类别数修改，比如：洪武通宝、永乐通宝、宣德通宝等）
NUM_CLASSES = 26
# 数据集路径（请替换为你的实际路径）
DATA_DIR = r"C:\PythonProject\数据可视化古钱币识别\dataset"
# 模型保存路径
SAVE_DIR = r"C:\PythonProject\数据可视化古钱币识别"
# 是否使用预训练权重
USE_PRETRAINED = True
# 设备配置（优先使用GPU）
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ====================== 5. 训练与验证函数（提前定义，避免主逻辑包裹问题） ======================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练单轮函数"""
    model.train()  # 切换训练模式：启用Dropout/BatchNorm训练模式（更新运行均值）
    running_loss = 0.0  # 累计损失（乘以批次大小，最后求平均）
    running_corrects = 0     # 累计正确预测数

    # 进度条遍历训练数据
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)  # 计算损失：CrossEntropyLoss自动处理logits→概率

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)  # 获取预测类别
        loss = criterion(outputs, labels)

        # 反向传播+优化
        loss.backward()
        optimizer.step()

        # 累计损失和正确数
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # 计算本轮训练损失和准确率
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)

    return epoch_loss, epoch_acc

def validate_one_epoch(model, loader, criterion, device):
    """验证单轮函数"""
    model.eval()  # 切换验证模式
    running_loss = 0.0
    running_corrects = 0

    # 验证阶段禁用梯度计算（节省内存）
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 累计损失和正确数
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    # 计算本轮验证损失和准确率
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)

    return epoch_loss, epoch_acc

# ====================== 主逻辑（包裹在if __name__中，适配Windows多进程） ======================
if __name__ == '__main__':
    print(f"使用设备: {DEVICE}")

    # ====================== 2. 数据预处理与加载 ======================
    # 训练集数据增强（防止过拟合）
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50输入尺寸为224x224
        transforms.RandomHorizontalFlip(p=0.5),   # 50%概率水平翻转（模拟钱币摆放角度）
        transforms.RandomRotation(degrees=15),  # 随机旋转±15°
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度/对比度调整
        transforms.ToTensor(),  # 转为张量  PIL→张量：(H,W,C)→(C,H,W)，值从0-255→0-1
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet归一化参数
                             std=[0.229, 0.224, 0.225])
    ])

    # 验证集仅做基础预处理（无增强）
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集（需保证数据集按指定结构组织）
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transforms)

    # 创建数据加载器（num_workers=0 适配Windows）
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 打印数据集信息
    print(f"训练集类别数: {len(train_dataset.classes)} | 训练集样本数: {len(train_dataset)}")
    print(f"验证集类别数: {len(val_dataset.classes)} | 验证集样本数: {len(val_dataset)}")
    print(f"类别映射: {train_dataset.class_to_idx}")

    # ====================== 3. 构建ResNet50模型 ======================
    # 加载预训练ResNet50
    model = models.resnet50(pretrained=USE_PRETRAINED)
    # 替换最后一层全连接层（适配自定义分类数）
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    # 将模型移至指定设备
    model = model.to(DEVICE)

    # ====================== 4. 定义损失函数与优化器 ======================
    # 交叉熵损失（适合分类任务）
    criterion = nn.CrossEntropyLoss()
    # Adam优化器（带权重衰减防止过拟合）
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # 学习率调度器（训练后期降低学习率）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # ====================== 6. 主训练循环 ======================
    # 记录训练过程中的指标
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # 保存最优模型（基于验证集准确率）
    best_val_acc = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n====================== 开始训练 ======================")
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        print("-" * 30)

        # 训练轮
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        # 验证轮
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, DEVICE)

        # 更新学习率
        scheduler.step()

        # 记录指标
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc.cpu().numpy())
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc.cpu().numpy())

        # 打印本轮指标
        print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
        print(f"当前学习率: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, os.path.join(SAVE_DIR, "best_ming_coins_model.pth"))
            print(f"保存最优模型（验证准确率: {best_val_acc:.4f}）")

    # ====================== 7. 训练完成后可视化 ======================
    print("\n====================== 训练完成 ======================")
    print(f"最优验证准确率: {best_val_acc:.4f}")

    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(val_acc_history, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 保存可视化结果
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"))
    plt.show()

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_ming_coins_model.pth"))
    print(f"最终模型已保存至: {os.path.join(SAVE_DIR, 'final_ming_coins_model.pth')}")