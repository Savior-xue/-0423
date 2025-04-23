# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from convTransformer import ConvTransformer
import os
from datetime import datetime
import csv
import psutil
from tqdm import tqdm
import matplotlib.pyplot as plt

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cpu')
print("运行模式：CPU模式")


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024  **  2
    print(f"当前内存占用: {mem:.1f} MB")


def load_data():
    try:
        X_train = np.fromfile("compressed/X_train_compressed.dat", dtype=np.float32).reshape(-1, 256)
        X_test = np.fromfile("compressed/X_test_compressed.dat", dtype=np.float32).reshape(-1, 256)
        y_train = np.load("compressed/y_train.npy").astype(np.float32)
        y_test = np.load("compressed/y_test.npy").astype(np.float32)

        x_mean, x_std = X_train.mean(), X_train.std()
        y_mean, y_std = y_train.mean(), y_train.std()

        X_train = (X_train - x_mean) / x_std
        X_test = (X_test - x_mean) / x_std
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        return X_train, y_train, X_test, y_test, x_mean, x_std, y_mean, y_std
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit(1)


def save_checkpoint(epoch, model, config, normalization_params, timestamp):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': config,
        'normalization_params': normalization_params
    }, f"results/best_model_{timestamp}.pth")


if __name__ == "__main__":
    # 数据准备
    print("正在加载数据...")
    X_train_all, y_train_all, X_test, y_test, x_mean, x_std, y_mean, y_std = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)

    # 配置参数
    config = {
        "batch_size": 1024,
        "lr": 5e-4,
        "epochs": 20,
        "input_dim": X_train.shape[1],
        "d_model": 64,
        "nhead": 4,
        "num_layers": 4,
        "conv_kernel": 5,
        "early_stop_patience": 20,
        "grad_clip": 1.0
    }

    # 模型初始化
    model = ConvTransformer(
        input_dim=config["input_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        conv_kernel=config["conv_kernel"]
    ).to(device)

    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 数据加载器
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
                              batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
                            batch_size=config["batch_size"] * 2)

    # 训练准备
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4, betas=(0.9, 0.98))
    criterion = nn.HuberLoss(delta=0.5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        epochs=config["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    training_log = []
    best_val_loss = float('inf')
    early_stop_counter = 0

    # 训练循环
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        # 记录日志
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr']
        })

        # 学习率调整和早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            save_checkpoint(epoch, model, config, (x_mean, x_std, y_mean, y_std), timestamp)
        else:
            early_stop_counter += 1
            if early_stop_counter >= config["early_stop_patience"]:
                print("\n早停触发")
                break

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    # 保存训练日志
    with open(f'results/training_log_{timestamp}.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Learning Rate'])
        for log in training_log:
            writer.writerow([log['epoch'], log['train_loss'], log['val_loss'], log['lr']])

    print_memory_usage()
    print("训练完成，模型和日志已保存在results目录")