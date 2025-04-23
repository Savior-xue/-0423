# test.py
import csv

import torch
import numpy as np
from sklearn.metrics import r2_score
import glob
import os
import matplotlib.pyplot as plt
from convTransformer import ConvTransformer

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_latest_model():
    checkpoint_files = glob.glob("results/best_model_*.pth")
    if not checkpoint_files:
        raise FileNotFoundError("未找到训练好的模型")

    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"正在加载模型: {os.path.basename(latest_checkpoint)}")

    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    model = ConvTransformer(
        input_dim=checkpoint['config']["input_dim"],
        d_model=checkpoint['config']["d_model"],
        nhead=checkpoint['config']["nhead"],
        num_layers=checkpoint['config']["num_layers"],
        conv_kernel=checkpoint['config']["conv_kernel"]
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

if __name__ == "__main__":
    # 设备配置
    device = torch.device('cpu')

    os.makedirs("results", exist_ok=True)

    # 可视化结果前再次确保目录存在
    plt.savefig('results/prediction_comparison.png')  # 原代码已有，确保前置目录创建
    # 加载模型
    model, checkpoint = load_latest_model()
    model.to(device).eval()

    # 加载测试数据
    X_test = np.fromfile("compressed/X_test_compressed.dat", dtype=np.float32).reshape(-1, 256)
    y_test = np.load("compressed/y_test.npy").astype(np.float32)

    # 标准化处理
    x_mean, x_std = checkpoint['normalization_params'][0], checkpoint['normalization_params'][1]
    y_mean, y_std = checkpoint['normalization_params'][2], checkpoint['normalization_params'][3]
    X_test = (X_test - x_mean) / x_std
    y_test = (y_test - y_mean) / y_std

    # 创建DataLoader
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=2048
    )

    # 进行预测
    predictions, true_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.extend((outputs.cpu().numpy() * y_std + y_mean).tolist())
            true_labels.extend((labels.cpu().numpy() * y_std + y_mean).tolist())

    # 计算指标
    mae = np.mean(np.abs(np.array(predictions) - np.array(true_labels)))
    mse = np.mean((np.array(predictions) - np.array(true_labels)) ** 2)
    r2 = r2_score(true_labels, predictions)

    print(f"\n测试结果:")
    print(f"MAE: {mae:.4f} 秒")
    print(f"MSE: {mse:.4f} 秒²")
    print(f"R² Score: {r2:.4f}")

    # 可视化结果
    plt.figure(figsize=(8, 6))
    plt.scatter(true_labels, predictions, alpha=0.5)
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], 'r--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测结果对比')
    plt.tight_layout()
    plt.savefig('results/prediction_comparison.png')
    plt.close()

    # 保存测试结果
    with open('results/test_results.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['指标', '值'])
        writer.writerow(['MAE', mae])
        writer.writerow(['MSE', mse])
        writer.writerow(['R²', r2])

    print("测试完成，结果已保存在results目录")