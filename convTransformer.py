import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # 不再需要维度调整

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0)  # 显式扩展维度到 (1, seq_len, d_model)
        x = x + pe.to(x.device)
        return x


class ConvTransformer(nn.Module):
    """Convolutional Transformer模型"""

    def __init__(self, input_dim=256, d_model=128, nhead=8, num_layers=4, conv_kernel=5):
        super().__init__()

        # 增强卷积模块
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(50)  # 统一输出长度
        )

        # 改进的位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # 更高效的Transformer配置
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,  # 标准配置比例
            dropout=0.2,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 优化回归头
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # 输入形状处理
        x = x.unsqueeze(1)  # [batch, 1, input_dim]
        x = self.conv(x)  # [batch, d_model, 50]

        # 维度调整
        x = x.permute(0, 2, 1)  # [batch, 50, d_model]
        x = self.pos_encoder(x)

        # Transformer处理
        x = self.transformer(x)  # [batch, 50, d_model]

        # 注意力池化代替简单平均
        attn_weights = torch.mean(x, dim=-1, keepdim=True)  # [batch, 50, 1]
        x = torch.sum(x * attn_weights.softmax(dim=1), dim=1)

        return self.regressor(x).squeeze(-1)