"""
简单 CNN 模型：2 卷积层 + 1 全连接层，用于 MNIST 手写数字识别。

网络结构:
    输入  1×28×28
    Conv1 (1→8, 3×3, pad1) → ReLU → MaxPool(2)   => 8×14×14
    Conv2 (8→16, 3×3, pad1) → ReLU → MaxPool(2)  => 16×7×7
    FC    (16*7*7 → 10)

forward 中把各层中间结果存入 self.features，供可视化使用。
"""
import torch.nn as nn


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        # 记录中间特征图（用于可视化）
        self.features = {}

        c1 = self.conv1(x)          # 8×28×28
        self.features['conv1'] = c1
        p1 = self.pool(self.relu(c1))  # 8×14×14
        self.features['pool1'] = p1

        c2 = self.conv2(p1)         # 16×14×14
        self.features['conv2'] = c2
        p2 = self.pool(self.relu(c2))  # 16×7×7
        self.features['pool2'] = p2

        flat = p2.view(p2.size(0), -1)
        return self.fc(flat)        # 10
