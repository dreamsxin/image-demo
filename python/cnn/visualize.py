"""
CNN 可视化：学习到的卷积核 + 各层特征图。

前置: 先运行 python -m cnn.train 生成 output/cnn_model.pt

输出:
    output/cnn_kernels.png       第1/2层卷积核
    output/cnn_feature_maps.png  单张测试图各层特征图
"""
import os
import sys

import torch
import numpy as np
from torchvision import datasets, transforms

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cnn.model import MnistCNN

DATA_DIR = os.path.join(HERE, "..", "data")
OUTPUT_DIR = os.path.join(HERE, "..", "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "cnn_model.pt")
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def load_model():
    model = MnistCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def _grid(tensors, nrow=8):
    """(N,H,W) numpy → 网格拼接的 2D 数组"""
    n, h, w = tensors.shape
    ncols = nrow
    nrows = (n + ncols - 1) // ncols
    canvas = np.zeros((nrows * h, ncols * w))
    for i in range(n):
        r, c = i // ncols, i % ncols
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = tensors[i]
    return canvas


def plot_kernels(model):
    """第1层卷积核 (8×1×3×3) 与第2层卷积核 (16×8×3×3，取输入通道0)"""
    k1 = model.conv1.weight.data.numpy()   # (8,1,3,3)
    k2 = model.conv2.weight.data.numpy()   # (16,8,3,3)

    fig, axes = plt.subplots(3, 8, figsize=(14, 5))
    for i in range(8):
        axes[0, i].imshow(k1[i, 0], cmap='RdBu', vmin=-abs(k1).max(), vmax=abs(k1).max())
        axes[0, i].set_title(f'conv1 #{i}', fontsize=9)
        axes[0, i].axis('off')
    for i in range(16):
        r, c = 1 + i // 8, i % 8
        axes[r, c].imshow(k2[i, 0], cmap='RdBu', vmin=-abs(k2).max(), vmax=abs(k2).max())
        axes[r, c].set_title(f'conv2 #{i}', fontsize=9)
        axes[r, c].axis('off')
    # 隐藏多余子图
    for j in range(16, 16):
        pass
    fig.suptitle('学习到的卷积核（红=正、蓝=负）', fontsize=14)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'cnn_kernels.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"卷积核可视化已保存: {os.path.abspath(out)}")


def plot_feature_maps(model, sample_idx=0):
    """选一张测试图，展示 conv1 / conv2 / pool2 各层特征图"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    img, label = test_set[sample_idx]

    with torch.no_grad():
        model(img.unsqueeze(0))
    feats = {k: v[0].numpy() for k, v in model.features.items()}

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # 输入图
    axes[0].imshow(img[0].numpy(), cmap='gray', aspect='auto')
    axes[0].set_title(f'输入图像（标签 {label}）', fontsize=12)
    axes[0].axis('off')

    # conv1 (8, 28, 28)
    axes[1].imshow(_grid(feats['conv1'], nrow=8), cmap='viridis', aspect='auto')
    axes[1].set_title('Conv1 特征图（8 通道，28×28）', fontsize=12)
    axes[1].axis('off')

    # conv2 (16, 14, 14)
    axes[2].imshow(_grid(feats['conv2'], nrow=8), cmap='viridis', aspect='auto')
    axes[2].set_title('Conv2 特征图（16 通道，14×14）', fontsize=12)
    axes[2].axis('off')

    # pool2 (16, 7, 7)
    axes[3].imshow(_grid(feats['pool2'], nrow=8), cmap='viridis', aspect='auto')
    axes[3].set_title('Pool2 特征图（16 通道，7×7 → 送入全连接层）', fontsize=12)
    axes[3].axis('off')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'cnn_feature_maps.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"特征图可视化已保存: {os.path.abspath(out)}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("模型不存在，先运行: python -m cnn.train")
        sys.exit(1)
    model = load_model()
    plot_kernels(model)
    plot_feature_maps(model, sample_idx=0)
