"""
CNN 预测演示：选若干测试图，展示预测数字与置信度分布。

前置: 先运行 python -m cnn.train 生成 output/cnn_model.pt

输出:
    output/cnn_predictions.png
"""
import os
import sys

import torch
import torch.nn.functional as F
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


def predict_samples(n=8):
    model = load_model()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 5))
    correct = 0
    for i in range(n):
        img, label = test_set[i]
        with torch.no_grad():
            logits = model(img.unsqueeze(0))
            prob = F.softmax(logits, dim=1)[0].numpy()

        pred = int(prob.argmax())
        correct += (pred == label)

        # 上排：图片
        axes[0, i].imshow(img[0].numpy(), cmap='gray')
        color = '#0F6E56' if pred == label else '#A32D2D'
        axes[0, i].set_title(f'真:{label} 预:{pred}', fontsize=10, color=color)
        axes[0, i].axis('off')

        # 下排：置信度条
        colors = ['#0F6E56' if j == pred else '#D3D1C7' for j in range(10)]
        axes[1, i].bar(range(10), prob, color=colors)
        axes[1, i].set_xticks(range(10))
        axes[1, i].set_ylim(0, 1)
        axes[1, i].tick_params(labelsize=8)
        if i == 0:
            axes[1, i].set_ylabel('置信度', fontsize=10)

    fig.suptitle(f'预测演示（{correct}/{n} 正确）', fontsize=14)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'cnn_predictions.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"预测演示已保存: {os.path.abspath(out)}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("模型不存在，先运行: python -m cnn.train")
        sys.exit(1)
    predict_samples(n=8)
