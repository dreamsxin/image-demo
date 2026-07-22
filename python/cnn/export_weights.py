"""
导出 MnistCNN 权重为 JSON，供 Web 端纯 TypeScript 推理使用。
同时导出若干 MNIST 测试样本，供 Web demo 展示与识别对比。

前置: 先运行 `python -m cnn.train` 生成 output/cnn_model.pt

输出:
    web/public/cnn_weights.json    模型权重(conv1/conv2/fc) + 归一化常量
    web/public/mnist_samples.json  10 个 MNIST 测试样本(像素 + 标签)

Web 端用纯 TypeScript 实现前向传播(Conv2d -> ReLU -> MaxPool -> Linear)，
不依赖任何运行时推理库，原理透明，复用 core/convolution.ts 的卷积思想。
"""
import os
import sys
import json

import torch
import numpy as np
from torchvision import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cnn.model import MnistCNN

OUTPUT_DIR = os.path.join(HERE, "..", "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "cnn_model.pt")
DATA_DIR = os.path.join(HERE, "..", "data")
WEB_PUBLIC = os.path.join(HERE, "..", "..", "web", "public")

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def export_weights():
    model = MnistCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    def t(w):
        return w.detach().cpu().numpy().tolist()

    weights = {
        "meta": {
            "mean": MNIST_MEAN,
            "std": MNIST_STD,
            "input_size": 28,
            "conv1": {"in_channels": 1, "out_channels": 8, "kernel_size": 3, "padding": 1},
            "conv2": {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "padding": 1},
            "fc": {"in_features": 784, "out_features": 10},
        },
        "conv1_weight": t(model.conv1.weight),  # [8,1,3,3]
        "conv1_bias":   t(model.conv1.bias),    # [8]
        "conv2_weight": t(model.conv2.weight),  # [16,8,3,3]
        "conv2_bias":   t(model.conv2.bias),    # [16]
        "fc_weight":    t(model.fc.weight),     # [10,784]
        "fc_bias":      t(model.fc.bias),       # [10]
    }

    os.makedirs(WEB_PUBLIC, exist_ok=True)
    out = os.path.join(WEB_PUBLIC, "cnn_weights.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(weights, f)
    size_kb = os.path.getsize(out) / 1024
    print(f"权重已导出: {os.path.abspath(out)} ({size_kb:.1f} KB)")


def export_samples(n=10):
    """导出 n 个 MNIST 测试样本(原始像素 0-255 + 标签)，供 Web demo 展示"""
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True)
    samples = []
    for i in range(n):
        img, label = test_set[i]  # PIL Image (L 模式, 28x28)
        arr = np.array(img, dtype=np.uint8)  # 28x28
        samples.append({"label": int(label), "pixels": arr.tolist()})

    os.makedirs(WEB_PUBLIC, exist_ok=True)
    out = os.path.join(WEB_PUBLIC, "mnist_samples.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f)
    print(f"样本已导出: {os.path.abspath(out)} ({n} 个)")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("模型不存在，先运行: python -m cnn.train")
        sys.exit(1)
    export_weights()
    export_samples(n=10)
