"""
训练 MNIST CNN 模型。

用法:
    python -m cnn.train            # 默认 2 epoch
    python -m cnn.train --epochs 3

训练后保存:
    output/cnn_model.pt        模型权重
    output/training_curve.png  训练 loss / 测试准确率曲线
"""
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
CURVE_PATH = os.path.join(OUTPUT_DIR, "training_curve.png")

# MNIST 标准化常量（全局均值/标准差）
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def get_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    return train_loader, test_loader


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            correct += output.argmax(dim=1).eq(target).sum().item()
    return correct / len(test_loader.dataset)


def train(epochs=2):
    device = torch.device("cpu")
    model = MnistCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_loaders()
    print(f"训练集 {len(train_loader.dataset)} 样本，测试集 {len(test_loader.dataset)} 样本")

    history = {'loss': [], 'acc': []}
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 300 == 0:
                print(f"  epoch {epoch} | batch {batch_idx + 1}/{len(train_loader)} | loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, test_loader)
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        print(f"== epoch {epoch} 完成: avg_loss={avg_loss:.4f}  test_acc={acc:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存: {os.path.abspath(MODEL_PATH)}")
    return model, history


def plot_curve(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, len(history['loss']) + 1), history['loss'], 'o-', color='#185FA5')
    axes[0].set_title('训练 Loss', fontsize=13)
    axes[0].set_xlabel('epoch')
    axes[0].grid(alpha=0.3)
    axes[1].plot(range(1, len(history['acc']) + 1), history['acc'], 'o-', color='#0F6E56')
    axes[1].set_title('测试准确率', fontsize=13)
    axes[1].set_xlabel('epoch')
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(CURVE_PATH, dpi=120, bbox_inches='tight')
    print(f"训练曲线已保存: {os.path.abspath(CURVE_PATH)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    _, history = train(epochs=args.epochs)
    plot_curve(history)
