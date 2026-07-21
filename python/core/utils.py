"""图像 IO 与可视化工具"""
import numpy as np
from PIL import Image
import matplotlib

# 中文显示：优先使用系统中文字体（Windows: 微软雅黑/黑体）
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt


def load_image(path, grayscale=False):
    """加载图片为 numpy 数组：(H,W) 灰度 或 (H,W,3) 彩色"""
    img = Image.open(path)
    img = img.convert("L" if grayscale else "RGB")
    return np.array(img)


def save_image(image, path):
    """保存 numpy 数组为图片"""
    arr = np.asarray(image)
    if arr.ndim == 2:
        Image.fromarray(arr.astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def show_images(images, titles=None, cols=4, figsize=None):
    """
    并排展示多张图片。

    参数:
        images : list of ndarray (2D 灰度 或 3D 彩色)
        titles : list of str
        cols   : 每行最多几张
    返回:
        matplotlib Figure
    """
    n = len(images)
    cols = min(cols, n) if n > 0 else 1
    rows = (n + cols - 1) // cols
    if figsize is None:
        figsize = (cols * 3, rows * 3)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.reshape(-1)
    for idx, ax in enumerate(axes):
        if idx < n:
            img = images[idx]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255)
            ax.set_title(titles[idx] if titles else "", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    return fig
