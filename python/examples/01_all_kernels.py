"""
对测试图应用全部 10 种卷积核，生成对比图。
验证 Python 卷积核心实现正确。

用法:
    python examples/01_all_kernels.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from core.convolution import convolve2d
from core.utils import load_image, show_images
from kernels.kernels import KERNELS

# 边缘检测类核：取绝对值，让边缘以亮像素呈现（负梯度也可见）
EDGE_KEYS = {"laplacian", "sobel_x", "sobel_y", "prewitt_x", "outline"}


def apply(image, kernel, divisor, bias=0.0, abs_norm=False):
    """卷积 + 可选绝对值归一化"""
    raw = convolve2d(image, kernel, divisor, border="reflect", bias=bias, clip=False)
    if abs_norm:
        raw = np.abs(raw)
    return np.clip(raw, 0, 255).astype(np.uint8)


def main():
    img_path = os.path.join(HERE, "..", "..", "assets", "test_image.png")
    if not os.path.exists(img_path):
        print("测试图不存在，先运行: python examples/generate_test_image.py")
        return

    gray = load_image(img_path, grayscale=True)

    images = [gray]
    titles = ["原图 (灰度)"]
    for key, kernel, divisor, name, category, bias in KERNELS:
        res = apply(gray, kernel, divisor, bias=bias, abs_norm=(key in EDGE_KEYS))
        images.append(res)
        titles.append(f"{name}\n[{category}]")

    fig = show_images(images, titles, cols=4)
    out = os.path.join(HERE, "..", "output", "all_kernels.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"对比图已生成: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
