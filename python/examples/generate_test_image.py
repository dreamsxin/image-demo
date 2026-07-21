"""
生成程序化测试图：渐变背景 + 几何形状 + 细节纹理 + 对角线。
不同区域可分别展示模糊、锐化、边缘检测的效果差异。
"""
import os
import numpy as np
from PIL import Image


def generate_test_image(size=256):
    img = np.zeros((size, size, 3), dtype=np.float64)

    # 渐变背景（左暗右亮），便于观察模糊/锐化在渐变上的变化
    grad = np.linspace(40, 210, size)
    for c in range(3):
        img[:, :, c] = np.tile(grad[None, :], (size, 1)) * (0.5 + 0.25 * c)

    # 实心矩形（清晰边缘，测边缘检测）
    img[35:90, 35:120] = [225, 80, 60]

    # 实心圆
    cy, cx = size // 2, size // 2 + 50
    yy, xx = np.ogrid[:size, :size]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 < 48 ** 2] = [60, 130, 210]

    # 细条纹（高频细节，测锐化与模糊对比）
    for x in range(150, 225, 6):
        img[150:215, x : x + 2] = 250

    # 对角线（测方向性核 Sobel-X/Y 的差异）
    for i in range(25, size - 25):
        img[i, i] = [245, 245, 245]
        img[i, min(i + 1, size - 1)] = [245, 245, 245]

    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    img = generate_test_image()
    here = os.path.dirname(os.path.abspath(__file__))

    # 存到共享 assets（供 web 使用）和 python/output 两处
    targets = [
        os.path.join(here, "..", "..", "assets", "test_image.png"),
        os.path.join(here, "..", "output", "test_image.png"),
    ]
    for t in targets:
        os.makedirs(os.path.dirname(t), exist_ok=True)
        Image.fromarray(img).save(t)
        print(f"已保存: {os.path.abspath(t)}")


if __name__ == "__main__":
    main()
