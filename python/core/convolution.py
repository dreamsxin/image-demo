"""
二维卷积核心实现（从零手写）

图像处理中的"卷积"实际上是互相关（cross-correlation）：
核不翻转，直接滑动覆盖图像区域做加权求和。这是行业惯例，本项目遵循之。

数学定义（核 K 大小 kh x kw，中心在核中央）：

    O(x, y) = (1/div) * Σ_{i,j} I(x + i - ph, y + j - pw) * K(i, j)  + bias

其中 ph = kh//2, pw = kw//2，div 为归一化除数，bias 为偏移。
边界超出部分按 border 策略填充。
"""
import numpy as np


def convolve2d(image, kernel, divisor=1.0, border="zero", bias=0.0, clip=True):
    """
    对图像做二维卷积（互相关）。

    参数:
        image   : 2D ndarray (H,W) 灰度，或 3D ndarray (H,W,C) 彩色
        kernel  : 2D array-like，卷积核
        divisor : float，归一化除数（均值核 9，高斯 16）
        border  : 'zero' 超界补零 | 'reflect' 镜像反射
        bias    : float，结果偏移（浮雕用 128 居中到灰度范围）
        clip    : True 返回 uint8 并截断 [0,255]；False 返回 float（便于边缘核取绝对值）

    返回:
        与输入同形状的 uint8 图像，或 float64 数组
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    # 彩色图：逐通道卷积后合并
    if image.ndim == 3:
        channels = [
            convolve2d(image[:, :, c], kernel, divisor, border, bias, clip)
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=2)

    H, W = image.shape
    image = image.astype(np.float64)

    # 边界填充：在四周补 ph/pw 圈像素
    if border == "reflect":
        padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
    else:
        padded = np.pad(image, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)

    out = np.zeros((H, W), dtype=np.float64)

    # 核心循环：遍历核每个位置，把对应偏移的图像块乘以权重累加。
    # 这等价于"对每个输出像素，把核覆盖区域加权求和"，但用 numpy 切片向量化更快。
    for i in range(kh):
        for j in range(kw):
            w = kernel[i, j]
            if w == 0:
                continue
            # padded[i:i+H, j:j+W] 正是核 (i,j) 位置覆盖的图像块
            out += padded[i : i + H, j : j + W] * w

    # 归一化 + 偏移
    if divisor != 0:
        out = out / divisor
    out += bias

    if clip:
        return np.clip(out, 0, 255).astype(np.uint8)
    return out
