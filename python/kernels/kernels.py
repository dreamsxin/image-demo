"""
卷积核定义：10 种，5 大类。
与 web/src/core/kernels.ts 保持一致。

元组格式: (key, kernel, divisor, 中文名, 分类, bias)
"""
import numpy as np

IDENTITY = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
BOX_BLUR = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float64)
GAUSSIAN_BLUR = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
EMBOSS = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float64)
OUTLINE = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)

KERNELS = [
    ("identity",      IDENTITY,      1,  "恒等核",      "基础", 0),
    ("box_blur",      BOX_BLUR,      9,  "均值模糊",    "模糊", 0),
    ("gaussian_blur", GAUSSIAN_BLUR, 16, "高斯模糊",    "模糊", 0),
    ("sharpen",       SHARPEN,       1,  "锐化",        "锐化", 0),
    ("laplacian",     LAPLACIAN,     1,  "Laplacian",  "边缘", 0),
    ("sobel_x",       SOBEL_X,       1,  "Sobel-X",    "边缘", 0),
    ("sobel_y",       SOBEL_Y,       1,  "Sobel-Y",    "边缘", 0),
    ("prewitt_x",     PREWITT_X,     1,  "Prewitt-X",  "边缘", 0),
    ("emboss",        EMBOSS,        1,  "浮雕",        "特效", 128),
    ("outline",       OUTLINE,       1,  "轮廓",        "特效", 0),
]
