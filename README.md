# image-demo — 图像处理教学仓库

通过**卷积核**理解各类图像处理原理。双层实现：

- **Python 核心原理层** (`python/`)：NumPy 从零手写卷积，代码即公式，看得见滑动加权求和。
- **Web 交互演示层** (`web/`)：TypeScript + Canvas，实时调参看效果。

## 卷积核一览

| 分类 | 核 | 归一化除数 |
|------|----|----|
| 基础 | 恒等核 | 1 |
| 模糊 | 均值模糊 / 高斯模糊 | 9 / 16 |
| 锐化 | 锐化 | 1 |
| 边缘检测 | Laplacian / Sobel-X / Sobel-Y / Prewitt-X | 1 |
| 特效 | 浮雕 / 轮廓 | 1 |

## 快速开始

### Python（看原理）

```bash
cd python
pip install -r requirements.txt

# 生成测试图
python examples/generate_test_image.py
# 跑全部核，输出对比图到 output/all_kernels.png
python examples/01_all_kernels.py
```

### Web（看交互）

```bash
cd web
npm install
npm run dev
```

## 卷积原理

图像处理中的"卷积"实为互相关：核滑过图像，每个输出像素 = 核覆盖区域的加权求和。

```
O(x, y) = (1/div) * Σ  I(x+i-ph, y+j-pw) * K(i, j)  + bias
                    i,j
```

- `ph = kh//2, pw = kw//2`：核半径
- `div`：归一化除数（均值核用 9，高斯用 16）
- `bias`：偏移（浮雕用 128 居中到灰度范围）
- 边界超出部分按 `zero`（补零）或 `reflect`（镜像）填充
