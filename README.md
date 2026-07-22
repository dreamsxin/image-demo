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

## CNN 图像识别（从固定核到学习型核）

卷积核实验室里的 10 种核都是**人工设定**的。CNN 则让网络从数据中**学到**合适的卷积核——这正是「固定卷积核 → 学习型卷积核」的教学闭环。

### 模型结构

```
输入 1×28×28
Conv1(1→8, 3×3, pad1) → ReLU → MaxPool(2)   → 8×14×14
Conv2(8→16, 3×3, pad1) → ReLU → MaxPool(2)  → 16×7×7
FC(16*7*7 → 10)
```

数据集 MNIST（28×28 灰度手写数字，10 类）。Adam 优化器 lr=1e-3，CPU 训练 2 epoch 即可达 ~97.9% 测试准确率。

### Python 端

```bash
cd python
pip install -r requirements.txt        # 含 torch / torchvision

python -m cnn.train                    # 训练，保存 output/cnn_model.pt + 训练曲线
python -m cnn.visualize                # 可视化学习到的卷积核 + 各层特征图
python -m cnn.predict                  # 预测演示（图片 + 置信度）
python -m cnn.export_weights           # 导出权重为 JSON，供 Web 端推理
```

`export_weights` 会在 `web/public/` 生成：
- `cnn_weights.json`：conv1 / conv2 / fc 全部权重 + 归一化常量
- `mnist_samples.json`：10 个 MNIST 测试样本（像素 + 标签）

### Web 端（浏览器内推理）

Web CNN demo 用**纯 TypeScript** 重写了前向传播（Conv2d → ReLU → MaxPool → FC），不依赖任何运行时推理库，复用 `core/convolution.ts` 的卷积思想。权重来自上一步导出的 JSON，浏览器内实时推理。

```bash
cd web
npm install
npm run dev
# 打开 http://127.0.0.1:5173/cnn.html
```

功能：
- 手绘数字（280×280 黑底白笔，自动降采样到 28×28），或点选 MNIST 示例样本
- 点击「识别」触发浏览器内前向传播，显示预测数字与 10 类置信度分布
- 展示 Conv1 的 8 个特征图，直观看到**学习到的卷积核**对输入提取了什么

> 三层共用同一套卷积思想：Python 层看得见公式，Web 层摸得着交互，CNN 层把「人工设定核」推进到「数据驱动核」。

