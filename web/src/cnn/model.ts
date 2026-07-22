/**
 * MnistCNN 纯 TypeScript 前向传播 — 浏览器端推理引擎。
 *
 * 与 python/cnn/model.py 网络结构完全一致，不依赖任何运行时推理库：
 *   输入 1×28×28
 *   Conv1(1→8, 3×3, pad1) → ReLU → MaxPool(2)   => 8×14×14
 *   Conv2(8→16, 3×3, pad1) → ReLU → MaxPool(2)  => 16×7×7
 *   FC(16*7*7 → 10)
 *
 * 权重来自 cnn_weights.json（由 python/cnn/export_weights.py 导出）。
 * 卷积采用互相关(cross-correlation)，与 PyTorch Conv2d 定义一致。
 * Flatten 顺序为 [C,H,W] 行优先，对应 PyTorch 的 view(batch,-1)。
 */

/** 三维张量: [channel][height][width] */
export type Tensor3 = number[][][];

export interface CnnWeights {
  meta: {
    mean: number;
    std: number;
    input_size: number;
    conv1: { in_channels: number; out_channels: number; kernel_size: number; padding: number };
    conv2: { in_channels: number; out_channels: number; kernel_size: number; padding: number };
    fc: { in_features: number; out_features: number };
  };
  conv1_weight: number[][][][]; // [8,1,3,3]
  conv1_bias: number[];         // [8]
  conv2_weight: number[][][][]; // [16,8,3,3]
  conv2_bias: number[];         // [16]
  fc_weight: number[][];        // [10,784]
  fc_bias: number[];            // [10]
}

export interface ForwardResult {
  logits: number[];   // [10]
  probs: number[];    // [10] softmax
  pred: number;       // argmax
  conv1: Tensor3;     // [8,28,28] 卷积输出(ReLU 前)，供可视化
  pool1: Tensor3;     // [8,14,14]
}

/**
 * Conv2d 互相关运算:
 *   out[oc][oy][ox] = bias[oc] + Σ_ic Σ_i Σ_j in[ic][oy+i-pad][ox+j-pad] * w[oc][ic][i][j]
 * padding 用零填充，越界位置直接跳过(等效补零)。
 */
function conv2d(input: Tensor3, weight: number[][][][], bias: number[], padding: number): Tensor3 {
  const oc = weight.length;
  const ic = weight[0].length;
  const kH = weight[0][0].length;
  const kW = weight[0][0][0].length;
  const inH = input[0].length;
  const inW = input[0][0].length;
  const outH = inH + 2 * padding - kH + 1;
  const outW = inW + 2 * padding - kW + 1;
  const out: Tensor3 = [];
  for (let o = 0; o < oc; o++) {
    const chan: number[][] = [];
    const wO = weight[o];
    for (let y = 0; y < outH; y++) {
      const row: number[] = new Array(outW);
      for (let x = 0; x < outW; x++) {
        let sum = bias[o];
        for (let c = 0; c < ic; c++) {
          const wChan = wO[c];
          const inChan = input[c];
          for (let i = 0; i < kH; i++) {
            const sy = y + i - padding;
            if (sy < 0 || sy >= inH) continue;
            const wRow = wChan[i];
            const inRow = inChan[sy];
            for (let j = 0; j < kW; j++) {
              const sx = x + j - padding;
              if (sx < 0 || sx >= inW) continue;
              sum += inRow[sx] * wRow[j];
            }
          }
        }
        row[x] = sum;
      }
      chan.push(row);
    }
    out.push(chan);
  }
  return out;
}

/** ReLU: max(0, x) */
function relu(t: Tensor3): Tensor3 {
  return t.map(chan => chan.map(row => row.map(v => (v > 0 ? v : 0))));
}

/** MaxPool2d(size, stride=size): 无 padding，取每个 size×size 窗口最大值 */
function maxPool2d(t: Tensor3, size: number): Tensor3 {
  const c = t.length;
  const h = t[0].length;
  const w = t[0][0].length;
  const outH = Math.floor(h / size);
  const outW = Math.floor(w / size);
  const out: Tensor3 = [];
  for (let ch = 0; ch < c; ch++) {
    const chan: number[][] = [];
    const tCh = t[ch];
    for (let y = 0; y < outH; y++) {
      const row: number[] = new Array(outW);
      for (let x = 0; x < outW; x++) {
        let m = -Infinity;
        for (let i = 0; i < size; i++) {
          const inRow = tCh[y * size + i];
          for (let j = 0; j < size; j++) {
            const v = inRow[x * size + j];
            if (v > m) m = v;
          }
        }
        row[x] = m;
      }
      chan.push(row);
    }
    out.push(chan);
  }
  return out;
}

/** 全连接: out[o] = bias[o] + Σ_i x[i] * w[o][i] */
function linear(x: number[], weight: number[][], bias: number[]): number[] {
  const outF = weight.length;
  const inF = weight[0].length;
  const out: number[] = new Array(outF);
  for (let o = 0; o < outF; o++) {
    let sum = bias[o];
    const wRow = weight[o];
    for (let i = 0; i < inF; i++) sum += x[i] * wRow[i];
    out[o] = sum;
  }
  return out;
}

/** 数值稳定的 softmax */
function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

/** 把 28×28 像素(0-255) 预处理为归一化张量 [1,28,28]，与 PyTorch Normalize 一致 */
export function preprocess(pixels: number[][], mean: number, std: number): Tensor3 {
  const chan = pixels.map(row => row.map(v => (v / 255 - mean) / std));
  return [chan];
}

/** 完整前向传播，返回 logits/概率/预测及中间特征图 */
export function forward(weights: CnnWeights, input: Tensor3): ForwardResult {
  const pad1 = weights.meta.conv1.padding;
  const pad2 = weights.meta.conv2.padding;

  const c1 = conv2d(input, weights.conv1_weight, weights.conv1_bias, pad1); // 8×28×28
  const p1 = maxPool2d(relu(c1), 2);                                       // 8×14×14
  const c2 = conv2d(p1, weights.conv2_weight, weights.conv2_bias, pad2);   // 16×14×14
  const p2 = maxPool2d(relu(c2), 2);                                       // 16×7×7

  // Flatten: [C,H,W] 行优先 → [784]，对应 PyTorch view(batch,-1)
  const flat: number[] = [];
  const fc = p2[0].length; // 7
  const fw = p2[0][0].length; // 7
  for (let ch = 0; ch < p2.length; ch++)
    for (let y = 0; y < fc; y++)
      for (let x = 0; x < fw; x++) flat.push(p2[ch][y][x]);

  const logits = linear(flat, weights.fc_weight, weights.fc_bias);
  const probs = softmax(logits);
  let pred = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[pred]) pred = i;

  return { logits, probs, pred, conv1: c1, pool1: p1 };
}
