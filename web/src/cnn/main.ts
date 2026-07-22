/**
 * CNN 手写数字识别 demo 交互。
 * 加载 cnn_weights.json，手绘/示例输入 → 纯 TS 前向传播 → 预测 + 置信度 + Conv1 特征图。
 */
import { preprocess, forward } from './model';
import type { CnnWeights, ForwardResult } from './model';

const DRAW = 280;
const SIZE = 28;

let weightsReady: CnnWeights | null = null;
let samples: { label: number; pixels: number[][] }[] = [];

const drawCanvas = document.getElementById('drawCanvas') as HTMLCanvasElement;
const dctx = drawCanvas.getContext('2d')!;
const sampleGrid = document.getElementById('sampleGrid')!;
const predDigit = document.getElementById('predDigit')!;
const predLabel = document.getElementById('predLabel')!;
const confBar = document.getElementById('confBar')!;
const featGrid = document.getElementById('featGrid')!;
const statusEl = document.getElementById('status')!;

// 置信度柱状图 10 条
const bars: HTMLDivElement[] = [];
const barVals: HTMLDivElement[] = [];
for (let i = 0; i < 10; i++) {
  const col = document.createElement('div');
  col.className = 'bar-col';
  const bar = document.createElement('div');
  bar.className = 'bar';
  bar.style.height = '2px';
  const lab = document.createElement('div');
  lab.className = 'bar-label';
  lab.textContent = String(i);
  const val = document.createElement('div');
  val.className = 'bar-val';
  col.appendChild(bar);
  col.appendChild(lab);
  col.appendChild(val);
  confBar.appendChild(col);
  bars.push(bar);
  barVals.push(val);
}

// Conv1 特征图 8 个画布
const featCanvases: HTMLCanvasElement[] = [];
for (let i = 0; i < 8; i++) {
  const wrap = document.createElement('div');
  wrap.className = 'feat';
  const c = document.createElement('canvas');
  c.width = SIZE;
  c.height = SIZE;
  const span = document.createElement('span');
  span.textContent = '#' + i;
  wrap.appendChild(c);
  wrap.appendChild(span);
  featGrid.appendChild(wrap);
  featCanvases.push(c);
}

// 黑底白笔手绘
dctx.fillStyle = '#000';
dctx.fillRect(0, 0, DRAW, DRAW);
dctx.strokeStyle = '#fff';
dctx.lineWidth = 18;
dctx.lineCap = 'round';
dctx.lineJoin = 'round';

let drawing = false;
let lastX = 0;
let lastY = 0;

function pos(e: PointerEvent) {
  const r = drawCanvas.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) * (DRAW / r.width),
    y: (e.clientY - r.top) * (DRAW / r.height),
  };
}

drawCanvas.addEventListener('pointerdown', e => {
  drawing = true;
  const p = pos(e);
  lastX = p.x;
  lastY = p.y;
  dctx.beginPath();
  dctx.moveTo(p.x, p.y);
  dctx.lineTo(p.x + 0.1, p.y + 0.1); // 单击也留点
  dctx.stroke();
  drawCanvas.setPointerCapture(e.pointerId);
});
drawCanvas.addEventListener('pointermove', e => {
  if (!drawing) return;
  const p = pos(e);
  dctx.beginPath();
  dctx.moveTo(lastX, lastY);
  dctx.lineTo(p.x, p.y);
  dctx.stroke();
  lastX = p.x;
  lastY = p.y;
});
const stopDraw = () => { drawing = false; };
drawCanvas.addEventListener('pointerup', stopDraw);
drawCanvas.addEventListener('pointercancel', stopDraw);
drawCanvas.addEventListener('pointerleave', stopDraw);

document.getElementById('clearBtn')!.addEventListener('click', () => {
  dctx.fillStyle = '#000';
  dctx.fillRect(0, 0, DRAW, DRAW);
  document.querySelectorAll('.sample').forEach(x => x.classList.remove('active'));
  resetResult();
});

document.getElementById('recognizeBtn')!.addEventListener('click', () => recognize());

async function init() {
  statusEl.textContent = '加载模型权重中...';
  try {
    const [wRes, sRes] = await Promise.all([
      fetch('/cnn_weights.json'),
      fetch('/mnist_samples.json'),
    ]);
    weightsReady = (await wRes.json()) as CnnWeights;
    samples = (await sRes.json()) as { label: number; pixels: number[][] }[];
    statusEl.textContent = '模型就绪 · 测试准确率 ~97.9%';
    renderSamples();
  } catch {
    statusEl.textContent = '加载失败：请先运行 python -m cnn.export_weights';
  }
}

function renderSamples() {
  samples.forEach(s => {
    const div = document.createElement('div');
    div.className = 'sample';
    const c = document.createElement('canvas');
    c.width = SIZE;
    c.height = SIZE;
    drawPixels(c, s.pixels);
    div.appendChild(c);
    div.title = `标签 ${s.label}`;
    div.addEventListener('click', () => {
      document.querySelectorAll('.sample').forEach(x => x.classList.remove('active'));
      div.classList.add('active');
      loadSample(s.pixels);
    });
    sampleGrid.appendChild(div);
  });
}

function loadSample(pixels: number[][]) {
  // 把 28×28 放大显示到手绘画布，并直接识别
  const tmp = document.createElement('canvas');
  tmp.width = SIZE;
  tmp.height = SIZE;
  drawPixels(tmp, pixels);
  dctx.fillStyle = '#000';
  dctx.fillRect(0, 0, DRAW, DRAW);
  dctx.imageSmoothingEnabled = true;
  dctx.drawImage(tmp, 0, 0, DRAW, DRAW);
  recognize(pixels);
}

function drawPixels(c: HTMLCanvasElement, pixels: number[][]) {
  const ctx = c.getContext('2d')!;
  const w = c.width;
  const h = c.height;
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = pixels[y][x];
      const o = (y * w + x) * 4;
      img.data[o] = img.data[o + 1] = img.data[o + 2] = v;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// 从手绘画布降采样到 28×28 像素（取 R 通道）
function sampleFromCanvas(): number[][] {
  const tmp = document.createElement('canvas');
  tmp.width = SIZE;
  tmp.height = SIZE;
  const tctx = tmp.getContext('2d')!;
  tctx.imageSmoothingEnabled = true;
  tctx.drawImage(drawCanvas, 0, 0, SIZE, SIZE);
  const img = tctx.getImageData(0, 0, SIZE, SIZE);
  const px: number[][] = [];
  for (let y = 0; y < SIZE; y++) {
    const row: number[] = [];
    for (let x = 0; x < SIZE; x++) {
      row.push(img.data[(y * SIZE + x) * 4]);
    }
    px.push(row);
  }
  return px;
}

function recognize(forcePixels?: number[][]) {
  if (!weightsReady) {
    statusEl.textContent = '模型尚未加载完成';
    return;
  }
  statusEl.textContent = '推理中...';
  const pixels = forcePixels ?? sampleFromCanvas();
  // 让状态文案先渲染
  setTimeout(() => {
    const t0 = performance.now();
    const input = preprocess(pixels, weightsReady!.meta.mean, weightsReady!.meta.std);
    const res = forward(weightsReady!, input);
    const ms = (performance.now() - t0).toFixed(1);
    renderResult(res, ms);
  }, 0);
}

function renderResult(res: ForwardResult, ms: string) {
  predDigit.textContent = String(res.pred);
  predLabel.textContent = `推理耗时 ${ms} ms`;
  res.probs.forEach((p, i) => {
    bars[i].style.height = `${Math.max(2, p * 130)}px`;
    bars[i].classList.toggle('top', i === res.pred);
    barVals[i].textContent = p > 0.01 ? Math.round(p * 100) + '%' : '';
  });
  res.conv1.forEach((chan, i) => drawFeat(featCanvases[i], chan));
  statusEl.textContent = '';
}

// 特征图 min-max 归一化为灰度（Conv1 输出含负值，ReLU 前）
function drawFeat(c: HTMLCanvasElement, chan: number[][]) {
  const ctx = c.getContext('2d')!;
  const h = chan.length;
  const w = chan[0].length;
  let mn = Infinity;
  let mx = -Infinity;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = chan[y][x];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  const range = mx - mn || 1;
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const g = Math.round(((chan[y][x] - mn) / range) * 255);
      const o = (y * w + x) * 4;
      img.data[o] = img.data[o + 1] = img.data[o + 2] = g;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function resetResult() {
  predDigit.textContent = '-';
  predLabel.textContent = '点击「识别」';
  bars.forEach(b => {
    b.style.height = '2px';
    b.classList.remove('top');
  });
  barVals.forEach(v => (v.textContent = ''));
}

init();
