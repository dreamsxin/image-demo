/**
 * 卷积核可视化交互：原图缩略图选择 + 核卡片网格 + 实时卷积渲染。
 */
import { convolve } from './core/convolution';
import { KERNELS } from './core/kernels';

const CAT_COLOR: Record<string, string> = {
  '基础': '#5F5E5A', '模糊': '#0F6E56', '锐化': '#BA7517', '边缘': '#534AB7', '特效': '#993C1D'
};

const IMAGES = [
  { file: 'test_image.png',  label: '几何测试' },
  { file: 'test_image2.jpg', label: '微距 · 紫' },
  { file: 'test_image3.jpg', label: '微距 · 蓝' },
];

const MAX_DIM = 480; // 缩放最大边，平衡性能与细节

let origImageData: ImageData | null = null;
let currentImgFile = IMAGES[0].file;
let currentKey = 'sobel_x';

// 缩略图
const thumbsEl = document.getElementById('thumbs')!;
IMAGES.forEach(im => {
  const div = document.createElement('div');
  div.className = 'thumb' + (im.file === currentImgFile ? ' active' : '');
  div.innerHTML = `<img src="/${im.file}" alt=""><div class="label">${im.label}</div>`;
  div.addEventListener('click', () => {
    if (im.file === currentImgFile) return;
    currentImgFile = im.file;
    document.querySelectorAll('.thumb').forEach(t => t.classList.remove('active'));
    div.classList.add('active');
    loadImage();
  });
  thumbsEl.appendChild(div);
});

// 核网格
const gridEl = document.getElementById('kernelGrid')!;
KERNELS.forEach(def => {
  const c = document.createElement('div');
  c.className = 'kc' + (def.key === currentKey ? ' active' : '');
  const matHtml = def.kernel.map(r => r.map(v => {
    const bg = v > 0 ? '#E6F1FB' : v < 0 ? '#FCEBEB' : '#F1EFE8';
    const fg = v > 0 ? '#185FA5' : v < 0 ? '#A32D2D' : '#888780';
    return `<div style="background:${bg};color:${fg}">${v}</div>`;
  }).join('')).join('');
  c.innerHTML = `
    <div class="krow">
      <span class="kname">${def.name}</span>
      <span class="kcat" style="background:${CAT_COLOR[def.category] || '#888'}">${def.category}</span>
    </div>
    <div class="kmat">${matHtml}</div>`;
  c.addEventListener('click', () => {
    if (def.key === currentKey) return;
    currentKey = def.key;
    document.querySelectorAll('.kc').forEach(x => x.classList.remove('active'));
    c.classList.add('active');
    apply();
  });
  gridEl.appendChild(c);
});

const originalCanvas = document.getElementById('original') as HTMLCanvasElement;
const resultCanvas   = document.getElementById('result') as HTMLCanvasElement;
const matrixDiv      = document.getElementById('matrix') as HTMLDivElement;
const metaDiv        = document.getElementById('meta') as HTMLDivElement;
const loadingDiv     = document.getElementById('loading') as HTMLDivElement;

async function loadImage() {
  loadingDiv.textContent = '加载图片中...';
  const img = new Image();
  img.src = '/' + currentImgFile;
  await new Promise<void>((res, rej) => {
    img.onload  = () => res();
    img.onerror = () => rej(new Error('加载失败: ' + img.src));
  });
  const scale = Math.min(1, MAX_DIM / Math.max(img.width, img.height));
  const w = Math.round(img.width * scale);
  const h = Math.round(img.height * scale);
  originalCanvas.width = w; originalCanvas.height = h;
  resultCanvas.width   = w; resultCanvas.height   = h;
  const ctx = originalCanvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, w, h);
  origImageData = ctx.getImageData(0, 0, w, h);
  apply();
}

function apply() {
  if (!origImageData) return;
  loadingDiv.textContent = '卷积计算中...';
  const def = KERNELS.find(k => k.key === currentKey)!;
  // 用 setTimeout 让 loading 文案先渲染
  setTimeout(() => {
    const t0 = performance.now();
    const out = convolve(origImageData!, def.kernel, {
      divisor: def.divisor, bias: def.bias, absNorm: def.absNorm, border: 'reflect'
    });
    const ms = (performance.now() - t0).toFixed(0);
    resultCanvas.getContext('2d')!.putImageData(out, 0, 0);
    loadingDiv.textContent = '';
    renderMatrix(def);
    metaDiv.innerHTML = `
      <div class="name">${def.name}</div>
      <div>分类：${def.category}</div>
      <div>÷ ${def.divisor}${def.bias ? ' · +' + def.bias : ''}${def.absNorm ? ' · |·|' : ''}</div>
      <div class="timing">卷积耗时 ${ms} ms（${origImageData!.width}×${origImageData!.height}）</div>`;
  }, 0);
}

function renderMatrix(def: typeof KERNELS[number]) {
  matrixDiv.innerHTML = def.kernel.map(r => r.map(v => {
    const bg = v > 0 ? '#E6F1FB' : v < 0 ? '#FCEBEB' : '#F1EFE8';
    const fg = v > 0 ? '#185FA5' : v < 0 ? '#A32D2D' : '#888780';
    return `<div style="background:${bg};color:${fg}">${v}</div>`;
  }).join('')).join('');
}

loadImage().catch(err => { loadingDiv.textContent = err.message; });
