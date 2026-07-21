/**
 * 二维卷积（互相关）核心 — TypeScript 版。
 * 与 python/core/convolution.py 算法一致。
 *
 * 原理:
 *   O(x, y) = (1/div) * Σ I(x+i-ph, y+j-pw) * K(i, j)  + bias
 *
 * 边界策略: 'zero' 补零 | 'reflect' 镜像反射
 */

export type Kernel = number[][];

export interface ConvolveOptions {
  divisor?: number;
  border?: 'zero' | 'reflect';
  bias?: number;
  /** 对原始结果取绝对值（边缘核用，避免负值被 clamp 掉） */
  absNorm?: boolean;
}

export function convolve(src: ImageData, kernel: Kernel, opts: ConvolveOptions = {}): ImageData {
  const { divisor = 1, border = 'reflect', bias = 0, absNorm = false } = opts;
  const kh = kernel.length;
  const kw = kernel[0].length;
  const ph = Math.floor(kh / 2);
  const pw = Math.floor(kw / 2);
  const W = src.width;
  const H = src.height;
  const s = src.data;
  const out = new Uint8ClampedArray(s.length);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let r = 0, g = 0, b = 0;
      for (let i = 0; i < kh; i++) {
        for (let j = 0; j < kw; j++) {
          const w = kernel[i][j];
          if (w === 0) continue;
          let sy = y + i - ph;
          let sx = x + j - pw;
          // 边界处理
          if (sy < 0 || sy >= H || sx < 0 || sx >= W) {
            if (border === 'zero') continue;
            // reflect: v → -v-1 (左侧) 或 2*max-v-1 (右侧)
            if (sy < 0) sy = -sy - 1; else if (sy >= H) sy = 2 * H - sy - 1;
            if (sx < 0) sx = -sx - 1; else if (sx >= W) sx = 2 * W - sx - 1;
          }
          const idx = (sy * W + sx) * 4;
          r += s[idx] * w;
          g += s[idx + 1] * w;
          b += s[idx + 2] * w;
        }
      }
      let rOut = r / divisor + bias;
      let gOut = g / divisor + bias;
      let bOut = b / divisor + bias;
      if (absNorm) {
        rOut = Math.abs(rOut);
        gOut = Math.abs(gOut);
        bOut = Math.abs(bOut);
      }
      const o = (y * W + x) * 4;
      out[o]     = clamp(rOut);
      out[o + 1] = clamp(gOut);
      out[o + 2] = clamp(bOut);
      out[o + 3] = s[o + 3]; // alpha 保持
    }
  }
  return new ImageData(out, W, H);
}

function clamp(v: number): number {
  if (v < 0) return 0;
  if (v > 255) return 255;
  return v | 0;
}
