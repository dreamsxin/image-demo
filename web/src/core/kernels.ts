/**
 * 卷积核定义 — 10 种，5 大类。
 * 与 python/kernels/kernels.py 保持完全一致。
 */

export const IDENTITY: number[][]      = [[0, 0, 0], [0, 1, 0], [0, 0, 0]];
export const BOX_BLUR: number[][]      = [[1, 1, 1], [1, 1, 1], [1, 1, 1]];
export const GAUSSIAN_BLUR: number[][] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]];
export const SHARPEN: number[][]       = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]];
export const LAPLACIAN: number[][]     = [[0, 1, 0], [1, -4, 1], [0, 1, 0]];
export const SOBEL_X: number[][]       = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
export const SOBEL_Y: number[][]       = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
export const PREWITT_X: number[][]     = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]];
export const EMBOSS: number[][]        = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]];
export const OUTLINE: number[][]       = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]];

export interface KernelDef {
  key: string;
  name: string;
  category: string;
  kernel: number[][];
  divisor: number;
  bias: number;
  /** 边缘类核通常取 abs 避免负值被 clamp */
  absNorm: boolean;
}

export const KERNELS: KernelDef[] = [
  { key: 'identity',      name: '恒等核',   category: '基础', kernel: IDENTITY,      divisor: 1,  bias: 0,   absNorm: false },
  { key: 'box_blur',      name: '均值模糊', category: '模糊', kernel: BOX_BLUR,      divisor: 9,  bias: 0,   absNorm: false },
  { key: 'gaussian_blur', name: '高斯模糊', category: '模糊', kernel: GAUSSIAN_BLUR, divisor: 16, bias: 0,   absNorm: false },
  { key: 'sharpen',       name: '锐化',     category: '锐化', kernel: SHARPEN,       divisor: 1,  bias: 0,   absNorm: false },
  { key: 'laplacian',     name: 'Laplacian', category: '边缘', kernel: LAPLACIAN,    divisor: 1,  bias: 0,   absNorm: true  },
  { key: 'sobel_x',       name: 'Sobel-X',  category: '边缘', kernel: SOBEL_X,       divisor: 1,  bias: 0,   absNorm: true  },
  { key: 'sobel_y',       name: 'Sobel-Y',  category: '边缘', kernel: SOBEL_Y,       divisor: 1,  bias: 0,   absNorm: true  },
  { key: 'prewitt_x',     name: 'Prewitt-X', category: '边缘', kernel: PREWITT_X,    divisor: 1,  bias: 0,   absNorm: true  },
  { key: 'emboss',        name: '浮雕',     category: '特效', kernel: EMBOSS,        divisor: 1,  bias: 128, absNorm: false },
  { key: 'outline',       name: '轮廓',     category: '特效', kernel: OUTLINE,       divisor: 1,  bias: 0,   absNorm: true  },
];
