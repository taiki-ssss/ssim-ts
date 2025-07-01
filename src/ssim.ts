import { SSIMOptions, ImageData, SSIMResult } from './types.js';

// Constants
const DEFAULT_K1 = 0.01;
const DEFAULT_K2 = 0.03;
const DEFAULT_WINDOW_SIZE = 11;
const DEFAULT_L = 255;
const DEFAULT_SIGMA = 1.5;

function createGaussianKernel(windowSize: number, sigma: number = DEFAULT_SIGMA): Float32Array {
  const kernel = new Float32Array(windowSize * windowSize);
  const center = Math.floor(windowSize / 2);
  let sum = 0;

  for (let y = 0; y < windowSize; y++) {
    for (let x = 0; x < windowSize; x++) {
      const dx = x - center;
      const dy = y - center;
      const value = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      const idx = y * windowSize + x;
      kernel[idx] = value;
      sum += value;
    }
  }

  // Normalize
  for (let i = 0; i < kernel.length; i++) {
    kernel[i] /= sum;
  }

  return kernel;
}

function applyGaussianWindow(
  data: number[] | Uint8Array | Float32Array,
  width: number,
  height: number,
  x: number,
  y: number,
  windowSize: number,
  kernel: Float32Array
): number {
  const halfSize = Math.floor(windowSize / 2);
  let sum = 0;
  let kernelSum = 0;

  const xStart = Math.max(0, x - halfSize);
  const xEnd = Math.min(width, x + halfSize + 1);
  const yStart = Math.max(0, y - halfSize);
  const yEnd = Math.min(height, y + halfSize + 1);

  for (let py = yStart; py < yEnd; py++) {
    for (let px = xStart; px < xEnd; px++) {
      const kx = px - x + halfSize;
      const ky = py - y + halfSize;
      const idx = py * width + px;
      const kernelIdx = ky * windowSize + kx;
      const weight = kernel[kernelIdx];
      
      sum += data[idx] * weight;
      kernelSum += weight;
    }
  }

  return sum / kernelSum;
}

function calculateLocalSSIM(
  img1: number[] | Uint8Array | Float32Array,
  img2: number[] | Uint8Array | Float32Array,
  width: number,
  height: number,
  x: number,
  y: number,
  windowSize: number,
  kernel: Float32Array,
  C1: number,
  C2: number
): number {
  const mu1 = applyGaussianWindow(img1, width, height, x, y, windowSize, kernel);
  const mu2 = applyGaussianWindow(img2, width, height, x, y, windowSize, kernel);
  const mu1Sq = mu1 * mu1;
  const mu2Sq = mu2 * mu2;
  const mu1Mu2 = mu1 * mu2;

  let sigma1Sq = 0;
  let sigma2Sq = 0;
  let sigma12 = 0;
  let weightSum = 0;

  const halfSize = Math.floor(windowSize / 2);
  const xStart = Math.max(0, x - halfSize);
  const xEnd = Math.min(width, x + halfSize + 1);
  const yStart = Math.max(0, y - halfSize);
  const yEnd = Math.min(height, y + halfSize + 1);

  for (let py = yStart; py < yEnd; py++) {
    for (let px = xStart; px < xEnd; px++) {
      const kx = px - x + halfSize;
      const ky = py - y + halfSize;
      const idx = py * width + px;
      const kernelIdx = ky * windowSize + kx;
      const weight = kernel[kernelIdx];
      
      const val1 = img1[idx];
      const val2 = img2[idx];
      
      sigma1Sq += weight * (val1 - mu1) * (val1 - mu1);
      sigma2Sq += weight * (val2 - mu2) * (val2 - mu2);
      sigma12 += weight * (val1 - mu1) * (val2 - mu2);
      weightSum += weight;
    }
  }

  sigma1Sq /= weightSum;
  sigma2Sq /= weightSum;
  sigma12 /= weightSum;

  const numerator = (2 * mu1Mu2 + C1) * (2 * sigma12 + C2);
  const denominator = (mu1Sq + mu2Sq + C1) * (sigma1Sq + sigma2Sq + C2);

  return numerator / denominator;
}

// Cache for Gaussian kernels
const kernelCache = new Map<string, Float32Array>();

function getOrCreateKernel(windowSize: number, sigma: number = DEFAULT_SIGMA): Float32Array {
  const key = `${windowSize}_${sigma}`;
  let kernel = kernelCache.get(key);
  
  if (!kernel) {
    kernel = createGaussianKernel(windowSize, sigma);
    kernelCache.set(key, kernel);
  }
  
  return kernel;
}

export function calculateSSIM(img1: ImageData, img2: ImageData, options?: SSIMOptions): SSIMResult {
  if (img1.width !== img2.width || img1.height !== img2.height) {
    throw new Error('Images must have the same dimensions');
  }

  const k1 = options?.k1 ?? DEFAULT_K1;
  const k2 = options?.k2 ?? DEFAULT_K2;
  const windowSize = options?.windowSize ?? DEFAULT_WINDOW_SIZE;
  const L = options?.L ?? DEFAULT_L;
  const C1 = Math.pow(k1 * L, 2);
  const C2 = Math.pow(k2 * L, 2);

  const { width, height } = img1;
  const kernel = getOrCreateKernel(windowSize);
  
  const halfSize = Math.floor(windowSize / 2);
  const validWidth = width - 2 * halfSize;
  const validHeight = height - 2 * halfSize;
  const ssimMap = new Float32Array(validWidth * validHeight);
  
  let idx = 0;
  for (let y = halfSize; y < height - halfSize; y++) {
    for (let x = halfSize; x < width - halfSize; x++) {
      ssimMap[idx++] = calculateLocalSSIM(
        img1.data,
        img2.data,
        width,
        height,
        x,
        y,
        windowSize,
        kernel,
        C1,
        C2
      );
    }
  }

  let sum = 0;
  for (let i = 0; i < ssimMap.length; i++) {
    sum += ssimMap[i];
  }
  const mssim = sum / ssimMap.length;

  return {
    mssim,
    ssim_map: Array.from(ssimMap)
  };
}

export function ssim(img1: ImageData, img2: ImageData, options?: SSIMOptions): number {
  return calculateSSIM(img1, img2, options).mssim;
}