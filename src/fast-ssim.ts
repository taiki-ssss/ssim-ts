import { SSIMOptions, ImageData, SSIMResult } from './types.js';

// Constants
const DEFAULT_K1 = 0.01;
const DEFAULT_K2 = 0.03;
const DEFAULT_WINDOW_SIZE = 8; // Using 8x8 for fast SSIM (power of 2)
const DEFAULT_L = 255;

// Optimized integral image with single-pass computation
class IntegralImage {
  private readonly integral: Float64Array;
  private readonly integralSq: Float64Array;
  private readonly width: number;
  private readonly height: number;
  private readonly stride: number;

  constructor(data: number[] | Uint8Array | Float32Array, width: number, height: number) {
    this.width = width;
    this.height = height;
    this.stride = width + 1;
    
    const size = this.stride * (height + 1);
    this.integral = new Float64Array(size);
    this.integralSq = new Float64Array(size);
    
    this.buildBoth(data);
  }

  // Build both integral and squared integral in a single pass
  private buildBoth(data: number[] | Uint8Array | Float32Array): void {
    const stride = this.stride;
    const integral = this.integral;
    const integralSq = this.integralSq;
    
    // Process in a single pass with better cache locality
    let dataIdx = 0;
    for (let y = 1; y <= this.height; y++) {
      const yOffset = y * stride;
      const prevYOffset = (y - 1) * stride;
      
      let rowSum = 0;
      let rowSumSq = 0;
      
      for (let x = 1; x <= this.width; x++) {
        const value = data[dataIdx++];
        const valueSq = value * value;
        
        rowSum += value;
        rowSumSq += valueSq;
        
        const idx = yOffset + x;
        integral[idx] = rowSum + integral[prevYOffset + x];
        integralSq[idx] = rowSumSq + integralSq[prevYOffset + x];
      }
    }
  }

  // Get mean of rectangular region
  getMean(x1: number, y1: number, x2: number, y2: number): number {
    const stride = this.stride;
    const integral = this.integral;
    
    const a = integral[y1 * stride + x1];
    const b = integral[y1 * stride + x2 + 1];
    const c = integral[(y2 + 1) * stride + x1];
    const d = integral[(y2 + 1) * stride + x2 + 1];
    
    const area = (x2 - x1 + 1) * (y2 - y1 + 1);
    return (d - b - c + a) / area;
  }

  getMeanSq(x1: number, y1: number, x2: number, y2: number): number {
    const stride = this.stride;
    const integralSq = this.integralSq;
    
    const a = integralSq[y1 * stride + x1];
    const b = integralSq[y1 * stride + x2 + 1];
    const c = integralSq[(y2 + 1) * stride + x1];
    const d = integralSq[(y2 + 1) * stride + x2 + 1];
    
    const area = (x2 - x1 + 1) * (y2 - y1 + 1);
    return (d - b - c + a) / area;
  }
}

// Optimized product integral with combined operations
class ProductIntegralImage {
  private readonly integral: Float64Array;
  private readonly stride: number;

  constructor(
    data1: number[] | Uint8Array | Float32Array,
    data2: number[] | Uint8Array | Float32Array,
    width: number,
    height: number
  ) {
    this.stride = width + 1;
    this.integral = new Float64Array(this.stride * (height + 1));
    this.buildProduct(data1, data2, width, height);
  }

  private buildProduct(
    data1: number[] | Uint8Array | Float32Array,
    data2: number[] | Uint8Array | Float32Array,
    width: number,
    height: number
  ): void {
    const stride = this.stride;
    const integral = this.integral;
    
    let dataIdx = 0;
    for (let y = 1; y <= height; y++) {
      const yOffset = y * stride;
      const prevYOffset = (y - 1) * stride;
      
      let rowSum = 0;
      
      for (let x = 1; x <= width; x++) {
        rowSum += data1[dataIdx] * data2[dataIdx];
        dataIdx++;
        
        const idx = yOffset + x;
        integral[idx] = rowSum + integral[prevYOffset + x];
      }
    }
  }

  getMean(x1: number, y1: number, x2: number, y2: number): number {
    const stride = this.stride;
    const integral = this.integral;
    
    const a = integral[y1 * stride + x1];
    const b = integral[y1 * stride + x2 + 1];
    const c = integral[(y2 + 1) * stride + x1];
    const d = integral[(y2 + 1) * stride + x2 + 1];
    
    const area = (x2 - x1 + 1) * (y2 - y1 + 1);
    return (d - b - c + a) / area;
  }
}

export function calculateSSIMFast(img1: ImageData, img2: ImageData, options?: SSIMOptions): SSIMResult {
  if (img1.width !== img2.width || img1.height !== img2.height) {
    throw new Error('Images must have the same dimensions');
  }

  const k1 = options?.k1 ?? DEFAULT_K1;
  const k2 = options?.k2 ?? DEFAULT_K2;
  const windowSize = options?.windowSize ?? DEFAULT_WINDOW_SIZE;
  const L = options?.L ?? DEFAULT_L;
  
  // Pre-compute constants
  const C1 = (k1 * L) * (k1 * L);
  const C2 = (k2 * L) * (k2 * L);
  const halfSize = windowSize >> 1;

  const { width, height, data: data1 } = img1;
  const { data: data2 } = img2;
  
  // Build optimized integral images
  const integral1 = new IntegralImage(data1, width, height);
  const integral2 = new IntegralImage(data2, width, height);
  const integralProd = new ProductIntegralImage(data1, data2, width, height);
  
  // Pre-calculate bounds
  const startX = halfSize;
  const startY = halfSize;
  const endX = width - halfSize;
  const endY = height - halfSize;
  const validWidth = endX - startX;
  const validHeight = endY - startY;
  
  const ssimMap = new Float32Array(validWidth * validHeight);
  
  // Optimized loop with better cache locality
  let idx = 0;
  for (let y = startY; y < endY; y++) {
    // Pre-calculate y bounds
    const y1 = Math.max(0, y - halfSize);
    const y2 = Math.min(height - 1, y + halfSize - 1);
    
    for (let x = startX; x < endX; x++) {
      // Pre-calculate x bounds
      const x1 = Math.max(0, x - halfSize);
      const x2 = Math.min(width - 1, x + halfSize - 1);
      
      // Calculate means
      const mu1 = integral1.getMean(x1, y1, x2, y2);
      const mu2 = integral2.getMean(x1, y1, x2, y2);
      
      // Calculate variances and covariance
      const meanSq1 = integral1.getMeanSq(x1, y1, x2, y2);
      const meanSq2 = integral2.getMeanSq(x1, y1, x2, y2);
      const meanProd = integralProd.getMean(x1, y1, x2, y2);
      
      const mu1Sq = mu1 * mu1;
      const mu2Sq = mu2 * mu2;
      const mu1Mu2 = mu1 * mu2;
      
      const sigma1Sq = meanSq1 - mu1Sq;
      const sigma2Sq = meanSq2 - mu2Sq;
      const sigma12 = meanProd - mu1Mu2;
      
      // SSIM formula with fused multiply-add where possible
      const numerator = (2 * mu1Mu2 + C1) * (2 * sigma12 + C2);
      const denominator = (mu1Sq + mu2Sq + C1) * (sigma1Sq + sigma2Sq + C2);
      
      ssimMap[idx++] = numerator / denominator;
    }
  }

  // Calculate mean SSIM with optimized summation
  let sum = 0;
  const len = ssimMap.length;
  
  // Unroll loop for better performance
  let i = 0;
  for (; i < len - 3; i += 4) {
    sum += ssimMap[i] + ssimMap[i + 1] + ssimMap[i + 2] + ssimMap[i + 3];
  }
  for (; i < len; i++) {
    sum += ssimMap[i];
  }
  
  const mssim = sum / len;

  return {
    mssim,
    ssim_map: Array.from(ssimMap)
  };
}

export function ssimFast(img1: ImageData, img2: ImageData, options?: SSIMOptions): number {
  return calculateSSIMFast(img1, img2, options).mssim;
}