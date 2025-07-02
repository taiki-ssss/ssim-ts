import { bench, describe } from 'vitest';
import { calculateSSIM, ssim } from '../src/ssim.js';
import { calculateSSIMFast, ssimFast } from '../src/fast-ssim.js';
import type { ImageData } from '../src/types.js';

// Generate synthetic test images
function generateTestImage(width: number, height: number, pattern: 'random' | 'gradient' | 'checkerboard'): ImageData {
  const data = new Float32Array(width * height);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      
      switch (pattern) {
        case 'random':
          data[idx] = Math.random() * 255;
          break;
        case 'gradient':
          data[idx] = ((x / width) + (y / height)) * 127.5;
          break;
        case 'checkerboard':
          data[idx] = ((x >> 3) + (y >> 3)) % 2 ? 255 : 0;
          break;
      }
    }
  }
  
  return { width, height, data };
}

// Add noise to an image
function addNoise(image: ImageData, noiseLevel: number): ImageData {
  const noisyData = new Float32Array(image.data.length);
  
  for (let i = 0; i < image.data.length; i++) {
    const noise = (Math.random() - 0.5) * 2 * noiseLevel;
    noisyData[i] = Math.max(0, Math.min(255, image.data[i] + noise));
  }
  
  return { ...image, data: noisyData };
}

describe('SSIM Performance Benchmarks', () => {
  // Small images (64x64)
  const smallImg1 = generateTestImage(64, 64, 'gradient');
  const smallImg2 = addNoise(smallImg1, 10);
  
  // Medium images (256x256)
  const mediumImg1 = generateTestImage(256, 256, 'gradient');
  const mediumImg2 = addNoise(mediumImg1, 10);
  
  // Large images (512x512)
  const largeImg1 = generateTestImage(512, 512, 'gradient');
  const largeImg2 = addNoise(largeImg1, 10);
  
  // Extra large images (1024x1024)
  const xlargeImg1 = generateTestImage(1024, 1024, 'gradient');
  const xlargeImg2 = addNoise(xlargeImg1, 10);

  // Original implementation
  bench('ssim - small images (64x64)', () => {
    ssim(smallImg1, smallImg2);
  });

  bench('ssim - medium images (256x256)', () => {
    ssim(mediumImg1, mediumImg2);
  });

  bench('ssim - large images (512x512)', () => {
    ssim(largeImg1, largeImg2);
  });

  bench('ssim - extra large images (1024x1024)', () => {
    ssim(xlargeImg1, xlargeImg2);
  });
  
  // Fast SSIM with integral images
  bench('ssimFast - small images (64x64)', () => {
    ssimFast(smallImg1, smallImg2);
  });

  bench('ssimFast - medium images (256x256)', () => {
    ssimFast(mediumImg1, mediumImg2);
  });

  bench('ssimFast - large images (512x512)', () => {
    ssimFast(largeImg1, largeImg2);
  });

  bench('ssimFast - extra large images (1024x1024)', () => {
    ssimFast(xlargeImg1, xlargeImg2);
  });

  bench('calculateSSIM with map - medium images', () => {
    calculateSSIM(mediumImg1, mediumImg2);
  });

  // Window size comparison
  bench('ssim - medium images - window size 7', () => {
    ssim(mediumImg1, mediumImg2, { windowSize: 7 });
  });

  bench('ssim - medium images - window size 11 (default)', () => {
    ssim(mediumImg1, mediumImg2, { windowSize: 11 });
  });

  bench('ssim - medium images - window size 15', () => {
    ssim(mediumImg1, mediumImg2, { windowSize: 15 });
  });

  // Different data types
  const uint8Img1 = { ...mediumImg1, data: new Uint8Array(mediumImg1.data) };
  const uint8Img2 = { ...mediumImg2, data: new Uint8Array(mediumImg2.data) };

  bench('ssim - Uint8Array data', () => {
    ssim(uint8Img1, uint8Img2);
  });

  bench('ssim - Float32Array data', () => {
    ssim(mediumImg1, mediumImg2);
  });

  // Cache effectiveness test (run same calculation multiple times)
  bench('ssim - cache test (5 runs)', () => {
    for (let i = 0; i < 5; i++) {
      ssim(smallImg1, smallImg2);
    }
  });
});