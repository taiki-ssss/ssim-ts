import { describe, it, expect } from 'vitest';
import { calculateSSIMFast, ssimFast } from '../src/fast-ssim.js';
import { calculateSSIM, ssim } from '../src/ssim.js';
import type { ImageData } from '../src/types.js';

describe('Fast SSIM', () => {
  const createTestImage = (width: number, height: number, value: number): ImageData => {
    const data = new Float32Array(width * height).fill(value);
    return { width, height, data };
  };

  const createGradientImage = (width: number, height: number): ImageData => {
    const data = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        data[y * width + x] = (x / width) * 255;
      }
    }
    return { width, height, data };
  };

  const createNoiseImage = (width: number, height: number, baseValue: number, noiseLevel: number): ImageData => {
    const data = new Float32Array(width * height);
    for (let i = 0; i < data.length; i++) {
      data[i] = baseValue + (Math.random() - 0.5) * noiseLevel;
    }
    return { width, height, data };
  };

  describe('calculateSSIMFast', () => {
    it('should use default parameters', () => {
      const img1 = createTestImage(32, 32, 128);
      const img2 = createTestImage(32, 32, 128);
      const result = calculateSSIMFast(img1, img2);
      expect(result.mssim).toBeCloseTo(1, 3);
    });

    it('should accept custom parameters', () => {
      const img1 = createTestImage(32, 32, 0.5);
      const img2 = createTestImage(32, 32, 0.5);
      const result = calculateSSIMFast(img1, img2, { k1: 0.02, k2: 0.04, windowSize: 8, L: 1 });
      expect(result.mssim).toBeCloseTo(1, 3);
    });

    it('should return 1 for identical images', () => {
      const img1 = createTestImage(64, 64, 128);
      const img2 = createTestImage(64, 64, 128);
      const result = calculateSSIMFast(img1, img2);
      expect(result.mssim).toBeCloseTo(1, 3);
    });

    it('should return lower value for different images', () => {
      const img1 = createTestImage(64, 64, 0);
      const img2 = createTestImage(64, 64, 255);
      const result = calculateSSIMFast(img1, img2);
      expect(result.mssim).toBeLessThan(0.1);
      expect(result.mssim).toBeGreaterThan(-1);
    });

    it('should handle gradient images', () => {
      const img1 = createGradientImage(64, 64);
      const img2 = createGradientImage(64, 64);
      const result = calculateSSIMFast(img1, img2);
      expect(result.mssim).toBeCloseTo(1, 3);
    });

    it('should detect structural differences', () => {
      const img1 = createGradientImage(64, 64);
      const img2 = createTestImage(64, 64, 128);
      const result = calculateSSIMFast(img1, img2);
      expect(result.mssim).toBeLessThan(1);
      expect(result.mssim).toBeGreaterThan(0);
    });

    it('should be sensitive to noise', () => {
      const img1 = createTestImage(64, 64, 128);
      const img2 = createNoiseImage(64, 64, 128, 10);
      const result = calculateSSIMFast(img1, img2);
      expect(result.mssim).toBeLessThan(1);
      expect(result.mssim).toBeGreaterThan(0.8);
    });

    it('should throw error for different image dimensions', () => {
      const img1 = createTestImage(64, 64, 128);
      const img2 = createTestImage(32, 32, 128);
      expect(() => calculateSSIMFast(img1, img2)).toThrow('Images must have the same dimensions');
    });

    it('should handle edge cases with small images', () => {
      const img1 = createTestImage(16, 16, 128);
      const img2 = createTestImage(16, 16, 128);
      const result = calculateSSIMFast(img1, img2, { windowSize: 3 });
      expect(result.mssim).toBeCloseTo(1, 3);
    });

    it('should return ssim_map with correct size', () => {
      const windowSize = 8;
      const width = 32;
      const height = 32;
      const img1 = createTestImage(width, height, 128);
      const img2 = createTestImage(width, height, 128);
      const result = calculateSSIMFast(img1, img2, { windowSize });

      const halfSize = Math.floor(windowSize / 2);
      const expectedMapSize = (width - 2 * halfSize) * (height - 2 * halfSize);
      expect(result.ssim_map.length).toBe(expectedMapSize);
    });

    it('should work with different data types', () => {
      const width = 32;
      const height = 32;

      const img1 = {
        width,
        height,
        data: new Uint8Array(width * height).fill(128)
      };

      const img2 = {
        width,
        height,
        data: new Float32Array(width * height).fill(128)
      };

      const result = calculateSSIMFast(img1, img2);
      expect(result.mssim).toBeCloseTo(1, 3);
    });
  });

  describe('ssimFast function', () => {
    it('should work as a standalone function', () => {
      const img1 = createTestImage(64, 64, 128);
      const img2 = createTestImage(64, 64, 128);
      const result = ssimFast(img1, img2);
      expect(result).toBeCloseTo(1, 3);
    });

    it('should accept options', () => {
      const img1 = createTestImage(64, 64, 128);
      const img2 = createNoiseImage(64, 64, 128, 20);
      const result1 = ssimFast(img1, img2);
      const result2 = ssimFast(img1, img2, { k1: 0.02, k2: 0.04 });
      expect(result1).not.toBe(result2);
    });
  });

  describe('Accuracy comparison with standard SSIM', () => {
    it('should produce similar results to standard SSIM for identical images', () => {
      const img1 = createTestImage(32, 32, 128);
      const img2 = createTestImage(32, 32, 128);
      
      // Both using window size 8 for fair comparison
      const standardResult = ssim(img1, img2, { windowSize: 8 });
      const fastResult = ssimFast(img1, img2, { windowSize: 8 });
      
      expect(standardResult).not.toBeNaN();
      expect(fastResult).not.toBeNaN();
      expect(standardResult).toBeCloseTo(1, 3);
      expect(fastResult).toBeCloseTo(1, 3);
      expect(Math.abs(standardResult - fastResult)).toBeLessThan(0.001);
    });

    it('should produce similar results for slightly different images', () => {
      const img1 = createTestImage(32, 32, 100);
      const img2 = createTestImage(32, 32, 110);
      
      const standardResult = ssim(img1, img2, { windowSize: 8 });
      const fastResult = ssimFast(img1, img2, { windowSize: 8 });
      
      // Fast SSIM uses rectangular window, so some difference is expected
      expect(Math.abs(standardResult - fastResult)).toBeLessThan(0.1);
    });

    it('should maintain accuracy for various test patterns', () => {
      const testCases = [
        { desc: 'identical', img1Val: 128, img2Val: 128, tolerance: 0.001 },
        { desc: 'slight difference', img1Val: 100, img2Val: 105, tolerance: 0.05 },
        { desc: 'moderate difference', img1Val: 50, img2Val: 100, tolerance: 0.1 },
        { desc: 'large difference', img1Val: 0, img2Val: 255, tolerance: 0.1 }
      ];

      for (const testCase of testCases) {
        const img1 = createTestImage(32, 32, testCase.img1Val);
        const img2 = createTestImage(32, 32, testCase.img2Val);
        
        const standardResult = ssim(img1, img2, { windowSize: 8 });
        const fastResult = ssimFast(img1, img2, { windowSize: 8 });
        
        expect(Math.abs(standardResult - fastResult)).toBeLessThan(testCase.tolerance);
      }
    });
  });
});