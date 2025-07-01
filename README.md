# ssim-ts

A TypeScript implementation of the Structural Similarity Index Measure (SSIM) algorithm for comparing image similarity.

## Installation

```bash
npm install ssim-ts
```

## Usage

```typescript
import { ssim, calculateSSIM } from 'ssim-ts';

// Simple usage - returns just the MSSIM value
const similarity = ssim(image1, image2);
console.log(`Similarity: ${similarity}`); // Value between -1 and 1

// Detailed usage - returns MSSIM value and similarity map
const result = calculateSSIM(image1, image2, {
  windowSize: 11,  // Gaussian window size (default: 11)
  k1: 0.01,        // Algorithm parameter (default: 0.01)
  k2: 0.03,        // Algorithm parameter (default: 0.03)
  L: 255           // Dynamic range of pixel values (default: 255)
});

console.log(`MSSIM: ${result.mssim}`);
console.log(`Similarity map: ${result.ssim_map}`);
```

## API

### `ssim(img1, img2, options?)`

Calculates the mean SSIM between two images.

- **Returns**: `number` - The MSSIM value between -1 and 1, where 1 indicates identical images.

### `calculateSSIM(img1, img2, options?)`

Calculates detailed SSIM information including a similarity map.

- **Returns**: `SSIMResult` object containing:
  - `mssim`: The mean SSIM value
  - `ssim_map`: Array of local SSIM values for each pixel

### Types

```typescript
interface ImageData {
  width: number;
  height: number;
  data: number[] | Uint8Array | Float32Array;
}

interface SSIMOptions {
  k1?: number;        // Default: 0.01
  k2?: number;        // Default: 0.03
  windowSize?: number; // Default: 11
  L?: number;         // Default: 255
}
```