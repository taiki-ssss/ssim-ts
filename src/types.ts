export interface SSIMOptions {
  readonly k1?: number;
  readonly k2?: number;
  readonly windowSize?: number;
  readonly L?: number;
}

export interface ImageData {
  readonly width: number;
  readonly height: number;
  readonly data: number[] | Uint8Array | Float32Array;
}

export interface SSIMResult {
  readonly mssim: number;
  readonly ssim_map: ReadonlyArray<number>;
}