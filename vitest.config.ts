import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['tests/*.{spec,test}.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/**',
        'dist/**',
        'benchmarks/**',
        '**/*.config.ts',
        '**/*.test.ts',
        '**/index.ts',
        '**/types.ts'
      ],
      thresholds: {
        statements: 100,
        branches: 100,
        functions: 100,
        lines: 100,
      }
    },
    benchmark: {
      include: ['benchmarks/*.bench.ts'],
      reporters: ['default']
    }
  }
})