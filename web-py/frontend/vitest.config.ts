/// <reference types="vitest" />
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'happy-dom',
    include: ['src/**/*.test.{ts,tsx}'],
    globals: true,
    coverage: {
      provider: 'v8',
      include: ['src/lib/**/*.ts'],
      // CoachApp is a React component glued together with WebSocket + audio +
      // TTS — the units it's composed from are tested directly, and full UI
      // rendering tests don't add signal worth the harness complexity here.
      exclude: ['src/components/**', 'src/pages/**', 'src/env.d.ts'],
      reporter: ['text', 'text-summary'],
      thresholds: {
        statements: 90,
        branches: 85,
        functions: 90,
        lines: 90,
      },
    },
  },
});
