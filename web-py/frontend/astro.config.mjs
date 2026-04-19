import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwind from '@astrojs/tailwind';

// Static-rendered shell with one React island. Audio + WS run in the browser;
// the FastAPI backend at ws://127.0.0.1:8765 does inference on the same machine.
export default defineConfig({
  integrations: [react(), tailwind({ applyBaseStyles: false })],
  server: { host: '127.0.0.1', port: 4321 },
  devToolbar: { enabled: false },
});
