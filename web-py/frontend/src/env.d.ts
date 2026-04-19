/// <reference path="../.astro/types.d.ts" />
/// <reference types="astro/client" />

interface ImportMetaEnv {
  readonly PUBLIC_BACKEND_WS_URL?: string;
  readonly PUBLIC_BACKEND_PORT?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
