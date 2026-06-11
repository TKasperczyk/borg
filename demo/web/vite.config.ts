import react from "@vitejs/plugin-react";
import type { ProxyOptions } from "vite";
import { defineConfig } from "vitest/config";

const apiTarget = "http://127.0.0.1:7740";

const proxy: Record<string, string | ProxyOptions> = {
  "/api/live": {
    target: apiTarget,
    changeOrigin: true,
    ws: true,
  },
  "/api": {
    target: apiTarget,
    changeOrigin: true,
  },
};

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy,
  },
  preview: {
    port: 5173,
    strictPort: true,
    proxy,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
  },
});
