import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev/build config only. Test config lives in vitest.config.ts.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:7740",
        changeOrigin: true,
        ws: true
      }
    }
  }
});
