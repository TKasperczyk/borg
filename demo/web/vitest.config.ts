import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Dedicated vitest config (separate from vite.config.ts) so static analyzers
// and tooling can read an explicit `test.include`. Mirrors demo/server's
// vitest.config.ts. Carries the React plugin + jsdom env + setup so the test
// runner behaves identically to the previous vite.config.ts `test` block.
export default defineConfig({
  plugins: [react()],
  test: {
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
  },
});
