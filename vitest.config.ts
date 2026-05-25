import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      borg: new URL("./src/index.ts", import.meta.url).pathname,
    },
  },
  test: {
    globals: false,
    environment: "node",
    include: [
      "src/**/*.test.ts",
      "tests/**/*.test.ts",
      "scripts/**/*.test.ts",
      "assessor/**/*.test.ts",
      "simulator/**/*.test.ts",
      "eval/**/*.test.ts",
      "demo/server/src/**/*.test.ts",
    ],
    testTimeout: 15_000,
    hookTimeout: 15_000,
    coverage: {
      provider: "v8",
      reporter: ["text", "html"],
      include: ["src/**/*.ts"],
      exclude: ["**/*.test.ts", "src/cli/**", "src/**/*.d.ts"],
    },
  },
});
