import { defineConfig } from "tsup";

export default defineConfig([
  {
    entry: { index: "src/index.ts" },
    format: ["esm"],
    target: "node22",
    dts: true,
    clean: true,
    sourcemap: true,
    splitting: false,
    shims: false,
  },
  {
    entry: { "cli/index": "src/cli/index.ts" },
    format: ["esm"],
    target: "node22",
    dts: false,
    clean: false,
    sourcemap: true,
    splitting: false,
    banner: { js: "#!/usr/bin/env node" },
    shims: false,
  },
]);
