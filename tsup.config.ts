import { defineConfig } from "tsup";

export default defineConfig([
  {
    entry: {
      index: "src/index.ts",
      "suppression-outcome": "src/cognition/generation/suppression-outcome.ts",
    },
    format: ["esm"],
    target: "node22",
    // tsup 8 strips node: prefixes from output imports by default, which
    // resolves for legacy builtins but breaks node:-only modules: bare
    // "sqlite" is not a package. Keep the prefixes verbatim.
    removeNodeProtocol: false,
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
    // tsup 8 strips node: prefixes from output imports by default, which
    // resolves for legacy builtins but breaks node:-only modules: bare
    // "sqlite" is not a package. Keep the prefixes verbatim.
    removeNodeProtocol: false,
    dts: false,
    clean: false,
    sourcemap: true,
    splitting: false,
    banner: { js: "#!/usr/bin/env node" },
    shims: false,
  },
]);
