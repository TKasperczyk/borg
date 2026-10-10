import {
  chmodSync,
  cpSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  symlinkSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { expect, it } from "vitest";

it.skipIf(process.getuid?.() === 0)(
  "runs the documented pod setup with a read-only app and writable tsx cache",
  () => {
    const root = mkdtempSync(join(tmpdir(), "embedding-pod-layout-"));
    const app = join(root, "app");
    const nodeBin = join(root, "layers", "node", "bin");
    const data = join(root, "data");
    const blockedTmp = join(root, "blocked-tmp");
    for (const path of [app, nodeBin, data, blockedTmp]) mkdirSync(path, { recursive: true });
    for (const name of ["src", "node_modules", "package.json"])
      symlinkSync(resolve(name), join(app, name));
    mkdirSync(join(app, "scripts"));
    cpSync(resolve("scripts/migrate-embeddings.ts"), join(app, "scripts/migrate-embeddings.ts"));
    cpSync(resolve("scripts/embedding-migration"), join(app, "scripts/embedding-migration"), {
      recursive: true,
    });
    symlinkSync(process.execPath, join(nodeBin, "node"));
    chmodSync(app, 0o555);
    chmodSync(blockedTmp, 0o555);
    try {
      const env = { ...process.env, TMPDIR: blockedTmp };
      const broken = spawnSync(
        process.execPath,
        ["--import", "tsx", "scripts/migrate-embeddings.ts", "--help"],
        { cwd: app, env, encoding: "utf8", timeout: 15_000 },
      );
      expect(broken.status).not.toBe(0);
      expect(broken.stderr).toContain("EACCES");
      const runbook = readFileSync(resolve("docs/embedding-migration.md"), "utf8");
      const setup = /```sh\n(# Pod shell setup[\s\S]*?)```/.exec(runbook)?.[1];
      expect(setup).toBeDefined();
      const command = setup!
        .replaceAll("/workspace/workspace/repos/app", app)
        .replaceAll("/layers/paketo-buildpacks_node-engine/node/bin", nodeBin)
        .replaceAll("/data/tmp", join(data, "tmp"));
      const result = spawnSync("sh", ["-eu", "-c", command], {
        cwd: app,
        env,
        encoding: "utf8",
        timeout: 15_000,
      });
      expect(result.status, result.stderr).toBe(0);
      expect(result.stdout).toContain("Usage: node --import tsx");
      expect(readdirSync(app).sort()).toEqual(["node_modules", "package.json", "scripts", "src"]);
      expect(readdirSync(join(data, "tmp")).some((name) => name.startsWith("tsx-"))).toBe(true);
    } finally {
      chmodSync(app, 0o700);
      chmodSync(blockedTmp, 0o700);
      rmSync(root, { recursive: true, force: true });
    }
  },
);
