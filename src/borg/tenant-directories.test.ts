import { chmodSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it } from "vitest";
import { listBankTenantIds } from "./tenant-directories.js";

const directories: string[] = [];
afterEach(() => {
  for (const dir of directories.splice(0)) rmSync(dir, { recursive: true, force: true });
});

it.skipIf(process.getuid?.() === 0)(
  "strict discovery reports unreadable candidates while pool discovery remains best effort",
  async () => {
    const root = mkdtempSync(join(tmpdir(), "tenant-discovery-"));
    directories.push(root);
    for (const id of ["readable", "unreadable", "unreadable-db"]) {
      mkdirSync(join(root, id));
      writeFileSync(join(root, id, "borg.db"), "fixture");
    }
    chmodSync(join(root, "unreadable"), 0);
    chmodSync(join(root, "unreadable-db", "borg.db"), 0);
    try {
      expect(await listBankTenantIds(root)).toEqual(["readable", "unreadable-db"]);
      await expect(listBankTenantIds(root, undefined, { strict: true })).rejects.toMatchObject({
        code: "EMBEDDING_TENANT_DISCOVERY_FAILED",
        message: expect.stringMatching(/(?=.*unreadable:)(?=.*unreadable-db:)/),
      });
    } finally {
      chmodSync(join(root, "unreadable"), 0o700);
      chmodSync(join(root, "unreadable-db", "borg.db"), 0o600);
    }
  },
);

it("strict discovery refuses symlink candidates and includes every accessible bank", async () => {
  const root = mkdtempSync(join(tmpdir(), "tenant-discovery-links-"));
  directories.push(root);
  mkdirSync(join(root, "bank"));
  writeFileSync(join(root, "bank", "borg.db"), "fixture");
  expect(await listBankTenantIds(root, undefined, { strict: true })).toEqual(["bank"]);
  symlinkSync(join(root, "bank"), join(root, "linked"));
  await expect(listBankTenantIds(root, undefined, { strict: true })).rejects.toThrow(
    "linked: tenant candidate is a symlink",
  );
});
