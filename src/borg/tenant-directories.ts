import { readdir } from "node:fs/promises";
import { join } from "node:path";
import { BANK_DB_FILENAME } from "./storage-setup.js";

export const DEFAULT_TENANT_ID_PATTERN = /^[a-z0-9][a-z0-9_-]{0,63}$/;

export async function listBankTenantIds(
  root: string,
  pattern = DEFAULT_TENANT_ID_PATTERN,
): Promise<string[]> {
  const entries = await readdir(root, { withFileTypes: true });
  const candidates = entries
    .filter((entry) => entry.isDirectory() && entry.name !== "backups" && pattern.test(entry.name))
    .map((entry) => entry.name);
  const banked = await Promise.all(
    candidates.map(async (tenantId) => {
      // A bank is identified by its sqlite file; readdir of the tenant dir
      // (rather than stat of the file) keeps this to one syscall class and
      // treats an unreadable directory as "not a bank" instead of throwing.
      try {
        const files = await readdir(join(root, tenantId));
        return files.includes(BANK_DB_FILENAME) ? tenantId : null;
      } catch {
        return null;
      }
    }),
  );

  return banked.filter((tenantId): tenantId is string => tenantId !== null).sort();
}
