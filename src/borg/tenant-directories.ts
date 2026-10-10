import { access, readdir, stat } from "node:fs/promises";
import { constants } from "node:fs";
import { join } from "node:path";
import { BANK_DB_FILENAME } from "./storage-setup.js";
import { ConfigError } from "../util/errors.js";

export const DEFAULT_TENANT_ID_PATTERN = /^[a-z0-9][a-z0-9_-]{0,63}$/;

export async function listBankTenantIds(
  root: string,
  pattern = DEFAULT_TENANT_ID_PATTERN,
  options: { strict?: boolean } = {},
): Promise<string[]> {
  const entries = await readdir(root, { withFileTypes: true });
  const candidates = entries
    .filter(
      (entry) =>
        (entry.isDirectory() || (options.strict && entry.isSymbolicLink())) &&
        entry.name !== "backups" &&
        pattern.test(entry.name),
    )
    .map((entry) => entry.name);
  const failures: string[] = [];
  const banked = await Promise.all(
    candidates.map(async (tenantId) => {
      try {
        if (options.strict && entries.find((entry) => entry.name === tenantId)?.isSymbolicLink())
          throw new Error("tenant candidate is a symlink");
        const files = await readdir(join(root, tenantId));
        if (!files.includes(BANK_DB_FILENAME)) return null;
        if (options.strict) {
          const path = join(root, tenantId, BANK_DB_FILENAME);
          await access(path, constants.R_OK);
          if (!(await stat(path)).isFile()) throw new Error("bank database is not a regular file");
        }
        return tenantId;
      } catch (error) {
        if (options.strict)
          failures.push(`${tenantId}: ${error instanceof Error ? error.message : String(error)}`);
        return null;
      }
    }),
  );

  if (failures.length)
    throw new ConfigError(`Cannot inspect tenant candidates: ${failures.sort().join("; ")}`, {
      code: "EMBEDDING_TENANT_DISCOVERY_FAILED",
    });

  return banked.filter((tenantId): tenantId is string => tenantId !== null).sort();
}
