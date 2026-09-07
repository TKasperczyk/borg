import {
  closeSync,
  existsSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  openSync,
  readdirSync,
  renameSync,
  statfsSync,
} from "node:fs";
import { cp } from "node:fs/promises";
import { basename, dirname, join, relative } from "node:path";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { z } from "zod";
import { openReadOnlyDatabase } from "../../src/storage/sqlite/index.js";
import { readJsonFile, syncDirectory, writeJsonFileAtomic } from "../../src/util/atomic-write.js";
import {
  EMBEDDING_ACCESS_FILE,
  EMBEDDING_FENCE_FILE,
  EmbeddingBankError,
} from "../../src/embeddings/bank-profile.js";
import { fingerprintCanonicalValue } from "../../src/cognition/deliberation/request-fingerprint.js";
import { inventoryBank, type BankInventory } from "./inventory.js";
import { FILE_LOCK_GUARD_SUFFIX } from "../../src/stream/file-lock-guard.js";

const backupMarkerSchema = z.object({
  version: z.literal(1),
  created_at: z.number().finite(),
  inventory_hash: z.string().length(64),
  files: z.record(z.string(), z.string().length(64)),
});

export type DiskSpaceReader = (path: string) => { bavail: number; bsize: number };

// These describe a particular migration attempt, never the bank being restored.
function isMigrationBookkeeping(name: string): boolean {
  return (
    name === ".embedding-backup.json" ||
    name.startsWith(".embedding-migration") ||
    name.startsWith("..embedding-migration-owner.lock.") ||
    name === EMBEDDING_ACCESS_FILE ||
    name.startsWith(`.${EMBEDDING_ACCESS_FILE}.`) ||
    name === EMBEDDING_FENCE_FILE
  );
}

async function fileChecksum(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

/** Reject symlinks/special files: a whole-bank backup must be self contained. */
export function* bankFiles(directory: string): Generator<{ path: string; bytes: number }> {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) yield* bankFiles(path);
    else if (entry.isFile()) yield { path, bytes: lstatSync(path).size };
    else
      throw new EmbeddingBankError(`Bank contains a symlink or special file: ${path}`, {
        code: "EMBEDDING_MIGRATION_UNSAFE_PATH",
      });
  }
}

export function migrationHeadroom(
  tenantDir: string,
  backupDir: string,
  inventory: BankInventory,
  dimensions: number,
  work: { backup?: boolean; remainingRows?: number; remainingTables?: readonly string[] } = {},
  diskSpace: DiskSpaceReader = statfsSync,
): { bank_bytes: number; staging_bytes: number; backup_bytes: number; available_bytes: number } {
  const files = [...bankFiles(tenantDir)];
  const bankBytes = files.reduce((sum, file) => sum + file.bytes, 0);
  const totalRows = inventory.tables.reduce((sum, table) => sum + table.rows.length, 0);
  const remainingRows = work.remainingRows ?? totalRows;
  const vectorBytes = remainingRows * dimensions * 4;
  // Keep a whole-table allowance until that table is complete: the last row
  // may be much larger than the average, and Lance can rewrite its fragment.
  const remainingTables = new Set(
    work.remainingTables ?? inventory.tables.map((table) => table.name),
  );
  const sourceBytes = inventory.tables.reduce((sum, table) => {
    const directory = join(tenantDir, "lancedb", `${table.name}.lance`);
    return (
      sum +
      (remainingTables.has(table.name) && existsSync(directory)
        ? [...bankFiles(directory)].reduce((bytes, file) => bytes + file.bytes, 0)
        : 0)
    );
  }, 0);
  const stagingBytes = 2 * sourceBytes + 3 * vectorBytes + 64 * 1024 * 1024;
  const backupBytes = work.backup === false ? 0 : bankBytes;
  let existingBackupParent = backupDir;
  while (!existsSync(existingBackupParent)) existingBackupParent = dirname(existingBackupParent);
  const local = diskSpace(tenantDir);
  const remote = diskSpace(existingBackupParent);
  const localFree = local.bavail * local.bsize;
  const remoteFree = remote.bavail * remote.bsize;
  const sameDevice = lstatSync(tenantDir).dev === lstatSync(existingBackupParent).dev;
  if (localFree < stagingBytes + (sameDevice ? backupBytes : 0) || remoteFree < backupBytes) {
    throw new EmbeddingBankError(
      `Insufficient disk headroom: staging=${stagingBytes}, backup=${backupBytes}, local available=${localFree}, backup available=${remoteFree}`,
      { code: "EMBEDDING_MIGRATION_DISK_FULL" },
    );
  }
  return {
    bank_bytes: bankBytes,
    staging_bytes: stagingBytes,
    backup_bytes: backupBytes,
    available_bytes: localFree,
  };
}

export async function backupTenant(
  tenantDir: string,
  destination: string,
  inventory: BankInventory,
  targetDims: number,
  diskSpace?: DiskSpaceReader,
): Promise<void> {
  const markerPath = join(destination, ".embedding-backup.json");
  const inventoryHash = fingerprintCanonicalValue(inventory).canonicalSha256;
  const existing = readJsonFile<{ inventory_hash: string }>(markerPath);
  if (existing) {
    if (existing.inventory_hash !== inventoryHash)
      throw new EmbeddingBankError("Backup inventory changed; refusing reuse", {
        code: "EMBEDDING_MIGRATION_BACKUP_INVALID",
      });
    await verifyTenantBackup(destination, inventory, targetDims);
    return;
  }
  if (existsSync(destination))
    throw new EmbeddingBankError(`Unverified backup destination exists: ${destination}`, {
      code: "EMBEDDING_MIGRATION_BACKUP_INVALID",
    });
  // A fresh path for each incomplete attempt; never delete a partial backup.
  migrationHeadroom(tenantDir, destination, inventory, targetDims, {}, diskSpace);
  const partial = `${destination}.partial-${process.pid}-${Date.now()}`;
  mkdirSync(dirname(destination), { recursive: true, mode: 0o700 });
  const sqliteFiles = new Set(["borg.db", "borg.db-wal", "borg.db-shm", "borg.db-journal"]);
  await cp(tenantDir, partial, {
    recursive: true,
    errorOnExist: true,
    force: false,
    preserveTimestamps: true,
    filter: (path) =>
      // Advisory-lock anchors and their write-probe journals are process state.
      // Opening/closing an anchor via fs here would release its POSIX lock.
      !basename(path).endsWith(FILE_LOCK_GUARD_SUFFIX) &&
      !basename(path).endsWith(`${FILE_LOCK_GUARD_SUFFIX}-journal`) &&
      !(
        dirname(path) === tenantDir &&
        (sqliteFiles.has(basename(path)) || isMigrationBookkeeping(basename(path)))
      ),
  });
  const db = openReadOnlyDatabase(join(tenantDir, "borg.db"));
  try {
    await db.raw.backup(join(partial, "borg.db"));
  } finally {
    db.close();
  }
  const hashes: Record<string, string> = {};
  // Copy verification is streaming; no large attachment or Lance file in RAM.
  for (const file of bankFiles(partial)) {
    const fd = openSync(file.path, "r");
    try {
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
    const name = relative(partial, file.path);
    hashes[name] = await fileChecksum(file.path);
    if (!sqliteFiles.has(name)) {
      if ((await fileChecksum(join(tenantDir, name))) !== hashes[name])
        throw new EmbeddingBankError(`Backup copy differs: ${name}`, {
          code: "EMBEDDING_MIGRATION_BACKUP_INVALID",
        });
    }
    syncDirectory(dirname(file.path));
  }
  writeJsonFileAtomic(
    join(partial, ".embedding-backup.json"),
    { version: 1, created_at: Date.now(), inventory_hash: inventoryHash, files: hashes },
    { mode: 0o600 },
  );
  await verifyTenantBackup(partial, inventory, targetDims);
  renameSync(partial, destination);
  syncDirectory(dirname(destination));
}

export async function verifyTenantBackup(
  destination: string,
  inventory: BankInventory,
  targetDims: number,
): Promise<void> {
  const marker = backupMarkerSchema.safeParse(
    readJsonFile(join(destination, ".embedding-backup.json")),
  );
  if (
    !marker.success ||
    marker.data.inventory_hash !== fingerprintCanonicalValue(inventory).canonicalSha256
  ) {
    throw new EmbeddingBankError("Backup is missing a valid completion manifest", {
      code: "EMBEDDING_MIGRATION_BACKUP_INVALID",
    });
  }
  const snapshot = await inventoryBank(destination, targetDims);
  if (
    fingerprintCanonicalValue(snapshot).canonicalSha256 !==
    fingerprintCanonicalValue(inventory).canonicalSha256
  ) {
    throw new EmbeddingBankError(
      "Backup failed SQLite counts/integrity or exact bank inventory verification",
      { code: "EMBEDDING_MIGRATION_BACKUP_INVALID" },
    );
  }
  for (const [name, expected] of Object.entries(marker.data.files)) {
    if ((await fileChecksum(join(destination, name))) !== expected)
      throw new EmbeddingBankError(`Backup checksum failed: ${name}`, {
        code: "EMBEDDING_MIGRATION_BACKUP_INVALID",
      });
  }
}
