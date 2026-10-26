import { copyFileSync, mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { createMigrations } from "../src/borg/storage-setup.js";
import { openDatabase } from "../src/storage/sqlite/index.js";
import { FixedClock } from "../src/util/clock.js";
import { AuditLog } from "../src/offline/audit-log.js";

export function repairFixture(liveShape = false) {
  const fixtureRoot = mkdtempSync(join(process.cwd(), ".borg-test-repair-"));
  // A synthetic path under this disposable fixture, never the real demo store.
  const dataDir = liveShape
    ? join(fixtureRoot, "demo", "server", ".borg-data", "fixture")
    : fixtureRoot;
  mkdirSync(dataDir, { recursive: true });
  const db = openDatabase(join(dataDir, "borg.db"), { migrations: createMigrations() });
  const clock = new FixedClock(1000);
  return {
    dataDir,
    fixtureRoot,
    db,
    clock,
    audit: new AuditLog({ db, clock }),
    cleanup() {
      db.close();
      rmSync(fixtureRoot, { recursive: true, force: true });
    },
  };
}

export function repairBackupFixture(dataDir: string, filename = "borg.db") {
  const path = mkdtempSync(join(process.cwd(), ".borg-test-repair-backup-"));
  copyFileSync(join(dataDir, "borg.db"), join(path, filename));
  return { path, cleanup: () => rmSync(path, { recursive: true, force: true }) };
}
