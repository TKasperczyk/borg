import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { createMigrations } from "../src/borg/storage-setup.js";
import { openDatabase } from "../src/storage/sqlite/index.js";
import { FixedClock } from "../src/util/clock.js";
import { AuditLog } from "../src/offline/audit-log.js";

export function repairFixture() {
  const dataDir = mkdtempSync(join(process.cwd(), ".borg-test-repair-"));
  const db = openDatabase(join(dataDir, "borg.db"), { migrations: createMigrations() });
  const clock = new FixedClock(1000);
  return {
    dataDir,
    db,
    clock,
    audit: new AuditLog({ db, clock }),
    cleanup() {
      db.close();
      rmSync(dataDir, { recursive: true, force: true });
    },
  };
}
