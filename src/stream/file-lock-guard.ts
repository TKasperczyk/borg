import { DatabaseSync } from "node:sqlite";

// Keep this inode forever: unlinking it while a connection is open would let
// another process lock a different inode at the same path. No application data
// is stored here; a verified write transaction holds SQLite's OS-backed RESERVED lock.
// Rollback mode (the default), not WAL, supports cross-host filesystem locking.
export const FILE_LOCK_GUARD_SUFFIX = ".guard.sqlite";

/**
 * Serialize the entire lease lifetime, including its metadata compare/unlink or
 * atomic replacement. Exactly one writable connection can hold BEGIN IMMEDIATE. Closing
 * the connection or process death releases it without a stale claim to reap.
 *
 * The filesystem must implement SQLite's advisory locks across all PVC clients.
 * Never open/copy the guard using ordinary fs APIs in a holder process: on POSIX,
 * closing any fd for this inode releases that process's advisory locks. Backups
 * exclude these empty companions. See https://sqlite.org/lockingv3.html.
 */
export function tryAcquireFileLockGuard(lockPath: string): DatabaseSync | null {
  const guardPath = `${lockPath}${FILE_LOCK_GUARD_SUFFIX}`;
  const guard = new DatabaseSync(guardPath, { readOnly: false });
  try {
    // Do not block the event loop (and other leases' heartbeats) on contention.
    guard.exec("PRAGMA busy_timeout = 0; BEGIN IMMEDIATE");
    // SQLite may silently fall back to a read-only handle even when read-write
    // was requested; BEGIN IMMEDIATE alone can then succeed without a RESERVED
    // lock. Node 22 exposes no sqlite3_db_readonly(), so force a main-database
    // header write and verify it inside the retained transaction. This cannot
    // succeed read-only and is rolled back on close (never committed).
    guard.exec("PRAGMA main.user_version = 1");
    if (
      !guard.isTransaction ||
      guard.prepare("PRAGMA main.user_version").get()?.user_version !== 1
    ) {
      throw new Error(`File lock guard has no verified write transaction at ${guardPath}`);
    }
    return guard;
  } catch (error) {
    guard.close();
    if (error instanceof Error && "errcode" in error && error.errcode === 5) {
      // SQLITE_BUSY: another holder/reaper owns the stable guard inode.
      return null;
    }
    if (error instanceof Error && "errcode" in error && Number(error.errcode) % 256 === 8) {
      // SQLITE_READONLY (including extended codes): never report ownership.
      throw new Error(`File lock guard is not writable at ${guardPath}`, { cause: error });
    }
    throw error;
  }
}
