import { DatabaseSync } from "node:sqlite";

// Keep this inode forever: unlinking it while a connection is open would let
// another process lock a different inode at the same path. No application data
// is stored here; BEGIN IMMEDIATE only takes SQLite's OS-backed RESERVED lock.
// Rollback mode (the default), not WAL, supports cross-host filesystem locking.
export const FILE_LOCK_GUARD_SUFFIX = ".guard.sqlite";

/**
 * Serialize the entire lease lifetime, including its metadata compare/unlink or
 * atomic replacement. Exactly one connection can hold BEGIN IMMEDIATE. Closing
 * the connection or process death releases it without a stale claim to reap.
 *
 * The filesystem must implement SQLite's advisory locks across all PVC clients.
 * Never open/copy the guard using ordinary fs APIs in a holder process: on POSIX,
 * closing any fd for this inode releases that process's advisory locks. Backups
 * exclude these empty companions. See https://sqlite.org/lockingv3.html.
 */
export function tryAcquireFileLockGuard(lockPath: string): DatabaseSync | null {
  const guard = new DatabaseSync(`${lockPath}${FILE_LOCK_GUARD_SUFFIX}`);
  try {
    // Do not block the event loop (and other leases' heartbeats) on contention.
    guard.exec("PRAGMA busy_timeout = 0; BEGIN IMMEDIATE");
    return guard;
  } catch (error) {
    guard.close();
    if (error instanceof Error && "errcode" in error && error.errcode === 5) {
      // SQLITE_BUSY: another holder/reaper owns the stable guard inode.
      return null;
    }
    throw error;
  }
}
