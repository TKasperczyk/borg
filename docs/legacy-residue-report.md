# Legacy residue report

Run `node --import tsx scripts/legacy-residue-report.ts --bank /data/banks/tenant-1`, or use `--all-tenants /data/banks` for one JSON document per immediate tenant directory containing `borg.db` or `lancedb` (symlink directories are excluded). Each check includes its SQL or precise counting logic, a count, the indicated population total where available, and at most three IDs redacted as `sha256:` plus the first 12 hex characters of SHA-256 of the string ID. No stored text is printed. `count: null` means unavailable or explicitly skipped, never zero residue; failures are explained in `query` and on stderr. Exit 1 indicates a failed read or validation; detected residue and checks skipped for missing operator inputs exit 0. The report never opens Borg, migrates, repairs, or opens bank lock files. It copies only SQLite's main file and WAL to external scratch, rejects a nonempty rollback journal and changes detected during that copy, and opens the copy with `DatabaseSync` read-only via the existing opener; any SQLite sidecars stay in scratch. LanceDB uses a fixed-version read-only checkout. Scratch is removed on normal exit; an interrupted process can leave a private `borg-legacy-residue-*` directory in the selected cache. Use an idle bank or filesystem snapshot for deletion evidence: SQLite, LanceDB and capture reads are separate observations, not an atomic cross-store snapshot. Ensure scratch has room for the database plus WAL. `--cache-dir` overrides `XDG_CACHE_HOME`, otherwise the OS temp directory is used; bank/app paths and symlink aliases into them are rejected.

Inside a pod, use the app's installed `tsx` without `npx` or a package install. Set HOME before launching it, since the loader starts before the script. Adjust `/app` and the tenant root to the mounted paths:

```sh
mkdir -p /tmp/x/cache
HOME=/tmp/x XDG_CACHE_HOME=/tmp/x/cache TMPDIR=/tmp/x TSX_DISABLE_CACHE=1 \
  /app/node_modules/.bin/tsx /app/scripts/legacy-residue-report.ts \
  --all-tenants /data/banks > /tmp/x/residue.jsonl
```

If only `/data` is writable, replace `/tmp/x` everywhere with `/data/legacy-report-scratch`, outside every tenant bank (it may be a sibling under the `--all-tenants` root). Keep diagnostics on stderr. For all five production banks, retain all five JSON documents and check every unavailable count before removing fallbacks. `E3` is a presence/completion indicator (1 is complete), and C4 buckets count records and files by schema version, so positive numbers there are not universally residue. Archived episodes are included. L6 accepts legitimate null critical domains. L8 status is derived from `resolved_at`. S4 treats nonempty inbox external ID, audience ID, labels and a valid conversation kind as complete source metadata; nullable URL/last-turn fields are not missing metadata. Its backlog checks use the indexed durable order: entries at/before the chat-response watermark are the covered prefix; pending pre-inbox entries lie after it or in sessions without a watermark. These counts do not hydrate streams or reconcile terminal stamps. Capture scanning reads one JSONL record at a time (maximum 128 MiB), supports gzip rotations without extracting them, ignores partial compression files, and counts a plain/gzip archive pair once.

The existing SQLite repair planners and outcome-corpus dry-run runner are reused with read-only handles; no CLI apply path is invoked. Outcome planning uses the original default corpus specification and can require more memory because its existing planner materializes episodes, including vectors. Candidate-operation counts and unsafe-item counts are separate. Audience-scoping and deadline repair require explicit operator-selected IDs; they are skipped unless `--repair-inputs /tmp/x/repair-inputs.json` provides a tenant-keyed document like this (substitute real handles). The input document's SHA-256 is recorded for audit; its contents are not printed.

```json
{
  "tenant-1": {
    "audience": {
      "fromEntityIds": ["ent_0000000000000001"],
      "toEntityId": "ent_0000000000000002"
    },
    "targetGoalIds": ["goal_0000000000000001"]
  }
}
```
