import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import {
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { gzipSync } from "node:zlib";
import { connect } from "@lancedb/lancedb";
import { Bool, Field, Schema, Utf8 } from "apache-arrow";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createCommitmentId,
  createEntityId,
  createEpisodeId,
  createGoalId,
} from "../src/util/ids.js";
import { createEpisodesTableSchema } from "../src/memory/episodic/repository.js";
import { generateLegacyResidueReport, main } from "./legacy-residue-report.js";
import { cacheDirectory, openEpisodeTable, openSqliteSnapshot } from "./legacy-residue/bank.js";
import { episodeChecks, fileChecks } from "./legacy-residue/file-checks.js";
import { outcomeChecks, repairChecks } from "./legacy-residue/repair-checks.js";
import { ReportChecks, type ResidueCheck } from "./legacy-residue/report.js";
import { sqliteChecks } from "./legacy-residue/sqlite-checks.js";

const temporary: string[] = [];
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  for (const path of temporary.splice(0)) rmSync(path, { recursive: true, force: true });
});

function temporaryRoot() {
  const root = mkdtempSync(join(tmpdir(), "borg-legacy-residue-test-"));
  temporary.push(root);
  return root;
}

function fixture(root = temporaryRoot(), name = "tenant-1") {
  const bank = join(root, name);
  const cache = join(root, "cache");
  mkdirSync(bank);
  const db = new DatabaseSync(join(bank, "borg.db"));
  db.exec(`
    CREATE TABLE stream_entry_index (entry_id TEXT PRIMARY KEY, session_id TEXT, kind TEXT, turn_id TEXT, entry_index INTEGER, timestamp INTEGER, source_message_key_source_type TEXT);
    CREATE TABLE commitments (id TEXT PRIMARY KEY, kind TEXT, enforcement_class TEXT, critical_domain TEXT, updated_at INTEGER, restricted_audience TEXT, priority INTEGER DEFAULT 1, directive TEXT DEFAULT 'Private test content', revoked_at INTEGER, expired_at INTEGER, superseded_by TEXT);
    CREATE TABLE identity_events (id INTEGER PRIMARY KEY, record_type TEXT, record_id TEXT, action TEXT, old_value_json TEXT, new_value_json TEXT, ts INTEGER DEFAULT 1, provenance_kind TEXT DEFAULT 'manual', provenance_episode_ids TEXT DEFAULT '[]', provenance_stream_entry_ids TEXT DEFAULT '[]', provenance_process TEXT, review_item_id INTEGER, overwrite_without_review INTEGER DEFAULT 0, reason TEXT);
    CREATE TABLE maintenance_audit (id INTEGER PRIMARY KEY, reversal TEXT, reverted_at INTEGER, run_id TEXT, process TEXT DEFAULT 'consolidator', action TEXT DEFAULT 'consolidate', targets TEXT DEFAULT '{}', applied_at INTEGER DEFAULT 1, reverted_by TEXT);
    CREATE TABLE review_queue (id INTEGER PRIMARY KEY, kind TEXT, refs TEXT, resolved_at INTEGER, resolution TEXT);
    CREATE TABLE stream_watermarks (process_name TEXT, session_id TEXT, last_entry_id TEXT, last_ts INTEGER);
    CREATE TABLE shared_state_entries (id TEXT PRIMARY KEY, state_key TEXT);
    CREATE TABLE sessions (session_id TEXT PRIMARY KEY, source_type TEXT, source_external_id TEXT, source_url TEXT, label TEXT, audience_label TEXT, audience_entity_id TEXT, conversation_kind TEXT, last_turn_id TEXT);
    CREATE TABLE entities (id TEXT PRIMARY KEY, kind TEXT, created_at INTEGER DEFAULT 1, canonical_name TEXT DEFAULT 'Synthetic entity');
    CREATE TABLE goals (id TEXT PRIMARY KEY, record_version INTEGER DEFAULT 1, description TEXT DEFAULT 'Private test content', status TEXT DEFAULT 'active', priority REAL DEFAULT 1, created_at INTEGER DEFAULT 1, target_at INTEGER, owner_entity_id TEXT, audience_entity_id TEXT);
    CREATE TABLE open_question_ruminations (id INTEGER PRIMARY KEY, tensions TEXT);
  `);
  const self = createEntityId();
  const speaker = createEntityId();
  const from = createEntityId();
  const to = createEntityId();
  for (const [id, kind] of [
    [self, "self"],
    [speaker, "person"],
    [from, "group"],
    [to, "group"],
  ])
    db.prepare("INSERT INTO entities (id, kind) VALUES (?, ?)").run(id!, kind!);
  const commitmentIds = Array.from({ length: 7 }, createCommitmentId);
  const commitmentFields = [
    [null, "advisory", null, 1],
    ["assistant_commitment", null, null, 1],
    ["boundary", "critical", null, 1],
    ["boundary", "critical", "privacy", 1],
    ["assistant_commitment", "advisory", null, 1],
    ["assistant_commitment", "advisory", null, null],
    ["assistant_commitment", "critical", null, 1],
  ];
  for (const [index, fields] of commitmentFields.entries())
    db.prepare(
      "INSERT INTO commitments (id, kind, enforcement_class, critical_domain, updated_at) VALUES (?, ?, ?, ?, ?)",
    ).run(
      commitmentIds[index]!,
      ...(fields as [string | null, string | null, string | null, number | null]),
    );
  db.prepare("UPDATE commitments SET restricted_audience = ? WHERE id = ?").run(
    from,
    commitmentIds[0]!,
  );
  const modern = {
    kind: "assistant_commitment",
    enforcement_class: "advisory",
    critical_domain: null,
    updated_at: 1,
  };
  const snapshots = [
    [{}, modern],
    [modern, { ...modern, updated_at: null }],
    [modern, modern],
    [null, null],
    [{}, {}],
  ];
  for (const [index, pair] of snapshots.entries())
    db.prepare(
      "INSERT INTO identity_events (id, record_type, record_id, action, old_value_json, new_value_json) VALUES (?, 'commitment', 'snapshot', 'update', ?, ?)",
    ).run(index + 1, JSON.stringify(pair[0]), JSON.stringify(pair[1]));
  const promoted = createGoalId();
  const other = createGoalId();
  const orphan = createGoalId();
  db.prepare(
    "INSERT INTO goals (id, target_at, owner_entity_id, audience_entity_id) VALUES (?, 10, ?, ?)",
  ).run(promoted, speaker, from);
  db.prepare("INSERT INTO goals (id, status, owner_entity_id) VALUES (?, 'done', ?)").run(
    other,
    speaker,
  );
  for (const [index, goal] of [promoted, other, orphan].entries())
    db.prepare(
      "INSERT INTO identity_events (id, record_type, record_id, action, new_value_json, provenance_process) VALUES (?, 'goal', ?, 'create', ?, ?)",
    ).run(
      10 + index,
      goal,
      JSON.stringify({ owner_entity_id: speaker, status: "active" }),
      index === 0 ? "goal-promotion-extractor" : null,
    );
  for (const [index, reversal] of [
    { newEpisodeId: "private-episode-1" },
    { newEpisodeId: "private-episode-2" },
    { versionEpisodeId: "current-version" },
    { nested: { newEpisodeId: "not-root" } },
  ].entries())
    db.prepare("INSERT INTO maintenance_audit (id, reversal, reverted_at) VALUES (?, ?, ?)").run(
      index + 1,
      JSON.stringify(reversal),
      index === 1 ? 100 : null,
    );
  const completeRefs = {
    target_type: "episode",
    target_id: createEpisodeId(),
    overseer_flag: {
      kind: "misattribution",
      flag_kind: "misattribution",
      reason: "Synthetic reason",
      confidence: 1,
      audience_entities: [],
      quoted_span: "Synthetic quote",
      cited_stream_ids: [],
      source_assessment: "supports_flag",
    },
  };
  const reviews = [
    ["relationship_claim_ungrounded", {}, null],
    ["relationship_claim_ungrounded", {}, 1],
    ["misattribution", {}, null],
    [
      "misattribution",
      { ...completeRefs, overseer_flag: { ...completeRefs.overseer_flag, quoted_span: undefined } },
      null,
    ],
    [
      "misattribution",
      {
        ...completeRefs,
        overseer_flag: { ...completeRefs.overseer_flag, cited_stream_ids: undefined },
      },
      null,
    ],
    [
      "misattribution",
      {
        ...completeRefs,
        overseer_flag: { ...completeRefs.overseer_flag, source_assessment: undefined },
      },
      1,
    ],
    ["misattribution", completeRefs, null],
    ["misattribution", { ...completeRefs, target_type: "unsupported" }, null],
    [
      "misattribution",
      { ...completeRefs, overseer_flag: { ...completeRefs.overseer_flag, quoted_span: "" } },
      null,
    ],
    ["other", { __borg_review_resolver_diagnostic: { reason: "Synthetic" } }, null],
    ["other", { __borg_review_resolver_diagnostic: { attempts: 1 } }, null],
    ["other", { __borg_review_resolver_diagnostic: { attempts: 0 } }, null],
    ["other", { __borg_review_resolver_diagnostic: null }, null],
  ];
  for (const [index, row] of reviews.entries())
    db.prepare("INSERT INTO review_queue (id, kind, refs, resolved_at) VALUES (?, ?, ?, ?)").run(
      index + 1,
      row[0] as string,
      JSON.stringify(row[1]),
      row[2] as number | null,
    );
  db.exec(`
    INSERT INTO stream_watermarks VALUES ('autonomy:goal-followup-due:old', 's', 'old', 0), ('autonomy:goal-followup-due:new:deadline', 's', 'old', 0), ('autonomy:goal-followup-due:new:stale', 's', 'old', 0), ('unrelated:old', 's', 'old', 0), ('autonomy:goal-followup-due:new:deadline:extra', 's', 'old', 0), ('chat-response', 'inbox', 'cursor', 20);
    INSERT INTO shared_state_entries VALUES ('private-state-1', NULL), ('private-state-2', 'key');
    INSERT INTO sessions VALUES ('inbox', 'teams_inbox', 'conversation-1', NULL, 'Label', 'Audience', 'entity-1', 'dm', NULL), ('broken', 'teams_inbox', NULL, NULL, '', '', NULL, 'dm', NULL), ('empty', 'demo', '', '', 'Label', 'Audience', NULL, 'demo', ''), ('plain', 'demo', NULL, NULL, 'Label', 'Audience', NULL, 'demo', NULL), ('unwatermarked', 'teams_inbox', 'conversation-2', NULL, 'Label', 'Audience', 'entity-1', 'channel', NULL);
    INSERT INTO stream_entry_index VALUES ('private-index-1', 'plain', NULL, NULL, 0, 0, NULL), ('private-index-2', 'plain', NULL, NULL, 1, 1, NULL), ('before', 'inbox', 'user_msg', NULL, 1, 10, NULL), ('cursor', 'inbox', 'user_msg', NULL, 2, 20, 'teams_inbox'), ('after', 'inbox', 'user_msg', NULL, 3, 30, NULL), ('modern', 'inbox', 'user_msg', NULL, 4, 40, 'teams_inbox'), ('unwatermarked-user', 'unwatermarked', 'user_msg', NULL, 0, 10, NULL), ('assigned', 'inbox', 'user_msg', 'turn-1', 0, 0, NULL);
  `);
  for (const [index, tensions] of [
    ['<parameter name="tensions">["Synthetic one","Synthetic two"]</parameter>'],
    ['<parameter name="growth_marker">Synthetic marker</parameter>'],
    ["Ordinary synthetic tension"],
  ].entries())
    db.prepare("INSERT INTO open_question_ruminations VALUES (?, ?)").run(
      index + 1,
      JSON.stringify(tensions),
    );
  db.close();
  return { bank, cache, from, to, promoted, other };
}

function check(checks: ResidueCheck[], id: string) {
  const value = checks.find((item) => item.id === id);
  expect(value, id).toBeDefined();
  return value!;
}

function fingerprint(directory: string): unknown {
  return readdirSync(directory, { withFileTypes: true })
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((entry) => {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) return [entry.name, fingerprint(path)];
      const stat = statSync(path);
      return [
        entry.name,
        stat.size,
        stat.mtimeMs,
        createHash("sha256").update(readFileSync(path)).digest("hex"),
      ];
    });
}

async function withSqliteChecks(bank: string, cache: string) {
  const opened = openSqliteSnapshot(bank, cache);
  const report = new ReportChecks(() => {});
  try {
    await sqliteChecks(opened.db, report);
    return report.checks;
  } finally {
    opened.close();
  }
}

async function seedEpisodes(bank: string, legacyColumns = false) {
  const connection = await connect(join(bank, "lancedb"));
  const names = legacyColumns
    ? ["id", "audience_entity_id", "episode_kind"]
    : [
        "id",
        "audience_entity_id",
        "origin_audience_entity_ids",
        "episode_kind",
        "consolidation_embedding_input",
      ];
  const schema = new Schema([
    ...names.map((name) => new Field(name, new Utf8(), true)),
    new Field("archived", new Bool(), false),
  ]);
  const rows = legacyColumns
    ? [
        {
          id: "private-episode",
          audience_entity_id: "entity",
          episode_kind: "consolidation_version",
          archived: true,
        },
      ]
    : [
        ["one", "entity", null, "consolidation_version", null, true],
        ["two", "entity", "", "raw", null, false],
        ["three", "entity", " [ ] ", "raw", null, false],
        ["four", "entity", '["entity"]', "consolidation_version", "{}", false],
        ["five", null, "[]", "raw", null, false],
        ["six", "entity", "null", "consolidation_version", "null", false],
        ["seven", "entity", '["entity"]', "consolidation_version", null, false],
      ].map((values) =>
        Object.fromEntries([...names, "archived"].map((name, index) => [name, values[index]])),
      );
  const table = await connection.createTable("episodes", rows, { schema });
  table.close();
  connection.close();
}

describe("legacy residue report", () => {
  it("counts SQLite legacy rows and their real status/cursor semantics", async () => {
    const { bank, cache } = fixture();
    const checks = await withSqliteChecks(bank, cache);
    const expected: Record<string, number> = {
      L2: 2,
      "L6.commitments": 4,
      "L6.identity_events": 3,
      L7: 2,
      "L7.unreverted": 1,
      L8: 2,
      "L8.open": 1,
      "L8.resolved": 1,
      L9: 6,
      "L9.open": 5,
      L10: 2,
      L11: 2,
      L14: 1,
      "S4.metadata": 1,
      S5: 2,
      "S4.backlog_unknown_order": 0,
      "S4.pending_pre_inbox_backlog": 2,
      "S4.backlog_before_watermark": 1,
    };
    for (const [id, count] of Object.entries(expected))
      expect(check(checks, id).count, id).toBe(count);
    expect(check(checks, "L2").total).toBe(8);
    expect(check(checks, "L6.commitments").total).toBe(7);
    expect(check(checks, "L6.identity_events").total).toBe(5);
    for (const row of checks) {
      expect(row.query.length).toBeGreaterThan(10);
      expect(row.sample?.length ?? 0).toBeLessThanOrEqual(3);
      for (const sample of row.sample ?? []) expect(sample).toMatch(/^sha256:[a-f0-9]{12}$/);
    }
    expect(JSON.stringify(checks)).not.toContain("private-index");
    expect(JSON.stringify(checks)).not.toContain("Private test content");
  });

  it("opens SQLite with readOnly even when query_only is disabled and preserves WAL sidecars", () => {
    const root = temporaryRoot();
    const bank = join(root, "tenant");
    mkdirSync(bank);
    const writer = new DatabaseSync(join(bank, "borg.db"));
    writer.exec(
      "PRAGMA journal_mode = WAL; PRAGMA wal_autocheckpoint = 0; CREATE TABLE evidence (id INTEGER); INSERT INTO evidence VALUES (1)",
    );
    const before = fingerprint(bank);
    const snapshot = openSqliteSnapshot(bank, join(root, "cache"));
    try {
      expect(snapshot.db.prepare("SELECT COUNT(*) AS n FROM evidence").get()?.n).toBe(1);
      snapshot.db.pragma("query_only = OFF");
      expect(() => snapshot.db.exec("INSERT INTO evidence VALUES (2)")).toThrow(
        /readonly|read-only/i,
      );
      expect(fingerprint(bank)).toEqual(before);
    } finally {
      snapshot.close();
      writer.close();
    }
    expect(readdirSync(join(root, "cache"))).toEqual([]);
  });

  it("honors the cache environment and refuses bank/app paths and symlink aliases", () => {
    const { bank, cache } = fixture();
    vi.stubEnv("XDG_CACHE_HOME", cache);
    const snapshot = openSqliteSnapshot(bank);
    try {
      expect(snapshot.directory.startsWith(cache)).toBe(true);
    } finally {
      snapshot.close();
    }
    expect(() => openSqliteSnapshot(bank, join(bank, "cache"))).toThrow(/outside/);
    expect(() => cacheDirectory(bank, process.cwd())).toThrow(/outside/);
    const alias = join(temporaryRoot(), "alias");
    symlinkSync(bank, alias);
    expect(() => openSqliteSnapshot(bank, join(alias, "cache"))).toThrow(/outside/);
  });

  it("refuses a bank with a rollback journal instead of reading a possibly uncommitted database", () => {
    const { bank, cache } = fixture();
    writeFileSync(join(bank, "borg.db-journal"), "synthetic unfinished transaction");
    const before = fingerprint(bank);
    expect(() => openSqliteSnapshot(bank, cache)).toThrow(/rollback journal/);
    expect(fingerprint(bank)).toEqual(before);
  });

  it("reports absent commitment columns as residue and missing tables as unavailable", async () => {
    const { bank, cache } = fixture();
    const db = new DatabaseSync(join(bank, "borg.db"));
    db.exec(
      "DROP TABLE commitments; CREATE TABLE commitments (id TEXT); INSERT INTO commitments VALUES ('old'); DROP TABLE shared_state_entries",
    );
    db.close();
    const checks = await withSqliteChecks(bank, cache);
    expect(check(checks, "L6.commitments")).toMatchObject({ count: 1, total: 1 });
    expect(check(checks, "L14").count).toBeNull();
  });

  it("fails a malformed JSON check without hiding other counts or exposing its payload", async () => {
    const { bank, cache } = fixture();
    const db = new DatabaseSync(join(bank, "borg.db"));
    db.exec("UPDATE identity_events SET old_value_json = 'private-invalid-json' WHERE id = 1");
    db.close();
    const checks = await withSqliteChecks(bank, cache);
    expect(check(checks, "L6.identity_events").count).toBeNull();
    expect(check(checks, "L2").count).toBe(2);
    expect(JSON.stringify(checks)).not.toContain("private-invalid-json");
  });

  it("does not guess pending backlog when its watermark cannot be resolved", async () => {
    const { bank, cache } = fixture();
    const db = new DatabaseSync(join(bank, "borg.db"));
    db.exec(
      "UPDATE stream_watermarks SET last_entry_id = 'missing' WHERE process_name = 'chat-response'",
    );
    db.close();
    const checks = await withSqliteChecks(bank, cache);
    expect(check(checks, "S4.backlog_unknown_order").count).toBe(2);
    expect(check(checks, "S4.pending_pre_inbox_backlog").count).toBeNull();
  });

  it("reuses all SQLite repair planners on scratch without applying their candidates", async () => {
    const data = fixture();
    const before = fingerprint(data.bank);
    const snapshot = openSqliteSnapshot(data.bank, data.cache);
    const report = new ReportChecks(() => {});
    try {
      await repairChecks(snapshot, report, {
        audience: { fromEntityIds: [data.from], toEntityId: data.to },
        targetGoalIds: [data.promoted, data.other],
      });
    } finally {
      snapshot.close();
    }
    const expected: Record<string, number> = {
      "R.migrate-audience-scoping": 2,
      "R.repair-goal-target-at": 1,
      "R.repair-goal-target-at.refusals": 0,
      "R.repair-goal-speaker-owner": 1,
      "R.repair-goal-rollback-audit": 1,
      "R.repair-goal-rollback-audit.status_drift": 1,
      "R.repair-rumination-scaffolding": 1,
      "R.repair-rumination-scaffolding.manual": 1,
    };
    for (const [id, count] of Object.entries(expected))
      expect(check(report.checks, id).count, check(report.checks, id).query).toBe(count);
    expect(fingerprint(data.bank)).toEqual(before);
  });

  it("counts archived LanceDB residue and enforces a read-only checkout", async () => {
    const { bank } = fixture();
    await seedEpisodes(bank);
    const before = fingerprint(bank);
    const opened = await openEpisodeTable(bank);
    const report = new ReportChecks(() => {});
    try {
      await episodeChecks(opened, report);
      expect(check(report.checks, "L5")).toMatchObject({ count: 4, total: 7 });
      expect(check(report.checks, "E2")).toMatchObject({ count: 3, total: 7 });
      await expect(opened.table.delete("true")).rejects.toThrow();
    } finally {
      opened.table.close();
      opened.connection.close();
    }
    expect(fingerprint(bank)).toEqual(before);
  });

  it("counts missing LanceDB columns without adding them", async () => {
    const { bank } = fixture();
    await seedEpisodes(bank, true);
    const before = fingerprint(bank);
    const opened = await openEpisodeTable(bank);
    const report = new ReportChecks(() => {});
    try {
      await episodeChecks(opened, report);
      expect(check(report.checks, "L5").count).toBe(1);
      expect(check(report.checks, "E2").count).toBe(1);
    } finally {
      opened.table.close();
      opened.connection.close();
    }
    expect(fingerprint(bank)).toEqual(before);
  });

  it("makes malformed episode origins unavailable instead of interpreting an object as an array", async () => {
    const { bank } = fixture();
    await seedEpisodes(bank);
    const connection = await connect(join(bank, "lancedb"));
    const table = await connection.openTable("episodes");
    await table.update({
      where: "id = 'one'",
      values: { origin_audience_entity_ids: '{"length":0}' },
    });
    table.close();
    connection.close();
    const opened = await openEpisodeTable(bank);
    const report = new ReportChecks(() => {});
    try {
      await episodeChecks(opened, report);
      expect(check(report.checks, "L5").count).toBeNull();
      expect(check(report.checks, "E2").count).toBe(3);
    } finally {
      opened.table.close();
      opened.connection.close();
    }
  });

  it("runs the outcome corpus dry-run through read-only dependencies without schema evolution", async () => {
    const { bank, cache } = fixture();
    const writer = new DatabaseSync(join(bank, "borg.db"));
    writer.exec(`
      DELETE FROM maintenance_audit;
      CREATE TABLE episode_stats (episode_id TEXT, retrieval_count INTEGER, use_count INTEGER, last_retrieved INTEGER, win_rate REAL, tier TEXT, promoted_at INTEGER, promoted_from TEXT, gist TEXT, gist_generated_at INTEGER, last_decayed_at INTEGER, heat_multiplier REAL, valence_mean REAL, archived INTEGER);
      CREATE TABLE episode_index (episode_id TEXT, archived INTEGER, episode_kind TEXT, consolidation_family_id TEXT);
      CREATE TABLE consolidation_families (family_id TEXT, current_version_episode_id TEXT, coverage_hash TEXT, policy_version INTEGER, created_at INTEGER, updated_at INTEGER);
      CREATE TABLE consolidation_members (family_id TEXT, raw_episode_id TEXT, source_stream_ids_json TEXT, added_by_version_episode_id TEXT);
      CREATE TABLE semantic_nodes (id TEXT, label TEXT, description TEXT, source_episode_ids TEXT, archived INTEGER, status TEXT, corrected_by TEXT, superseded_at INTEGER);
    `);
    writer.close();
    const connection = await connect(join(bank, "lancedb"));
    const empty = await connection.createEmptyTable("episodes", createEpisodesTableSchema(2));
    empty.close();
    connection.close();
    const before = fingerprint(bank);
    const snapshot = openSqliteSnapshot(bank, cache);
    const opened = await openEpisodeTable(bank);
    const report = new ReportChecks(() => {});
    try {
      await outcomeChecks(snapshot, opened, report);
      expect(
        check(report.checks, "R.migrate-outcome-corpus").count,
        JSON.stringify(report.checks),
      ).toBe(0);
      expect(check(report.checks, "R.migrate-outcome-corpus.raw_outcomes").count).toBe(0);
      expect(check(report.checks, "R.migrate-outcome-corpus.unsafe_items").count).toBeGreaterThan(
        0,
      );
    } finally {
      opened.table.close();
      opened.connection.close();
      snapshot.close();
    }
    expect(fingerprint(bank)).toEqual(before);
  });

  it("counts capture versions across plain/gzip rotations and reads migration completion", async () => {
    const { bank } = fixture();
    const directory = join(bank, "captures");
    mkdirSync(directory);
    writeFileSync(join(bank, ".embedding-migration.json"), JSON.stringify({ phase: "complete" }));
    writeFileSync(
      join(directory, "finalizer-contexts.jsonl"),
      '{"schema_version":2}\n{"schema_version":1}\n{}\ninvalid-json\n',
    );
    const rotated = join(directory, "finalizer-contexts.jsonl.rotated-20260101T000000.000Z");
    writeFileSync(rotated, '{"schema_version":1}\n');
    writeFileSync(`${rotated}.gz`, gzipSync('{"schema_version":1}\n'));
    writeFileSync(
      join(directory, "finalizer-contexts.jsonl.rotated-20260102T000000.000Z.gz"),
      gzipSync('{"schema_version":2}\n{"schema_version":3}'),
    );
    writeFileSync(`${rotated}.gz.partial`, "ignored");
    symlinkSync(rotated, join(directory, "finalizer-contexts.jsonl.1"));
    const before = fingerprint(bank);
    const report = new ReportChecks(() => {});
    await fileChecks(bank, report);
    for (const [id, count] of Object.entries({
      E3: 1,
      "E3.exists": 1,
      C4: 7,
      "C4.files": 3,
      "C4.schema_version.1": 2,
      "C4.schema_version.2": 2,
      "C4.schema_version.3": 1,
      "C4.schema_version.missing": 1,
      "C4.schema_version.invalid_json": 1,
      "C4.files.schema_version.1": 2,
    }))
      expect(check(report.checks, id).count, id).toBe(count);
    expect(fingerprint(bank)).toEqual(before);
  });

  it("leaves bank files and lock sentinels unchanged during a complete report", async () => {
    const { bank, cache } = fixture();
    await seedEpisodes(bank);
    writeFileSync(join(bank, "borg.lock"), "lock-sentinel");
    writeFileSync(join(bank, "embedding-migration.lock"), "migration-lock-sentinel");
    const before = fingerprint(bank);
    const report = await generateLegacyResidueReport(bank, {
      cacheDir: cache,
      diagnostic: () => {},
    });
    expect(report.tenant).toBe("tenant-1");
    expect(Number.isFinite(Date.parse(report.generated_at))).toBe(true);
    expect(check(report.checks, "R.migrate-audience-scoping")).toMatchObject({
      count: null,
      query: expect.stringContaining("SKIPPED"),
    });
    expect(fingerprint(bank)).toEqual(before);
    expect(readdirSync(cache)).toEqual([]);
  });

  it("prints one JSON document per tenant, keeps diagnostics separate, and continues on unavailable storage", async () => {
    const root = temporaryRoot();
    fixture(root, "tenant-2");
    fixture(root, "tenant-1");
    mkdirSync(join(root, "not-a-bank"));
    const cache = join(root, "cache");
    let stdout = "";
    let stderr = "";
    const code = await main(["--all-tenants", root, "--cache-dir", cache], {
      stdout: (text) => {
        stdout += text;
        return true;
      },
      stderr: (text) => {
        stderr += text;
        return true;
      },
    });
    const documents = stdout
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as { tenant: string; checks: ResidueCheck[] });
    expect(documents.map((report) => report.tenant)).toEqual(["tenant-1", "tenant-2"]);
    expect(code).toBe(1);
    expect(stderr).toContain("reading bank");
    expect(check(documents[0]!.checks, "storage.episodes").count).toBeNull();
    await expect(main(["--bank", root, "--apply"])).rejects.toThrow();
  });

  it("launches with the app's installed tsx and scratch HOME while stdout stays JSON", () => {
    const { bank, cache } = fixture();
    const home = temporaryRoot();
    const before = fingerprint(bank);
    const child = spawnSync(
      resolve("node_modules/.bin/tsx"),
      [resolve("scripts/legacy-residue-report.ts"), "--bank", bank, "--cache-dir", cache],
      {
        cwd: home,
        env: {
          ...process.env,
          HOME: home,
          XDG_CACHE_HOME: cache,
          TMPDIR: home,
          TSX_DISABLE_CACHE: "1",
        },
        encoding: "utf8",
      },
    );
    expect(child.error).toBeUndefined();
    expect(child.status).toBe(1); // This synthetic bank deliberately has no LanceDB.
    const document = JSON.parse(child.stdout) as { tenant: string; checks: ResidueCheck[] };
    expect(document.tenant).toBe("tenant-1");
    expect(check(document.checks, "L2").count).toBe(2);
    expect(child.stderr).toContain("[legacy-residue]");
    expect(fingerprint(bank)).toEqual(before);
  });
});
