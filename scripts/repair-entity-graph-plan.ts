import { quoteSqlString } from "../src/storage/codecs.js";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { createMigrations } from "../src/borg/storage-setup.js";
import { createEmbeddingClient } from "../src/borg/clients.js";
import { loadConfig } from "../src/config/index.js";
import type { EmbeddingClient } from "../src/embeddings/index.js";
import { EntityRepository } from "../src/memory/commitments/repository.js";
import { IdentityEventRepository } from "../src/memory/identity/repository.js";
import { buildObservedEventRecurrenceKey } from "../src/memory/observed-events/emission.js";
import { GoalsRepository } from "../src/memory/self/goals-repository.js";
import {
  buildOpenQuestionDedupeKey,
  mapOpenQuestionRow,
} from "../src/memory/self/open-questions.js";
import {
  SemanticNodeRepository,
  SemanticEdgeRepository,
  edgeFromRow,
} from "../src/memory/semantic/repository.js";
import { buildNodeEmbeddingText } from "../src/memory/semantic/extractor.js";
import { semanticObservationMetadataSchema } from "../src/memory/semantic/types.js";
import { AuditLog } from "../src/offline/audit-log.js";
import { LanceDbTable } from "../src/storage/lancedb/index.js";
import { runMigrations, type SqliteDatabase } from "../src/storage/sqlite/index.js";
import { ManualClock } from "../src/util/clock.js";
import {
  entityIdHelpers,
  semanticNodeIdHelpers,
  parseEntityId,
  parseSemanticNodeId,
  createMaintenanceRunId,
  type MaintenanceRunId,
  type GoalId,
  type SessionId,
} from "../src/util/ids.js";
import type { JsonValue } from "../src/util/json-value.js";
import {
  type Row,
  type SqlChange,
  type LanceChange,
  fail,
  same,
  remap,
  sqlInventory,
  openRepairDatabase,
  openLance,
  planLance,
  quoteIdentifier,
  cloneDatabase,
  applySqlChanges,
  assertLeafDeletion,
  diffSql,
  digest,
  assertOne,
} from "./repair-entity-graph-storage.js";
import { validateSqlChanges, validateLanceChanges } from "./repair-entity-graph-validation.js";

export type Merge = { from: string; into: string; renameInto?: string };
export type RepairInput = { dataDir: string; merges: Merge[]; nodeMerges: Merge[] };
export type GoalGap = {
  id: string;
  version: number;
  before: Row;
  after: Row;
  ts: number;
  reason: string;
};
export type RepairPlan = {
  input: RepairInput;
  plannedAt: number;
  runId: MaintenanceRunId | null;
  migrations: Array<{ id: number; name: string }>;
  migrationSql: SqlChange[];
  schemaChanges: Array<{ name: string; type: string; before: string | null; after: string | null }>;
  protectedIndexHash: string;
  drainSql: SqlChange[];
  sql: SqlChange[];
  additionalSql: SqlChange[];
  completionSql: SqlChange[];
  lance: LanceChange[];
  files: never[];
  protectedFiles: Array<{ path: string; sha256: string }>;
  embedding: { model: string; dimensions: number; endpoint: string; injectedClient: boolean };
  goals: GoalGap[];
  touchedNodes: string[];
  inventory: { sqlite: Record<string, string[]>; lance: Record<string, string[]> };
  reports: Array<{ category: string; rows: Row[] }>;
};
function mergeMap(merges: Merge[], valid: (id: string) => boolean): Map<string, string> {
  const map = new Map<string, string>();
  for (const merge of merges) {
    if (
      !valid(merge.from) ||
      !valid(merge.into) ||
      merge.from === merge.into ||
      map.has(merge.from)
    )
      fail("Invalid or repeated merge source");
    map.set(merge.from, merge.into);
  }
  // Deliberately reject chains: each command names the final, reviewed survivor.
  for (const into of map.values())
    if (map.has(into)) fail("Merge chains/cycles are not allowed; use final survivors");
  return map;
}

function array(row: Row, key: string): unknown[] {
  const value: unknown = JSON.parse(String(row[key] ?? "[]"));
  if (!Array.isArray(value)) fail(`Expected an array in ${key}`);
  return value;
}

function union(a: unknown[], b: unknown[]): unknown[] {
  return [...new Map([...a, ...b].map((item) => [JSON.stringify(item), item])).values()];
}

function goalGaps(
  db: SqliteDatabase,
  ids: Map<string, string>,
  now: number,
): { gaps: GoalGap[]; unresolved: Row[] } {
  const gaps: GoalGap[] = [];
  const unresolved: Row[] = [];
  const goals = new GoalsRepository({ db });
  const rows = db.prepare("SELECT id, record_version FROM goals ORDER BY id").all() as Row[];
  for (const goal of rows) {
    const events = db
      .prepare(
        "SELECT * FROM identity_events WHERE record_type = 'goal' AND record_id = ? ORDER BY ts, id",
      )
      .all(goal.id) as Row[];
    const versions = new Map<number, { value: Row; ts: number }>();
    const observedBefore = new Map<number, { value: Row; ts: number }>();
    for (const event of events) {
      for (const [field, target] of [
        ["new_value_json", versions],
        ["old_value_json", observedBefore],
      ] as const) {
        const value = JSON.parse(String(event[field] ?? "null")) as Row | null;
        if (value !== null && Number.isInteger(value.record_version))
          target.set(Number(value.record_version), { value, ts: Number(event.ts) });
      }
    }
    const current = goals.get(String(goal.id) as GoalId)!;
    observedBefore.set(Number(goal.record_version), { value: current, ts: now });
    for (let version = 1; version <= Number(goal.record_version); version++) {
      if (versions.has(version)) continue;
      const before = versions.get(version - 1);
      const after = observedBefore.get(version);
      if (before === undefined || after === undefined || before.ts >= after.ts) {
        unresolved.push({
          id: goal.id,
          version,
          reason:
            "No pair of adjacent stored snapshots with an ordered time window; no history invented",
        });
        continue;
      }
      gaps.push({
        id: String(goal.id),
        version,
        before: remap(before.value, ids) as Row,
        after: remap(after.value, ids) as Row,
        ts: before.ts + Math.min(1, (after.ts - before.ts) / 2),
        reason: `Offline entity-graph repair recorded at ${now}: reconstructed missing version ${version} from stored adjacent snapshots. Original actor and exact write time are unknown; ts is an ordering placeholder within (${before.ts}, ${after.ts}).`,
      });
    }
  }
  return { gaps, unresolved };
}

export function protectedFiles(dataDir: string): RepairPlan["protectedFiles"] {
  const files: RepairPlan["protectedFiles"] = [];
  const walk = (directory: string) => {
    if (!existsSync(join(dataDir, directory))) return;
    for (const entry of readdirSync(join(dataDir, directory), { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) walk(path);
      else files.push({ path, sha256: digest(readFileSync(join(dataDir, path))) });
    }
  };
  walk("stream");
  walk("working");
  return files.sort((a, b) => a.path.localeCompare(b.path));
}

export async function planEntityGraphRepair(
  input: RepairInput,
  embeddingClient?: EmbeddingClient,
): Promise<RepairPlan> {
  const ids = mergeMap(input.merges, entityIdHelpers.is);
  const nodeIds = mergeMap(input.nodeMerges, semanticNodeIdHelpers.is);
  const source = openRepairDatabase(input.dataDir);
  const db = cloneDatabase(source);
  const connection = await openLance(input.dataDir);
  try {
    const now = Date.now();
    const config = loadConfig({ dataDir: input.dataDir });
    const appliedMigrations = new Set(db.listAppliedMigrations().map((item) => item.id));
    const migrations = createMigrations().filter((item) => !appliedMigrations.has(item.id));
    // Only the reviewed additive migration may be introduced by this repair.
    if (migrations.some((item) => item.name !== "entity_identity_redirects"))
      fail("The store needs other migrations; migrate a copy and review those separately first");
    const migrationBefore = sqlInventory(db, ["_migrations"]);
    const schemaBefore = new Map(
      db
        .prepare("SELECT name,type,sql FROM sqlite_master WHERE sql IS NOT NULL")
        .all()
        .map((row) => [String(row.name), row]),
    );
    runMigrations(db, migrations, now);
    const schemaChanges = db
      .prepare("SELECT name,type,sql FROM sqlite_master WHERE sql IS NOT NULL")
      .all()
      .flatMap((row) =>
        same(schemaBefore.get(String(row.name)), row)
          ? []
          : [
              {
                name: String(row.name),
                type: String(row.type),
                before: (schemaBefore.get(String(row.name))?.sql as string) ?? null,
                after: String(row.sql),
              },
            ],
      );
    const migrationSql = diffSql(migrationBefore, sqlInventory(db, ["_migrations"]));
    const inventory = sqlInventory(db);
    const entities = new EntityRepository({ db });
    for (const merge of input.merges)
      entities.merge(parseEntityId(merge.from), parseEntityId(merge.into), {
        renameInto: merge.renameInto,
      });
    if (entities.getSelf() === null) fail("Expected exactly one surviving self entity");
    const projected = new Map<string, Map<number, Row | null>>();
    for (const [table, rows] of sqlInventory(db))
      projected.set(table, new Map(rows.map(({ rowid, row }) => [rowid, { ...row }])));
    const tableRows = (table: string) =>
      projected.get(table) ?? fail(`Missing required table ${table}`);
    const find = (table: string, id: string): [number, Row] => {
      for (const [rowid, row] of tableRows(table)) if (row?.id === id) return [rowid, row];
      return fail(`Missing ${table} row ${id}`);
    };
    const rawId = /\bent_[a-z0-9]{16}\b/g;
    for (const row of tableRows("semantic_nodes").values()) {
      if (row === null) continue;
      row.aliases = JSON.stringify(
        array(row, "aliases").filter((alias) => String(alias).match(rawId) === null),
      );
      row.label = String(row.label).replace(
        rawId,
        (id) =>
          entities.get(parseEntityId(id))?.canonical_name ??
          fail(`Unknown entity ID in semantic label ${row.id}: ${id}`),
      );
    }
    for (const merge of input.nodeMerges) {
      const [, sourceNode] = find("semantic_nodes", merge.from);
      const [, target] = find("semantic_nodes", merge.into);
      if (sourceNode.kind !== "entity" || target.kind !== "entity")
        fail("Explicit node merges require entity nodes");
      if (target.status !== "active" || target.archived) fail("Node survivor must be active");
      if (
        sourceNode.archived &&
        sourceNode.superseded_by === target.id &&
        sourceNode.status === "superseded"
      )
        continue;
      target.aliases = JSON.stringify(
        union(array(target, "aliases"), [sourceNode.label, ...array(sourceNode, "aliases")]).filter(
          (name) => name !== target.label,
        ),
      );
      target.source_episode_ids = JSON.stringify(
        union(array(target, "source_episode_ids"), array(sourceNode, "source_episode_ids")),
      );
      Object.assign(sourceNode, {
        archived: 1,
        status: "superseded",
        superseded_by: target.id,
        superseded_at: now,
      });
    }
    const allIds = new Map([...ids, ...nodeIds]);
    const retiredSelfEdges = new Set<string>();
    const triples = new Map<string, Row>();
    const edgeRepository = new SemanticEdgeRepository({ db, clock: new ManualClock(now) });
    // Prefer an existing canonical triple, avoiding any transient unique-key collision.
    const edgeRows = [...tableRows("semantic_edges")].sort(([, a], [, b]) => {
      const moved = (row: Row | null) =>
        Number(nodeIds.has(String(row?.from_node_id)) || nodeIds.has(String(row?.to_node_id)));
      return moved(a) - moved(b);
    });
    for (const [, edge] of edgeRows) {
      if (edge === null) continue;
      const from = nodeIds.get(String(edge.from_node_id)) ?? edge.from_node_id;
      const to = nodeIds.get(String(edge.to_node_id)) ?? edge.to_node_id;
      if (from === to && (from !== edge.from_node_id || to !== edge.to_node_id)) {
        // Retain the historical endpoints on an explicitly closed edge. Never create a self-edge.
        retiredSelfEdges.add(String(edge.id));
        if (edge.valid_to === null)
          Object.assign(edge, {
            valid_to: Math.max(now, Number(edge.valid_from)),
            invalidated_at: now,
            invalidated_by_process: "maintenance",
            invalidated_reason: "Endpoints coalesced by explicit offline node merge",
          });
        continue;
      }
      edge.from_node_id = from;
      edge.to_node_id = to;
      if (edge.valid_to !== null) continue;
      const key = JSON.stringify([from, to, edge.relation]);
      const survivor = triples.get(key);
      if (survivor === undefined) {
        triples.set(key, edge);
        assertOne(
          db
            .prepare("UPDATE semantic_edges SET from_node_id = ?, to_node_id = ? WHERE id = ?")
            .run(from, to, edge.id),
          `semantic_edges/${edge.id}`,
        );
        continue;
      }
      const merged =
        edgeRepository.mergeOpenEdgeEvidence({
          ...edgeFromRow(edge),
          from_node_id: parseSemanticNodeId(String(from)),
          to_node_id: parseSemanticNodeId(String(to)),
        }) ?? fail("Open edge survivor disappeared");
      Object.assign(
        survivor,
        db.prepare("SELECT * FROM semantic_edges WHERE id = ?").get(merged.id),
      );
      allIds.set(String(edge.id), String(survivor.id));
      Object.assign(edge, {
        valid_to: Math.max(now, Number(edge.valid_from)),
        invalidated_at: now,
        invalidated_by_process: "maintenance",
        invalidated_reason: "Evidence transferred to surviving edge by offline node merge",
      });
    }
    for (const [table, rows] of projected) {
      for (const [rowid, row] of rows) {
        if (row === null || table === "stream_entry_index" || table === "_migrations") continue;
        if (table === "maintenance_audit" && String(row.action).startsWith("entity_graph_"))
          continue;
        const next = remap(row, allIds) as Row;
        if (["entities", "semantic_nodes", "semantic_edges"].includes(table)) next.id = row.id;
        if (table === "entities") next.merged_into = row.merged_into;
        if (table === "semantic_node_vector_sync_outbox") next.node_id = row.node_id;
        if (table === "semantic_edges" && retiredSelfEdges.has(String(row.id))) {
          next.from_node_id = row.from_node_id;
          next.to_node_id = row.to_node_id;
        }
        if (table === "review_queue") {
          const refs = JSON.parse(String(next.refs)) as Row;
          if (
            Array.isArray(refs.node_ids) &&
            refs.node_ids.length === 2 &&
            refs.node_ids[0] === refs.node_ids[1] &&
            row.resolved_at === null
          ) {
            next.resolved_at = now;
            next.resolution = "dismiss";
            next.reason = `${row.reason}\nRetired by explicit offline node merge: both pair handles now identify the same node.`;
          }
        }
        if (table === "observed_events" && row.speaker_entity_id !== next.speaker_entity_id)
          next.recurrence_key = buildObservedEventRecurrenceKey({
            sessionId: String(next.session_id) as SessionId,
            speakerEntityId:
              next.speaker_entity_id === null
                ? null
                : parseEntityId(String(next.speaker_entity_id)),
            classificationKind: String(next.classification_kind),
          });
        if (
          table === "open_questions" &&
          (row.audience_entity_id !== next.audience_entity_id ||
            row.related_semantic_node_ids !== next.related_semantic_node_ids)
        ) {
          const question = mapOpenQuestionRow(next);
          next.dedupe_key = buildOpenQuestionDedupeKey({
            question: question.question,
            relatedEpisodeIds: question.related_episode_ids,
            relatedSemanticNodeIds: question.related_semantic_node_ids,
            audienceEntityId: question.audience_entity_id,
          });
        }
        rows.set(rowid, next);
      }
      // Every UNIQUE key participates, not just primary keys. Parent collisions
      // and unequal leaf collisions require an operator decision, never cascades.
      const primary = db
        .prepare(`PRAGMA table_info(${quoteIdentifier(table)})`)
        .all()
        .filter((c) => Number(c.pk) > 0)
        .sort((a, b) => Number(a.pk) - Number(b.pk))
        .map((c) => String(c.name));
      const uniqueKeys = [primary];
      for (const index of db.prepare(`PRAGMA index_list(${quoteIdentifier(table)})`).all()) {
        if (!index.unique || index.partial) continue;
        const columns = db
          .prepare(`PRAGMA index_info(${quoteIdentifier(String(index.name))})`)
          .all()
          .map((c) => String(c.name));
        if (!uniqueKeys.some((key) => same(key, columns))) uniqueKeys.push(columns);
      }
      for (const columns of uniqueKeys.filter((key) => key.length > 0)) {
        const seen = new Map<string, Row>();
        for (const [rowid, row] of rows) {
          if (row === null || columns.some((column) => row[column] === null)) continue;
          const key = JSON.stringify(columns.map((column) => row[column]));
          const survivor = seen.get(key);
          if (survivor === undefined) {
            seen.set(key, row);
            continue;
          }
          assertLeafDeletion(db, table);
          const left = { ...survivor },
            right = { ...row };
          if (table === "semantic_belief_dependencies") {
            delete left.created_at;
            delete right.created_at;
          }
          if (!same(left, right))
            fail(`Conflicting ${table} unique key ${key}; requires an explicit data decision`);
          if (table === "semantic_belief_dependencies")
            survivor.created_at = Math.min(Number(survivor.created_at), Number(row.created_at));
          rows.set(rowid, null);
        }
      }
    }
    const touchedNodes = new Set<string>();
    const projectedInventory = new Map(
      [...projected].map(([table, rows]) => [
        table,
        [...rows].flatMap(([rowid, row]) => (row === null ? [] : [{ rowid, row }])),
      ]),
    );
    for (const change of diffSql(inventory, projectedInventory)) {
      if (change.table === "semantic_nodes" && change.after !== null) {
        change.after.updated_at = now;
        touchedNodes.add(String(change.after.id));
      }
    }
    // Apply to the private projection; triggers and every FK/UNIQUE constraint run here.
    const currentInventory = sqlInventory(db);
    const projectedChanges = diffSql(currentInventory, projectedInventory);
    validateSqlChanges(projectedChanges);
    db.transaction(() => {
      db.pragma("defer_foreign_keys = ON");
      applySqlChanges(db, projectedChanges);
    }).immediate();
    const sql = diffSql(inventory, sqlInventory(db));
    validateSqlChanges(sql);
    const violations = db.prepare("PRAGMA foreign_key_check").all();
    if (violations.length) fail(`Projected foreign key violations: ${JSON.stringify(violations)}`);
    const nodes = new Map(
      [...tableRows("semantic_nodes").values()]
        .filter((row): row is Row => row !== null)
        .map((row) => [String(row.id), row]),
    );
    currentInventory.clear();
    projectedInventory.clear();
    projected.clear();
    const lance = await planLance(connection, allIds, nodes, touchedNodes);
    const client =
      embeddingClient ?? (touchedNodes.size ? createEmbeddingClient(config) : undefined);
    if (connection !== null) {
      for (const name of new Set(
        lance.changes.filter((item) => item.reembed).map((item) => item.table),
      )) {
        const table = new LanceDbTable(await connection.openTable(name));
        try {
          const idsToEmbed = lance.changes
            .filter((item) => item.table === name && item.reembed)
            .map((item) => quoteSqlString(item.id));
          const rows = new Map(
            (await table.list({ where: `id IN (${idsToEmbed.join(",")})` })).map((row) => [
              String(row.id),
              row,
            ]),
          );
          for (const change of lance.changes.filter(
            (item) => item.table === name && item.reembed,
          )) {
            const vector = rows.get(change.id)?.embedding as Iterable<number>;
            change.before.embedding = Array.from(vector);
            const next = await client!.embed(
              buildNodeEmbeddingText({
                label: String(change.after.label),
                description: String(change.after.description),
                aliases: JSON.parse(String(change.after.aliases)),
                observationMetadata: semanticObservationMetadataSchema
                  .nullable()
                  .parse(JSON.parse(String(change.after.observation_metadata ?? "null"))),
              }),
            );
            if (
              next.length !== (change.before.embedding as number[]).length ||
              (embeddingClient === undefined && next.length !== config.embedding.dims) ||
              !next.every(Number.isFinite)
            )
              fail(`Embedding shape changed for ${change.id}`);
            change.after.embedding = Array.from(next);
          }
        } finally {
          table.close();
        }
      }
    }
    const rowsById = new Map(
      [...sqlInventory(db, [...new Set(lance.changes.map((row) => row.table))])].map(
        ([table, rows]) => [table, new Map(rows.map(({ row }) => [String(row.id), row]))],
      ),
    );
    validateLanceChanges(lance.changes, rowsById);
    const gaps = goalGaps(db, allIds, now);
    const plan: RepairPlan = {
      input,
      plannedAt: now,
      runId: null,
      migrations: migrations.map(({ id, name }) => ({ id, name })),
      migrationSql,
      schemaChanges,
      protectedIndexHash: digest(JSON.stringify(inventory.get("stream_entry_index"))),
      drainSql: [],
      sql,
      additionalSql: [],
      completionSql: [],
      lance: lance.changes,
      files: [],
      protectedFiles: protectedFiles(input.dataDir),
      embedding: {
        model: config.embedding.model,
        dimensions: config.embedding.dims,
        endpoint: config.embedding.baseUrl,
        injectedClient: embeddingClient !== undefined,
      },
      goals: gaps.gaps,
      touchedNodes: [...touchedNodes],
      inventory: {
        sqlite: Object.fromEntries(
          [...inventory].map(([table]) => [
            table,
            db
              .prepare(`PRAGMA table_info(${quoteIdentifier(table)})`)
              .all()
              .map((c) => String(c.name)),
          ]),
        ),
        lance: lance.inventory,
      },
      reports: [
        { category: "goal_history_not_reconstructable", rows: gaps.unresolved },
        {
          category: "scheduled_actions_need_outcome_evidence_archive_policy_exempts_scheduled",
          rows: db
            .prepare("SELECT * FROM action_records WHERE state = 'scheduled' ORDER BY id")
            .all(),
        },
      ],
    };
    if (!sql.length && !lance.changes.length && !gaps.gaps.length && !migrations.length)
      return plan;
    plan.runId = createMaintenanceRunId();
    const audit = new AuditLog({ db, clock: new ManualClock(now) });
    const identity = new IdentityEventRepository({ db, clock: new ManualClock(now) });
    const extraMutations: SqlChange[] = [];
    for (const gap of gaps.gaps) {
      const event = identity.record({
        record_type: "goal",
        record_id: gap.id,
        action: "repair_missing_update",
        old_value: gap.before as JsonValue,
        new_value: gap.after as JsonValue,
        reason: gap.reason,
        ts: gap.ts,
        provenance: { kind: "manual" },
      });
      extraMutations.push({
        table: "identity_events",
        rowid: event.id,
        before: null,
        after: db.prepare("SELECT * FROM identity_events WHERE id = ?").get(event.id)!,
        reason: "backfill evidenced goal history",
      });
    }
    if (touchedNodes.size) {
      const table = new LanceDbTable(await connection!.openTable("semantic_nodes"));
      try {
        const repository = new SemanticNodeRepository({ db, table });
        for (const id of touchedNodes) {
          const before =
            db
              .prepare("SELECT * FROM semantic_node_vector_sync_outbox WHERE node_id = ?")
              .get(id) ?? null;
          repository.enqueueVectorSyncRow({
            nodeId: parseSemanticNodeId(id),
            reason: "entity_graph_repair",
            createdAt: now,
          });
          const after = db
            .prepare("SELECT * FROM semantic_node_vector_sync_outbox WHERE node_id = ?")
            .get(id)!;
          extraMutations.push({
            table: "semantic_node_vector_sync_outbox",
            rowid: Number(after.id),
            before,
            after,
            reason: "enqueue vector mirror refresh",
          });
        }
      } finally {
        table.close();
      }
    }
    plan.additionalSql = [...extraMutations];
    plan.drainSql = extraMutations
      .filter((row) => row.table === "semantic_node_vector_sync_outbox")
      .map((row) => ({
        ...row,
        before: row.after,
        after: null,
        reason: "repository generation-guarded outbox drain after Lance commit",
      }));
    const recordAudit = (input: Parameters<AuditLog["record"]>[0], destination: SqlChange[]) => {
      const record = audit.record(input);
      destination.push({
        table: "maintenance_audit",
        rowid: record.id,
        before: null,
        after: db.prepare("SELECT * FROM maintenance_audit WHERE id = ?").get(record.id)!,
        reason: "maintenance audit with reversal payload",
      });
    };
    const allSql = [...sql, ...extraMutations];
    for (const table of new Set(allSql.map((change) => change.table))) {
      const changes = allSql.filter((change) => change.table === table);
      recordAudit(
        {
          run_id: plan.runId,
          process: "correction",
          action: "entity_graph_sql_rows",
          targets: { table, rows: changes.map(({ rowid, after }) => ({ rowid, after })) },
          reversal: { table, rows: changes.map(({ rowid, before }) => ({ rowid, before })) },
        },
        plan.additionalSql,
      );
    }
    for (const table of new Set(lance.changes.map((change) => change.table))) {
      const changes = lance.changes.filter((change) => change.table === table);
      recordAudit(
        {
          run_id: plan.runId,
          process: "correction",
          action: "entity_graph_lance_rows",
          targets: { table, rows: changes.map(({ id, after }) => ({ id, after })) },
          reversal: { table, rows: changes.map(({ id, before }) => ({ id, before })) },
        },
        plan.additionalSql,
      );
    }
    if (plan.drainSql.length)
      recordAudit(
        {
          run_id: plan.runId,
          process: "correction",
          action: "entity_graph_outbox_drains",
          targets: { rows: plan.drainSql.map(({ rowid, after }) => ({ rowid, after })) },
          reversal: {
            rows: plan.drainSql.map(({ rowid, before }) => ({ rowid, before })),
            restored_by: "entity_graph_sql_rows for the prior queue state",
          },
        },
        plan.additionalSql,
      );
    recordAudit(
      {
        run_id: plan.runId,
        process: "correction",
        action: "entity_graph_sql_committed",
        targets: { input, migrations: plan.migrations },
        reversal: { restore_run_id: plan.runId },
      },
      plan.additionalSql,
    );
    recordAudit(
      {
        run_id: plan.runId,
        process: "correction",
        action: "entity_graph_complete",
        targets: { input },
        reversal: { restore_run_id: plan.runId },
      },
      plan.completionSql,
    );
    validateSqlChanges([...plan.additionalSql, ...plan.completionSql]);
    return plan;
  } finally {
    connection?.close();
    db.close();
    source.close();
  }
}
