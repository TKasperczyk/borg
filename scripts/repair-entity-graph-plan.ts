import { GoalsRepository } from "../src/memory/self/goals-repository.js";
import { entityIdHelpers, semanticNodeIdHelpers, type GoalId } from "../src/util/ids.js";
import type { SqliteDatabase } from "../src/storage/sqlite/index.js";
import {
  type Row,
  type SqlChange,
  type FileChange,
  type LanceChange,
  fail,
  same,
  remap,
  sqlInventory,
  openRepairDatabase,
  openLance,
  planLance,
  planFiles,
  quoteIdentifier,
} from "./repair-entity-graph-storage.js";

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
  sql: SqlChange[];
  lance: LanceChange[];
  files: FileChange[];
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

export async function planEntityGraphRepair(input: RepairInput): Promise<RepairPlan> {
  const ids = mergeMap(input.merges, entityIdHelpers.is);
  const nodeIds = mergeMap(input.nodeMerges, semanticNodeIdHelpers.is);
  const db = openRepairDatabase(input.dataDir);
  const connection = await openLance(input.dataDir);
  try {
    const now = Date.now();
    const inventory = sqlInventory(db);
    const projected = new Map<string, Map<number, Row | null>>();
    for (const [table, rows] of inventory)
      projected.set(table, new Map(rows.map(({ rowid, row }) => [rowid, { ...row }])));
    const tableRows = (table: string) =>
      projected.get(table) ?? fail(`Missing required table ${table}`);
    const find = (table: string, id: string): [number, Row] => {
      for (const [rowid, row] of tableRows(table)) if (row?.id === id) return [rowid, row];
      return fail(`Missing ${table} row ${id}`);
    };
    const alreadyApplied = (inventory.get("maintenance_audit") ?? []).some(({ row }) => {
      if (row.action !== "entity_graph_complete") return false;
      const prior = (JSON.parse(String(row.targets)) as { input: RepairInput }).input;
      return same(prior.merges, input.merges) && same(prior.nodeMerges, input.nodeMerges);
    });
    for (const merge of input.merges) {
      if (
        alreadyApplied &&
        ![...tableRows("entities").values()].some((row) => row?.id === merge.from)
      )
        continue;
      const [sourceRowid, source] = find("entities", merge.from);
      const [, target] = find("entities", merge.into);
      if (source.kind === "self" && target.kind !== "self")
        fail("The self entity must survive a merge");
      const canonical = merge.renameInto?.trim() ?? String(target.canonical_name);
      if (canonical.length === 0) fail("Empty survivor name");
      // Rename adopts a known canonical name/provenance, never invents attribution.
      if (merge.renameInto !== undefined && canonical !== source.canonical_name)
        fail("--rename-into must equal the merged entity's canonical name");
      target.aliases = JSON.stringify(
        union(array(target, "aliases"), [
          target.canonical_name,
          source.canonical_name,
          ...array(source, "aliases"),
        ]).filter((name) => name !== canonical),
      );
      target.canonical_name = canonical;
      if (merge.renameInto !== undefined) target.name_provenance = source.name_provenance;
      target.borg_role ??= source.borg_role;
      tableRows("entities").set(sourceRowid, null);
    }
    const selfCount = [...tableRows("entities").values()].filter(
      (row) => row?.kind === "self",
    ).length;
    if (selfCount !== 1) fail(`Expected exactly one surviving self entity, found ${selfCount}`);
    const survivingEntities = new Map(
      [...tableRows("entities").values()]
        .filter((row): row is Row => row !== null)
        .map((row) => [String(row.id), row]),
    );
    // An ID alias conveys no identity evidence (including an incorrectly bound ID).
    const rawId = /\bent_[a-z0-9]{16}\b/g;
    const unresolvedLabels: Row[] = [];
    for (const row of tableRows("semantic_nodes").values()) {
      if (row === null) continue;
      row.aliases = JSON.stringify(
        array(row, "aliases").filter((alias) => String(alias).match(rawId) === null),
      );
      row.label = String(row.label).replace(rawId, (id) => {
        const entity = survivingEntities.get(ids.get(id) ?? id);
        if (entity === undefined) {
          unresolvedLabels.push({ id: row.id, entityId: id });
          return id;
        }
        return String(entity.canonical_name);
      });
    }
    if (unresolvedLabels.length > 0)
      fail(`Unknown entity IDs in semantic labels: ${JSON.stringify(unresolvedLabels)}`);
    for (const merge of input.nodeMerges) {
      const [, source] = find("semantic_nodes", merge.from);
      const [, target] = find("semantic_nodes", merge.into);
      if (source.kind !== "entity" || target.kind !== "entity")
        fail("Explicit node merges require entity nodes");
      if (target.status !== "active" || target.archived) fail("Node survivor must be active");
      if (source.archived && source.superseded_by === target.id && source.status === "superseded")
        continue;
      target.aliases = JSON.stringify(
        union(array(target, "aliases"), [source.label, ...array(source, "aliases")]).filter(
          (name) => name !== target.label,
        ),
      );
      target.source_episode_ids = JSON.stringify(
        union(array(target, "source_episode_ids"), array(source, "source_episode_ids")),
      );
      source.archived = 1;
      source.status = "superseded";
      source.superseded_by = target.id;
      source.superseded_at = now;
    }
    const allIds = new Map([...ids, ...nodeIds]);
    // Open-edge uniqueness is repository-enforced. Folding exact structural
    // triples unions evidence, like mergeOpenEdgeEvidence, without judging text.
    const triples = new Map<string, Row>();
    for (const [rowid, edge] of tableRows("semantic_edges")) {
      if (edge === null) continue;
      edge.from_node_id = nodeIds.get(String(edge.from_node_id)) ?? edge.from_node_id;
      edge.to_node_id = nodeIds.get(String(edge.to_node_id)) ?? edge.to_node_id;
      if (edge.valid_to !== null) continue;
      const key = JSON.stringify([edge.from_node_id, edge.to_node_id, edge.relation]);
      const survivor = triples.get(key);
      if (survivor === undefined) {
        triples.set(key, edge);
        continue;
      }
      survivor.evidence_episode_ids = JSON.stringify(
        union(array(survivor, "evidence_episode_ids"), array(edge, "evidence_episode_ids")),
      );
      survivor.confidence = Math.max(Number(survivor.confidence), Number(edge.confidence));
      survivor.last_verified_at = Math.max(
        Number(survivor.last_verified_at),
        Number(edge.last_verified_at),
      );
      survivor.valid_from = Math.min(Number(survivor.valid_from), Number(edge.valid_from));
      allIds.set(String(edge.id), String(survivor.id));
      tableRows("semantic_edges").set(rowid, null);
    }
    const { files, offsets } = planFiles(input.dataDir, allIds);
    for (const [table, rows] of projected) {
      for (const [rowid, row] of rows) {
        if (row === null) continue;
        // Repair reversals intentionally retain the original IDs and snapshots.
        if (table === "maintenance_audit" && String(row.action).startsWith("entity_graph_"))
          continue;
        const next = remap(row, allIds) as Row;
        // Definitions retain their IDs. References, including JSON keys, move.
        if (table === "entities" || table === "semantic_nodes" || table === "semantic_edges")
          next.id = row.id;
        // A queued mirror write belongs to its original node, including an
        // archived duplicate, whose tombstone also needs to reach Lance.
        if (table === "semantic_node_vector_sync_outbox") next.node_id = row.node_id;
        if (table === "stream_entry_index" && offsets.has(String(row.entry_id)))
          next.byte_offset = offsets.get(String(row.entry_id));
        rows.set(rowid, next);
      }
      const primary = (db.prepare(`PRAGMA table_info(${quoteIdentifier(table)})`).all() as Row[])
        .filter((column) => Number(column.pk) > 0)
        .sort((a, b) => Number(a.pk) - Number(b.pk))
        .map((column) => String(column.name));
      if (primary.length === 0) continue;
      const seen = new Map<string, Row>();
      for (const [rowid, row] of rows) {
        if (row === null) continue;
        const key = JSON.stringify(primary.map((column) => row[column]));
        const survivor = seen.get(key);
        if (survivor === undefined) {
          seen.set(key, row);
          continue;
        }
        const left = { ...survivor },
          right = { ...row };
        if (table === "semantic_belief_dependencies") {
          delete left.created_at;
          delete right.created_at;
        }
        if (!same(left, right))
          fail(`Conflicting ${table} primary key ${key}; requires an explicit data decision`);
        if (table === "semantic_belief_dependencies")
          survivor.created_at = Math.min(Number(survivor.created_at), Number(row.created_at));
        rows.set(rowid, null);
      }
    }
    const touchedNodes = new Set<string>();
    const sql: SqlChange[] = [];
    for (const [table, rows] of inventory) {
      for (const { rowid, row: before } of rows) {
        const after = tableRows(table).get(rowid)!;
        if (same(before, after)) continue;
        if (table === "semantic_nodes" && after !== null) {
          after.updated_at = now;
          touchedNodes.add(String(after.id));
        }
        sql.push({
          table,
          rowid,
          before,
          after,
          reason: after === null ? "merge duplicate" : "repair structural references",
        });
      }
    }
    const nodes = new Map(
      [...tableRows("semantic_nodes").values()]
        .filter((row): row is Row => row !== null)
        .map((row) => [String(row.id), row]),
    );
    const lance = await planLance(connection, allIds, nodes, touchedNodes);
    const gaps = goalGaps(db, allIds, now);
    const scheduled = db
      .prepare("SELECT * FROM action_records WHERE state = 'scheduled' ORDER BY id")
      .all() as Row[];
    const reports: RepairPlan["reports"] = [
      { category: "goal_history_not_reconstructable", rows: gaps.unresolved },
      {
        category: "scheduled_actions_need_outcome_evidence_archive_policy_exempts_scheduled",
        rows: scheduled,
      },
    ];
    // Name equality is never used to choose identities or semantic-node merges.
    return {
      input,
      plannedAt: now,
      sql,
      lance: lance.changes,
      files,
      goals: gaps.gaps,
      touchedNodes: [...touchedNodes],
      reports,
      inventory: {
        sqlite: Object.fromEntries(
          [...inventory].map(([table, rows]) => [
            table,
            rows[0] === undefined
              ? (db.prepare(`PRAGMA table_info(${quoteIdentifier(table)})`).all() as Row[]).map(
                  (column) => String(column.name),
                )
              : Object.keys(rows[0].row),
          ]),
        ),
        lance: lance.inventory,
      },
    };
  } finally {
    connection?.close();
    db.close();
  }
}
