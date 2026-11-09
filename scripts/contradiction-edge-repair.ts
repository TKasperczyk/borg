/** Stop-window repairs. No model judgments or boot-time data mutation. */
import {
  SemanticEdgeRepository,
  type SemanticEdgeInvalidationInput,
} from "../src/memory/semantic/repository.js";
import {
  CONTRADICTION_RETIREMENT_ACTION,
  isContradictionForNodePair,
  retireContradictionEdges,
} from "../src/memory/semantic/contradiction-retirement.js";
import { semanticEdgeSchema, type SemanticEdge } from "../src/memory/semantic/types.js";
import { semanticPairReviewRefsSchema } from "../src/memory/review-queue/review-handlers/semantic-pair.js";
import { stripReviewResolverRefs } from "../src/memory/review-queue/review-queue.js";
import type { SqliteDatabase } from "../src/storage/sqlite/index.js";
import { FixedClock } from "../src/util/clock.js";
import type { SemanticEdgeId } from "../src/util/ids.js";
import {
  applyDataRepair,
  assertNoPending,
  assertSame,
  checkedPath,
  hasStep,
  loadRepairPlan,
  newPlan,
  openDataRepairDatabase,
  printPlan,
  repairArgs,
  type DataRepairPlan,
  type RepairApplyOptions,
} from "./audited-data-repair.js";

export type ContradictionRepairKind =
  | "backfill-judged-contradiction-edges"
  | "backfill-superseded-contradiction-edges";
type ReviewSnapshot = {
  id: number;
  kind: string;
  refs: string;
  reason: string;
  resolved_at: number;
  resolution: string;
};
type NodeSnapshot = {
  id: string;
  status: string;
  corrected_by: string | null;
  superseded_at: number | null;
};
type RepairRow = {
  before: SemanticEdge;
  after: SemanticEdge;
  provenance: SemanticEdgeInvalidationInput;
  review: ReviewSnapshot | null;
  nodes: NodeSnapshot[];
};
export type ContradictionRepairPlan = DataRepairPlan & {
  kind: ContradictionRepairKind;
  rows: RepairRow[];
};

function snapshotReview(db: SqliteDatabase, id: number): ReviewSnapshot | null {
  return (
    (db
      .prepare(
        "SELECT id, kind, refs, reason, resolved_at, resolution FROM review_queue WHERE id = ?",
      )
      .get(id) as ReviewSnapshot | undefined) ?? null
  );
}
function snapshotNodes(db: SqliteDatabase, edge: SemanticEdge): NodeSnapshot[] {
  return db
    .prepare(
      "SELECT id, status, corrected_by, superseded_at FROM semantic_nodes WHERE id IN (?, ?) ORDER BY id",
    )
    .all(edge.from_node_id, edge.to_node_id) as NodeSnapshot[];
}
function repairRow(
  before: SemanticEdge,
  provenance: SemanticEdgeInvalidationInput,
  review: ReviewSnapshot | null,
  nodes: NodeSnapshot[],
): RepairRow {
  const at = Math.max(provenance.at, before.valid_from);
  return {
    before,
    provenance: { ...provenance, at },
    review,
    nodes,
    after: semanticEdgeSchema.parse({
      ...before,
      valid_to: at,
      invalidated_at: provenance.at,
      invalidated_by_edge_id: provenance.by_edge_id ?? null,
      invalidated_by_review_id: provenance.by_review_id ?? null,
      invalidated_by_process: provenance.by_process,
      invalidated_reason: provenance.reason ?? null,
    }),
  };
}

export function planContradictionRepair(
  input: {
    kind: ContradictionRepairKind;
    dataDir: string;
    liveStoreConfirmed?: string;
  },
  now = Date.now(),
): ContradictionRepairPlan {
  const dataDir = checkedPath(input.dataDir, input.liveStoreConfirmed);
  assertNoPending(dataDir);
  const db = openDataRepairDatabase(dataDir, true, input.liveStoreConfirmed);
  try {
    const repo = new SemanticEdgeRepository({ db });
    const rows: RepairRow[] = [];
    let skippedDiagnostic = 0,
      skippedMismatch = 0;
    if (input.kind === "backfill-judged-contradiction-edges") {
      const candidates = db
        .prepare(
          "SELECT id, kind, refs, reason, resolved_at, resolution FROM review_queue WHERE kind = 'contradiction' AND resolved_at IS NOT NULL AND resolution IN ('keep_both', 'dismiss') ORDER BY resolved_at DESC, id DESC",
        )
        .all() as ReviewSnapshot[];
      const seen = new Set<string>();
      for (const review of candidates) {
        const refs = JSON.parse(review.refs) as Record<string, unknown>;
        // Only the finalization marker proves the handler was bypassed. Earlier
        // failed attempts can leave diagnostics on a later successful judgment.
        const repair = refs.__borg_review_resolver_repair as { mode?: string } | undefined;
        if (repair?.mode === "resolver_finalized_without_handler") {
          skippedDiagnostic++;
          continue;
        }
        const parsed = semanticPairReviewRefsSchema.safeParse(stripReviewResolverRefs(refs));
        if (!parsed.success || !("node_ids" in parsed.data) || parsed.data.edge_id === undefined) {
          skippedMismatch++;
          continue;
        }
        const edge = repo.getEdge(parsed.data.edge_id);
        if (edge === null || !isContradictionForNodePair(edge, parsed.data.node_ids)) {
          skippedMismatch++;
          continue;
        }
        if (edge.valid_to !== null || seen.has(edge.id)) continue;
        seen.add(edge.id);
        rows.push(
          repairRow(
            edge,
            {
              at: now,
              by_process: "review",
              by_review_id: review.id,
              reason: `judged contradiction backfill: review ${review.id}, resolution ${review.resolution}`,
            },
            review,
            [],
          ),
        );
      }
    } else {
      const candidates = db
        .prepare(
          "SELECT DISTINCT e.id FROM semantic_edges e JOIN semantic_nodes n ON n.id = e.from_node_id OR n.id = e.to_node_id WHERE e.relation = 'contradicts' AND e.valid_to IS NULL AND n.status = 'superseded' ORDER BY e.id",
        )
        .all() as Array<{ id: SemanticEdgeId }>;
      for (const { id } of candidates) {
        const edge = repo.getEdge(id)!;
        const nodes = snapshotNodes(db, edge);
        rows.push(
          repairRow(
            edge,
            {
              at: now,
              by_process: "maintenance",
              reason: `supersession contradiction backfill: ${nodes
                .filter((n) => n.status === "superseded")
                .map((n) => n.id)
                .join(", ")}`,
            },
            null,
            nodes,
          ),
        );
      }
    }
    return {
      ...newPlan(input.kind, dataDir, rows.length, now, input.liveStoreConfirmed),
      kind: input.kind,
      rows,
      summary: {
        changed: rows.length,
        skipped_diagnostic: skippedDiagnostic,
        skipped_mismatched_refs: skippedMismatch,
      },
    };
  } finally {
    db.close();
  }
}

function preflight(db: SqliteDatabase, plan: ContradictionRepairPlan): void {
  const done = plan.runId !== null && hasStep(db, plan, CONTRADICTION_RETIREMENT_ACTION);
  const repo = new SemanticEdgeRepository({ db });
  for (const row of plan.rows) {
    assertSame(repo.getEdge(row.before.id), done ? row.after : row.before, `edge ${row.before.id}`);
    if (row.review !== null)
      assertSame(snapshotReview(db, row.review.id), row.review, `review ${row.review.id}`);
    if (row.nodes.length > 0)
      assertSame(snapshotNodes(db, row.before), row.nodes, `endpoints ${row.before.id}`);
  }
}

export async function applyContradictionRepair(
  plan: ContradictionRepairPlan,
  options: RepairApplyOptions = {},
): Promise<void> {
  await applyDataRepair(
    plan,
    {
      preflight,
      async finish(db, current, checkpoint) {
        db.transaction(() => {
          preflight(db, current);
          if (hasStep(db, current, CONTRADICTION_RETIREMENT_ACTION)) return;
          const clock = new FixedClock(current.plannedAt);
          const repository = new SemanticEdgeRepository({ db, clock });
          for (const row of current.rows)
            retireContradictionEdges({
              db,
              clock,
              repository,
              edges: [row.before],
              provenance: row.provenance,
              runId: current.runId!,
              context: {
                repair_kind: current.kind,
                review_item_id: row.review?.id ?? null,
                resolution: row.review?.resolution ?? null,
                nodes: row.nodes,
                source_process: current.kind,
                live_store_override: current.live_store_override ?? null,
              },
            });
          preflight(db, current);
        }).immediate();
        await checkpoint?.("after_sqlite");
      },
    },
    options,
  );
}

export async function contradictionRepairCli(kind: ContradictionRepairKind): Promise<void> {
  const args = repairArgs(process.argv.slice(2));
  const plan = args.resume
    ? loadRepairPlan<ContradictionRepairPlan>(
        args.dataDir,
        kind,
        undefined,
        args.liveStoreConfirmed,
      )
    : planContradictionRepair({
        kind,
        dataDir: args.dataDir,
        liveStoreConfirmed: args.liveStoreConfirmed,
      });
  await printPlan(plan);
  if (args.apply) {
    await applyContradictionRepair(plan, {
      resume: args.resume,
      liveStoreConfirmed: args.liveStoreConfirmed,
    });
    console.log(
      JSON.stringify({
        run_id: plan.runId,
        ...plan.summary,
        status: plan.runId === null ? "no changes; nothing pending" : "applied",
      }),
    );
  }
}
