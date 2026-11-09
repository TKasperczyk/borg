import { z } from "zod";
import { AuditLog, type ReverserRegistry } from "../../offline/audit-log.js";
import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import type { Clock } from "../../util/clock.js";
import { SemanticError } from "../../util/errors.js";
import { createMaintenanceRunId, type MaintenanceRunId } from "../../util/ids.js";
import type { SemanticEdgeRepository, SemanticEdgeInvalidationInput } from "./repository.js";
import { semanticEdgeIdSchema, semanticEdgeSchema, type SemanticEdge } from "./types.js";

export const CONTRADICTION_RETIREMENT_ACTION = "retire_contradiction_edges";
export function isContradictionForNodePair(edge: SemanticEdge, ids: readonly string[]): boolean {
  return (
    edge.relation === "contradicts" &&
    ids.length === 2 &&
    ((edge.from_node_id === ids[0] && edge.to_node_id === ids[1]) ||
      (edge.from_node_id === ids[1] && edge.to_node_id === ids[0]))
  );
}
const reversalSchema = z.object({
  before: z.array(semanticEdgeSchema),
  expected_after: z.array(semanticEdgeSchema),
  invalidation_events: z
    .array(
      z.object({
        edge_id: semanticEdgeIdSchema,
        event_id: z.number().int().positive(),
      }),
    )
    .optional(),
});

/** Known edge handles only. The caller owns the judgment or lifecycle transition. */
export function retireContradictionEdges(input: {
  db: SqliteDatabase;
  clock: Clock;
  edges: readonly SemanticEdge[];
  repository: Pick<SemanticEdgeRepository, "getEdge" | "invalidateEdge">;
  provenance: SemanticEdgeInvalidationInput;
  context: Record<string, unknown>;
  runId?: MaintenanceRunId;
  close?: (edge: SemanticEdge) => void;
}): number | null {
  const apply = () => {
    const before: SemanticEdge[] = [];
    const after: SemanticEdge[] = [];
    const invalidationEvents: Array<{ edge_id: SemanticEdge["id"]; event_id: number }> = [];
    for (const id of new Set(input.edges.map((edge) => edge.id))) {
      const edge = input.repository.getEdge(id);
      if (edge === null || edge.valid_to !== null) continue;
      if (edge.relation !== "contradicts")
        throw new SemanticError("Contradiction retirement requires a contradicts edge", {
          code: "CONTRADICTION_RETIREMENT_INVALID",
        });
      // The transaction isolates the outbox writes. IDs created by this closure
      // identify its event even if another closure has identical timestamps.
      const priorEvent = input.db
        .prepare("SELECT COALESCE(MAX(id), 0) AS id FROM semantic_edge_invalidation_events")
        .get() as { id: number };
      if (input.close) input.close(edge);
      else
        input.repository.invalidateEdge(id, {
          ...input.provenance,
          at: Math.max(input.provenance.at, edge.valid_from),
        });
      const next = input.repository.getEdge(id);
      if (next?.valid_to == null)
        throw new SemanticError("Contradiction retirement did not close the edge", {
          code: "CONTRADICTION_RETIREMENT_INVALID",
        });
      const events = input.db
        .prepare("SELECT id FROM semantic_edge_invalidation_events WHERE edge_id = ? AND id > ?")
        .all(edge.id, priorEvent.id) as Array<{ id: number }>;
      if (events.length !== 1)
        throw new SemanticError("Contradiction retirement requires its own invalidation event", {
          code: "CONTRADICTION_RETIREMENT_EVENT_INVALID",
        });
      before.push(edge);
      after.push(next);
      invalidationEvents.push({ edge_id: edge.id, event_id: events[0]!.id });
    }
    if (before.length === 0) return null;
    return new AuditLog({ db: input.db, clock: input.clock }).record({
      run_id: input.runId ?? createMaintenanceRunId(),
      process: "correction",
      action: CONTRADICTION_RETIREMENT_ACTION,
      targets: {
        ...input.context,
        edge_ids: before.map((edge) => edge.id),
        provenance: input.provenance,
      },
      reversal: { before, expected_after: after, invalidation_events: invalidationEvents },
    }).id;
  };
  return input.db.raw.inTransaction ? apply() : input.db.transaction(apply).immediate();
}

export function registerContradictionRetirementReverser(
  registry: ReverserRegistry,
  repository: Pick<SemanticEdgeRepository, "getEdge" | "restoreEdge">,
): void {
  for (const kind of [
    "backfill-judged-contradiction-edges",
    "backfill-superseded-contradiction-edges",
  ])
    registry.register("correction", `${kind}:complete`, () => {});
  registry.register("correction", CONTRADICTION_RETIREMENT_ACTION, ({ reversal }) => {
    const parsed = reversalSchema.parse(reversal);
    const events = new Map(
      parsed.invalidation_events?.map((event) => [event.edge_id, event.event_id]),
    );
    if (events.size !== parsed.before.length || parsed.before.some((edge) => !events.has(edge.id)))
      throw new SemanticError("Contradiction reversal requires recorded invalidation event IDs", {
        code: "CONTRADICTION_REVERSAL_EVENT_IDS_MISSING",
      });
    // Refuse to overwrite any subsequent edge write, including another repair.
    for (const expected of parsed.expected_after) {
      if (JSON.stringify(repository.getEdge(expected.id)) !== JSON.stringify(expected))
        throw new SemanticError("Contradiction edge changed since retirement", {
          code: "CONTRADICTION_REVERSAL_STALE",
        });
    }
    // Invalidation never changes dependencies. Undo only its edge fields, keeping
    // every existing dependency (including derived_from rows and their timestamps).
    // Audited restore also cancels the recorded pending invalidation event in
    // this same audit-reversal transaction, before it can reach the reviser.
    for (const edge of parsed.before)
      repository.restoreEdge(edge, {
        auditedReversal: true,
        preserveDependencies: true,
        invalidationEventId: events.get(edge.id)!,
      });
  });
}
