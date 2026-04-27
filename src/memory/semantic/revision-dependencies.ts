import { z } from "zod";

import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { SemanticError } from "../../util/errors.js";
import { semanticEdgeIdSchema, semanticNodeIdSchema } from "./types.js";

export const semanticBeliefDependencyTargetTypeSchema = z.enum(["semantic_node", "semantic_edge"]);
export const semanticBeliefDependencyKindSchema = z.enum(["supports", "derived_from"]);

const semanticNodeBeliefDependencySchema = z.object({
  target_type: z.literal("semantic_node"),
  target_id: semanticNodeIdSchema,
  source_edge_id: semanticEdgeIdSchema,
  dependency_kind: semanticBeliefDependencyKindSchema,
  created_at: z.number().finite(),
});

const semanticEdgeBeliefDependencySchema = z.object({
  target_type: z.literal("semantic_edge"),
  target_id: semanticEdgeIdSchema,
  source_edge_id: semanticEdgeIdSchema,
  dependency_kind: semanticBeliefDependencyKindSchema,
  created_at: z.number().finite(),
});

export const semanticBeliefDependencySchema = z.discriminatedUnion("target_type", [
  semanticNodeBeliefDependencySchema,
  semanticEdgeBeliefDependencySchema,
]);
export const semanticBeliefDependencyInputSchema = z.discriminatedUnion("target_type", [
  semanticNodeBeliefDependencySchema.omit({ created_at: true }),
  semanticEdgeBeliefDependencySchema.omit({ created_at: true }),
]);

export type SemanticBeliefDependency = z.infer<typeof semanticBeliefDependencySchema>;
export type SemanticBeliefDependencyInput = z.input<typeof semanticBeliefDependencyInputSchema>;
export type SemanticBeliefDependencyTargetType = z.infer<
  typeof semanticBeliefDependencyTargetTypeSchema
>;
export type SemanticBeliefDependencyKind = z.infer<typeof semanticBeliefDependencyKindSchema>;

export type SemanticBeliefDependencyRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function dependencyFromRow(row: Record<string, unknown>): SemanticBeliefDependency {
  const parsed = semanticBeliefDependencySchema.safeParse({
    target_type: row.target_type,
    target_id: row.target_id,
    source_edge_id: row.source_edge_id,
    dependency_kind: row.dependency_kind,
    created_at: Number(row.created_at),
  });

  if (!parsed.success) {
    throw new SemanticError("Semantic belief dependency row failed validation", {
      cause: parsed.error,
      code: "SEMANTIC_BELIEF_DEPENDENCY_INVALID",
    });
  }

  return parsed.data;
}

export class SemanticBeliefDependencyRepository {
  private readonly clock: Clock;

  constructor(private readonly options: SemanticBeliefDependencyRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  addDependency(input: SemanticBeliefDependencyInput): SemanticBeliefDependency {
    const parsed = semanticBeliefDependencyInputSchema.parse(input);

    this.db
      .prepare(
        `
          INSERT OR IGNORE INTO semantic_belief_dependencies (
            target_type,
            target_id,
            source_edge_id,
            dependency_kind,
            created_at
          ) VALUES (?, ?, ?, ?, ?)
        `,
      )
      .run(
        parsed.target_type,
        parsed.target_id,
        parsed.source_edge_id,
        parsed.dependency_kind,
        this.clock.now(),
      );

    const row = this.db
      .prepare(
        `
          SELECT target_type, target_id, source_edge_id, dependency_kind, created_at
          FROM semantic_belief_dependencies
          WHERE target_type = ?
            AND target_id = ?
            AND source_edge_id = ?
            AND dependency_kind = ?
        `,
      )
      .get(parsed.target_type, parsed.target_id, parsed.source_edge_id, parsed.dependency_kind) as
      | Record<string, unknown>
      | undefined;

    if (row === undefined) {
      throw new SemanticError("Failed to read back semantic belief dependency", {
        code: "SEMANTIC_BELIEF_DEPENDENCY_INSERT_FAILED",
      });
    }

    return dependencyFromRow(row);
  }

  removeDependency(input: SemanticBeliefDependencyInput): boolean {
    const parsed = semanticBeliefDependencyInputSchema.parse(input);
    const result = this.db
      .prepare(
        `
          DELETE FROM semantic_belief_dependencies
          WHERE target_type = ?
            AND target_id = ?
            AND source_edge_id = ?
            AND dependency_kind = ?
        `,
      )
      .run(parsed.target_type, parsed.target_id, parsed.source_edge_id, parsed.dependency_kind);

    return result.changes > 0;
  }

  listBySourceEdge(sourceEdgeId: SemanticBeliefDependency["source_edge_id"]) {
    const parsedSourceEdgeId = semanticEdgeIdSchema.parse(sourceEdgeId);
    const rows = this.db
      .prepare(
        `
          SELECT target_type, target_id, source_edge_id, dependency_kind, created_at
          FROM semantic_belief_dependencies
          WHERE source_edge_id = ?
          ORDER BY created_at ASC, target_type ASC, target_id ASC, dependency_kind ASC
        `,
      )
      .all(parsedSourceEdgeId) as Record<string, unknown>[];

    return rows.map((row) => dependencyFromRow(row));
  }

  listByTarget(
    targetType: SemanticBeliefDependencyTargetType,
    targetId: SemanticBeliefDependency["target_id"],
  ) {
    const parsedTargetType = semanticBeliefDependencyTargetTypeSchema.parse(targetType);
    const parsedTarget =
      parsedTargetType === "semantic_node"
        ? semanticNodeIdSchema.parse(targetId)
        : semanticEdgeIdSchema.parse(targetId);
    const rows = this.db
      .prepare(
        `
          SELECT target_type, target_id, source_edge_id, dependency_kind, created_at
          FROM semantic_belief_dependencies
          WHERE target_type = ?
            AND target_id = ?
          ORDER BY created_at ASC, source_edge_id ASC, dependency_kind ASC
        `,
      )
      .all(parsedTargetType, parsedTarget) as Record<string, unknown>[];

    return rows.map((row) => dependencyFromRow(row));
  }
}
