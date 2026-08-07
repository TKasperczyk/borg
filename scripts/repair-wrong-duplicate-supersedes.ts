/*
 * Restore semantic nodes wrongly superseded by duplicate/contradiction reviews
 * whose enqueue-time labels carry disjoint machine identifiers.
 *
 * Run with every Borg writer stopped and take a verified backup first. Dry-run
 * is the default; --apply performs the repository restores and writes one
 * maintenance_audit row per restored node.
 *
 * Usage:
 *   pnpm tsx scripts/repair-wrong-duplicate-supersedes.ts --data-dir <bank-dir>
 *   pnpm tsx scripts/repair-wrong-duplicate-supersedes.ts --data-dir <bank-dir> --apply
 */
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { pathToFileURL } from "node:url";

import { z } from "zod";

import {
  ReviewQueueRepository,
  semanticPairReviewRefsSchema,
  stripReviewResolverRefs,
  type ReviewQueueItem,
} from "../src/memory/review-queue/index.js";
import {
  disjointDistinctIdentifiers,
  type DistinctIdentifierConflict,
} from "../src/memory/semantic/distinct-identifiers.js";
import {
  semanticNodeCorrectionRefSchema,
  semanticNodeIdSchema,
  semanticNodeStatusSchema,
  type SemanticNode,
  type SemanticNodeStatusTransition,
} from "../src/memory/semantic/index.js";
import { AuditLog } from "../src/offline/audit-log.js";
import { SqliteDatabase, SqliteRawDatabase } from "../src/storage/sqlite/index.js";
import { SystemClock, type Clock } from "../src/util/clock.js";
import { createMaintenanceRunId, type MaintenanceRunId } from "../src/util/ids.js";

export const WRONG_DUPLICATE_RESTORE_AUDIT_ACTION = "restore_distinct_identifier_supersede";

const repairSemanticNodeSchema = z.object({
  id: semanticNodeIdSchema,
  label: z.string().min(1),
  status: semanticNodeStatusSchema,
  corrected_by: semanticNodeCorrectionRefSchema.nullable(),
  superseded_at: z.number().finite().nullable(),
  archived: z.boolean(),
});

type RepairSemanticNode = z.infer<typeof repairSemanticNodeSchema>;

type RepairSemanticNodeRepository = {
  getMany: (
    ids: readonly SemanticNode["id"][],
    options?: { includeArchived?: boolean },
  ) => Promise<Array<RepairSemanticNode | null>>;
  restoreActive: (id: SemanticNode["id"]) => Promise<SemanticNodeStatusTransition | null>;
};

type RepairDependencies = {
  db: SqliteDatabase;
  reviewQueueRepository: Pick<ReviewQueueRepository, "list">;
  semanticNodeRepository: RepairSemanticNodeRepository;
  auditLog: Pick<AuditLog, "record">;
  runId: MaintenanceRunId;
};

export type WrongDuplicateRepairMatch = {
  reviewId: number;
  kind: Extract<ReviewQueueItem["kind"], "duplicate" | "contradiction">;
  nodeIds: [SemanticNode["id"], SemanticNode["id"]];
  labels: [string, string];
  identifiers: DistinctIdentifierConflict;
};

export type WrongDuplicateRepairCandidate = {
  nodeId: SemanticNode["id"];
  counterpartNodeId: SemanticNode["id"];
  label: string;
  reviewIds: number[];
  correctedBy: SemanticNode["id"];
  supersededAt: number | null;
  archived: boolean;
};

export type WrongDuplicateRepairMalformedReview = {
  reviewId: number;
  reason: string;
};

export type WrongDuplicateRepairOutOfScopeReview = {
  reviewId: number;
  reason: "semantic edge-closure refs";
};

export type WrongDuplicateRepairMissingTargets = {
  reviewId: number;
  nodeIds: [SemanticNode["id"], SemanticNode["id"]];
  missingNodeIds: SemanticNode["id"][];
};

export type WrongDuplicateRepairReport = {
  dryRun: boolean;
  matchingReviews: WrongDuplicateRepairMatch[];
  candidates: WrongDuplicateRepairCandidate[];
  restored: WrongDuplicateRepairCandidate[];
  currentOnlyReviewIds: number[];
  malformedReviews: WrongDuplicateRepairMalformedReview[];
  outOfScopeReviews: WrongDuplicateRepairOutOfScopeReview[];
  missingTargets: WrongDuplicateRepairMissingTargets[];
};

type RepairScan = Omit<WrongDuplicateRepairReport, "dryRun" | "restored">;

async function scanWrongDuplicateSupersedes(
  dependencies: Pick<RepairDependencies, "reviewQueueRepository" | "semanticNodeRepository">,
): Promise<RepairScan> {
  const reviews = (["duplicate", "contradiction"] as const)
    .flatMap((kind) =>
      dependencies.reviewQueueRepository.list({ kind }).map((item) => ({ item, kind })),
    )
    .filter(({ item }) => item.resolution === "supersede")
    .sort((left, right) => left.item.id - right.item.id);
  const matchingReviews: WrongDuplicateRepairMatch[] = [];
  const currentOnlyReviewIds: number[] = [];
  const malformedReviews: WrongDuplicateRepairMalformedReview[] = [];
  const outOfScopeReviews: WrongDuplicateRepairOutOfScopeReview[] = [];
  const missingTargets: WrongDuplicateRepairMissingTargets[] = [];
  const candidatesById = new Map<SemanticNode["id"], WrongDuplicateRepairCandidate>();

  for (const { item: review, kind } of reviews) {
    const parsed = semanticPairReviewRefsSchema.safeParse(stripReviewResolverRefs(review.refs));

    if (!parsed.success) {
      malformedReviews.push({
        reviewId: review.id,
        reason: "refs failed semantic-pair validation",
      });
      continue;
    }

    if (!("node_ids" in parsed.data)) {
      outOfScopeReviews.push({
        reviewId: review.id,
        reason: "semantic edge-closure refs",
      });
      continue;
    }

    const refs = parsed.data;
    const nodes = await dependencies.semanticNodeRepository.getMany(refs.node_ids, {
      includeArchived: true,
    });
    const first = nodes[0];
    const second = nodes[1];
    const missingNodeIds = refs.node_ids.filter((_, index) => nodes[index] == null);

    if (missingNodeIds.length > 0) {
      missingTargets.push({
        reviewId: review.id,
        nodeIds: refs.node_ids,
        missingNodeIds,
      });
    }

    const currentConflict =
      first === null || first === undefined || second === null || second === undefined
        ? null
        : disjointDistinctIdentifiers(first.label, second.label);
    const persistedLabels = refs.node_labels;
    const persistedConflict =
      persistedLabels === undefined
        ? null
        : disjointDistinctIdentifiers(persistedLabels[0], persistedLabels[1]);

    // Current labels cannot prove what the historical resolver saw. Surface
    // these rows for inspection, but only enqueue-time label conflicts mutate.
    if (persistedConflict === null) {
      if (currentConflict !== null) {
        currentOnlyReviewIds.push(review.id);
      }
      continue;
    }

    matchingReviews.push({
      reviewId: review.id,
      kind,
      nodeIds: refs.node_ids,
      labels: persistedLabels as [string, string],
      identifiers: persistedConflict,
    });

    if (first === null || first === undefined || second === null || second === undefined) {
      continue;
    }

    for (const [node, counterpart] of [
      [first, second],
      [second, first],
    ] as const) {
      if (node.status !== "superseded" || node.corrected_by !== counterpart.id) {
        continue;
      }

      const existing = candidatesById.get(node.id);

      if (existing !== undefined) {
        existing.reviewIds.push(review.id);
        continue;
      }

      candidatesById.set(node.id, {
        nodeId: node.id,
        counterpartNodeId: counterpart.id,
        label: node.label,
        reviewIds: [review.id],
        correctedBy: counterpart.id,
        supersededAt: node.superseded_at,
        archived: node.archived,
      });
    }
  }

  return {
    matchingReviews,
    candidates: [...candidatesById.values()].sort((left, right) =>
      left.nodeId.localeCompare(right.nodeId),
    ),
    currentOnlyReviewIds,
    malformedReviews,
    outOfScopeReviews,
    missingTargets,
  };
}

function sameLifecycleState(
  node: RepairSemanticNode,
  candidate: WrongDuplicateRepairCandidate,
): boolean {
  return (
    node.status === "superseded" &&
    node.corrected_by === candidate.correctedBy &&
    node.superseded_at === candidate.supersededAt &&
    node.archived === candidate.archived
  );
}

async function inImmediateTransaction<T>(
  db: SqliteDatabase,
  operation: () => Promise<T>,
): Promise<T> {
  db.exec("BEGIN IMMEDIATE");

  try {
    const result = await operation();
    db.exec("COMMIT");
    return result;
  } catch (error) {
    try {
      if (db.raw.inTransaction) {
        db.exec("ROLLBACK");
      }
    } catch {
      // Preserve the original failure.
    }

    throw error;
  }
}

export async function repairWrongDuplicateSupersedes(
  dependencies: RepairDependencies,
  options: { apply?: boolean } = {},
): Promise<WrongDuplicateRepairReport> {
  const scan = await scanWrongDuplicateSupersedes(dependencies);
  const apply = options.apply === true;
  const restored: WrongDuplicateRepairCandidate[] = [];

  if (apply) {
    for (const candidate of scan.candidates) {
      await inImmediateTransaction(dependencies.db, async () => {
        const currentPair = await dependencies.semanticNodeRepository.getMany(
          [candidate.nodeId, candidate.counterpartNodeId],
          {
            includeArchived: true,
          },
        );
        const current = currentPair[0];
        const counterpart = currentPair[1];

        if (current === null || current === undefined) {
          throw new Error(`Semantic node ${candidate.nodeId} disappeared before restore`);
        }

        if (counterpart === null || counterpart === undefined) {
          throw new Error(
            `Counterpart semantic node ${candidate.counterpartNodeId} disappeared before restore`,
          );
        }

        if (
          current.corrected_by !== counterpart.id ||
          counterpart.id !== candidate.counterpartNodeId
        ) {
          throw new Error(
            `Semantic node ${candidate.nodeId} is no longer superseded by its reviewed counterpart`,
          );
        }

        if (!sameLifecycleState(current, candidate)) {
          throw new Error(`Semantic node ${candidate.nodeId} changed after repair discovery`);
        }

        const transition = await dependencies.semanticNodeRepository.restoreActive(
          candidate.nodeId,
        );

        if (
          transition === null ||
          transition.fromStatus !== "superseded" ||
          transition.toStatus !== "active"
        ) {
          throw new Error(`Semantic node ${candidate.nodeId} was not restored from superseded`);
        }

        dependencies.auditLog.record({
          run_id: dependencies.runId,
          process: "review-resolver",
          action: WRONG_DUPLICATE_RESTORE_AUDIT_ACTION,
          targets: {
            semantic_node_id: candidate.nodeId,
            counterpart_semantic_node_id: candidate.counterpartNodeId,
            label: candidate.label,
            review_ids: candidate.reviewIds,
          },
          reversal: {
            no_reverser: true,
            previous: {
              status: "superseded",
              corrected_by: candidate.correctedBy,
              superseded_at: candidate.supersededAt,
              archived: candidate.archived,
            },
          },
        });
      });
      restored.push(candidate);
    }
  }

  return {
    dryRun: !apply,
    ...scan,
    restored,
  };
}

function sqliteBoolean(value: unknown): unknown {
  if (value === true || value === 1) {
    return true;
  }

  if (value === false || value === 0) {
    return false;
  }

  return value;
}

function repairSemanticNodeFromRow(row: Record<string, unknown>): RepairSemanticNode {
  return repairSemanticNodeSchema.parse({
    id: row.id,
    label: row.label,
    status: row.status ?? "active",
    corrected_by: row.corrected_by ?? null,
    superseded_at:
      row.superseded_at === null || row.superseded_at === undefined
        ? null
        : Number(row.superseded_at),
    archived: sqliteBoolean(row.archived),
  });
}

class SqliteRepairSemanticNodeRepository implements RepairSemanticNodeRepository {
  constructor(
    private readonly db: SqliteDatabase,
    private readonly clock: Clock,
  ) {}

  async getMany(
    ids: readonly SemanticNode["id"][],
    options: { includeArchived?: boolean } = {},
  ): Promise<Array<RepairSemanticNode | null>> {
    if (ids.length === 0) {
      return [];
    }

    const parsedIds = ids.map((id) => semanticNodeIdSchema.parse(id));
    const placeholders = parsedIds.map(() => "?").join(", ");
    const rows = this.db
      .prepare(
        `
          SELECT id, label, status, corrected_by, superseded_at, archived
          FROM semantic_nodes
          WHERE id IN (${placeholders})
        `,
      )
      .all(...parsedIds) as Record<string, unknown>[];
    const byId = new Map(
      rows.map((row) => {
        const node = repairSemanticNodeFromRow(row);
        return [node.id, node] as const;
      }),
    );

    return parsedIds.map((id) => {
      const node = byId.get(id) ?? null;

      return node !== null && node.archived && options.includeArchived !== true ? null : node;
    });
  }

  async restoreActive(id: SemanticNode["id"]): Promise<SemanticNodeStatusTransition | null> {
    const parsedId = semanticNodeIdSchema.parse(id);
    const current = (await this.getMany([parsedId], { includeArchived: true }))[0];

    if (current === null || current === undefined) {
      return null;
    }

    const result = this.db
      .prepare(
        `
          UPDATE semantic_nodes
          SET status = 'active',
              corrected_by = NULL,
              superseded_at = NULL,
              updated_at = ?
          WHERE id = ?
        `,
      )
      .run(this.clock.now(), parsedId);

    if (result.changes !== 1) {
      return null;
    }

    return {
      id: parsedId,
      fromStatus: current.status,
      toStatus: "active",
      correctedBy: null,
      supersededAt: null,
    };
  }
}

function openRepairDatabase(path: string, readOnly: boolean): SqliteDatabase {
  let raw: SqliteRawDatabase | undefined;

  try {
    raw = new SqliteRawDatabase(
      new DatabaseSync(path, {
        enableDoubleQuotedStringLiterals: true,
        readOnly,
      }),
    );
    const db = new SqliteDatabase(raw);
    db.pragma("busy_timeout = 5000");
    db.pragma("foreign_keys = ON");

    if (readOnly) {
      db.pragma("query_only = ON");
    }

    return db;
  } catch (error) {
    try {
      raw?.close();
    } catch {
      // Preserve the original open failure.
    }

    throw error;
  }
}

type RepairCliArgs =
  | { help: true }
  | {
      help: false;
      dataDir: string;
      apply: boolean;
    };

export function parseRepairCliArgs(argv: readonly string[]): RepairCliArgs {
  let dataDir: string | undefined;
  let apply = false;

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];

    if (argument === "--help" || argument === "-h") {
      return { help: true };
    }

    if (argument === "--apply") {
      apply = true;
      continue;
    }

    if (argument === "--data-dir") {
      const value = argv[index + 1];

      if (value === undefined || value.startsWith("--")) {
        throw new Error("--data-dir requires a path");
      }

      dataDir = value;
      index += 1;
      continue;
    }

    if (argument !== undefined && !argument.startsWith("--") && dataDir === undefined) {
      dataDir = argument;
      continue;
    }

    throw new Error(`Unknown argument: ${argument ?? ""}`);
  }

  if (dataDir === undefined || dataDir.trim().length === 0) {
    throw new Error("A bank data directory is required");
  }

  return {
    help: false,
    dataDir: resolve(dataDir),
    apply,
  };
}

function usage(): string {
  return [
    "Usage: pnpm tsx scripts/repair-wrong-duplicate-supersedes.ts --data-dir <bank-dir> [--apply]",
    "",
    "Dry-run is the default. Stop every Borg writer and take a verified backup before --apply.",
  ].join("\n");
}

export function formatRepairReport(report: WrongDuplicateRepairReport): string {
  const lines = [
    `mode=${report.dryRun ? "dry-run" : "apply"}`,
    `matching_reviews=${report.matchingReviews.length}`,
    `restore_candidates=${report.candidates.length}`,
    `restored=${report.restored.length}`,
    `malformed_reviews=${report.malformedReviews.length}`,
    `out_of_scope_reviews=${report.outOfScopeReviews.length}`,
    `missing_target_reviews=${report.missingTargets.length}`,
  ];
  const restoredIds = new Set(report.restored.map((candidate) => candidate.nodeId));

  for (const candidate of report.candidates) {
    const action = report.dryRun
      ? "would_restore"
      : restoredIds.has(candidate.nodeId)
        ? "restored"
        : "not_restored";
    lines.push(
      `${action} node=${candidate.nodeId} counterpart=${candidate.counterpartNodeId} label=${JSON.stringify(candidate.label)} corrected_by=${candidate.correctedBy} archived=${candidate.archived} reviews=${candidate.reviewIds.join(",")}`,
    );
  }

  for (const reviewId of report.currentOnlyReviewIds) {
    lines.push(`review=${reviewId} current_labels_only=no_change`);
  }

  for (const malformed of report.malformedReviews) {
    lines.push(`review=${malformed.reviewId} malformed=${JSON.stringify(malformed.reason)}`);
  }

  for (const outOfScope of report.outOfScopeReviews) {
    lines.push(`review=${outOfScope.reviewId} out_of_scope=${JSON.stringify(outOfScope.reason)}`);
  }

  for (const missing of report.missingTargets) {
    lines.push(
      `review=${missing.reviewId} missing_targets=${missing.missingNodeIds.join(",")} pair=${missing.nodeIds.join(",")}`,
    );
  }

  return `${lines.join("\n")}\n`;
}

export function repairReportExitCode(report: WrongDuplicateRepairReport): 0 | 1 {
  if (report.dryRun) {
    return 0;
  }

  return report.malformedReviews.length > 0 || report.missingTargets.length > 0 ? 1 : 0;
}

export async function main(argv: readonly string[] = process.argv.slice(2)): Promise<0 | 1> {
  const args = parseRepairCliArgs(argv);

  if (args.help) {
    process.stdout.write(`${usage()}\n`);
    return 0;
  }

  const databasePath = join(args.dataDir, "borg.db");

  if (!existsSync(databasePath)) {
    throw new Error(`No borg.db found in data directory ${args.dataDir}`);
  }

  process.stderr.write(
    "WARNING: this maintenance requires a verified backup and exclusive single-writer access.\n",
  );
  const db = openRepairDatabase(databasePath, !args.apply);
  const clock = new SystemClock();

  try {
    const reviewQueueRepository = new ReviewQueueRepository({ db, clock });
    const report = await repairWrongDuplicateSupersedes(
      {
        db,
        reviewQueueRepository,
        semanticNodeRepository: new SqliteRepairSemanticNodeRepository(db, clock),
        auditLog: new AuditLog({ db, clock }),
        runId: createMaintenanceRunId(),
      },
      { apply: args.apply },
    );
    process.stdout.write(formatRepairReport(report));
    const exitCode = repairReportExitCode(report);

    if (exitCode !== 0) {
      process.stderr.write(
        "ERROR: apply completed with review rows that could not be safely evaluated; inspect the report.\n",
      );
    }

    return exitCode;
  } finally {
    db.close();
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().then(
    (exitCode) => {
      process.exitCode = exitCode;
    },
    (error: unknown) => {
      process.stderr.write(
        `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
      );
      process.exitCode = 1;
    },
  );
}
