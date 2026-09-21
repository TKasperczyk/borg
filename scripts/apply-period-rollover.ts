import { statSync } from "node:fs";
import { userInfo } from "node:os";
import { join, relative, sep } from "node:path";
import { CommitmentRepository } from "../src/memory/commitments/repository.js";
import { IdentityEventRepository } from "../src/memory/identity/repository.js";
import { IdentityService } from "../src/memory/identity/service.js";
import {
  ReviewQueueHandlerRegistry,
  ReviewQueueRepository,
  stripReviewResolverRefs,
  type ReviewQueueItem,
} from "../src/memory/review-queue/review-queue.js";
import {
  createIdentityInconsistencyReviewQueueHandler,
  identityInconsistencyReviewRefsSchema,
  type IdentityInconsistencyReviewRefs,
} from "../src/memory/review-queue/review-handlers/identity-inconsistency.js";
import {
  AutobiographicalRepository,
  type AutobiographicalPeriod,
} from "../src/memory/self/autobiographical.js";
import { GoalsRepository } from "../src/memory/self/goals-repository.js";
import { GrowthMarkersRepository } from "../src/memory/self/growth-markers.js";
import { OpenQuestionsRepository } from "../src/memory/self/open-questions.js";
import { TraitsRepository } from "../src/memory/self/traits-repository.js";
import { ValuesRepository } from "../src/memory/self/values-repository.js";
import type { MaintenanceAuditRecordInput } from "../src/offline/audit-log.js";
import type { SqliteDatabase } from "../src/storage/sqlite/index.js";
import { FixedClock } from "../src/util/clock.js";
import { cloneDatabase, type Row } from "./repair-entity-graph-storage.js";
import {
  applyDataRepair,
  assertNoPending,
  assertSame,
  auditInput,
  checkedPath,
  hasStep,
  loadRepairPlan,
  newPlan,
  openDataRepairDatabase,
  printPlan,
  refuse,
  repairArgs,
  repairAudit,
  runRepairCli,
  type DataRepairPlan,
  type RepairCheckpoint,
} from "./audited-data-repair.js";

const KIND = "apply-period-rollover";
export const MANUAL_RESOLUTION_REF = "__borg_review_resolver_manual_resolution";
type PeriodRefs = Extract<
  IdentityInconsistencyReviewRefs,
  { target_type: "autobiographical_period" }
>;
type RolloverRefs = PeriodRefs & { next_period_open_payload: AutobiographicalPeriod };
type ReviewChange = { before: ReviewQueueItem; after: ReviewQueueItem; resolution_text: string };
type Dismissal = ReviewChange & { classification: "duplicate_closure" | "superseded_narrative" };
export type PeriodRolloverPlan = DataRepairPlan & {
  review_id: number;
  operator: string;
  backup_verified: string | null;
  boundary: number;
  boundary_iso: string;
  applied_refs: RolloverRefs;
  review: ReviewChange;
  predecessor: { before: AutobiographicalPeriod; after: AutobiographicalPeriod };
  successor: { before: null; after: AutobiographicalPeriod };
  periods_before: AutobiographicalPeriod[];
  dismissals: Dismissal[];
  identity_events: Row[];
  audits: MaintenanceAuditRecordInput[];
};
export type RolloverInput = {
  dataDir: string;
  reviewId: number;
  dismiss?: number[];
  operator?: string;
  backupVerified?: string;
};

function reviewId(value: unknown): number {
  const id = Number(value);
  if (!Number.isSafeInteger(id) || id <= 0) refuse("Review IDs must be positive safe integers");
  return id;
}
function periodRefs(item: ReviewQueueItem): PeriodRefs {
  if (
    item.kind !== "identity_inconsistency" ||
    item.resolved_at !== null ||
    item.resolution !== null
  )
    refuse(`Review ${item.id} must be an open identity_inconsistency row`);
  const refs = identityInconsistencyReviewRefsSchema.parse(stripReviewResolverRefs(item.refs));
  if (refs.target_type !== "autobiographical_period")
    refuse(`Review ${item.id} does not target an autobiographical period`);
  return refs;
}
function sorted(periods: AutobiographicalPeriod[]): AutobiographicalPeriod[] {
  return [...periods].sort((a, b) => a.start_ts - b.start_ts || a.id.localeCompare(b.id));
}
function periods(db: SqliteDatabase): AutobiographicalPeriod[] {
  return sorted(new AutobiographicalRepository({ db }).listPeriods({ limit: -1 }));
}
function validateTimeline(rows: AutobiographicalPeriod[]): void {
  const ordered = sorted(rows);
  if (ordered.filter((p) => p.end_ts === null).length !== 1)
    refuse("Exactly one period must be open");
  for (let i = 0; i < ordered.length; i++) {
    const period = ordered[i]!,
      next = ordered[i + 1];
    if (period.end_ts !== null && period.end_ts <= period.start_ts)
      refuse("Period has an empty or reversed interval");
    if (!next) {
      if (period.end_ts !== null) refuse("Orphan timeline: the last period must be open");
    } else {
      if (period.end_ts === null || period.end_ts > next.start_ts) refuse("Overlapping periods");
      if (period.end_ts < next.start_ts) refuse("Orphan interval between periods");
    }
  }
}

export function verifyRolloverBackup(dataDir: string, path: string | undefined): string {
  if (!path)
    refuse("--apply requires --backup-verified <path> for an operator-verified fresh backup");
  const backup = checkedPath(path),
    bank = checkedPath(dataDir);
  const inside = (parent: string, child: string) => {
    const rel = relative(parent, child);
    return rel === "" || (rel !== ".." && !rel.startsWith(`..${sep}`) && !rel.startsWith(sep));
  };
  if (inside(bank, backup) || inside(backup, bank))
    refuse("Backup must be independent of the repair bank");
  const stat = statSync(backup),
    source = statSync(join(bank, "borg.db"));
  if (
    (!stat.isFile() && !stat.isDirectory()) ||
    (stat.dev === source.dev && stat.ino === source.ino)
  )
    refuse("Backup must be an independent file or directory");
  return backup;
}

function reviewChange(
  before: ReviewQueueItem,
  decision: "accept" | "dismiss",
  text: string,
  refs: Record<string, unknown>,
  operator: string,
  boundary: number,
  now: number,
): ReviewChange {
  return {
    before,
    resolution_text: text,
    after: {
      ...before,
      resolved_at: now,
      resolution: decision,
      refs: { ...refs, [MANUAL_RESOLUTION_REF]: { operator, boundary, text } },
    },
  };
}

/** Register an offline adapter around the SAME handler and queue transaction used by the resolver.
 * The adapter supplies the reviewed manual refs, persists explanation metadata and audits inside
 * that transaction; period mutations and review finalization remain in their normal repositories.
 */
function reviewQueue(
  db: SqliteDatabase,
  plan: PeriodRolloverPlan,
  projection = false,
): ReviewQueueRepository {
  const clock = new FixedClock(plan.plannedAt),
    options = { db, clock };
  const autobiographicalRepository = new AutobiographicalRepository(options);
  const identityEventRepository = new IdentityEventRepository(options);
  const identityService = new IdentityService({
    autobiographicalRepository,
    identityEventRepository,
    valuesRepository: new ValuesRepository(options),
    goalsRepository: new GoalsRepository(options),
    traitsRepository: new TraitsRepository(options),
    growthMarkersRepository: new GrowthMarkersRepository(options),
    openQuestionsRepository: new OpenQuestionsRepository(options),
    commitmentRepository: new CommitmentRepository(options),
  });
  const handlers = new ReviewQueueHandlerRegistry();
  const handler = createIdentityInconsistencyReviewQueueHandler();
  handlers.register({
    ...handler,
    async apply(input) {
      const primary = input.item.id === plan.review_id;
      const change = primary
        ? plan.review
        : (plan.dismissals.find((d) => d.before.id === input.item.id) ??
          refuse("Unplanned review resolution"));
      assertSame(input.item, change.before, `review ${input.item.id}`);
      assertSame(input.resolution.decision, change.after.resolution, "review decision");
      if (!projection) preflight(db, plan);
      await handler.apply({
        ...input,
        item: { ...input.item, reason: change.resolution_text },
        refs: primary ? plan.applied_refs : input.refs,
      });
      if (primary) {
        validateTimeline(periods(db));
        if (!projection) {
          assertSame(
            autobiographicalRepository.getPeriod(plan.predecessor.before.id),
            plan.predecessor.after,
            "closed period",
          );
          assertSame(
            autobiographicalRepository.getPeriod(plan.successor.after.id),
            plan.successor.after,
            "opened successor",
          );
        }
      }
      if (!projection) {
        const audit =
          plan.audits.find((a) => a.action === `${KIND}:review:${input.item.id}`) ??
          refuse("Missing review audit plan");
        repairAudit(db, plan).record(audit);
      }
      return { refs: change.after.refs };
    },
  });
  return new ReviewQueueRepository({
    ...options,
    handlers,
    autobiographicalRepository,
    identityEventRepository,
    identityService,
  });
}

export async function planPeriodRollover(
  input: RolloverInput,
  now = Date.now(),
): Promise<PeriodRolloverPlan> {
  const dataDir = checkedPath(input.dataDir),
    id = reviewId(input.reviewId);
  const operator = (input.operator ?? userInfo().username).trim();
  if (!operator) refuse("Operator name must be nonempty");
  assertNoPending(dataDir);
  const backup =
    input.backupVerified === undefined ? null : verifyRolloverBackup(dataDir, input.backupVerified);
  const dismissIds = (input.dismiss ?? []).map(reviewId);
  if (new Set([id, ...dismissIds]).size !== dismissIds.length + 1)
    refuse("Review IDs must be distinct; the applied row cannot be dismissed");
  const db = openDataRepairDatabase(dataDir);
  try {
    const queue = new ReviewQueueRepository({ db });
    const item = queue.get(id) ?? refuse(`Missing review ${id}`),
      refs = periodRefs(item);
    const next = refs.next_period_open_payload;
    if (typeof refs.patch.end_ts !== "number" || next === undefined)
      refuse("Target review must propose a closure and next_period_open_payload");
    if (next.end_ts !== null)
      refuse("Successor must be open; a closed successor would orphan the timeline");
    const before = periods(db);
    if (before.filter((p) => p.end_ts === null).length !== 1)
      refuse("Exactly one period must be open");
    const predecessor =
      before.find((p) => p.id === refs.target_id) ?? refuse("Missing target period");
    if (predecessor.end_ts !== null) refuse("Target period is not open");
    if (before.some((p) => p.id === next.id)) refuse("Successor period already exists");
    if (refs.patch.start_ts !== undefined && refs.patch.start_ts !== predecessor.start_ts)
      refuse("Changing the predecessor start would orphan its prior boundary");
    const boundary = next.start_ts;
    if (boundary <= predecessor.start_ts || !Number.isFinite(new Date(boundary).getTime()))
      refuse("Boundary must be a valid instant strictly after predecessor start");
    validateTimeline(before);
    const appliedRefs: RolloverRefs = {
      ...refs,
      patch: { ...refs.patch, end_ts: boundary, provenance: { kind: "manual" } },
      proposed_provenance: { kind: "manual" },
      next_period_open_payload: { ...next, provenance: { kind: "manual" } },
    };
    const boundaryIso = new Date(boundary).toISOString();
    const resolution = `operator ${operator} applied rollover ${id} at boundary ${boundary} (${boundaryIso})`;
    const plan: PeriodRolloverPlan = {
      ...newPlan(KIND, dataDir, 2, now),
      review_id: id,
      operator,
      backup_verified: backup,
      boundary,
      boundary_iso: boundaryIso,
      applied_refs: appliedRefs,
      review: reviewChange(
        item,
        "accept",
        resolution,
        { ...item.refs, ...appliedRefs },
        operator,
        boundary,
        now,
      ),
      predecessor: { before: predecessor, after: predecessor },
      successor: { before: null, after: appliedRefs.next_period_open_payload },
      periods_before: before,
      dismissals: [],
      identity_events: [],
      audits: [],
    };
    for (const dismissId of dismissIds) {
      const row = queue.get(dismissId) ?? refuse(`Missing dismissal review ${dismissId}`),
        dismissRefs = periodRefs(row);
      if (dismissRefs.target_id !== predecessor.id)
        refuse(`Dismissal ${dismissId} targets another period`);
      let classification: Dismissal["classification"], text: string;
      if (typeof dismissRefs.patch.end_ts === "number") {
        classification = "duplicate_closure";
        text = `dismissed as duplicate of ${id}`;
      } else {
        const keys = Object.keys(dismissRefs.patch);
        const narrativeFields = ["label", "narrative", "key_episode_ids", "themes"];
        if (
          dismissRefs.next_period_open_payload !== undefined ||
          !keys.some((k) => narrativeFields.includes(k)) ||
          keys.some((k) => ![...narrativeFields, "provenance", "disclosure_label"].includes(k))
        )
          refuse(`Dismissal ${dismissId} is neither a closure nor a narrative patch`);
        classification = "superseded_narrative";
        text = `superseded by the operator-applied rollover ${id}; content after the boundary belongs to the successor`;
      }
      plan.dismissals.push({
        ...reviewChange(row, "dismiss", text, row.refs, operator, boundary, now),
        classification,
      });
    }
    // Execute the actual handler on an in-memory clone to plan all repository effects.
    const projection = cloneDatabase(db);
    try {
      const eventIds = new Set(
        projection
          .prepare("SELECT id FROM identity_events")
          .all()
          .map((r) => r.id),
      );
      const projectedQueue = reviewQueue(projection, plan, true);
      for (const change of [plan.review, ...plan.dismissals]) {
        await projectedQueue.resolve(
          change.before.id,
          { decision: change.after.resolution!, reason: change.resolution_text },
          { source: "manual" },
        );
        assertSame(
          projectedQueue.get(change.before.id),
          change.after,
          `projected review ${change.before.id}`,
        );
      }
      const repo = new AutobiographicalRepository({ db: projection });
      plan.predecessor.after = repo.getPeriod(predecessor.id)!;
      plan.successor.after = repo.getPeriod(next.id)!;
      plan.identity_events = projection
        .prepare("SELECT * FROM identity_events ORDER BY id")
        .all()
        .filter((r) => !eventIds.has(r.id));
    } finally {
      projection.close();
    }
    plan.audits.push(
      auditInput(
        plan,
        `${KIND}:review:${id}`,
        {
          operator,
          boundary,
          backup_verified: backup,
          predecessor: plan.predecessor,
          successor: plan.successor,
          review: plan.review,
        },
        {
          predecessor_before: predecessor,
          predecessor_expected_after: plan.predecessor.after,
          successor_before: null,
          successor_expected_after: plan.successor.after,
          review_before: item,
          review_expected_after: plan.review.after,
        },
      ),
    );
    for (const dismissal of plan.dismissals)
      plan.audits.push(
        auditInput(
          plan,
          `${KIND}:review:${dismissal.before.id}`,
          { ...dismissal },
          { review_before: dismissal.before, review_expected_after: dismissal.after },
        ),
      );
    plan.summary = {
      changed: 2,
      accepted_reviews: 1,
      dismissed_reviews: plan.dismissals.length,
      identity_events: plan.identity_events.length,
    };
    return plan;
  } finally {
    db.close();
  }
}

function preflight(db: SqliteDatabase, plan: PeriodRolloverPlan): void {
  const committed = hasStep(db, plan, `${KIND}:review:${plan.review_id}`);
  const expected = committed
    ? sorted([
        ...plan.periods_before.filter((p) => p.id !== plan.predecessor.before.id),
        plan.predecessor.after,
        plan.successor.after,
      ])
    : plan.periods_before;
  assertSame(periods(db), expected, "period timeline");
  validateTimeline(expected);
  const queue = new ReviewQueueRepository({ db });
  for (const change of [plan.review, ...plan.dismissals]) {
    const done = hasStep(db, plan, `${KIND}:review:${change.before.id}`);
    if (done && !committed) refuse("Dismissal committed before the rollover");
    assertSame(
      queue.get(change.before.id),
      done ? change.after : change.before,
      `review ${change.before.id}`,
    );
  }
}
export async function applyPeriodRollover(
  plan: PeriodRolloverPlan,
  options: { backupVerified?: string; resume?: boolean; checkpoint?: RepairCheckpoint } = {},
): Promise<void> {
  const backup = verifyRolloverBackup(plan.dataDir, options.backupVerified);
  assertSame(
    backup,
    plan.backup_verified,
    "verified backup; include --backup-verified when planning apply",
  );
  await applyDataRepair(
    plan,
    {
      preflight,
      async finish(db, plan, checkpoint) {
        const queue = reviewQueue(db, plan);
        for (const change of [plan.review, ...plan.dismissals]) {
          if (!hasStep(db, plan, `${KIND}:review:${change.before.id}`)) {
            await queue.resolve(
              change.before.id,
              { decision: change.after.resolution!, reason: change.resolution_text },
              { source: "manual", sourceProcess: "correction" },
            );
          }
          await checkpoint?.(`after_review:${change.before.id}`);
        }
      },
    },
    options,
  );
}

runRepairCli(import.meta.url, async () => {
  const args = repairArgs(process.argv.slice(2), {
    "review-id": { type: "string" },
    dismiss: { type: "string", multiple: true },
    operator: { type: "string" },
    "backup-verified": { type: "string" },
  });
  const backup =
    typeof args.values["backup-verified"] === "string" ? args.values["backup-verified"] : undefined;
  let plan: PeriodRolloverPlan;
  if (args.resume) {
    plan = loadRepairPlan<PeriodRolloverPlan>(args.dataDir, KIND);
    if (args.values["review-id"] !== undefined)
      assertSame(reviewId(args.values["review-id"]), plan.review_id, "--review-id");
    if (args.values.operator !== undefined)
      assertSame(args.values.operator, plan.operator, "--operator");
    if (Array.isArray(args.values.dismiss))
      assertSame(
        args.values.dismiss.map(reviewId),
        plan.dismissals.map((d) => d.before.id),
        "--dismiss",
      );
  } else {
    plan = await planPeriodRollover({
      dataDir: args.dataDir,
      reviewId: reviewId(args.values["review-id"]),
      dismiss: Array.isArray(args.values.dismiss) ? args.values.dismiss.map(reviewId) : [],
      operator: typeof args.values.operator === "string" ? args.values.operator : undefined,
      backupVerified: backup,
    });
  }
  await printPlan(plan);
  if (args.apply) {
    await applyPeriodRollover(plan, { backupVerified: backup, resume: args.resume });
    console.log(JSON.stringify({ run_id: plan.runId, ...plan.summary }));
  }
});
