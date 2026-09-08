import { z } from "zod";
import { planAudienceScopingMigration } from "../migrate-audience-scoping.js";
import { planGoalTargetAtRepair } from "../repair-goal-target-at.js";
import { planGoalSpeakerOwnerRepair } from "../repair-goal-speaker-owner.js";
import { planGoalRollbackAuditRepair } from "../repair-goal-rollback-audit.js";
import { planRuminationScaffoldingRepairs } from "../repair-rumination-scaffolding.js";
import {
  migrateOutcomeCorpus,
  type OutcomeCorpusMigrationReport,
} from "../migrate-outcome-corpus.js";
import { EpisodicRepository } from "../../src/memory/episodic/repository.js";
import { AuditLog } from "../../src/offline/audit-log.js";
import { LanceDbTable } from "../../src/storage/lancedb/index.js";
import { createMaintenanceRunId } from "../../src/util/ids.js";
import type { EpisodeTable, SqliteSnapshot } from "./bank.js";
import { columns } from "./sqlite-checks.js";
import { counter, ReportChecks, ResidueReportError } from "./report.js";

export const repairInputsSchema = z.record(
  z.string().min(1),
  z
    .object({
      audience: z
        .object({
          fromEntityIds: z.array(z.string().min(1)).min(1),
          toEntityId: z.string().min(1),
        })
        .strict()
        .optional(),
      targetGoalIds: z.array(z.string().min(1)).min(1).optional(),
    })
    .strict(),
);
export type TenantRepairInputs = z.infer<typeof repairInputsSchema>[string];

function candidates(ids: Iterable<unknown>, total?: number) {
  const matches = counter();
  for (const id of ids) matches.add(id);
  return { ...matches.result, ...(total === undefined ? {} : { total }) };
}

export async function repairChecks(
  snapshot: SqliteSnapshot,
  report: ReportChecks,
  inputs: TenantRepairInputs = {},
): Promise<void> {
  const { db, directory: dataDir } = snapshot;
  const prefix =
    "Only the exported planner is called, against the read-only scratch SQLite copy. No CLI main/apply/Borg.open. ";
  if (inputs.audience === undefined) {
    report.skip(
      "R.migrate-audience-scoping",
      "Audience-scoping migration candidates",
      "planAudienceScopingMigration requires operator-selected fromEntityIds and toEntityId. Supply --repair-inputs; audience identity is not inferred from stored text.",
    );
  } else {
    await report.run(
      "R.migrate-audience-scoping",
      "Audience-scoping migration candidates",
      `${prefix}scripts/migrate-audience-scoping.ts planAudienceScopingMigration({dataDir: scratch, ...repairInputs[tenant].audience}).candidates.length; selects active commitments restricted to fromEntityIds and active goals with audience_entity_id in fromEntityIds; validates destination group.`,
      () => {
        const plan = planAudienceScopingMigration({ dataDir, ...inputs.audience });
        return candidates(plan.candidates.map((candidate) => candidate.id));
      },
    );
  }
  if (inputs.targetGoalIds === undefined) {
    report.skip(
      "R.repair-goal-target-at",
      "Goal deadline repair candidates",
      "planGoalTargetAtRepair requires explicit operator-selected goal IDs whose deadlines were guessed. Supply --repair-inputs targetGoalIds; a non-null deadline alone is not evidence of residue.",
    );
  } else {
    let plan: ReturnType<typeof planGoalTargetAtRepair> | undefined;
    await report.run(
      "R.repair-goal-target-at",
      "Goal deadline repair candidates",
      `${prefix}scripts/repair-goal-target-at.ts planGoalTargetAtRepair({dataDir: scratch, goalIds: repairInputs[tenant].targetGoalIds}).candidates.length; supplied goals with target_at IS NOT NULL.`,
      () => {
        plan = planGoalTargetAtRepair({ dataDir, goalIds: inputs.targetGoalIds });
        return candidates(
          plan.candidates.map((candidate) => candidate.id),
          plan.requestedGoalIds.length,
        );
      },
    );
    if (plan !== undefined) {
      const value = plan;
      await report.run(
        "R.repair-goal-target-at.refusals",
        "Requested deadline-repair goals that do not exist",
        "Same plan; refusals.length",
        () =>
          candidates(
            value.refusals.map((row) => row.id),
            value.requestedGoalIds.length,
          ),
      );
    }
  }

  await report.run(
    "R.repair-goal-speaker-owner",
    "Goal speaker-as-owner repair candidates",
    `${prefix}scripts/repair-goal-speaker-owner.ts planGoalSpeakerOwnerRepair({dataDir: scratch}).candidates.length; earliest creation event from goal-promotion-extractor, recorded non-self/non-null owner still matches current owner, no later owner-changing identity event. Total = plan.counts.total.`,
    () => {
      for (const table of ["entities", "goals", "identity_events"]) columns(db, table);
      if (db.prepare("SELECT id FROM entities WHERE kind = 'self' LIMIT 1").get() === undefined)
        throw new ResidueReportError("Speaker-owner planner requires an existing self entity");
      const plan = planGoalSpeakerOwnerRepair({ dataDir });
      return candidates(
        plan.candidates.map((candidate) => candidate.id),
        plan.counts.total,
      );
    },
  );

  let rollback: ReturnType<typeof planGoalRollbackAuditRepair> | undefined;
  await report.run(
    "R.repair-goal-rollback-audit",
    "Stranded goal creation chains needing terminal audit events",
    `${prefix}scripts/repair-goal-rollback-audit.ts planGoalRollbackAuditRepair({dataDir: scratch}).candidates.length; missing live goal, first identity event is create, no delete/forget event.`,
    () => {
      for (const table of ["goals", "identity_events"]) columns(db, table);
      rollback = planGoalRollbackAuditRepair({ dataDir });
      return candidates(rollback.candidates.map((candidate) => candidate.goalId));
    },
  );
  if (rollback !== undefined) {
    const plan = rollback;
    await report.run(
      "R.repair-goal-rollback-audit.status_drift",
      "Live goals differing from their latest audited status",
      "Same planGoalRollbackAuditRepair result; statusDrifts.length (report only).",
      () => candidates(plan.statusDrifts.map((row) => row.goalId)),
    );
  }

  const manual = counter();
  let ruminationTotal = 0;
  let ruminationComplete = false;
  const ruminationQuery =
    "SELECT id, tensions FROM open_question_ruminations ORDER BY id ASC; pass each raw row to scripts/repair-rumination-scaffolding.ts planRuminationScaffoldingRepairs([row]); sum candidates and manualDecisions separately. Pure planning only.";
  await report.run(
    "R.repair-rumination-scaffolding",
    "Recoverable rumination scaffolding repair candidates",
    ruminationQuery,
    () => {
      const matches = counter();
      for (const row of db
        .prepare("SELECT id, tensions FROM open_question_ruminations ORDER BY id ASC")
        .iterate()) {
        ruminationTotal += 1;
        const plan = planRuminationScaffoldingRepairs([row]);
        for (const candidate of plan.candidates) matches.add(candidate.id);
        for (const decision of plan.manualDecisions) manual.add(decision.id);
      }
      ruminationComplete = true;
      return { ...matches.result, total: ruminationTotal };
    },
  );
  if (ruminationComplete)
    await report.run(
      "R.repair-rumination-scaffolding.manual",
      "Rumination scaffolding requiring manual decisions",
      ruminationQuery,
      () => ({ ...manual.result, total: ruminationTotal }),
    );
}

export async function outcomeChecks(
  snapshot: SqliteSnapshot,
  opened: EpisodeTable,
  report: ReportChecks,
): Promise<void> {
  const query = `scripts/migrate-outcome-corpus.ts migrateOutcomeCorpus(dependencies, {apply: false}), default specification. Dependencies use read-only scratch SQLite and episodes checkout(${opened.version}). Bypass CLI main, LanceDbStore.openTable and schema evolution. Repository exposes only listAll/listStats/listConsolidationFamilies/listConsolidationMembers/isEpisodeEffectivelyVisible; no backfill or mutation methods. No embedding client. Count = new multi-source rollup groups + legacyFamiliesToDissolve + versionsToReembed + toxicEpisodes in would_archive state. Counts describe planned operations even if unsafeItems prevents applying the original migration.`;
  let plan: OutcomeCorpusMigrationReport | undefined;
  await report.run(
    "R.migrate-outcome-corpus",
    "Outcome-corpus candidate operations",
    query,
    async () => {
      for (const table of [
        "episode_stats",
        "episode_index",
        "consolidation_families",
        "consolidation_members",
        "maintenance_audit",
        "semantic_nodes",
      ])
        columns(snapshot.db, table);
      const repository = new EpisodicRepository({
        db: snapshot.db,
        table: new LanceDbTable(opened.table),
      });
      const allowed = new Set<PropertyKey>([
        "listAll",
        "listStats",
        "listConsolidationFamilies",
        "listConsolidationMembers",
        "isEpisodeEffectivelyVisible",
      ]);
      const readRepository = new Proxy(repository, {
        get(target, property) {
          if (!allowed.has(property))
            throw new ResidueReportError(
              "Outcome planner requested a repository operation outside its read-only contract",
            );
          const method = Reflect.get(target, property) as (...args: unknown[]) => unknown;
          return method.bind(target);
        },
      });
      const audit = new AuditLog({ db: snapshot.db });
      plan = await migrateOutcomeCorpus(
        {
          db: snapshot.db,
          episodicRepository: readRepository,
          auditLog: {
            list: audit.list.bind(audit),
            record() {
              throw new ResidueReportError("Outcome planner attempted an audit write");
            },
          },
          clock: { now: () => Date.now() },
          runId: createMaintenanceRunId(),
        },
        { apply: false },
      );
      const ids = [
        ...plan.groups
          .filter((group) => group.sources.length > 1 && group.existingFamily === null)
          .map((group) => group.key),
        ...plan.legacyFamiliesToDissolve.map((item) => item.family.family_id),
        ...plan.versionsToReembed.map((item) => item.episode.id),
        ...plan.toxicEpisodes
          .filter((item) => item.state === "would_archive")
          .map((item) => item.id),
      ];
      return candidates(ids);
    },
  );
  if (plan === undefined) return;
  const value = plan;
  const counts: Record<string, number> = {
    raw_outcomes: value.rawOutcomeEpisodeCount,
    rollup_groups: value.groups.filter(
      (group) => group.sources.length > 1 && group.existingFamily === null,
    ).length,
    dissolve_families: value.legacyFamiliesToDissolve.length,
    reembed_versions: value.versionsToReembed.length,
    archive_episodes: value.toxicEpisodes.filter((item) => item.state === "would_archive").length,
    unsafe_items: value.unsafeItems.length,
    unsafe_families: value.unsafeFamilies.length,
  };
  for (const [name, count] of Object.entries(counts))
    await report.run(
      `R.migrate-outcome-corpus.${name}`,
      `Outcome-corpus ${name}`,
      `${query}\nBreakdown: ${name}.`,
      () => ({ count }),
    );
}
