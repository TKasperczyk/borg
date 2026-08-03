/*
 * One-off: roll up scheduled OUTCOME episodes by role and UTC day, backfill
 * surviving OUTCOME consolidation embeddings, and archive the validated toxic
 * corpus rows listed below.
 *
 * Run with every Borg writer stopped and take a verified backup first. Dry-run
 * is the default; --apply performs repository writes. The script opens only
 * borg.db, the episodes LanceDB table, and the embedding client needed by apply.
 *
 * This migration is intentionally not resumable. Its SQLite and LanceDB writes
 * cannot form one atomic commit. Any thrown mutation error, missing migration
 * audit row, or other unsafe partial-state report means: stop, do not rerun the
 * damaged bank, restore the entire bank from the verified pre-surgery backup,
 * and only then investigate or retry. Best-effort in-process restoration is
 * diagnostic containment, not a supported rollback mechanism.
 *
 * The custom migration audit actions have no registered AuditLog reversers.
 * Their `reversal` objects retain complete restoration payloads for manual and
 * forensic inspection and explicitly carry `no_reverser: true`; the supported
 * rollback for the one-off surgery is restoration of the pre-surgery backup.
 *
 * Usage:
 *   pnpm tsx scripts/migrate-outcome-corpus.ts --data-dir <bank-dir>
 *   KRATOS_BASE_URL=<url> LLM_API_KEY=<key> EMBEDDING_MODEL=<model> \
 *     EMBEDDING_DIMS=<dims> pnpm tsx scripts/migrate-outcome-corpus.ts \
 *     --data-dir <freshly-extracted-bank-dir> --apply
 */
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { pathToFileURL } from "node:url";

import { connect } from "@lancedb/lancedb";

import { OpenAICompatibleEmbeddingClient, type EmbeddingClient } from "../src/embeddings/index.js";
import {
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromEpisodeAccess,
} from "../src/memory/common/index.js";
import {
  buildConsolidationCoverageHash,
  buildConsolidationEpisodeEmbeddingText,
  collectProtectedEpisodeTokenLines,
  createEpisodesTableSchema,
  EpisodicRepository,
  normalizeEpisodeAccess,
  type ConsolidationFamilyRecord,
  type ConsolidationMemberRecord,
  type Episode,
  type EpisodeStats,
  type EpisodeTier,
} from "../src/memory/episodic/index.js";
import { AuditLog } from "../src/offline/audit-log.js";
import { LanceDbStore, type LanceDbTable } from "../src/storage/lancedb/index.js";
import { SqliteDatabase, SqliteRawDatabase } from "../src/storage/sqlite/index.js";
import { SystemClock, type Clock } from "../src/util/clock.js";
import { uniqueStrings } from "../src/util/collections.js";
import {
  createConsolidationFamilyId,
  createEpisodeId,
  createMaintenanceRunId,
  parseConsolidationFamilyId,
  parseEpisodeId,
  parseSemanticNodeId,
  type ConsolidationFamilyId,
  type EpisodeId,
  type MaintenanceRunId,
  type SemanticNodeId,
} from "../src/util/ids.js";
import { utcDayKey } from "../src/util/utc-day.js";

export const OUTCOME_ROLLUP_AUDIT_ACTION = "outcome_corpus_rollup";
export const OUTCOME_FAMILY_DISSOLVE_AUDIT_ACTION = "outcome_corpus_dissolve_family";
export const OUTCOME_VERSION_REEMBED_AUDIT_ACTION = "outcome_corpus_reembed_version";
export const OUTCOME_ROLLUP_MARKER_TAG = "outcome-corpus-rollup-v1";

const CUSTOM_MIGRATION_AUDIT_ACTIONS = new Set([
  OUTCOME_ROLLUP_AUDIT_ACTION,
  OUTCOME_FAMILY_DISSOLVE_AUDIT_ACTION,
  OUTCOME_VERSION_REEMBED_AUDIT_ACTION,
]);
const BACKUP_ROLLBACK_INSTRUCTION = "restore_verified_pre_surgery_backup";
const PARTIAL_STATE_RECOVERY_MESSAGE =
  "restore the entire bank from the verified pre-surgery backup before retrying";

const CONSOLIDATION_POLICY_VERSION = 1;
const OUTCOME_ROLLUP_SCRIPT_VERSION = 1;
const RECALL_LIMIT = 8;
// The fp is an already-known machine source handle. Keep two exact-handle
// rescue slots beside vector recall so the write-dedup check cannot be crowded
// out by semantically similar OUTCOME records.
const RECALL_EXACT_FP_RESERVED_SLOTS = 2;
const DEFAULT_OUTCOME_FINGERPRINT_PATTERN = /OUTCOME fp=(\S+)/gu;
const DEFAULT_OUTCOME_ROLE_FIELD_PATTERN = /(?:^|[\t ])role=(\S+)/gu;
const DEFAULT_OUTCOME_ACTION_FIELD_PATTERN =
  /(?:^|[\t ])(decision|ticket|action|transition|verdict|summary|mr|teams_card|card_count|sample_ids|run|thread_id)=/gu;
const DEFAULT_SINGLE_TOKEN_OUTCOME_FIELDS = new Set([
  "decision",
  "ticket",
  "action",
  "mr",
  "teams_card",
  "card_count",
  "sample_ids",
  "run",
  "thread_id",
]);
const DEFAULT_SCHEDULED_FINGERPRINT_PATTERN = /^scheduled:/u;
const TIER_RANK = {
  T1: 1,
  T2: 2,
  T3: 3,
  T4: 4,
} as const satisfies Record<EpisodeTier, number>;

const SCHEDULED_FP_SELF_CHECKS = [
  "scheduled:drafter:team-agent-ai",
  "scheduled:aiops:team-agent-ai",
  "scheduled:triage:team-agent-ai",
  "scheduled:autoheal:team-agent-ai",
  "scheduled:openshift:team-agent-ai",
] as const;

export type ToxicEpisodeSpec = {
  id: EpisodeId;
  expectedBodySha256: string;
  expectedKind: "raw" | "consolidation_version";
  conditional: boolean;
  reason: string;
};

const DEFAULT_TOXIC_EPISODE_SPECS: readonly ToxicEpisodeSpec[] = [
  {
    id: parseEpisodeId("ep_3vspwxifw2ahk16h"),
    expectedBodySha256: "e191e1f8f887b9cd7297a2db23c9a61ce0187cefe4c8f875fa625d6833ab41d6",
    expectedKind: "raw",
    conditional: false,
    reason: "validated 2026-07-15 mis-voice episode",
  },
  {
    id: parseEpisodeId("ep_q5jn1a2wbf9dupyl"),
    expectedBodySha256: "0fc7240eb4eaa88b7942433bc0d132bd61cb1dcb100f2737585d43006feb503b",
    expectedKind: "raw",
    conditional: false,
    reason: "validated 2026-07-15 mis-voice episode",
  },
  {
    id: parseEpisodeId("ep_rpqhgi6yfg820kac"),
    expectedBodySha256: "a2de14c6a24d0394d3667d452d938c131a5a100f2ed8c899edb21bbb03570905",
    expectedKind: "raw",
    conditional: false,
    reason: "validated 2026-07-15 mis-voice episode",
  },
  {
    id: parseEpisodeId("ep_4pgb2wvvsz89ibfl"),
    expectedBodySha256: "b011f83f5ffca1a9095d6a6752c1337a916aa4c497ed4bef77c94105e2509bf1",
    expectedKind: "raw",
    conditional: false,
    reason: "validated punishment episode",
  },
  {
    id: parseEpisodeId("ep_1ih966l33t4djq6s"),
    expectedBodySha256: "c27cb7e625768db9ebd9bbe949216a7e209394f1065043a5b4207d8f450e65d6",
    expectedKind: "consolidation_version",
    conditional: false,
    reason: "validated stale non-current consolidation mis-voice version",
  },
  {
    id: parseEpisodeId("ep_c8bxspa8tn9324m5"),
    expectedBodySha256: "7eb596cc0ed4bd64cbdf2efdf912ad6d0d9487861b4b520c698f121d7d753257",
    expectedKind: "raw",
    conditional: true,
    reason: "content-verified punishment episode",
  },
];

const DEFAULT_EXPLICIT_KEEP_EPISODE_ID = parseEpisodeId("ep_3945b0jha998wvxn");
const DEFAULT_EXPLICIT_KEEP_BODY_SHA256 =
  "a8c9a6e5a2c5119dc55df1baea63b843f86f23b5da10e7d44295e61610a8bf1b";
const DEFAULT_PUNISHMENT_SEMANTIC_NODE_ID = parseSemanticNodeId("semn_j5pfpq2ud2byv1ss");

export type OutcomeCorpusGrammar = {
  outcomeFingerprintPattern: RegExp;
  outcomeRoleFieldPattern: RegExp;
  outcomeActionFieldPattern: RegExp;
  singleTokenOutcomeFields: ReadonlySet<string>;
  scheduledFingerprintPattern: RegExp;
};

export type OutcomeCorpusSpecification = {
  toxicEpisodeSpecs: readonly ToxicEpisodeSpec[];
  explicitKeepEpisodeId: EpisodeId;
  explicitKeepBodySha256: string;
  punishmentSemanticNodeId: SemanticNodeId;
  scheduledFpSelfChecks: readonly string[];
  grammar: OutcomeCorpusGrammar;
};

export const DEFAULT_OUTCOME_CORPUS_GRAMMAR: OutcomeCorpusGrammar = {
  outcomeFingerprintPattern: DEFAULT_OUTCOME_FINGERPRINT_PATTERN,
  outcomeRoleFieldPattern: DEFAULT_OUTCOME_ROLE_FIELD_PATTERN,
  outcomeActionFieldPattern: DEFAULT_OUTCOME_ACTION_FIELD_PATTERN,
  singleTokenOutcomeFields: DEFAULT_SINGLE_TOKEN_OUTCOME_FIELDS,
  scheduledFingerprintPattern: DEFAULT_SCHEDULED_FINGERPRINT_PATTERN,
};

export const DEFAULT_OUTCOME_CORPUS_SPECIFICATION: OutcomeCorpusSpecification = {
  toxicEpisodeSpecs: DEFAULT_TOXIC_EPISODE_SPECS,
  explicitKeepEpisodeId: DEFAULT_EXPLICIT_KEEP_EPISODE_ID,
  explicitKeepBodySha256: DEFAULT_EXPLICIT_KEEP_BODY_SHA256,
  punishmentSemanticNodeId: DEFAULT_PUNISHMENT_SEMANTIC_NODE_ID,
  scheduledFpSelfChecks: SCHEDULED_FP_SELF_CHECKS,
  grammar: DEFAULT_OUTCOME_CORPUS_GRAMMAR,
};

type OutcomeActionField = {
  key: string;
  value: string;
};

export type OutcomeActionRecord = {
  action: string;
  decision?: string;
  ticket?: string;
  transition?: string;
  verdict?: string;
  summary?: string;
  mr?: string;
  teamsCard?: string;
  cardCount?: string;
};

export type ScheduledOutcomeSource = {
  episode: Episode;
  stats: EpisodeStats;
  role: string;
  utcDay: string;
  fingerprints: string[];
  protectedLines: string[];
  actions: OutcomeActionRecord[];
};

export type OutcomeRollupRender = {
  title: string;
  prose: string;
  protectedLines: string[];
  narrative: string;
};

type OutcomeRollupGroup = OutcomeRollupRender & {
  key: string;
  role: string;
  utcDay: string;
  sources: ScheduledOutcomeSource[];
  tags: string[];
  participants: string[];
  sourceStreamIds: Episode["source_stream_ids"];
  coverageHash: string;
  embeddingText: string;
  existingFamily: ConsolidationFamilyRecord | null;
  plannedFamilyId: ConsolidationFamilyId | null;
  plannedVersionId: EpisodeId | null;
};

type FamilyDissolution = {
  family: ConsolidationFamilyRecord;
  members: ConsolidationMemberRecord[];
  currentVersion: Episode;
  currentStats: EpisodeStats;
};

export type UnsafeFamilyMember = {
  episodeId: EpisodeId;
  intendedState: string;
};

export type UnsafeFamilyPlan = {
  familyId: ConsolidationFamilyId;
  reason: "contains_explicit_keep" | "mixed_membership";
  members: UnsafeFamilyMember[];
};

type VersionReembed = {
  episode: Episode;
  embeddingText: string;
  embeddingTextSha256: string;
};

export type ToxicEpisodePlan = {
  id: EpisodeId;
  reason: string;
  conditional: boolean;
  bodySha256: string | null;
  state: "would_archive" | "already_archived" | "unsafe";
  effectivelyVisibleBefore: boolean;
};

export type OutcomeFpSelfCheck = {
  fingerprint: string;
  latestUtcDay: string | null;
  requiredDecisionLineCount: number;
  passed: boolean;
  matchedEpisodeId: EpisodeId | null;
  rank: number | null;
  matchSource: "vector" | "exact_fp_rescue" | null;
  reason: string | null;
};

export type PunishmentSemanticNodeState = {
  id: string;
  found: boolean;
  label: string | null;
  description: string | null;
  sourceEpisodeIds: string[];
  archived: boolean | null;
  status: string | null;
  correctedBy: string | null;
  supersededAt: number | null;
  action: "report_only_no_safe_repository_primitive";
};

export type OutcomeCorpusMigrationReport = {
  dryRun: boolean;
  rawOutcomeEpisodeCount: number;
  scheduledOutcomeSourceCount: number;
  nonScheduledOutcomeEpisodeCount: number;
  visibleNonScheduledOutcomeEpisodeCount: number;
  nonScheduledOutcomeEpisodeIds: EpisodeId[];
  malformedScheduledOutcomeEpisodeIds: EpisodeId[];
  groupCount: number;
  multiMemberGroupCount: number;
  singletonGroupCount: number;
  largestGroupSize: number;
  projectedLiveOutcomeRecordCount: number;
  legacyFamiliesToDissolve: FamilyDissolution[];
  unsafeFamilies: UnsafeFamilyPlan[];
  groups: OutcomeRollupGroup[];
  rollupsCreated: Array<{ key: string; familyId: ConsolidationFamilyId; episodeId: EpisodeId }>;
  tokenBearingVersionCount: number;
  versionsToReembed: VersionReembed[];
  versionsReembedded: EpisodeId[];
  toxicEpisodes: ToxicEpisodePlan[];
  toxicEpisodesArchived: EpisodeId[];
  explicitKeep: {
    id: EpisodeId;
    found: boolean;
    bodySha256: string | null;
    currentVersion: boolean;
    state: "kept" | "unsafe";
  };
  semanticNode: PunishmentSemanticNodeState;
  unsafeItems: string[];
  acknowledgedMixedItems: string[];
  fpSelfChecks: OutcomeFpSelfCheck[];
  liveRolledOutcomeRecordCountAfter: number | null;
  liveOutcomeRecordCountAfter: number | null;
  expectedVisibleOutcomeEpisodeIds: EpisodeId[];
  actualVisibleOutcomeEpisodeIdsAfter: EpisodeId[] | null;
  missingVisibleOutcomeEpisodeIdsAfter: EpisodeId[];
  extraVisibleOutcomeEpisodeIdsAfter: EpisodeId[];
  toxicEpisodesInvisibleAfter: boolean | null;
  auditRowsWritten: number;
  customAuditRowsWritten: number;
  customAuditRowsWithNoReverser: number;
};

export type OutcomeCorpusMigrationDependencies = {
  db: SqliteDatabase;
  episodicRepository: EpisodicRepository;
  auditLog: Pick<AuditLog, "list" | "record">;
  embeddingClient?: EmbeddingClient;
  clock: Clock;
  runId: MaintenanceRunId;
  specification?: OutcomeCorpusSpecification;
  // Mixed families (targeted rollup members consolidated together with
  // unrelated episodes) are unsafe by default. An operator who has inspected
  // the dry-run report can acknowledge specific family ids for dissolution;
  // their unrelated members return to effective visibility and are reported.
  acknowledgedMixedFamilyIds?: readonly string[];
};

function sha256Text(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function episodeBodySha256(episode: Episode): string {
  return sha256Text(`${episode.title}\n${episode.narrative}`);
}

function isRawEpisode(episode: Episode): boolean {
  return (episode.episode_kind ?? "raw") === "raw";
}

function sameStringSet(left: readonly string[], right: readonly string[]): boolean {
  const leftSet = new Set(left);
  const rightSet = new Set(right);

  return leftSet.size === rightSet.size && [...leftSet].every((value) => rightSet.has(value));
}

function globalPattern(pattern: RegExp): RegExp {
  return new RegExp(
    pattern.source,
    pattern.flags.includes("g") ? pattern.flags : `${pattern.flags}g`,
  );
}

function matchesPattern(value: string, pattern: RegExp): boolean {
  return new RegExp(pattern.source, pattern.flags.replaceAll("g", "").replaceAll("y", "")).test(
    value,
  );
}

function outcomeFingerprints(
  lines: readonly string[],
  grammar: OutcomeCorpusGrammar = DEFAULT_OUTCOME_CORPUS_GRAMMAR,
): string[] {
  const fingerprints: string[] = [];

  for (const line of lines) {
    for (const match of line.matchAll(globalPattern(grammar.outcomeFingerprintPattern))) {
      const fingerprint = match[1];

      if (fingerprint !== undefined && !fingerprints.includes(fingerprint)) {
        fingerprints.push(fingerprint);
      }
    }
  }

  return fingerprints;
}

function outcomeRoles(
  lines: readonly string[],
  grammar: OutcomeCorpusGrammar = DEFAULT_OUTCOME_CORPUS_GRAMMAR,
): string[] {
  const roles: string[] = [];

  for (const line of lines) {
    for (const match of line.matchAll(globalPattern(grammar.outcomeRoleFieldPattern))) {
      const role = match[1];

      if (role !== undefined && !roles.includes(role)) {
        roles.push(role);
      }
    }
  }

  return roles;
}

function singleTokenValue(value: string): string {
  return /^\S+/u.exec(value)?.[0] ?? "";
}

function outcomeActionFields(line: string, grammar: OutcomeCorpusGrammar): OutcomeActionField[] {
  const matches = [...line.matchAll(globalPattern(grammar.outcomeActionFieldPattern))];

  return matches.map((match, index) => {
    const key = match[1] ?? "";
    const rawValue = line
      .slice((match.index ?? 0) + match[0].length, matches[index + 1]?.index ?? line.length)
      .trim();

    return {
      key,
      value: grammar.singleTokenOutcomeFields.has(key) ? singleTokenValue(rawValue) : rawValue,
    };
  });
}

export function parseOutcomeActionRecords(
  lines: readonly string[],
  grammar: OutcomeCorpusGrammar = DEFAULT_OUTCOME_CORPUS_GRAMMAR,
): OutcomeActionRecord[] {
  const records: OutcomeActionRecord[] = [];

  for (const line of lines) {
    let current: Partial<OutcomeActionRecord> = {};
    const flush = () => {
      if (current.action !== undefined && current.action.length > 0) {
        records.push(current as OutcomeActionRecord);
      }
      current = {};
    };

    for (const field of outcomeActionFields(line, grammar)) {
      switch (field.key) {
        case "decision":
          flush();
          current.decision = field.value;
          break;
        case "ticket":
          if (current.ticket !== undefined) {
            flush();
          }
          current.ticket = field.value;
          break;
        case "action":
          if (current.action !== undefined) {
            flush();
          }
          current.action = field.value;
          break;
        case "transition":
          current.transition = field.value;
          break;
        case "verdict":
          current.verdict = field.value;
          break;
        case "summary":
          current.summary = field.value;
          break;
        case "mr":
          current.mr = field.value;
          break;
        case "teams_card":
          current.teamsCard = field.value;
          break;
        case "card_count":
          current.cardCount = field.value;
          break;
      }
    }

    flush();
  }

  return records;
}

function cleanedProtocolDescription(value: string | undefined): string | null {
  if (value === undefined) {
    return null;
  }

  const collapsed = value
    .replace(/[\t\r\n ]+/gu, " ")
    .trim()
    .replace(/[.;:,]+$/u, "");

  if (collapsed.length === 0) {
    return null;
  }

  return collapsed.length <= 160 ? collapsed : `${collapsed.slice(0, 159)}…`;
}

function uniqueActions(
  sources: readonly ScheduledOutcomeSource[],
  action: string,
): OutcomeActionRecord[] {
  const unique = new Map<string, OutcomeActionRecord>();

  for (const source of sources) {
    for (const record of source.actions) {
      if (record.action !== action) {
        continue;
      }

      const key =
        action === "created"
          ? `${action}|${record.ticket ?? ""}`
          : action === "transition"
            ? `${action}|${record.ticket ?? ""}|${record.transition ?? ""}|${record.verdict ?? ""}`
            : action === "mr"
              ? `${action}|${record.ticket ?? ""}|${record.mr ?? ""}`
              : `${source.episode.id}|${action}|${record.ticket ?? ""}`;

      if (!unique.has(key)) {
        unique.set(key, record);
      }
    }
  }

  return [...unique.values()];
}

function actionClauses(sources: readonly ScheduledOutcomeSource[]): string[] {
  const created = uniqueActions(sources, "created").filter((record) => record.ticket !== undefined);
  const transitioned = uniqueActions(sources, "transition").filter(
    (record) => record.ticket !== undefined && record.transition !== undefined,
  );
  const mergeRequests = uniqueActions(sources, "mr").filter(
    (record) => record.ticket !== undefined,
  );
  const teamsCards = sources.filter((source) =>
    source.actions.some((record) => record.action === "teams_card"),
  ).length;
  const knownActions = new Set(["created", "transition", "mr", "teams_card"]);
  const other = sources.flatMap((source) =>
    source.actions.filter((record) => !knownActions.has(record.action)),
  );
  const clauses: string[] = [];

  if (created.length > 0) {
    clauses.push(
      `created ${created
        .map((record) => {
          const summary = cleanedProtocolDescription(record.summary);
          return summary === null ? record.ticket! : `${record.ticket!} (${summary})`;
        })
        .join("; ")}`,
    );
  }

  if (transitioned.length > 0) {
    clauses.push(
      `transitioned ${transitioned
        .map((record) => {
          const verdict = cleanedProtocolDescription(record.verdict);
          const transition = cleanedProtocolDescription(record.transition) ?? record.transition!;
          return verdict === null
            ? `${record.ticket!} to ${transition}`
            : `${record.ticket!} to ${transition} (${verdict})`;
        })
        .join("; ")}`,
    );
  }

  if (mergeRequests.length > 0) {
    clauses.push(
      `opened merge requests for ${mergeRequests
        .map((record) =>
          record.mr === undefined ? record.ticket! : `${record.ticket!} (${record.mr})`,
        )
        .join("; ")}`,
    );
  }

  if (teamsCards > 0) {
    clauses.push(`posted ${teamsCards} Teams card notification${teamsCards === 1 ? "" : "s"}`);
  }

  if (other.length > 0) {
    clauses.push(
      `recorded ${other
        .map((record) =>
          record.ticket === undefined
            ? `action ${record.action}`
            : `${record.action} for ${record.ticket}`,
        )
        .join("; ")}`,
    );
  }

  return clauses;
}

export function renderOutcomeRollup(input: {
  role: string;
  utcDay: string;
  sources: readonly ScheduledOutcomeSource[];
}): OutcomeRollupRender {
  const sources = [...input.sources].sort(
    (left, right) =>
      left.episode.start_time - right.episode.start_time ||
      left.episode.id.localeCompare(right.episode.id),
  );
  const clauses = actionClauses(sources);
  const sentences: string[] = [];

  if (clauses.length === 0) {
    sentences.push(
      `On ${input.utcDay}, ${input.role} recorded ${sources.length} autonomous outcome${sources.length === 1 ? "" : "s"}.`,
    );
  } else {
    const renderedClauses = clauses.slice(0, 3);

    if (clauses.length > 3) {
      renderedClauses[2] = `${renderedClauses[2]}; and ${clauses.slice(3).join("; and ")}`;
    }

    for (const [index, clause] of renderedClauses.entries()) {
      sentences.push(
        index === 0
          ? `On ${input.utcDay}, ${input.role} ${clause}.`
          : `The ${input.role} role ${clause}.`,
      );
    }
  }

  const protectedLines = collectProtectedEpisodeTokenLines(
    sources.map((source) => source.episode.narrative),
  );

  if (sentences.length < 2) {
    sentences.push(
      `This daily rollup covers ${sources.length} source episode${sources.length === 1 ? "" : "s"} and preserves ${protectedLines.length} distinct protocol record${protectedLines.length === 1 ? "" : "s"}.`,
    );
  }

  const prose = sentences.join(" ");

  return {
    title: `OUTCOME rollup: ${input.role} on ${input.utcDay}`,
    prose,
    protectedLines,
    narrative: `${prose}\n${protectedLines.join("\n")}`,
  };
}

function maxTier(stats: readonly EpisodeStats[]): EpisodeTier {
  return stats.reduce<EpisodeTier>(
    (best, current) => (TIER_RANK[current.tier] > TIER_RANK[best] ? current.tier : best),
    "T1",
  );
}

function coverageHash(sourceStreamIds: readonly string[]): string {
  return buildConsolidationCoverageHash([
    ...sourceStreamIds,
    `consolidation_policy_version:${CONSOLIDATION_POLICY_VERSION}`,
  ]);
}

function buildRollupEpisode(
  group: OutcomeRollupGroup,
  embedding: Float32Array,
  nowMs: number,
): Episode {
  const rawEpisodes = group.sources.map((source) => source.episode);
  const locations = uniqueStrings(
    rawEpisodes.flatMap((episode) => (episode.location === null ? [] : [episode.location.trim()])),
  ).filter((location) => location.length > 0);
  const disclosure = combineMemoryDisclosureLabels(
    rawEpisodes.map(memoryDisclosureLabelFromEpisodeAccess),
  );

  return normalizeEpisodeAccess({
    id: group.plannedVersionId ?? createEpisodeId(),
    title: group.title,
    narrative: group.narrative,
    participants: group.participants,
    location: locations.length === 1 ? (locations[0] ?? null) : null,
    start_time: Math.min(...rawEpisodes.map((episode) => episode.start_time)),
    end_time: Math.max(...rawEpisodes.map((episode) => episode.end_time)),
    source_stream_ids: group.sourceStreamIds,
    significance: Math.max(...rawEpisodes.map((episode) => episode.significance)),
    tags: group.tags,
    confidence: Math.min(...rawEpisodes.map((episode) => episode.confidence)),
    lineage: {
      derived_from: rawEpisodes.map((episode) => episode.id),
      supersedes: rawEpisodes.map((episode) => episode.id),
    },
    emotional_arc:
      rawEpisodes.find((episode) => episode.emotional_arc !== null)?.emotional_arc ?? null,
    origin_audience_entity_ids: [...disclosure.originAudienceEntityIds],
    shared: disclosure.disclosureClass === "public",
    episode_kind: "consolidation_version",
    consolidation_family_id: group.plannedFamilyId ?? createConsolidationFamilyId(),
    consolidation_coverage_hash: group.coverageHash,
    embedding,
    created_at: nowMs,
    updated_at: nowMs,
  });
}

function episodeForAudit(episode: Episode): Record<string, unknown> {
  return {
    ...episode,
    embedding: Array.from(episode.embedding),
  };
}

function statsPatchForRestore(stats: EpisodeStats): Omit<EpisodeStats, "episode_id" | "archived"> {
  const { episode_id: _episodeId, archived: _archived, ...patch } = stats;
  return patch;
}

function parseStoredStringArray(value: unknown): string[] {
  if (typeof value !== "string") {
    return [];
  }

  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === "string")
      : [];
  } catch {
    return [];
  }
}

function readPunishmentSemanticNode(
  db: SqliteDatabase,
  punishmentSemanticNodeId: SemanticNodeId,
): PunishmentSemanticNodeState {
  const row = db
    .prepare(
      `
        SELECT id, label, description, source_episode_ids, archived, status, corrected_by,
               superseded_at
        FROM semantic_nodes
        WHERE id = ?
      `,
    )
    .get(punishmentSemanticNodeId) as Record<string, unknown> | undefined;

  if (row === undefined) {
    return {
      id: punishmentSemanticNodeId,
      found: false,
      label: null,
      description: null,
      sourceEpisodeIds: [],
      archived: null,
      status: null,
      correctedBy: null,
      supersededAt: null,
      action: "report_only_no_safe_repository_primitive",
    };
  }

  return {
    id: punishmentSemanticNodeId,
    found: true,
    label: row.label === null || row.label === undefined ? null : String(row.label),
    description:
      row.description === null || row.description === undefined ? null : String(row.description),
    sourceEpisodeIds: parseStoredStringArray(row.source_episode_ids),
    archived: row.archived === true || Number(row.archived) === 1,
    status: row.status === null || row.status === undefined ? null : String(row.status),
    correctedBy:
      row.corrected_by === null || row.corrected_by === undefined ? null : String(row.corrected_by),
    supersededAt:
      row.superseded_at === null || row.superseded_at === undefined
        ? null
        : Number(row.superseded_at),
    action: "report_only_no_safe_repository_primitive",
  };
}

type RollupAuditFact = {
  familyId: ConsolidationFamilyId;
  versionEpisodeId: EpisodeId;
  groupKey: string | null;
  coverageHash: string | null;
  sourceEpisodeIds: string[];
};

type RollupAuditScan = {
  factsByArtifact: Map<string, RollupAuditFact>;
  duplicateArtifactKeys: Set<string>;
  malformedAuditIds: number[];
};

function rollupArtifactKey(familyId: ConsolidationFamilyId, versionEpisodeId: EpisodeId): string {
  return `${familyId}\u0000${versionEpisodeId}`;
}

function activeRollupAuditFacts(auditLog: Pick<AuditLog, "list">): RollupAuditScan {
  const factsByArtifact = new Map<string, RollupAuditFact>();
  const duplicateArtifactKeys = new Set<string>();
  const malformedAuditIds: number[] = [];

  for (const audit of auditLog.list({ process: "consolidator", reverted: false })) {
    if (audit.action !== OUTCOME_ROLLUP_AUDIT_ACTION) {
      continue;
    }

    if (audit.targets.migration_version !== OUTCOME_ROLLUP_SCRIPT_VERSION) {
      malformedAuditIds.push(audit.id);
      continue;
    }

    const familyId = audit.targets.family_id;
    const versionEpisodeId = audit.targets.version_episode_id;

    if (typeof familyId !== "string" || typeof versionEpisodeId !== "string") {
      malformedAuditIds.push(audit.id);
      continue;
    }

    try {
      const parsedFamilyId = parseConsolidationFamilyId(familyId);
      const parsedVersionEpisodeId = parseEpisodeId(versionEpisodeId);
      const artifactKey = rollupArtifactKey(parsedFamilyId, parsedVersionEpisodeId);

      if (factsByArtifact.has(artifactKey)) {
        duplicateArtifactKeys.add(artifactKey);
      }

      factsByArtifact.set(artifactKey, {
        familyId: parsedFamilyId,
        versionEpisodeId: parsedVersionEpisodeId,
        groupKey: typeof audit.targets.group_key === "string" ? audit.targets.group_key : null,
        coverageHash:
          typeof audit.targets.coverage_hash === "string" ? audit.targets.coverage_hash : null,
        sourceEpisodeIds: Array.isArray(audit.targets.source_episode_ids)
          ? audit.targets.source_episode_ids.filter(
              (episodeId): episodeId is string => typeof episodeId === "string",
            )
          : [],
      });
    } catch {
      malformedAuditIds.push(audit.id);
    }
  }

  return { factsByArtifact, duplicateArtifactKeys, malformedAuditIds };
}

function activeReembedAuditHashes(auditLog: Pick<AuditLog, "list">): Map<EpisodeId, string> {
  const hashes = new Map<EpisodeId, string>();

  for (const audit of auditLog.list({ process: "consolidator", reverted: false })) {
    if (audit.action !== OUTCOME_VERSION_REEMBED_AUDIT_ACTION) {
      continue;
    }

    const episodeId = audit.targets.episode_id;
    const embeddingTextSha256 = audit.targets.embedding_text_sha256;

    if (typeof episodeId !== "string" || typeof embeddingTextSha256 !== "string") {
      continue;
    }

    try {
      hashes.set(parseEpisodeId(episodeId), embeddingTextSha256);
    } catch {
      // Malformed historical audit rows cannot mark a current episode complete.
    }
  }

  return hashes;
}

async function buildMigrationReport(
  dependencies: OutcomeCorpusMigrationDependencies,
): Promise<OutcomeCorpusMigrationReport> {
  const specification = dependencies.specification ?? DEFAULT_OUTCOME_CORPUS_SPECIFICATION;
  const grammar = specification.grammar;
  const episodes = await dependencies.episodicRepository.listAll();
  const episodesById = new Map(episodes.map((episode) => [episode.id, episode]));
  const statsById = new Map(
    dependencies.episodicRepository.listStats().map((stats) => [stats.episode_id, stats]),
  );
  const families = dependencies.episodicRepository.listConsolidationFamilies();
  const members = dependencies.episodicRepository.listConsolidationMembers();
  const membersByFamilyId = new Map<ConsolidationFamilyId, ConsolidationMemberRecord[]>();
  const unsafeItems: string[] = [];

  for (const member of members) {
    const familyMembers = membersByFamilyId.get(member.family_id) ?? [];
    familyMembers.push(member);
    membersByFamilyId.set(member.family_id, familyMembers);
  }

  const rawOutcomeEpisodes: Episode[] = [];
  const scheduledSources: ScheduledOutcomeSource[] = [];
  const nonScheduledOutcomeEpisodes: Episode[] = [];
  const malformedScheduledOutcomeEpisodeIds: EpisodeId[] = [];

  for (const episode of episodes.filter(isRawEpisode)) {
    const protectedLines = collectProtectedEpisodeTokenLines([episode.narrative]);
    const fingerprints = outcomeFingerprints(protectedLines, grammar);

    if (fingerprints.length === 0) {
      continue;
    }

    rawOutcomeEpisodes.push(episode);
    const roles = outcomeRoles(protectedLines, grammar);

    // The corpus also contains role-bearing alert headers whose fp is a run
    // fingerprint rather than scheduled:<role>:<tenant>. The role field, not
    // the fp spelling, is the authoritative B1 grouping handle.
    if (roles.length === 0) {
      if (
        fingerprints.some((fingerprint) =>
          matchesPattern(fingerprint, grammar.scheduledFingerprintPattern),
        )
      ) {
        malformedScheduledOutcomeEpisodeIds.push(episode.id);
        unsafeItems.push(
          `scheduled-grammar OUTCOME episode ${episode.id} has no role field and is excluded from migration`,
        );
        continue;
      }

      nonScheduledOutcomeEpisodes.push(episode);
      continue;
    }

    const stats = statsById.get(episode.id);

    if (roles.length !== 1) {
      unsafeItems.push(
        `scheduled OUTCOME episode ${episode.id} has ${roles.length} distinct role fields`,
      );
      continue;
    }

    if (stats === undefined) {
      unsafeItems.push(`scheduled OUTCOME episode ${episode.id} has no episode_stats row`);
      continue;
    }

    if (stats.archived) {
      unsafeItems.push(`scheduled OUTCOME episode ${episode.id} is already archived`);
      continue;
    }

    scheduledSources.push({
      episode,
      stats,
      role: roles[0]!,
      utcDay: utcDayKey(episode.start_time),
      fingerprints,
      protectedLines,
      actions: parseOutcomeActionRecords(protectedLines, grammar),
    });
  }

  const sourcesByGroupKey = new Map<string, ScheduledOutcomeSource[]>();

  for (const source of scheduledSources) {
    const key = `${source.role}/${source.utcDay}`;
    const groupSources = sourcesByGroupKey.get(key) ?? [];
    groupSources.push(source);
    sourcesByGroupKey.set(key, groupSources);
  }

  const baseGroups = [...sourcesByGroupKey.entries()]
    .map(([key, sources]) => ({
      key,
      sources: sources.sort((left, right) => left.episode.start_time - right.episode.start_time),
    }))
    .sort((left, right) => left.key.localeCompare(right.key));
  const groupBySourceId = new Map<EpisodeId, (typeof baseGroups)[number]>();

  for (const group of baseGroups) {
    for (const source of group.sources) {
      groupBySourceId.set(source.episode.id, group);
    }
  }

  const familyById = new Map(families.map((family) => [family.family_id, family]));
  const rollupAuditScan = activeRollupAuditFacts(dependencies.auditLog);
  const rollupAuditFacts = rollupAuditScan.factsByArtifact;

  for (const auditId of rollupAuditScan.malformedAuditIds) {
    unsafeItems.push(
      `migration rollup audit row ${auditId} has malformed or unsupported completion metadata; unsafe partial state detected -- ${PARTIAL_STATE_RECOVERY_MESSAGE}`,
    );
  }

  for (const artifactKey of rollupAuditScan.duplicateArtifactKeys) {
    const fact = rollupAuditFacts.get(artifactKey);
    unsafeItems.push(
      `rollup family ${fact?.familyId ?? "unknown"} version ${fact?.versionEpisodeId ?? "unknown"} has duplicate active migration audit rows; unsafe partial state detected -- ${PARTIAL_STATE_RECOVERY_MESSAGE}`,
    );
  }

  for (const episode of episodes.filter((item) => item.tags.includes(OUTCOME_ROLLUP_MARKER_TAG))) {
    const family =
      episode.consolidation_family_id === null || episode.consolidation_family_id === undefined
        ? undefined
        : familyById.get(episode.consolidation_family_id);
    const isCurrent = family?.current_version_episode_id === episode.id;

    if (!isCurrent) {
      unsafeItems.push(
        `rollup marker version ${episode.id} has no matching current family; unsafe partial cross-store state detected -- ${PARTIAL_STATE_RECOVERY_MESSAGE}`,
      );
      continue;
    }

    if (rollupAuditFacts.get(rollupArtifactKey(family.family_id, episode.id)) === undefined) {
      unsafeItems.push(
        `rollup family ${family.family_id} version ${episode.id} has no corresponding migration audit row; unsafe partial cross-store state detected -- ${PARTIAL_STATE_RECOVERY_MESSAGE}`,
      );
    }
  }

  for (const fact of rollupAuditFacts.values()) {
    const family = familyById.get(fact.familyId);
    const version = episodesById.get(fact.versionEpisodeId);

    if (
      family?.current_version_episode_id !== fact.versionEpisodeId ||
      version?.tags.includes(OUTCOME_ROLLUP_MARKER_TAG) !== true
    ) {
      unsafeItems.push(
        `migration audit for rollup family ${fact.familyId} version ${fact.versionEpisodeId} has no matching current rollup artifact; unsafe partial cross-store state detected -- ${PARTIAL_STATE_RECOVERY_MESSAGE}`,
      );
    }
  }

  const markerFamilies = new Map<string, ConsolidationFamilyRecord[]>();
  const invalidMarkerFamilyIds = new Set<ConsolidationFamilyId>();

  for (const family of families) {
    const familyMembers = membersByFamilyId.get(family.family_id) ?? [];
    const targetMemberIds = familyMembers
      .map((member) => member.raw_episode_id)
      .filter((episodeId) => groupBySourceId.has(episodeId));

    if (targetMemberIds.length === 0) {
      continue;
    }

    const currentVersion = episodesById.get(family.current_version_episode_id);

    if (currentVersion?.tags.includes(OUTCOME_ROLLUP_MARKER_TAG) !== true) {
      continue;
    }

    const auditFact = rollupAuditFacts.get(rollupArtifactKey(family.family_id, currentVersion.id));

    if (auditFact === undefined) {
      invalidMarkerFamilyIds.add(family.family_id);
      continue;
    }

    const memberGroups = uniqueStrings(
      targetMemberIds.map((episodeId) => groupBySourceId.get(episodeId)!.key),
    );

    if (memberGroups.length !== 1) {
      invalidMarkerFamilyIds.add(family.family_id);
      unsafeItems.push(
        `rollup marker family ${family.family_id} spans ${memberGroups.length} role/day groups`,
      );
      continue;
    }

    const group = sourcesByGroupKey.get(memberGroups[0]!) ?? [];
    const expectedIds = group.map((source) => source.episode.id);

    if (
      !sameStringSet(
        familyMembers.map((member) => member.raw_episode_id),
        expectedIds,
      )
    ) {
      invalidMarkerFamilyIds.add(family.family_id);
      unsafeItems.push(`rollup marker family ${family.family_id} does not exactly cover its group`);
      continue;
    }

    if (
      auditFact.groupKey !== memberGroups[0] ||
      auditFact.coverageHash !== family.coverage_hash ||
      !sameStringSet(auditFact.sourceEpisodeIds, expectedIds)
    ) {
      invalidMarkerFamilyIds.add(family.family_id);
      unsafeItems.push(
        `rollup family ${family.family_id} migration audit row does not match its current group/version payload`,
      );
      continue;
    }

    const list = markerFamilies.get(memberGroups[0]!) ?? [];
    list.push(family);
    markerFamilies.set(memberGroups[0]!, list);
  }

  const groups: OutcomeRollupGroup[] = baseGroups.map(({ key, sources }) => {
    const role = sources[0]!.role;
    const utcDay = sources[0]!.utcDay;
    const rendered = renderOutcomeRollup({ role, utcDay, sources });
    const rawEpisodes = sources.map((source) => source.episode);
    const tags = uniqueStrings([
      ...rawEpisodes.flatMap((episode) => episode.tags.map((tag) => tag.trim())),
      "outcome-rollup",
      OUTCOME_ROLLUP_MARKER_TAG,
    ]).filter((tag) => tag.length > 0);
    const participants = uniqueStrings(
      rawEpisodes.flatMap((episode) =>
        episode.participants.map((participant) => participant.trim()),
      ),
    ).filter((participant) => participant.length > 0);
    const sourceStreamIds = uniqueStrings(
      rawEpisodes.flatMap((episode) => episode.source_stream_ids),
    ) as Episode["source_stream_ids"];
    const groupMarkerFamilies = markerFamilies.get(key) ?? [];

    if (groupMarkerFamilies.length > 1) {
      unsafeItems.push(`role/day group ${key} has multiple rollup marker families`);
    }

    const existingFamily = groupMarkerFamilies[0] ?? null;
    const computedCoverageHash = coverageHash(sourceStreamIds);

    if (existingFamily !== null) {
      const currentVersion = episodesById.get(existingFamily.current_version_episode_id);

      if (
        existingFamily.coverage_hash !== computedCoverageHash ||
        existingFamily.policy_version !== CONSOLIDATION_POLICY_VERSION ||
        currentVersion?.title !== rendered.title ||
        currentVersion.narrative !== rendered.narrative ||
        currentVersion.consolidation_coverage_hash !== computedCoverageHash
      ) {
        unsafeItems.push(
          `existing rollup family ${existingFamily.family_id} failed deterministic validation`,
        );
      }
    }

    return {
      ...rendered,
      key,
      role,
      utcDay,
      sources,
      tags,
      participants,
      sourceStreamIds,
      coverageHash: computedCoverageHash,
      embeddingText: buildConsolidationEpisodeEmbeddingText({
        title: rendered.title,
        synthesizedNarrative: rendered.prose,
        protectedSourceTexts: rawEpisodes.map((episode) => episode.narrative),
        tags,
        participants,
      }),
      existingFamily,
      plannedFamilyId:
        sources.length > 1 && existingFamily === null ? createConsolidationFamilyId() : null,
      plannedVersionId: sources.length > 1 && existingFamily === null ? createEpisodeId() : null,
    };
  });

  const targetSourceIds = new Set(scheduledSources.map((source) => source.episode.id));
  const existingRollupFamilyIds = new Set(
    groups.flatMap((group) =>
      group.existingFamily === null ? [] : [group.existingFamily.family_id],
    ),
  );
  const legacyFamiliesToDissolve: FamilyDissolution[] = [];
  const unsafeFamilies: UnsafeFamilyPlan[] = [];
  const acknowledgedMixedFamilyIds = new Set(dependencies.acknowledgedMixedFamilyIds ?? []);
  const consumedAcknowledgedMixedFamilyIds = new Set<string>();
  const acknowledgedMixedItems: string[] = [];
  const toxicEpisodeIds = new Set(specification.toxicEpisodeSpecs.map((spec) => spec.id));

  const describeMemberIntent = (episodeId: EpisodeId): string => {
    const group = groupBySourceId.get(episodeId);

    if (group !== undefined) {
      return `roll_up:${group.key}`;
    }

    if (episodeId === specification.explicitKeepEpisodeId) {
      return "explicit_keep";
    }

    if (toxicEpisodeIds.has(episodeId)) {
      return "archive_toxic";
    }

    return "preserve_unrelated";
  };

  for (const family of families) {
    const familyMembers = membersByFamilyId.get(family.family_id) ?? [];
    const targetedMembers = familyMembers.filter((member) =>
      targetSourceIds.has(member.raw_episode_id),
    );

    if (targetedMembers.length === 0) {
      continue;
    }

    const familyContainsExplicitKeep =
      family.current_version_episode_id === specification.explicitKeepEpisodeId ||
      familyMembers.some((member) => member.raw_episode_id === specification.explicitKeepEpisodeId);
    const memberPlans = familyMembers.map((member) => ({
      episodeId: member.raw_episode_id,
      intendedState: describeMemberIntent(member.raw_episode_id),
    }));

    if (familyContainsExplicitKeep) {
      unsafeFamilies.push({
        familyId: family.family_id,
        reason: "contains_explicit_keep",
        members: memberPlans,
      });
      unsafeItems.push(
        `family ${family.family_id} contains explicit-keep episode ${specification.explicitKeepEpisodeId} and ${targetedMembers.length} scheduled rollup member(s); excluded from dissolution`,
      );
      continue;
    }

    if (targetedMembers.length !== familyMembers.length) {
      if (acknowledgedMixedFamilyIds.has(family.family_id)) {
        consumedAcknowledgedMixedFamilyIds.add(family.family_id);
        const releasedMembers = familyMembers.filter(
          (member) => !targetSourceIds.has(member.raw_episode_id),
        );
        acknowledgedMixedItems.push(
          `acknowledged mixed family ${family.family_id}: dissolving; ${releasedMembers.length} unrelated member(s) return to effective visibility: ${releasedMembers.map((member) => member.raw_episode_id).join(", ")}`,
        );
      } else {
        unsafeFamilies.push({
          familyId: family.family_id,
          reason: "mixed_membership",
          members: memberPlans,
        });
        unsafeItems.push(
          `mixed family ${family.family_id} contains ${targetedMembers.length} targeted and ${familyMembers.length - targetedMembers.length} unrelated member(s); excluded from dissolution (acknowledge with --dissolve-mixed-family ${family.family_id})`,
        );
        continue;
      }
    }

    if (
      existingRollupFamilyIds.has(family.family_id) ||
      invalidMarkerFamilyIds.has(family.family_id)
    ) {
      continue;
    }

    const currentVersion = episodesById.get(family.current_version_episode_id);
    const currentStats = statsById.get(family.current_version_episode_id);

    if (currentVersion === undefined || currentStats === undefined) {
      unsafeItems.push(
        `legacy OUTCOME family ${family.family_id} has a missing current version or stats row`,
      );
      continue;
    }

    if (
      currentVersion.episode_kind !== "consolidation_version" ||
      currentVersion.consolidation_family_id !== family.family_id ||
      currentStats.archived
    ) {
      unsafeItems.push(`legacy OUTCOME family ${family.family_id} has an invalid current version`);
      continue;
    }

    legacyFamiliesToDissolve.push({
      family,
      members: familyMembers,
      currentVersion,
      currentStats,
    });
  }

  // Typo protection: an acknowledged id that never matched a mixed family is
  // operator error (wrong id or wrong bank) — fail safe instead of ignoring.
  for (const familyId of acknowledgedMixedFamilyIds) {
    if (!consumedAcknowledgedMixedFamilyIds.has(familyId)) {
      unsafeItems.push(
        `--dissolve-mixed-family ${familyId} did not match any mixed family in this bank`,
      );
    }
  }

  const currentVersionIds = new Set(families.map((family) => family.current_version_episode_id));
  const legacyFamilyIds = new Set(
    legacyFamiliesToDissolve.map((dissolution) => dissolution.family.family_id),
  );
  const tokenBearingVersions = episodes.filter(
    (episode) =>
      episode.episode_kind === "consolidation_version" &&
      outcomeFingerprints(collectProtectedEpisodeTokenLines([episode.narrative]), grammar).length >
        0,
  );
  const completedReembedHashes = activeReembedAuditHashes(dependencies.auditLog);
  const versionsToReembed: VersionReembed[] = [];

  for (const episode of tokenBearingVersions) {
    if (
      !currentVersionIds.has(episode.id) ||
      episode.tags.includes(OUTCOME_ROLLUP_MARKER_TAG) ||
      (episode.consolidation_family_id !== null &&
        episode.consolidation_family_id !== undefined &&
        legacyFamilyIds.has(episode.consolidation_family_id))
    ) {
      continue;
    }

    const embeddingText = buildConsolidationEpisodeEmbeddingText({
      title: episode.title,
      synthesizedNarrative: episode.narrative,
      protectedSourceTexts: [episode.narrative],
      tags: episode.tags,
      participants: episode.participants,
    });
    const embeddingTextSha256 = sha256Text(embeddingText);

    if (completedReembedHashes.get(episode.id) === embeddingTextSha256) {
      continue;
    }

    versionsToReembed.push({ episode, embeddingText, embeddingTextSha256 });
  }

  const currentFamilyVersionIds = new Set(
    families.map((family) => family.current_version_episode_id),
  );
  const toxicEpisodes: ToxicEpisodePlan[] = [];

  for (const spec of specification.toxicEpisodeSpecs) {
    const episode = episodesById.get(spec.id);
    const stats = statsById.get(spec.id);

    if (episode === undefined || stats === undefined) {
      unsafeItems.push(`toxic target ${spec.id} is missing its episode or stats row`);
      toxicEpisodes.push({
        id: spec.id,
        reason: spec.reason,
        conditional: spec.conditional,
        bodySha256: episode === undefined ? null : episodeBodySha256(episode),
        state: "unsafe",
        effectivelyVisibleBefore: false,
      });
      continue;
    }

    const bodySha256 = episodeBodySha256(episode);
    const kind = episode.episode_kind ?? "raw";
    const currentVersion = currentFamilyVersionIds.has(episode.id);
    const effectivelyVisibleBefore = dependencies.episodicRepository.isEpisodeEffectivelyVisible(
      episode.id,
    );

    if (bodySha256 !== spec.expectedBodySha256 || kind !== spec.expectedKind || currentVersion) {
      unsafeItems.push(
        `toxic target ${spec.id} failed content, kind, or non-current-version validation`,
      );
      toxicEpisodes.push({
        id: spec.id,
        reason: spec.reason,
        conditional: spec.conditional,
        bodySha256,
        state: "unsafe",
        effectivelyVisibleBefore,
      });
      continue;
    }

    toxicEpisodes.push({
      id: spec.id,
      reason: spec.reason,
      conditional: spec.conditional,
      bodySha256,
      state: stats.archived ? "already_archived" : "would_archive",
      effectivelyVisibleBefore,
    });
  }

  const keepEpisode = episodesById.get(specification.explicitKeepEpisodeId);
  const keepBodySha256 = keepEpisode === undefined ? null : episodeBodySha256(keepEpisode);
  const keepCurrent = currentFamilyVersionIds.has(specification.explicitKeepEpisodeId);
  const keepSafe =
    keepEpisode !== undefined &&
    keepBodySha256 === specification.explicitKeepBodySha256 &&
    keepCurrent &&
    statsById.get(specification.explicitKeepEpisodeId)?.archived === false;

  if (!keepSafe) {
    unsafeItems.push(
      `explicit keep episode ${specification.explicitKeepEpisodeId} failed validation`,
    );
  }

  const semanticNode = readPunishmentSemanticNode(
    dependencies.db,
    specification.punishmentSemanticNodeId,
  );
  const visibleNonScheduledOutcomeEpisodes = nonScheduledOutcomeEpisodes.filter((episode) =>
    dependencies.episodicRepository.isEpisodeEffectivelyVisible(episode.id),
  );

  if (!semanticNode.found) {
    unsafeItems.push(
      `punishment semantic node ${specification.punishmentSemanticNodeId} is missing`,
    );
  }

  const dissolvedCurrentVersionIds = new Set(
    legacyFamiliesToDissolve.map((item) => item.currentVersion.id),
  );
  const expectedVisibleOutcomeEpisodeIds = new Set<EpisodeId>();

  for (const episode of episodes) {
    if (
      !dependencies.episodicRepository.isEpisodeEffectivelyVisible(episode.id) ||
      outcomeFingerprints(collectProtectedEpisodeTokenLines([episode.narrative]), grammar)
        .length === 0 ||
      targetSourceIds.has(episode.id) ||
      dissolvedCurrentVersionIds.has(episode.id) ||
      toxicEpisodeIds.has(episode.id)
    ) {
      continue;
    }

    expectedVisibleOutcomeEpisodeIds.add(episode.id);
  }

  for (const group of groups) {
    const representativeId =
      group.sources.length === 1
        ? group.sources[0]!.episode.id
        : (group.existingFamily?.current_version_episode_id ?? group.plannedVersionId);

    if (representativeId !== null && !toxicEpisodeIds.has(representativeId)) {
      expectedVisibleOutcomeEpisodeIds.add(representativeId);
    }
  }

  const sortedExpectedVisibleOutcomeEpisodeIds = [...expectedVisibleOutcomeEpisodeIds].sort();

  return {
    dryRun: true,
    rawOutcomeEpisodeCount: rawOutcomeEpisodes.length,
    scheduledOutcomeSourceCount: scheduledSources.length,
    nonScheduledOutcomeEpisodeCount: nonScheduledOutcomeEpisodes.length,
    visibleNonScheduledOutcomeEpisodeCount: visibleNonScheduledOutcomeEpisodes.length,
    nonScheduledOutcomeEpisodeIds: nonScheduledOutcomeEpisodes.map((episode) => episode.id),
    malformedScheduledOutcomeEpisodeIds,
    groupCount: groups.length,
    multiMemberGroupCount: groups.filter((group) => group.sources.length > 1).length,
    singletonGroupCount: groups.filter((group) => group.sources.length === 1).length,
    largestGroupSize: Math.max(0, ...groups.map((group) => group.sources.length)),
    projectedLiveOutcomeRecordCount: sortedExpectedVisibleOutcomeEpisodeIds.length,
    legacyFamiliesToDissolve,
    unsafeFamilies,
    groups,
    rollupsCreated: [],
    tokenBearingVersionCount: tokenBearingVersions.length,
    versionsToReembed,
    versionsReembedded: [],
    toxicEpisodes,
    toxicEpisodesArchived: [],
    explicitKeep: {
      id: specification.explicitKeepEpisodeId,
      found: keepEpisode !== undefined,
      bodySha256: keepBodySha256,
      currentVersion: keepCurrent,
      state: keepSafe ? "kept" : "unsafe",
    },
    semanticNode,
    unsafeItems,
    acknowledgedMixedItems,
    fpSelfChecks: [],
    liveRolledOutcomeRecordCountAfter: null,
    liveOutcomeRecordCountAfter: null,
    expectedVisibleOutcomeEpisodeIds: sortedExpectedVisibleOutcomeEpisodeIds,
    actualVisibleOutcomeEpisodeIdsAfter: null,
    missingVisibleOutcomeEpisodeIdsAfter: [],
    extraVisibleOutcomeEpisodeIdsAfter: [],
    toxicEpisodesInvisibleAfter: null,
    auditRowsWritten: 0,
    customAuditRowsWritten: 0,
    customAuditRowsWithNoReverser: 0,
  };
}

async function restoreDissolvedFamily(
  dependencies: OutcomeCorpusMigrationDependencies,
  dissolution: FamilyDissolution,
): Promise<void> {
  await dependencies.episodicRepository.createEpisode(dissolution.currentVersion);
  dependencies.episodicRepository.updateStats(
    dissolution.currentVersion.id,
    statsPatchForRestore(dissolution.currentStats),
  );
  const existingFamily = dependencies.episodicRepository.getConsolidationFamily(
    dissolution.family.family_id,
  );

  if (existingFamily === null) {
    dependencies.episodicRepository.createConsolidationFamily({
      familyId: dissolution.family.family_id,
      currentVersionEpisodeId: dissolution.family.current_version_episode_id,
      coverageHash: dissolution.family.coverage_hash,
      policyVersion: dissolution.family.policy_version,
      members: dissolution.members,
    });
    dependencies.db
      .prepare(
        `
          UPDATE consolidation_families
          SET created_at = ?, updated_at = ?
          WHERE family_id = ?
        `,
      )
      .run(
        dissolution.family.created_at,
        dissolution.family.updated_at,
        dissolution.family.family_id,
      );
  } else if (
    existingFamily.current_version_episode_id !== dissolution.family.current_version_episode_id ||
    existingFamily.coverage_hash !== dissolution.family.coverage_hash ||
    existingFamily.policy_version !== dissolution.family.policy_version
  ) {
    throw new Error(
      `Family ${dissolution.family.family_id} changed while attempting payload restoration`,
    );
  }

  const restoredEpisode = await dependencies.episodicRepository.get(dissolution.currentVersion.id, {
    includeArchived: true,
  });
  const restoredStats = dependencies.episodicRepository.getStats(dissolution.currentVersion.id);
  const restoredMembers = dependencies.episodicRepository.listConsolidationMembers(
    dissolution.family.family_id,
  );
  const restoredFamily = dependencies.episodicRepository.getConsolidationFamily(
    dissolution.family.family_id,
  );

  if (
    restoredEpisode === null ||
    JSON.stringify(episodeForAudit(restoredEpisode)) !==
      JSON.stringify(episodeForAudit(dissolution.currentVersion)) ||
    restoredStats === null ||
    JSON.stringify(restoredStats) !== JSON.stringify(dissolution.currentStats) ||
    restoredFamily === null ||
    JSON.stringify(restoredFamily) !== JSON.stringify(dissolution.family) ||
    JSON.stringify(restoredMembers) !== JSON.stringify(dissolution.members)
  ) {
    throw new Error(`Payload restoration validation failed for ${dissolution.family.family_id}`);
  }
}

async function abortAfterDissolutionFailure(
  dependencies: OutcomeCorpusMigrationDependencies,
  dissolution: FamilyDissolution,
  stage: "cross_store_dissolution" | "audit_write",
  cause: unknown,
): Promise<never> {
  let restoreError: unknown;

  try {
    await restoreDissolvedFamily(dependencies, dissolution);
  } catch (error) {
    restoreError = error;
  }

  const restoreStatus = restoreError === undefined ? "succeeded" : "failed";
  const message =
    `Cross-store state report: family=${dissolution.family.family_id} stage=${stage} ` +
    `payload_restore_attempt=${restoreStatus}; migration aborted; ${PARTIAL_STATE_RECOVERY_MESSAGE}`;

  throw new AggregateError(restoreError === undefined ? [cause] : [cause, restoreError], message);
}

async function dissolveLegacyFamily(
  dependencies: OutcomeCorpusMigrationDependencies,
  dissolution: FamilyDissolution,
): Promise<void> {
  try {
    await dependencies.episodicRepository.revertConsolidationVersion({
      familyId: dissolution.family.family_id,
      versionEpisodeId: dissolution.family.current_version_episode_id,
      previousCurrentVersionEpisodeId: null,
      previousCoverageHash: null,
      previousPolicyVersion: null,
    });
  } catch (error) {
    return abortAfterDissolutionFailure(
      dependencies,
      dissolution,
      "cross_store_dissolution",
      error,
    );
  }

  try {
    dependencies.auditLog.record({
      run_id: dependencies.runId,
      process: "consolidator",
      action: OUTCOME_FAMILY_DISSOLVE_AUDIT_ACTION,
      targets: {
        migration_version: OUTCOME_ROLLUP_SCRIPT_VERSION,
        family_id: dissolution.family.family_id,
        current_version_episode_id: dissolution.family.current_version_episode_id,
        member_episode_ids: dissolution.members.map((member) => member.raw_episode_id),
      },
      reversal: {
        no_reverser: true,
        supported_rollback: BACKUP_ROLLBACK_INSTRUCTION,
        family: dissolution.family,
        members: dissolution.members,
        current_version: episodeForAudit(dissolution.currentVersion),
        current_stats: dissolution.currentStats,
      },
    });
  } catch (error) {
    return abortAfterDissolutionFailure(dependencies, dissolution, "audit_write", error);
  }
}

async function createRollupFamily(
  dependencies: OutcomeCorpusMigrationDependencies,
  group: OutcomeRollupGroup,
  embedding: Float32Array,
): Promise<{ familyId: ConsolidationFamilyId; episodeId: EpisodeId }> {
  const familyId = group.plannedFamilyId;
  const versionEpisodeId = group.plannedVersionId;

  if (familyId === null || versionEpisodeId === null) {
    throw new Error(`Rollup group ${group.key} has no planned ids`);
  }

  const nowMs = dependencies.clock.now();
  const episode = buildRollupEpisode(group, embedding, nowMs);
  let familyCreated = false;

  try {
    await dependencies.episodicRepository.createEpisode(episode);
    dependencies.episodicRepository.updateStats(episode.id, {
      tier: maxTier(group.sources.map((source) => source.stats)),
      promoted_at: nowMs,
      promoted_from: "consolidator",
    });
    dependencies.episodicRepository.createConsolidationFamily({
      familyId,
      currentVersionEpisodeId: episode.id,
      coverageHash: group.coverageHash,
      policyVersion: CONSOLIDATION_POLICY_VERSION,
      members: group.sources.map((source) => ({
        raw_episode_id: source.episode.id,
        source_stream_ids: source.episode.source_stream_ids,
        added_by_version_episode_id: episode.id,
      })),
    });
    familyCreated = true;
    const createdFamily = dependencies.episodicRepository.getConsolidationFamily(familyId);
    const createdMembers = dependencies.episodicRepository.listConsolidationMembers(familyId);
    const createdStats = dependencies.episodicRepository.getStats(episode.id);

    if (createdFamily === null || createdStats === null) {
      throw new Error(`Rollup family ${familyId} could not be read back for audit`);
    }

    dependencies.auditLog.record({
      run_id: dependencies.runId,
      process: "consolidator",
      action: OUTCOME_ROLLUP_AUDIT_ACTION,
      targets: {
        migration_version: OUTCOME_ROLLUP_SCRIPT_VERSION,
        group_key: group.key,
        role: group.role,
        utc_day: group.utcDay,
        family_id: familyId,
        version_episode_id: episode.id,
        source_episode_ids: group.sources.map((source) => source.episode.id),
        coverage_hash: group.coverageHash,
      },
      reversal: {
        no_reverser: true,
        supported_rollback: BACKUP_ROLLBACK_INSTRUCTION,
        created_family: createdFamily,
        created_members: createdMembers,
        created_version: episodeForAudit(episode),
        created_stats: createdStats,
      },
    });
  } catch (error) {
    if (familyCreated) {
      await dependencies.episodicRepository.revertConsolidationVersion({
        familyId,
        versionEpisodeId: episode.id,
        previousCurrentVersionEpisodeId: null,
        previousCoverageHash: null,
        previousPolicyVersion: null,
      });
    } else if (
      (await dependencies.episodicRepository.get(episode.id, { includeArchived: true })) !== null
    ) {
      await dependencies.episodicRepository.delete(episode.id);
    }
    throw error;
  }

  return { familyId, episodeId: episode.id };
}

async function reembedVersion(
  dependencies: OutcomeCorpusMigrationDependencies,
  item: VersionReembed,
  embedding: Float32Array,
): Promise<void> {
  const current = await dependencies.episodicRepository.get(item.episode.id, {
    includeArchived: true,
  });
  const family =
    item.episode.consolidation_family_id === null ||
    item.episode.consolidation_family_id === undefined
      ? null
      : dependencies.episodicRepository.getConsolidationFamily(
          item.episode.consolidation_family_id,
        );

  if (
    current === null ||
    episodeBodySha256(current) !== episodeBodySha256(item.episode) ||
    family?.current_version_episode_id !== item.episode.id
  ) {
    throw new Error(`OUTCOME version ${item.episode.id} changed after migration discovery`);
  }

  const currentStats = dependencies.episodicRepository.getStats(current.id);

  if (currentStats === null) {
    throw new Error(`OUTCOME version ${item.episode.id} has no stats row before re-embedding`);
  }

  const next = { ...current, embedding };
  await dependencies.episodicRepository.upsertEpisodeBodyPreservingStats(next);

  try {
    dependencies.auditLog.record({
      run_id: dependencies.runId,
      process: "consolidator",
      action: OUTCOME_VERSION_REEMBED_AUDIT_ACTION,
      targets: {
        migration_version: OUTCOME_ROLLUP_SCRIPT_VERSION,
        episode_id: current.id,
        family_id: current.consolidation_family_id,
        embedding_text_sha256: item.embeddingTextSha256,
      },
      reversal: {
        no_reverser: true,
        supported_rollback: BACKUP_ROLLBACK_INSTRUCTION,
        previous_episode: episodeForAudit(current),
        previous_stats: currentStats,
      },
    });
  } catch (error) {
    await dependencies.episodicRepository.upsertEpisodeBodyPreservingStats(current);
    throw error;
  }
}

function requiredDecisionLines(
  sources: readonly ScheduledOutcomeSource[],
  grammar: OutcomeCorpusGrammar,
): string[] {
  return collectProtectedEpisodeTokenLines(
    sources.map((source) => source.episode.narrative),
  ).filter((line) => parseOutcomeActionRecords([line], grammar).length > 0);
}

async function runFpSelfChecks(
  dependencies: OutcomeCorpusMigrationDependencies,
  groups: readonly OutcomeRollupGroup[],
): Promise<OutcomeFpSelfCheck[]> {
  const embeddingClient = dependencies.embeddingClient;

  if (embeddingClient === undefined) {
    throw new Error("Embedding client is required for fp recall self-checks");
  }

  const specification = dependencies.specification ?? DEFAULT_OUTCOME_CORPUS_SPECIFICATION;
  const queries = specification.scheduledFpSelfChecks.map(
    (fingerprint) => `OUTCOME fp=${fingerprint}`,
  );
  const vectors = await embeddingClient.embedBatch(queries);
  const visibleEpisodes = await dependencies.episodicRepository.listEffectivelyVisible();
  const checks: OutcomeFpSelfCheck[] = [];

  for (const [index, fingerprint] of specification.scheduledFpSelfChecks.entries()) {
    const matchingGroups = groups.filter((group) =>
      group.sources.some((source) => source.fingerprints.includes(fingerprint)),
    );
    const latestUtcDay =
      matchingGroups
        .map((group) => group.utcDay)
        .sort()
        .at(-1) ?? null;
    const latestSources = matchingGroups
      .filter((group) => group.utcDay === latestUtcDay)
      .flatMap((group) => group.sources);
    const decisionLines = requiredDecisionLines(latestSources, specification.grammar);
    const vector = vectors[index];

    if (latestUtcDay === null || vector === undefined || decisionLines.length === 0) {
      checks.push({
        fingerprint,
        latestUtcDay,
        requiredDecisionLineCount: decisionLines.length,
        passed: false,
        matchedEpisodeId: null,
        rank: null,
        matchSource: null,
        reason:
          latestUtcDay === null
            ? "no source group"
            : decisionLines.length === 0
              ? "latest day has no parseable decision lines"
              : "missing query embedding",
      });
      continue;
    }

    const vectorCandidates = await dependencies.episodicRepository.recallByVectorForCognition(
      vector,
      {
        limit: RECALL_LIMIT,
      },
    );
    const fpToken = `OUTCOME fp=${fingerprint}`;
    const exactFpRescues = visibleEpisodes
      .filter((episode) => `${episode.title}\n${episode.narrative}`.includes(fpToken))
      .sort(
        (left, right) =>
          right.start_time - left.start_time ||
          right.updated_at - left.updated_at ||
          left.id.localeCompare(right.id),
      )
      .slice(0, RECALL_EXACT_FP_RESERVED_SLOTS);
    const exactFpRescueIds = new Set(exactFpRescues.map((episode) => episode.id));
    const candidates = [
      ...vectorCandidates
        .filter((candidate) => !exactFpRescueIds.has(candidate.episode.id))
        .slice(0, RECALL_LIMIT - exactFpRescues.length)
        .map((candidate) => ({ episode: candidate.episode, source: "vector" as const })),
      ...exactFpRescues.map((episode) => ({ episode, source: "exact_fp_rescue" as const })),
    ];
    const matchedIndex = candidates.findIndex((candidate) => {
      const text = `${candidate.episode.title}\n${candidate.episode.narrative}`;
      return text.includes(fpToken) && decisionLines.every((line) => text.includes(line));
    });
    const matched = matchedIndex < 0 ? null : (candidates[matchedIndex] ?? null);

    checks.push({
      fingerprint,
      latestUtcDay,
      requiredDecisionLineCount: decisionLines.length,
      passed: matched !== null,
      matchedEpisodeId: matched?.episode.id ?? null,
      rank: matched === null ? null : matchedIndex + 1,
      matchSource: matched?.source ?? null,
      reason: matched === null ? `no matching episode in top ${RECALL_LIMIT}` : null,
    });
  }

  return checks;
}

async function revalidateExplicitKeep(
  dependencies: OutcomeCorpusMigrationDependencies,
  report: OutcomeCorpusMigrationReport,
): Promise<void> {
  const specification = dependencies.specification ?? DEFAULT_OUTCOME_CORPUS_SPECIFICATION;
  const keepEpisode = await dependencies.episodicRepository.get(
    specification.explicitKeepEpisodeId,
    { includeArchived: true },
  );
  const keepStats = dependencies.episodicRepository.getStats(specification.explicitKeepEpisodeId);
  const family =
    keepEpisode?.consolidation_family_id === null ||
    keepEpisode?.consolidation_family_id === undefined
      ? null
      : dependencies.episodicRepository.getConsolidationFamily(keepEpisode.consolidation_family_id);
  const bodySha256 = keepEpisode === null ? null : episodeBodySha256(keepEpisode);
  const currentVersion = family?.current_version_episode_id === specification.explicitKeepEpisodeId;
  const safe =
    keepEpisode !== null &&
    bodySha256 === specification.explicitKeepBodySha256 &&
    currentVersion &&
    keepStats?.archived === false;

  report.explicitKeep = {
    id: specification.explicitKeepEpisodeId,
    found: keepEpisode !== null,
    bodySha256,
    currentVersion,
    state: safe ? "kept" : "unsafe",
  };

  if (!safe) {
    report.unsafeItems.push(
      `post-apply explicit keep episode ${specification.explicitKeepEpisodeId} failed id/content-hash/current-version/non-archived revalidation`,
    );
  }
}

async function applyMigration(
  dependencies: OutcomeCorpusMigrationDependencies,
  report: OutcomeCorpusMigrationReport,
): Promise<OutcomeCorpusMigrationReport> {
  const embeddingClient = dependencies.embeddingClient;

  if (embeddingClient === undefined) {
    throw new Error("Embedding client is required for --apply");
  }

  const groupsToCreate = report.groups.filter(
    (group) => group.sources.length > 1 && group.existingFamily === null,
  );
  const embeddingInputs = [
    ...groupsToCreate.map((group) => group.embeddingText),
    ...report.versionsToReembed.map((item) => item.embeddingText),
  ];
  const embeddings =
    embeddingInputs.length === 0 ? [] : await embeddingClient.embedBatch(embeddingInputs);

  if (embeddings.length !== embeddingInputs.length) {
    throw new Error(
      `Embedding preflight returned ${embeddings.length} vectors for ${embeddingInputs.length} inputs`,
    );
  }

  for (const dissolution of report.legacyFamiliesToDissolve) {
    const current = dependencies.episodicRepository.getConsolidationFamily(
      dissolution.family.family_id,
    );

    if (current?.current_version_episode_id !== dissolution.family.current_version_episode_id) {
      throw new Error(`Legacy family ${dissolution.family.family_id} changed after discovery`);
    }

    await dissolveLegacyFamily(dependencies, dissolution);
  }

  for (const [index, group] of groupsToCreate.entries()) {
    const embedding = embeddings[index];

    if (embedding === undefined) {
      throw new Error(`Missing rollup embedding for ${group.key}`);
    }

    const created = await createRollupFamily(dependencies, group, embedding);
    report.rollupsCreated.push({ key: group.key, ...created });
  }

  const reembedOffset = groupsToCreate.length;

  for (const [index, item] of report.versionsToReembed.entries()) {
    const embedding = embeddings[reembedOffset + index];

    if (embedding === undefined) {
      throw new Error(`Missing backfill embedding for ${item.episode.id}`);
    }

    await reembedVersion(dependencies, item, embedding);
    report.versionsReembedded.push(item.episode.id);
  }

  for (const toxic of report.toxicEpisodes) {
    if (toxic.state !== "would_archive") {
      continue;
    }

    dependencies.episodicRepository.archiveEpisode(toxic.id, {
      caller: "scripts/migrate-outcome-corpus.ts",
      reason: toxic.reason,
      process: "consolidator",
      runId: dependencies.runId,
    });
    report.toxicEpisodesArchived.push(toxic.id);
  }

  await revalidateExplicitKeep(dependencies, report);
  report.fpSelfChecks = await runFpSelfChecks(dependencies, report.groups);
  const visible = await dependencies.episodicRepository.listEffectivelyVisible();
  const specification = dependencies.specification ?? DEFAULT_OUTCOME_CORPUS_SPECIFICATION;
  const actualVisibleOutcomeEpisodeIds = visible
    .filter(
      (episode) =>
        outcomeFingerprints(
          collectProtectedEpisodeTokenLines([episode.narrative]),
          specification.grammar,
        ).length > 0,
    )
    .map((episode) => episode.id)
    .sort();
  const actualVisibleOutcomeEpisodeIdSet = new Set(actualVisibleOutcomeEpisodeIds);
  const expectedVisibleOutcomeEpisodeIdSet = new Set(report.expectedVisibleOutcomeEpisodeIds);
  report.actualVisibleOutcomeEpisodeIdsAfter = actualVisibleOutcomeEpisodeIds;
  report.missingVisibleOutcomeEpisodeIdsAfter = report.expectedVisibleOutcomeEpisodeIds.filter(
    (episodeId) => !actualVisibleOutcomeEpisodeIdSet.has(episodeId),
  );
  report.extraVisibleOutcomeEpisodeIdsAfter = actualVisibleOutcomeEpisodeIds.filter(
    (episodeId) => !expectedVisibleOutcomeEpisodeIdSet.has(episodeId),
  );
  report.liveRolledOutcomeRecordCountAfter = actualVisibleOutcomeEpisodeIds.length;
  report.liveOutcomeRecordCountAfter = actualVisibleOutcomeEpisodeIds.length;
  report.toxicEpisodesInvisibleAfter = report.toxicEpisodes.every(
    (item) => !dependencies.episodicRepository.isEpisodeEffectivelyVisible(item.id),
  );
  const auditRows = dependencies.auditLog.list({ run_id: dependencies.runId });
  report.auditRowsWritten = auditRows.length;
  const customAuditRows = auditRows.filter((audit) =>
    CUSTOM_MIGRATION_AUDIT_ACTIONS.has(audit.action),
  ).length;
  report.customAuditRowsWritten = customAuditRows;
  report.customAuditRowsWithNoReverser = auditRows.filter(
    (audit) =>
      CUSTOM_MIGRATION_AUDIT_ACTIONS.has(audit.action) && audit.reversal.no_reverser === true,
  ).length;
  report.dryRun = false;
  return report;
}

export async function migrateOutcomeCorpus(
  dependencies: OutcomeCorpusMigrationDependencies,
  options: { apply?: boolean } = {},
): Promise<OutcomeCorpusMigrationReport> {
  const report = await buildMigrationReport(dependencies);

  if (options.apply !== true || report.unsafeItems.length > 0) {
    return report;
  }

  return applyMigration(dependencies, report);
}

function formatGroup(group: OutcomeRollupGroup, dryRun: boolean): string {
  const action =
    group.sources.length === 1
      ? "singleton_keep"
      : group.existingFamily !== null
        ? "existing_rollup"
        : dryRun
          ? "would_roll_up"
          : "rolled_up";

  return `${action} group=${group.key} members=${group.sources.length} protected_lines=${group.protectedLines.length} family=${group.existingFamily?.family_id ?? group.plannedFamilyId ?? "none"}`;
}

export function formatOutcomeCorpusMigrationReport(report: OutcomeCorpusMigrationReport): string {
  const lines = [
    `mode=${report.dryRun ? "dry-run" : "apply"}`,
    `raw_outcome_episodes=${report.rawOutcomeEpisodeCount}`,
    `role_bearing_outcome_sources=${report.scheduledOutcomeSourceCount}`,
    `non_scheduled_outcome_episodes=${report.nonScheduledOutcomeEpisodeCount}`,
    `visible_non_scheduled_outcome_episodes=${report.visibleNonScheduledOutcomeEpisodeCount}`,
    `malformed_scheduled_outcome_episodes=${report.malformedScheduledOutcomeEpisodeIds.length}`,
    `role_day_groups=${report.groupCount}`,
    `multi_member_groups=${report.multiMemberGroupCount}`,
    `singleton_groups=${report.singletonGroupCount}`,
    `largest_group=${report.largestGroupSize}`,
    `projected_live_outcome_records=${report.projectedLiveOutcomeRecordCount}`,
    `legacy_families_to_dissolve=${report.legacyFamiliesToDissolve.length}`,
    `rollups_created=${report.rollupsCreated.length}`,
    `token_bearing_consolidation_versions=${report.tokenBearingVersionCount}`,
    `versions_to_reembed=${report.versionsToReembed.length}`,
    `versions_reembedded=${report.versionsReembedded.length}`,
    `toxic_archive_candidates=${report.toxicEpisodes.filter((item) => item.state === "would_archive").length}`,
    `toxic_archived=${report.toxicEpisodesArchived.length}`,
    `explicit_keep=${report.explicitKeep.id} state=${report.explicitKeep.state} current_version=${report.explicitKeep.currentVersion}`,
    `semantic_node=${report.semanticNode.id} found=${report.semanticNode.found} status=${report.semanticNode.status ?? "missing"} archived=${report.semanticNode.archived ?? "missing"} action=${report.semanticNode.action}`,
    `unsafe_items=${report.unsafeItems.length}`,
    `fp_self_checks=${report.fpSelfChecks.length} passed=${report.fpSelfChecks.filter((check) => check.passed).length}`,
    `live_rolled_outcome_records_after=${report.liveRolledOutcomeRecordCountAfter ?? "not-run"}`,
    `live_outcome_records_after=${report.liveOutcomeRecordCountAfter ?? "not-run"}`,
    `expected_visible_outcome_ids=${report.expectedVisibleOutcomeEpisodeIds.length}`,
    `actual_visible_outcome_ids_after=${report.actualVisibleOutcomeEpisodeIdsAfter?.length ?? "not-run"}`,
    `missing_visible_outcome_ids_after=${report.missingVisibleOutcomeEpisodeIdsAfter.length}`,
    `extra_visible_outcome_ids_after=${report.extraVisibleOutcomeEpisodeIdsAfter.length}`,
    `toxic_invisible_after=${report.toxicEpisodesInvisibleAfter ?? "not-run"}`,
    `audit_rows_written=${report.auditRowsWritten}`,
    `custom_audit_rows_written=${report.customAuditRowsWritten}`,
    `custom_audit_rows_no_reverser=${report.customAuditRowsWithNoReverser}`,
  ];

  for (const episodeId of report.malformedScheduledOutcomeEpisodeIds) {
    lines.push(`malformed_scheduled episode=${episodeId} intended_state=unsafe_preserve`);
  }

  for (const group of report.groups) {
    lines.push(formatGroup(group, report.dryRun));
  }

  for (const dissolution of report.legacyFamiliesToDissolve) {
    lines.push(
      `${report.dryRun ? "would_dissolve" : "dissolved"} family=${dissolution.family.family_id} current=${dissolution.family.current_version_episode_id} members=${dissolution.members.length}`,
    );
  }

  for (const family of report.unsafeFamilies) {
    lines.push(`unsafe_family=${family.familyId} reason=${family.reason}`);
    for (const member of family.members) {
      lines.push(
        `unsafe_family_member family=${family.familyId} episode=${member.episodeId} intended_state=${member.intendedState}`,
      );
    }
  }

  for (const item of report.acknowledgedMixedItems) {
    lines.push(`acknowledged_mixed=${JSON.stringify(item)}`);
  }

  const reembeddedIds = new Set(report.versionsReembedded);
  for (const item of report.versionsToReembed) {
    lines.push(
      `${report.dryRun ? "would_reembed" : reembeddedIds.has(item.episode.id) ? "reembedded" : "not_reembedded"} episode=${item.episode.id} family=${item.episode.consolidation_family_id ?? "none"} embedding_text_sha256=${item.embeddingTextSha256}`,
    );
  }

  const archivedIds = new Set(report.toxicEpisodesArchived);
  for (const toxic of report.toxicEpisodes) {
    const action =
      toxic.state === "would_archive"
        ? report.dryRun
          ? "would_archive"
          : archivedIds.has(toxic.id)
            ? "archived"
            : "not_archived"
        : toxic.state;
    lines.push(
      `${action} episode=${toxic.id} conditional=${toxic.conditional} visible_before=${toxic.effectivelyVisibleBefore} body_sha256=${toxic.bodySha256 ?? "missing"} reason=${JSON.stringify(toxic.reason)}`,
    );
  }

  for (const check of report.fpSelfChecks) {
    lines.push(
      `fp_check fingerprint=${check.fingerprint} passed=${check.passed} latest_day=${check.latestUtcDay ?? "missing"} decision_lines=${check.requiredDecisionLineCount} episode=${check.matchedEpisodeId ?? "none"} rank=${check.rank ?? "none"} source=${check.matchSource ?? "none"} reason=${JSON.stringify(check.reason)}`,
    );
  }

  for (const episodeId of report.missingVisibleOutcomeEpisodeIdsAfter) {
    lines.push(`acceptance_missing_visible_outcome_episode=${episodeId}`);
  }

  for (const episodeId of report.extraVisibleOutcomeEpisodeIdsAfter) {
    lines.push(`acceptance_extra_visible_outcome_episode=${episodeId}`);
  }

  for (const unsafe of report.unsafeItems) {
    lines.push(`unsafe=${JSON.stringify(unsafe)}`);
  }

  return `${lines.join("\n")}\n`;
}

export function outcomeCorpusMigrationExitCode(report: OutcomeCorpusMigrationReport): 0 | 1 {
  if (report.unsafeItems.length > 0) {
    return 1;
  }

  if (
    !report.dryRun &&
    (report.fpSelfChecks.some((check) => !check.passed) ||
      report.actualVisibleOutcomeEpisodeIdsAfter === null ||
      report.missingVisibleOutcomeEpisodeIdsAfter.length > 0 ||
      report.extraVisibleOutcomeEpisodeIdsAfter.length > 0 ||
      report.toxicEpisodesInvisibleAfter !== true ||
      report.customAuditRowsWithNoReverser !== report.customAuditRowsWritten)
  ) {
    return 1;
  }

  return 0;
}

type OutcomeCorpusCliArgs =
  | { help: true }
  | {
      help: false;
      dataDir: string;
      apply: boolean;
      dissolveMixedFamilyIds: string[];
    };

export function parseOutcomeCorpusCliArgs(argv: readonly string[]): OutcomeCorpusCliArgs {
  let dataDir: string | undefined;
  let apply = false;
  const dissolveMixedFamilyIds: string[] = [];

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];

    if (argument === "--help" || argument === "-h") {
      return { help: true };
    }

    if (argument === "--apply") {
      apply = true;
      continue;
    }

    if (argument === "--dissolve-mixed-family") {
      const value = argv[index + 1];

      if (value === undefined || value.startsWith("--")) {
        throw new Error("--dissolve-mixed-family requires a consolidation family id");
      }

      dissolveMixedFamilyIds.push(value);
      index += 1;
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

  return { help: false, dataDir: resolve(dataDir), apply, dissolveMixedFamilyIds };
}

function usage(): string {
  return [
    "Usage: pnpm tsx scripts/migrate-outcome-corpus.ts --data-dir <bank-dir> [--apply]",
    "         [--dissolve-mixed-family <family-id>]...",
    "",
    "Dry-run is the default. Stop every Borg writer and take a verified backup before --apply.",
    "Mixed families are unsafe by default; after inspecting the dry-run report an operator can",
    "acknowledge specific ids with --dissolve-mixed-family (unrelated members become visible).",
    "Dry-run reads embedding dimensions from the existing episodes LanceDB schema.",
    "Apply requires KRATOS_BASE_URL, LLM_API_KEY, EMBEDDING_MODEL, and EMBEDDING_DIMS.",
    "Apply also requires EMBEDDING_DIMS to match the existing LanceDB schema.",
  ].join("\n");
}

function openMaintenanceDatabase(path: string, readOnly: boolean): SqliteDatabase {
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

function requiredEnv(env: NodeJS.ProcessEnv, name: string): string {
  const value = env[name]?.trim();

  if (value === undefined || value.length === 0) {
    throw new Error(`${name} is required for --apply`);
  }

  return value;
}

function embeddingDimensions(env: NodeJS.ProcessEnv): number {
  const raw = requiredEnv(env, "EMBEDDING_DIMS");
  const dims = Number(raw);

  if (!Number.isInteger(dims) || dims <= 0) {
    throw new Error(`EMBEDDING_DIMS must be a positive integer, received ${JSON.stringify(raw)}`);
  }

  return dims;
}

async function existingEpisodeEmbeddingDimensions(lancePath: string): Promise<number> {
  const connection = await connect(lancePath);

  try {
    const table = await connection.openTable("episodes");

    try {
      const tableSchema = await table.schema();
      const embeddingField = tableSchema.fields.find((field) => field.name === "embedding");
      const embeddingType = embeddingField?.type as { listSize?: unknown } | undefined;
      const listSize = embeddingType?.listSize;

      if (typeof listSize !== "number") {
        throw new Error("Existing episodes LanceDB schema has no fixed-size embedding vector");
      }

      if (!Number.isInteger(listSize) || listSize <= 0) {
        throw new Error(`Existing episodes LanceDB embedding dimension is invalid: ${listSize}`);
      }

      return listSize;
    } finally {
      table.close();
    }
  } finally {
    connection.close();
  }
}

function createScriptEmbeddingClient(env: NodeJS.ProcessEnv, dims: number): EmbeddingClient {
  return new OpenAICompatibleEmbeddingClient({
    baseUrl: requiredEnv(env, "KRATOS_BASE_URL"),
    apiKey: requiredEnv(env, "LLM_API_KEY"),
    model: requiredEnv(env, "EMBEDDING_MODEL"),
    dims,
  });
}

export async function main(
  argv: readonly string[] = process.argv.slice(2),
  env: NodeJS.ProcessEnv = process.env,
): Promise<0 | 1> {
  const args = parseOutcomeCorpusCliArgs(argv);

  if (args.help) {
    process.stdout.write(`${usage()}\n`);
    return 0;
  }

  const databasePath = join(args.dataDir, "borg.db");
  const lancePath = join(args.dataDir, "lancedb");
  const episodesTablePath = join(lancePath, "episodes.lance");

  if (!existsSync(databasePath)) {
    throw new Error(`No borg.db found in data directory ${args.dataDir}`);
  }

  if (!existsSync(episodesTablePath)) {
    throw new Error(`No episodes LanceDB table found in data directory ${args.dataDir}`);
  }

  process.stderr.write(
    "WARNING: this maintenance requires a verified backup and exclusive single-writer access.\n",
  );
  const schemaDims = await existingEpisodeEmbeddingDimensions(lancePath);
  const dims = args.apply ? embeddingDimensions(env) : schemaDims;

  if (args.apply && dims !== schemaDims) {
    throw new Error(
      `EMBEDDING_DIMS=${dims} does not match the existing episodes LanceDB schema dimension ${schemaDims}`,
    );
  }

  const db = openMaintenanceDatabase(databasePath, !args.apply);
  const lance = new LanceDbStore({ uri: lancePath });
  let episodesTable: LanceDbTable | undefined;

  try {
    episodesTable = await lance.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(dims),
    });
    const clock = new SystemClock();
    const report = await migrateOutcomeCorpus(
      {
        db,
        episodicRepository: new EpisodicRepository({ table: episodesTable, db, clock }),
        auditLog: new AuditLog({ db, clock }),
        ...(args.apply ? { embeddingClient: createScriptEmbeddingClient(env, dims) } : {}),
        clock,
        runId: createMaintenanceRunId(),
        ...(args.dissolveMixedFamilyIds.length === 0
          ? {}
          : { acknowledgedMixedFamilyIds: args.dissolveMixedFamilyIds }),
      },
      { apply: args.apply },
    );
    process.stdout.write(formatOutcomeCorpusMigrationReport(report));
    const exitCode = outcomeCorpusMigrationExitCode(report);

    if (exitCode !== 0) {
      process.stderr.write(
        "ERROR: migration validation or post-apply acceptance checks failed; inspect the report.\n",
      );
    }

    return exitCode;
  } finally {
    episodesTable?.close();
    await lance.close();
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
