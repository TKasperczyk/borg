import { BorgTransport, type AuditTranscriptEntry } from "../assessor/borg-transport.js";
import type { BorgDependencies } from "../src/borg/types.js";
import type { StreamEntry } from "../src/stream/index.js";
import { estimatePromptTokens } from "../src/util/token-estimate.js";
import type { SessionId } from "../src/util/ids.js";

const MAX_TEXT_CHARS = 240;
const LARGE_LIMIT = 1_000;
export const MEMORY_SNAPSHOT_TARGET_TOKEN_BUDGET = 80_000;
export const MEMORY_SNAPSHOT_FAILSAFE_TOKEN_CAP = 150_000;
const MEMORY_SNAPSHOT_ROW_TOKEN_BUDGET = MEMORY_SNAPSHOT_TARGET_TOKEN_BUDGET - 512;
const SECTION_ROW_LIMITS = {
  "Scope And Counts": 8,
  "Stream Transcript": 240,
  "Episodic Memory": 180,
  "Semantic Nodes": 180,
  "Semantic Edges": 240,
  "Identity And Self": 180,
  "Goals And Open Questions": 180,
  Commitments: 160,
  Actions: 240,
  "Relational And Social": 240,
  "Affective State": 180,
  "Procedural Memory": 180,
  "Working Memory": 32,
  "Review And Audit Diagnostics": 160,
} as const;

type BorgWithDeps = {
  deps?: BorgDependencies;
};

type RecordLike = Record<string, unknown>;
type SnapshotSection = {
  title: keyof typeof SECTION_ROW_LIMITS;
  rows: readonly string[];
  empty: string;
};

export type BuildMemorySnapshotOptions = {
  transport: BorgTransport;
  sessionIds?: readonly SessionId[];
};

function borgDeps(transport: BorgTransport): BorgDependencies | null {
  return (transport.getBorg() as unknown as BorgWithDeps).deps ?? null;
}

function asRecord(value: unknown): RecordLike | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as RecordLike)
    : null;
}

function oneLine(value: unknown, max = MAX_TEXT_CHARS): string {
  const text =
    typeof value === "string"
      ? value
      : value === null || value === undefined
        ? ""
        : JSON.stringify(value);
  const compact = text.replace(/\s+/g, " ").trim();

  if (compact.length <= max) {
    return compact;
  }

  return `${compact.slice(0, max - 1)}...`;
}

function scalar(value: unknown): string {
  if (value === null) return "null";
  if (value === undefined) return "n/a";
  if (Array.isArray(value)) return value.map((item) => scalar(item)).join(",");
  if (typeof value === "object") {
    const record = asRecord(value);
    const kind = record?.kind;
    const episodeIds = record?.episode_ids;
    const process = record?.process;
    const parts = [
      typeof kind === "string" ? kind : undefined,
      Array.isArray(episodeIds) && episodeIds.length > 0
        ? `episodes=${episodeIds.map((item) => scalar(item)).join(",")}`
        : undefined,
      typeof process === "string" ? `process=${process}` : undefined,
    ].filter((part): part is string => part !== undefined);

    return parts.length === 0 ? oneLine(value, 80) : parts.join(" ");
  }

  return String(value);
}

function ts(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? String(value) : "n/a";
}

function ids(value: unknown): string {
  return Array.isArray(value) && value.length > 0
    ? value.map((item) => scalar(item)).join(",")
    : "-";
}

function flattenGoalRows(goals: readonly RecordLike[]): RecordLike[] {
  const rows: RecordLike[] = [];
  const visit = (goal: RecordLike): void => {
    rows.push(goal);
    const children = goal.children;

    if (Array.isArray(children)) {
      for (const child of children) {
        const record = asRecord(child);
        if (record !== null) {
          visit(record);
        }
      }
    }
  };

  for (const goal of goals) {
    visit(goal);
  }

  return rows;
}

function numericRank(record: RecordLike, fields: readonly string[]): number {
  for (const field of fields) {
    const value = record[field];

    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }

  return Number.NEGATIVE_INFINITY;
}

function rankedRecords(
  rows: readonly RecordLike[],
  input: {
    timeFields?: readonly string[];
    salienceFields?: readonly string[];
  } = {},
): RecordLike[] {
  const timeFields = input.timeFields ?? [];
  const salienceFields = input.salienceFields ?? [];

  return [...rows].sort((left, right) => {
    const salienceDelta = numericRank(right, salienceFields) - numericRank(left, salienceFields);

    if (salienceDelta !== 0) {
      return salienceDelta;
    }

    const timeDelta = numericRank(right, timeFields) - numericRank(left, timeFields);

    if (timeDelta !== 0) {
      return timeDelta;
    }

    return scalar(left.id).localeCompare(scalar(right.id));
  });
}

function rankedTranscriptRows(entries: readonly AuditTranscriptEntry[]): string[] {
  return entries
    .map((entry, index) => ({ entry, index }))
    .sort(
      (left, right) =>
        right.entry.entry.timestamp - left.entry.entry.timestamp ||
        left.entry.entry.id.localeCompare(right.entry.entry.id),
    )
    .map(({ entry, index }) => streamRow(entry, index));
}

function cappedRows(
  sectionTitle: keyof typeof SECTION_ROW_LIMITS,
  rows: readonly string[],
): string[] {
  const limit = SECTION_ROW_LIMITS[sectionTitle];

  if (rows.length <= limit) {
    return [...rows];
  }

  return [
    ...rows.slice(0, limit),
    `- ... omitted ${rows.length - limit} older entries (showing ${limit} newest/highest-priority of ${rows.length}; per-section cap=${limit}).`,
  ];
}

function appendBudgetedRow(lines: string[], row: string): boolean {
  const next = [...lines, row].join("\n");

  if (estimatePromptTokens(next) > MEMORY_SNAPSHOT_ROW_TOKEN_BUDGET) {
    return false;
  }

  lines.push(row);
  return true;
}

function appendBudgetTrailer(lines: string[], sectionTitle: string, omitted: number): void {
  if (omitted <= 0) {
    return;
  }

  let omittedCount = omitted;
  let trailer = `- ... omitted ${omittedCount} additional rows from ${sectionTitle} due to global token budget (target=${MEMORY_SNAPSHOT_TARGET_TOKEN_BUDGET}).`;

  while (!appendBudgetedRow(lines, trailer) && lines.length > 3) {
    const previous = lines.at(-1);

    if (previous === undefined || previous.startsWith("### ")) {
      break;
    }

    lines.pop();
    omittedCount += 1;
    trailer = `- ... omitted ${omittedCount} additional rows from ${sectionTitle} due to global token budget (target=${MEMORY_SNAPSHOT_TARGET_TOKEN_BUDGET}).`;
  }
}

function renderSnapshotSections(sections: readonly SnapshotSection[]): string {
  const lines = ["## Memory Snapshot"];

  for (const snapshotSection of sections) {
    lines.push("", `### ${snapshotSection.title}`);
    const rows =
      snapshotSection.rows.length === 0
        ? [snapshotSection.empty]
        : cappedRows(snapshotSection.title, snapshotSection.rows);
    let omittedByBudget = 0;

    for (const row of rows) {
      if (!appendBudgetedRow(lines, row)) {
        omittedByBudget += 1;
      }
    }

    appendBudgetTrailer(lines, snapshotSection.title, omittedByBudget);
  }

  let rendered = lines.join("\n");

  if (estimatePromptTokens(rendered) <= MEMORY_SNAPSHOT_FAILSAFE_TOKEN_CAP) {
    return rendered;
  }

  const failSafeLines = rendered.split("\n");
  const trailer = `- ... snapshot truncated at fail-safe cap ${MEMORY_SNAPSHOT_FAILSAFE_TOKEN_CAP} estimated tokens.`;

  while (
    estimatePromptTokens([...failSafeLines, trailer].join("\n")) >
      MEMORY_SNAPSHOT_FAILSAFE_TOKEN_CAP &&
    failSafeLines.length > 3
  ) {
    failSafeLines.pop();
  }

  rendered = [...failSafeLines, trailer].join("\n");
  return rendered;
}

function streamRow(entry: AuditTranscriptEntry, index: number): string {
  const quarantine = entry.quarantined ? ` quarantine=${entry.quarantineReason ?? "unknown"}` : "";

  return `- [${index}] id=${entry.entry.id} ts=${entry.entry.timestamp} session=${entry.entry.session_id} kind=${entry.entry.kind}${quarantine} text="${oneLine(entry.entry.content)}"`;
}

function episodeRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} ts=${ts(record.start_time)}-${ts(record.end_time)} confidence=${scalar(record.confidence)} sources=${ids(record.source_stream_ids)} title="${oneLine(record.title, 120)}" narrative="${oneLine(record.narrative)}"`;
}

function semanticNodeRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} kind=${scalar(record.kind)} confidence=${scalar(record.confidence)} archived=${scalar(record.archived)} sources=${ids(record.source_episode_ids)} label="${oneLine(record.label, 120)}" description="${oneLine(record.description)}"`;
}

function semanticEdgeRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} ${scalar(record.from_node_id)} -[${scalar(record.relation)}]-> ${scalar(record.to_node_id)} confidence=${scalar(record.confidence)} evidence=${ids(record.evidence_episode_ids)} valid=${ts(record.valid_from)}..${ts(record.valid_to)} invalidated=${ts(record.invalidated_at)} reason="${oneLine(record.invalidated_reason, 120)}"`;
}

function valueRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} state=${scalar(record.state)} priority=${scalar(record.priority)} confidence=${scalar(record.confidence)} evidence=${ids(record.evidence_episode_ids)} provenance=${scalar(record.provenance)} label="${oneLine(record.label, 100)}" description="${oneLine(record.description)}"`;
}

function goalRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} status=${scalar(record.status)} priority=${scalar(record.priority)} progress_ts=${ts(record.last_progress_ts)} sources=${ids(record.source_stream_entry_ids)} provenance=${scalar(record.provenance)} description="${oneLine(record.description)}" progress="${oneLine(record.progress_notes, 120)}"`;
}

function traitRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} state=${scalar(record.state)} strength=${scalar(record.strength)} confidence=${scalar(record.confidence)} evidence=${ids(record.evidence_episode_ids)} provenance=${scalar(record.provenance)} label="${oneLine(record.label, 120)}"`;
}

function periodRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} start=${ts(record.start_ts)} end=${ts(record.end_ts)} episodes=${ids(record.key_episode_ids)} provenance=${scalar(record.provenance)} label="${oneLine(record.label, 120)}" narrative="${oneLine(record.narrative)}"`;
}

function growthMarkerRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} ts=${ts(record.ts)} category=${scalar(record.category)} confidence=${scalar(record.confidence)} evidence=${ids(record.evidence_episode_ids)} source_process=${scalar(record.source_process)} change="${oneLine(record.what_changed)}"`;
}

function openQuestionRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} status=${scalar(record.status)} urgency=${scalar(record.urgency)} source=${scalar(record.source)} episodes=${ids(record.related_episode_ids)} semantic_nodes=${ids(record.related_semantic_node_ids)} question="${oneLine(record.question)}"`;
}

function identityEventRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} ts=${ts(record.ts)} type=${scalar(record.record_type)} record=${scalar(record.record_id)} action=${scalar(record.action)} review=${scalar(record.review_item_id)} provenance=${scalar(record.provenance)} summary="${oneLine(record.summary)}"`;
}

function commitmentRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} type=${scalar(record.type)} family=${scalar(record.directive_family)} priority=${scalar(record.priority)} made_to=${scalar(record.made_to_entity)} audience=${scalar(record.restricted_audience)} about=${scalar(record.about_entity)} revoked=${ts(record.revoked_at)} superseded_by=${scalar(record.superseded_by)} sources=${ids(record.source_stream_entry_ids)} directive="${oneLine(record.directive)}"`;
}

function commitmentExpired(record: RecordLike, nowMs: number): boolean {
  return (
    (record.expired_at !== null && record.expired_at !== undefined) ||
    (typeof record.expires_at === "number" && record.expires_at <= nowMs)
  );
}

function commitmentActive(record: RecordLike, nowMs: number): boolean {
  return (
    (record.revoked_at === null || record.revoked_at === undefined) &&
    (record.superseded_by === null || record.superseded_by === undefined) &&
    !commitmentExpired(record, nowMs)
  );
}

type CommitmentLifecycleCounts = {
  active: number;
  revoked: number;
  expired: number;
  canonicalized: number;
  superseded: number;
};

function countFromSource(source: unknown, method: string): number | null {
  const record = asRecord(source);
  const candidate = record?.[method];

  if (typeof candidate !== "function") {
    return null;
  }

  const value = candidate();

  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function commitmentLifecycleCounts(input: {
  commitments: readonly RecordLike[];
  nowMs: number;
  source: unknown;
}): CommitmentLifecycleCounts {
  const active = input.commitments.filter((commitment) =>
    commitmentActive(commitment, input.nowMs),
  );
  const revoked = input.commitments.filter(
    (commitment) => commitment.revoked_at !== null && commitment.revoked_at !== undefined,
  );
  const expired = input.commitments.filter((commitment) =>
    commitmentExpired(commitment, input.nowMs),
  );
  const canonicalized = input.commitments.filter(
    (commitment) =>
      commitment.canonicalized_by_artifact_entry_id !== null &&
      commitment.canonicalized_by_artifact_entry_id !== undefined,
  );
  const superseded = input.commitments.filter(
    (commitment) => commitment.superseded_by !== null && commitment.superseded_by !== undefined,
  );

  return {
    active: countFromSource(input.source, "countActive") ?? active.length,
    revoked: countFromSource(input.source, "countRevoked") ?? revoked.length,
    expired: countFromSource(input.source, "countExpired") ?? expired.length,
    canonicalized: countFromSource(input.source, "countCanonicalized") ?? canonicalized.length,
    superseded: countFromSource(input.source, "countSuperseded") ?? superseded.length,
  };
}

function commitmentLifecycleRows(input: {
  commitments: readonly RecordLike[];
  nowMs: number;
  countsSource: unknown;
}): string[] {
  const commitments = input.commitments;
  const counts = commitmentLifecycleCounts({
    commitments,
    nowMs: input.nowMs,
    source: input.countsSource,
  });
  const active = commitments.filter((commitment) => commitmentActive(commitment, input.nowMs));

  return [
    `- total_commitments=${commitments.length}`,
    `- lifecycle_counts active=${counts.active} revoked=${counts.revoked} expired=${counts.expired} canonicalized=${counts.canonicalized} superseded=${counts.superseded}`,
    ...rankedRecords(active, {
      timeFields: ["last_reinforced_at", "created_at"],
    }).map(commitmentRow),
  ];
}

function actionRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} actor=${scalar(record.actor)} state=${scalar(record.state)} confidence=${scalar(record.confidence)} updated=${ts(record.updated_at)} completed=${ts(record.completed_at)} episodes=${ids(record.provenance_episode_ids)} streams=${ids(record.provenance_stream_entry_ids)} description="${oneLine(record.description)}"`;
}

function entityRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} kind=${scalar(record.kind)} provenance=${scalar(record.name_provenance)} created=${ts(record.created_at)} name="${oneLine(record.canonical_name, 120)}" aliases=${ids(record.aliases)}`;
}

function relationalSlotRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} subject=${scalar(record.subject_entity_id)} key=${scalar(record.slot_key)} state=${scalar(record.state)} updated=${ts(record.updated_at)} value="${oneLine(record.value)}" evidence=${ids(record.evidence_stream_entry_ids)}`;
}

function socialProfileRow(record: RecordLike): string {
  return `- entity=${scalar(record.entity_id)} trust=${scalar(record.trust)} attachment=${scalar(record.attachment)} interactions=${scalar(record.interaction_count)} commitments=${scalar(record.commitment_count)} updated=${ts(record.updated_at)} sentiment="${oneLine(record.sentiment_summary)}"`;
}

function socialEventRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} entity=${scalar(record.entity_id)} ts=${ts(record.ts)} kind=${scalar(record.kind)} trust_delta=${scalar(record.trust_delta)} attachment_delta=${scalar(record.attachment_delta)} valence=${scalar(record.valence)} provenance=${scalar(record.provenance)}`;
}

function moodStateRow(record: RecordLike): string {
  return `- session=${scalar(record.session_id)} updated=${ts(record.updated_at)} valence=${scalar(record.valence)} arousal=${scalar(record.arousal)} triggers=${ids(record.recent_triggers)}`;
}

function moodHistoryRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} session=${scalar(record.session_id)} ts=${ts(record.ts)} valence=${scalar(record.valence)} arousal=${scalar(record.arousal)} reason="${oneLine(record.trigger_reason, 120)}" provenance=${scalar(record.provenance)}`;
}

function skillRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} status=${scalar(record.status)} attempts=${scalar(record.attempts)} successes=${scalar(record.successes)} failures=${scalar(record.failures)} episodes=${ids(record.source_episode_ids)} applies_when="${oneLine(record.applies_when, 160)}" approach="${oneLine(record.approach, 160)}"`;
}

function skillContextStatsRow(record: RecordLike): string {
  return `- skill=${scalar(record.skill_id)} context=${scalar(record.context_key)} attempts=${scalar(record.attempts)} successes=${scalar(record.successes)} failures=${scalar(record.failures)} updated=${ts(record.updated_at)}`;
}

function proceduralEvidenceRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} classification=${scalar(record.classification)} grounded=${scalar(record.grounded)} consumed=${ts(record.consumed_at)} episodes=${ids(record.resolved_episode_ids)} audience=${scalar(record.audience_entity_id)} evidence="${oneLine(record.evidence_text)}"`;
}

function workingMemoryRow(record: RecordLike): string {
  return `- session=${scalar(record.session_id)} turn=${scalar(record.turn_counter)} updated=${ts(record.updated_at)} mode=${scalar(record.mode)} hot_entities=${ids(record.hot_entities)} pending_actions=${Array.isArray(record.pending_actions) ? record.pending_actions.length : 0} suppressed=${Array.isArray(record.suppressed) ? record.suppressed.length : 0} pending_procedural_attempts=${Array.isArray(record.pending_procedural_attempts) ? record.pending_procedural_attempts.length : 0} discourse="${oneLine(record.discourse_state)}"`;
}

function reviewRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} kind=${scalar(record.kind)} status=${scalar(record.status)} created=${ts(record.created_at)} target=${scalar(record.refs)} reason="${oneLine(record.reason)}"`;
}

function auditRow(record: RecordLike): string {
  return `- id=${scalar(record.id)} process=${scalar(record.process)} action=${scalar(record.action)} applied=${ts(record.applied_at)} reverted=${scalar(record.reverted_at)} summary="${oneLine(record.summary)}"`;
}

function records(values: readonly unknown[]): RecordLike[] {
  return values
    .map((value) => asRecord(value))
    .filter((value): value is RecordLike => value !== null);
}

function streamSessions(entries: readonly StreamEntry[]): SessionId[] {
  return [...new Set(entries.map((entry) => entry.session_id))];
}

function snapshotNowMs(deps: BorgDependencies | null): number {
  const clock = asRecord(deps)?.clock;
  const now = asRecord(clock)?.now;

  if (typeof now === "function") {
    const value = now.call(clock);

    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }

  return Date.now();
}

export async function buildMemorySnapshotMarkdown(
  options: BuildMemorySnapshotOptions,
): Promise<string> {
  const borg = options.transport.getBorg();
  const deps = borgDeps(options.transport);
  const nowMs = snapshotNowMs(deps);
  const auditTranscript = await options.transport.readAuditTranscript();
  const sessionIds = [
    ...new Set([
      ...(options.sessionIds ?? []),
      ...streamSessions(auditTranscript.map((entry) => entry.entry)),
    ]),
  ];

  const episodeList = await borg.episodic.list({ limit: LARGE_LIMIT });
  const episodes = records(episodeList.items);
  const semanticNodes = records(
    await borg.semantic.nodes.list({ includeArchived: true, limit: LARGE_LIMIT }),
  );
  const semanticEdges = records(await borg.semantic.edges.list({ includeInvalid: true }));
  const values = records(borg.self.values.list());
  const goals = flattenGoalRows(records(borg.self.goals.list({})));
  const traits = records(borg.self.traits.list());
  const periods = records(borg.self.autobiographical.listPeriods({ limit: LARGE_LIMIT }));
  const currentPeriod = borg.self.autobiographical.currentPeriod();
  const growthMarkers = records(borg.self.growthMarkers.list({ limit: LARGE_LIMIT }));
  const openQuestions = records(borg.self.openQuestions.list({ limit: LARGE_LIMIT }));
  const identityEvents = records(borg.identity.listEvents({ limit: LARGE_LIMIT }));
  const commitments = records(borg.commitments.list({ activeOnly: false }));
  const actions = records(borg.actions.list());
  const entities = deps === null ? [] : records(deps.entityRepository.list());
  const relationalSlots =
    deps === null ? [] : records(deps.relationalSlotRepository.list({ limit: LARGE_LIMIT }));
  const relationalCounts = deps === null ? null : deps.relationalSlotRepository.countByState();
  const socialProfiles = deps === null ? [] : records(deps.socialRepository.list(LARGE_LIMIT));
  const socialEvents = deps === null ? [] : records(deps.socialRepository.listEvents());
  const moodStates = deps === null ? [] : records(deps.moodRepository.listStates());
  const moodHistory =
    deps === null
      ? []
      : records(
          sessionIds.flatMap((sessionId) =>
            deps.moodRepository.history(sessionId, { limit: LARGE_LIMIT }),
          ),
        );
  const skills = records(borg.skills.list(LARGE_LIMIT));
  const skillContextStats =
    deps === null
      ? []
      : records(
          [
            ...deps.skillRepository
              .batchListContextStatsForSkills(
                skills
                  .map((skill) => skill.id)
                  .filter(
                    (
                      id,
                    ): id is Parameters<
                      BorgDependencies["skillRepository"]["batchListContextStatsForSkills"]
                    >[0][number] => typeof id === "string",
                  ),
              )
              .values(),
          ].flat(),
        );
  const proceduralEvidence =
    deps === null ? [] : records(deps.proceduralEvidenceRepository.list(LARGE_LIMIT));
  const workingMemory = records(sessionIds.map((sessionId) => borg.workmem.load(sessionId)));
  const reviewItems = records(borg.review.list({ openOnly: true }));
  const auditRows = records(borg.audit.list({}).slice(0, LARGE_LIMIT));

  const sections: SnapshotSection[] = [
    {
      title: "Scope And Counts",
      rows: [
        `- stream_entries=${auditTranscript.length} sessions=${sessionIds.length} episodes=${episodes.length} semantic_nodes=${semanticNodes.length} semantic_edges=${semanticEdges.length}`,
        `- values=${values.length} goals=${goals.length} traits=${traits.length} periods=${periods.length} growth_markers=${growthMarkers.length} open_questions=${openQuestions.length} identity_events=${identityEvents.length}`,
        `- actions=${actions.length} commitments=${commitments.length} entities=${entities.length} relational_slots=${relationalSlots.length} social_profiles=${socialProfiles.length} mood_states=${moodStates.length} skills=${skills.length} procedural_evidence=${proceduralEvidence.length}`,
        `- relational_slot_counts=${relationalCounts === null ? "unavailable" : scalar(relationalCounts)}`,
      ],
      empty: "No snapshot counts available.",
    },
    {
      title: "Stream Transcript",
      rows: rankedTranscriptRows(auditTranscript),
      empty: "No stream transcript entries recorded.",
    },
    {
      title: "Episodic Memory",
      rows: rankedRecords(episodes, {
        timeFields: ["updated_at", "end_time", "start_time", "created_at"],
        salienceFields: ["significance", "confidence"],
      }).map(episodeRow),
      empty: "No episodes recorded.",
    },
    {
      title: "Semantic Nodes",
      rows: rankedRecords(semanticNodes, {
        timeFields: ["updated_at", "created_at"],
        salienceFields: ["confidence"],
      }).map(semanticNodeRow),
      empty: "No semantic nodes recorded.",
    },
    {
      title: "Semantic Edges",
      rows: rankedRecords(semanticEdges, {
        timeFields: ["last_verified_at", "created_at", "valid_from"],
        salienceFields: ["confidence"],
      }).map(semanticEdgeRow),
      empty: "No semantic edges recorded.",
    },
    {
      title: "Identity And Self",
      rows: [
        ...rankedRecords(values, {
          timeFields: ["updated_at", "created_at"],
          salienceFields: ["priority", "confidence"],
        }).map(valueRow),
        ...rankedRecords(traits, {
          timeFields: ["updated_at", "created_at"],
          salienceFields: ["strength", "confidence"],
        }).map(traitRow),
        ...rankedRecords(periods, {
          timeFields: ["end_ts", "start_ts"],
        }).map(periodRow),
        ...(currentPeriod === null
          ? []
          : [
              `- current_period=${scalar((currentPeriod as unknown as RecordLike).id)} label="${oneLine((currentPeriod as unknown as RecordLike).label, 120)}"`,
            ]),
        ...rankedRecords(growthMarkers, {
          timeFields: ["ts"],
          salienceFields: ["confidence"],
        }).map(growthMarkerRow),
        ...rankedRecords(identityEvents, {
          timeFields: ["ts"],
        }).map(identityEventRow),
      ],
      empty: "No identity/self records recorded.",
    },
    {
      title: "Goals And Open Questions",
      rows: [
        ...rankedRecords(goals, {
          timeFields: ["last_progress_ts", "updated_at", "created_at"],
          salienceFields: ["priority"],
        }).map(goalRow),
        ...rankedRecords(openQuestions, {
          timeFields: ["updated_at", "created_at"],
          salienceFields: ["urgency"],
        }).map(openQuestionRow),
      ],
      empty: "No goals or open questions recorded.",
    },
    {
      title: "Commitments",
      rows: commitmentLifecycleRows({
        commitments,
        nowMs,
        countsSource: borg.commitments,
      }),
      empty: "No commitments recorded.",
    },
    {
      title: "Actions",
      rows: rankedRecords(actions, {
        timeFields: ["updated_at", "completed_at", "created_at"],
        salienceFields: ["confidence"],
      }).map(actionRow),
      empty: "No action records recorded.",
    },
    {
      title: "Relational And Social",
      rows: [
        ...rankedRecords(entities, {
          timeFields: ["updated_at", "created_at"],
        }).map(entityRow),
        ...rankedRecords(relationalSlots, {
          timeFields: ["updated_at", "created_at"],
        }).map(relationalSlotRow),
        ...rankedRecords(socialProfiles, {
          timeFields: ["updated_at"],
          salienceFields: ["interaction_count", "commitment_count"],
        }).map(socialProfileRow),
        ...rankedRecords(socialEvents, {
          timeFields: ["ts"],
        }).map(socialEventRow),
      ],
      empty: "No relational or social records recorded.",
    },
    {
      title: "Affective State",
      rows: [
        ...rankedRecords(moodStates, {
          timeFields: ["updated_at"],
        }).map(moodStateRow),
        ...rankedRecords(moodHistory, {
          timeFields: ["ts"],
        }).map(moodHistoryRow),
      ],
      empty: "No affective records recorded.",
    },
    {
      title: "Procedural Memory",
      rows: [
        ...rankedRecords(skills, {
          timeFields: ["updated_at", "created_at"],
        }).map(skillRow),
        ...rankedRecords(skillContextStats, {
          timeFields: ["updated_at"],
        }).map(skillContextStatsRow),
        ...rankedRecords(proceduralEvidence, {
          timeFields: ["created_at", "consumed_at"],
        }).map(proceduralEvidenceRow),
      ],
      empty: "No procedural records recorded.",
    },
    {
      title: "Working Memory",
      rows: rankedRecords(workingMemory, {
        timeFields: ["updated_at"],
      }).map(workingMemoryRow),
      empty: "No working memory sessions recorded.",
    },
    {
      title: "Review And Audit Diagnostics",
      rows: [
        ...rankedRecords(reviewItems, {
          timeFields: ["updated_at", "created_at"],
        }).map(reviewRow),
        ...rankedRecords(auditRows, {
          timeFields: ["applied_at"],
        }).map(auditRow),
      ],
      empty: "No open review or audit diagnostics recorded.",
    },
  ];

  return renderSnapshotSections(sections);
}
