import {
  actionMemoryDisclosureLabel,
  openQuestionMemoryDisclosureLabel,
  goalMemoryDisclosureLabel,
  observedEventMemoryDisclosureLabel,
} from "../memory/common/disclosure-serializers.js";
import type { PerceptionResult, TemporalCue } from "./types.js";
import type {
  ActivityAutobiographicalSourceEvent,
  ActivityRepository,
} from "../memory/activity/index.js";
import type { ActionRecord, ActionRepository } from "../memory/actions/index.js";
import type { EpisodicRepository } from "../memory/episodic/index.js";
import type { ObservedEventRepository } from "../memory/observed-events/index.js";
import { OBSERVED_EVENT_DISCLOSURE_CLASSES } from "../memory/observed-events/types.js";
import type { SelfDecisionRepository } from "../memory/self-decisions/index.js";
import type {
  AutobiographicalPeriod,
  AutobiographicalRepository,
  GoalRecord,
  GoalTreeNode,
  OpenQuestion,
  OpenQuestionsRepository,
  GoalsRepository,
} from "../memory/self/index.js";
import type { Provenance } from "../memory/common/provenance.js";
import {
  memoryDisclosureLabelFromEpisodeAccess,
  selfPrivateMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
} from "../retrieval/index.js";
import type { SessionAudienceRole, SessionRecord, SessionsRepository } from "../sessions/index.js";
import type { StreamEntry, StreamEntryKind, StreamReader } from "../stream/index.js";
import { OUTBOUND_POST_TOOL_NAME } from "../tools/internal/outbound-post-name.js";
import type { Clock } from "../util/clock.js";
import type { EntityId, EpisodeId, SessionId, StreamEntryId } from "../util/ids.js";
import { stripToolCallScaffolding } from "../util/prompt-tags.js";
import { formatRelativeAge } from "../util/relative-time.js";
import { utf16SafeSuffixStart } from "../util/utf16-boundary.js";

const DEFAULT_AUTOBIOGRAPHICAL_RECALL_WINDOW_MS = 7 * 24 * 60 * 60_000;
const DEFAULT_AUTOBIOGRAPHICAL_RECALL_SESSION_CAP = 24;
const DEFAULT_AUTOBIOGRAPHICAL_RECALL_SOURCE_CAP = 10;
const DEFAULT_AUTOBIOGRAPHICAL_RECALL_TOTAL_CAP = 48;
// Matches the budget reflection already gives this field, so a single retained note is rarely cut.
const GOAL_PROGRESS_NOTES_CHAR_BUDGET = 1_200;

const AUTOBIOGRAPHICAL_STREAM_KINDS = [
  "thought",
  "agent_suppressed",
  "agent_observed",
  "tool_call",
  "tool_result",
] as const satisfies readonly StreamEntryKind[];

// Aggregates structured autobiographical evidence for the finalizer. It intentionally does not
// write a first-person narrative; the model narrates from these labeled source records.
export type AutobiographicalRecallWindowSource = "perception_temporal_cue" | "recent_default";

export type AutobiographicalRecallWindow = {
  startMs: number;
  endMs: number;
  label: string;
  source: AutobiographicalRecallWindowSource;
};

export type AutobiographicalRecallSourceKind =
  | "activity"
  | "self_decision"
  | "stream_reflection"
  | "silence_decision"
  | "outbound_attempt"
  | "observed_presence"
  | "episode"
  | "observed_social_event"
  | "open_question"
  | "goal"
  | "action"
  | "autobiographical_period";

export type AutobiographicalRecallCandidateCount =
  | {
      candidateCount: number;
      candidateCountLowerBound?: never;
    }
  | {
      candidateCount?: never;
      candidateCountLowerBound: number;
    };

export type AutobiographicalRecallCapMetadata = {
  sourceGroup?: AutobiographicalRecallCandidateCount & {
    renderedCount: number;
    candidateScope?: "scanned_sessions";
  };
  total?: {
    candidateCount: number;
    renderedCount: number;
    candidateScope: "post_source_caps";
  };
};

export type AutobiographicalRecallEvidenceItem = {
  id: string;
  kind: AutobiographicalRecallSourceKind;
  groupId: string;
  groupLabel: string;
  occurredAt: number;
  relativeAge: string;
  score: number;
  text: string;
  disclosureLabel: MemoryDisclosureLabel;
  sourceStreamEntryIds: readonly StreamEntryId[];
  sourceEpisodeIds: readonly EpisodeId[];
  metadata: Record<string, unknown>;
  capMetadata?: AutobiographicalRecallCapMetadata;
};

export type AutobiographicalRecallResult = {
  window: AutobiographicalRecallWindow;
  evidence: readonly AutobiographicalRecallEvidenceItem[];
};

export type AutobiographicalRecallServiceOptions = {
  clock: Clock;
  activityRepository?: Partial<Pick<ActivityRepository, "listRecentGlobalEvents">>;
  selfDecisionRepository?: Pick<SelfDecisionRepository, "listRecentAutonomousSelfPrivate"> &
    Partial<Pick<SelfDecisionRepository, "countAutonomousSelfPrivateDecisions">>;
  observedEventRepository?: Pick<ObservedEventRepository, "listRecentGlobal">;
  episodicRepository?: Partial<Pick<EpisodicRepository, "listRecentForCognition">>;
  actionRepository?: Pick<ActionRepository, "list">;
  goalsRepository?: Pick<GoalsRepository, "list">;
  openQuestionsRepository?: Pick<OpenQuestionsRepository, "list">;
  autobiographicalRepository?: Pick<AutobiographicalRepository, "listPeriods">;
  sessionsRepository?: Pick<SessionsRepository, "list">;
  createStreamReader?: (sessionId: SessionId) => StreamReader;
  sourceCap?: number;
  sessionCap?: number;
  totalCap?: number;
};

export type AutobiographicalRecallInput = {
  sessionId: SessionId;
  temporalCue: TemporalCue | null;
  isSelfAudience: boolean;
  sessionAudienceRole: SessionAudienceRole;
  perceptionMode: PerceptionResult["mode"];
};

type AddItemInput = Omit<AutobiographicalRecallEvidenceItem, "relativeAge" | "capMetadata">;

type CandidateCountEstimate = {
  count: number;
  precision: "exact" | "lower_bound";
};

type GroupSelection = {
  candidateCount: CandidateCountEstimate;
  selectedCount: number;
  candidateScope?: "scanned_sessions";
};

type RecordGroupSelection = (input: {
  groupId: string;
  candidateCount: CandidateCountEstimate;
  selectedCount: number;
  candidateScope?: "scanned_sessions";
}) => void;

function stableGroupLabel(groupId: string): string {
  switch (groupId) {
    case "self_decisions":
      return "Self-decisions and autonomous choices";
    case "stream_reflection":
      return "Reflection and silence markers";
    case "outbound":
      return "Outbound attempts and observations";
    case "episodes":
      return "Recent episodic memory";
    case "observed_social_events":
      return "Observed social events";
    case "open_questions":
      return "Open questions considered";
    case "goals":
      return "Goals touched";
    case "autobiographical_periods":
      return "Autobiographical periods";
    default:
      if (groupId.startsWith("activity:")) {
        return `Session activity: ${groupId.slice("activity:".length)}`;
      }
      return groupId;
  }
}

function sanitizePromptText(value: unknown, maxChars = 520): string {
  const raw = typeof value === "string" ? value : JSON.stringify(value ?? null);
  const normalized = stripToolCallScaffolding(raw).replace(/\s+/g, " ").trim();

  if (normalized.length <= maxChars) {
    return normalized;
  }

  return `${normalized.slice(0, maxChars - 3).trimEnd()}...`;
}

// progress_notes is an append-only log: newest note last, one per line. Head truncation therefore
// drops exactly the note that describes the most recent act and keeps the oldest ones, so retain
// the tail instead, aligned to note boundaries, using the same elision marker as reflection.
function sanitizeAppendedPromptText(
  value: string,
  maxChars = GOAL_PROGRESS_NOTES_CHAR_BUDGET,
): string {
  const notes = stripToolCallScaffolding(value)
    .split(/\n+/)
    .map((note) => note.replace(/\s+/g, " ").trim())
    .filter((note) => note.length > 0);
  const normalized = notes.join(" ");

  if (normalized.length <= maxChars) {
    return normalized;
  }

  const renderMarker = (retainedTailChars: number) =>
    `[older progress_notes elided; total_chars=${normalized.length}; retained_tail_chars=${retainedTailChars}] `;
  // Reserving the widest possible retained-count field keeps the final marker within budget.
  const tailBudget = Math.max(0, maxChars - renderMarker(normalized.length).length);
  const retained: string[] = [];
  let retainedChars = 0;

  for (let index = notes.length - 1; index >= 0; index -= 1) {
    const note = notes[index] ?? "";
    const nextChars = retainedChars === 0 ? note.length : retainedChars + 1 + note.length;

    if (nextChars > tailBudget) {
      break;
    }

    retained.unshift(note);
    retainedChars = nextChars;
  }

  // A single note wider than the budget still yields its tail rather than its head.
  const tail =
    retained.length > 0
      ? retained.join(" ")
      : normalized.slice(utf16SafeSuffixStart(normalized, normalized.length - tailBudget));
  return `${renderMarker(tail.length)}${tail}`;
}

function boundedCap(value: number | undefined, fallback: number): number {
  return Math.max(1, Math.floor(value ?? fallback));
}

function selectWindowEligibleCandidatesWithinSourceCap<T>(
  eligibleCandidates: readonly T[],
  sourceCap: number,
): T[] {
  // Window eligibility must precede this slice: otherwise out-of-window rows consume source
  // slots, hiding in-window evidence and making the group's candidate/rendered counts dishonest.
  return eligibleCandidates.slice(0, sourceCap);
}

function boundedFetchCandidateCount(input: {
  fetchedCount: number;
  eligibleCount: number;
  fetchLimit: number;
}): CandidateCountEstimate {
  return {
    count: input.eligibleCount,
    precision: input.fetchedCount < input.fetchLimit ? "exact" : "lower_bound",
  };
}

function mergeCandidateCounts(
  left: CandidateCountEstimate,
  right: CandidateCountEstimate,
): CandidateCountEstimate {
  return {
    count: left.count + right.count,
    precision: left.precision === "exact" && right.precision === "exact" ? "exact" : "lower_bound",
  };
}

function recordGroupedSelection<T>(input: {
  candidates: readonly T[];
  selected: readonly T[];
  candidateCount: CandidateCountEstimate;
  groupIdFor: (value: T) => string;
  candidateScope?: "scanned_sessions";
  recordGroupSelection: RecordGroupSelection;
}): void {
  const candidateCounts = new Map<string, number>();
  const selectedCounts = new Map<string, number>();

  for (const candidate of input.candidates) {
    const groupId = input.groupIdFor(candidate);
    candidateCounts.set(groupId, (candidateCounts.get(groupId) ?? 0) + 1);
  }

  for (const candidate of input.selected) {
    const groupId = input.groupIdFor(candidate);
    selectedCounts.set(groupId, (selectedCounts.get(groupId) ?? 0) + 1);
  }

  for (const [groupId, candidateCount] of candidateCounts) {
    input.recordGroupSelection({
      groupId,
      candidateCount: {
        count: candidateCounts.size === 1 ? input.candidateCount.count : candidateCount,
        precision: input.candidateCount.precision,
      },
      selectedCount: selectedCounts.get(groupId) ?? 0,
      ...(input.candidateScope === undefined ? {} : { candidateScope: input.candidateScope }),
    });
  }
}

function publicCandidateCount(
  candidateCount: CandidateCountEstimate,
): AutobiographicalRecallCandidateCount {
  return candidateCount.precision === "exact"
    ? { candidateCount: candidateCount.count }
    : { candidateCountLowerBound: candidateCount.count };
}

function recencyScore(occurredAt: number, window: AutobiographicalRecallWindow): number {
  const span = Math.max(1, window.endMs - window.startMs);
  const age = Math.max(0, window.endMs - occurredAt);
  return Math.max(0, Math.min(1, 1 - age / span));
}

export function shouldRecallAutobiographicalEvidence(input: AutobiographicalRecallInput): boolean {
  return (
    input.temporalCue !== null ||
    input.isSelfAudience ||
    input.sessionAudienceRole === "operator" ||
    input.perceptionMode === "reflective"
  );
}

export function resolveAutobiographicalRecallWindow(
  temporalCue: TemporalCue | null,
  nowMs: number,
): AutobiographicalRecallWindow {
  if (temporalCue === null) {
    return {
      startMs: nowMs - DEFAULT_AUTOBIOGRAPHICAL_RECALL_WINDOW_MS,
      endMs: nowMs,
      label: "recent default",
      source: "recent_default",
    };
  }

  const endMs = Math.min(temporalCue.untilTs ?? nowMs, nowMs);
  const startMs = temporalCue.sinceTs ?? endMs - DEFAULT_AUTOBIOGRAPHICAL_RECALL_WINDOW_MS;

  return {
    startMs,
    endMs,
    label: temporalCue.label ?? "perception temporal cue",
    source: "perception_temporal_cue",
  };
}

function withinWindow(timestamp: number, window: AutobiographicalRecallWindow): boolean {
  return timestamp >= window.startMs && timestamp <= window.endMs;
}

function sourceLabelForActivity(event: ActivityAutobiographicalSourceEvent): string {
  return `${event.sessionSourceType}/${event.sessionAudienceRole}`;
}

function activityDisclosureLabel(
  event: ActivityAutobiographicalSourceEvent,
): MemoryDisclosureLabel {
  return selfPrivateMemoryDisclosureLabel(
    [event.audienceEntityId, ...event.participantEntityIds].filter(
      (id): id is EntityId => id !== null,
    ),
  );
}

function streamDisclosureLabel(entry: StreamEntry): MemoryDisclosureLabel {
  return selfPrivateMemoryDisclosureLabel(
    [entry.sender_entity_id, entry.reply_target_entity_id].filter(
      (id): id is EntityId => id !== null,
    ),
  );
}

function streamItemKind(entry: StreamEntry): AutobiographicalRecallSourceKind | null {
  if (entry.kind === "thought") {
    return "stream_reflection";
  }

  if (entry.kind === "agent_suppressed") {
    return "silence_decision";
  }

  if (entry.kind === "agent_observed") {
    return "observed_presence";
  }

  if (entry.kind === "tool_call") {
    const content = entry.content;

    if (
      typeof content === "object" &&
      content !== null &&
      "tool_name" in content &&
      content.tool_name === OUTBOUND_POST_TOOL_NAME
    ) {
      return "outbound_attempt";
    }
  }

  return null;
}

function streamGroupId(kind: AutobiographicalRecallSourceKind): string {
  return kind === "outbound_attempt" || kind === "observed_presence"
    ? "outbound"
    : "stream_reflection";
}

function actionTimestamp(action: ActionRecord): number {
  return (
    action.completed_at ??
    action.committed_at ??
    action.considering_at ??
    action.updated_at ??
    action.created_at
  );
}

function flattenGoals(goals: readonly GoalTreeNode[]): GoalRecord[] {
  return goals.flatMap((goal) => [goal, ...flattenGoals(goal.children)]);
}

type StreamEventCandidate = {
  entry: StreamEntry;
  kind: AutobiographicalRecallSourceKind;
  session: SessionRecord;
};

function streamContentRecord(entry: StreamEntry): Record<string, unknown> | null {
  return entry.content !== null &&
    typeof entry.content === "object" &&
    !Array.isArray(entry.content)
    ? (entry.content as Record<string, unknown>)
    : null;
}

function streamToolCallId(entry: StreamEntry): string | null {
  const callId = streamContentRecord(entry)?.call_id;
  return typeof callId === "string" && callId.length > 0 ? callId : null;
}

function streamToolResultKey(sessionId: SessionId, callId: string): string {
  return `${sessionId}:${callId}`;
}

// `outcome` below is derived from the dispatcher's `ok` flag, which records only that the tool
// call returned a schema-valid result -- not that anything reached a destination. A target session
// that was busy, a turn that emitted nothing, a session with no wired connector, and a connector
// that threw all return `ok: true` and so all read as "succeeded". The discriminator between them
// lives in the same tool result, so surface it rather than leaving one word to carry five states.
function outboundDeliveryFields(result: Record<string, unknown> | null): Record<string, unknown> {
  const output = result?.output;
  const outbound =
    typeof output === "object" && output !== null
      ? (output as Record<string, unknown>).outbound
      : undefined;

  if (typeof outbound !== "object" || outbound === null) {
    return {};
  }

  const record = outbound as Record<string, unknown>;
  const delivery = record.delivery;
  const deliveryStatus =
    typeof delivery === "object" && delivery !== null
      ? (delivery as Record<string, unknown>).status
      : undefined;
  // Older results predate `delivery_outcome`; absence is a schema generation, not a failure.
  const deliveryOutcome = record.delivery_outcome;
  const deliveryOutcomeState =
    typeof deliveryOutcome === "object" && deliveryOutcome !== null
      ? (deliveryOutcome as Record<string, unknown>).state
      : undefined;

  return {
    ...(typeof record.emitted === "boolean" ? { emitted: record.emitted } : {}),
    ...(typeof deliveryOutcomeState === "string"
      ? { delivery_outcome_state: deliveryOutcomeState }
      : {}),
    ...(typeof deliveryStatus === "string" ? { delivery_status: deliveryStatus } : {}),
  };
}

function outboundAttemptMetadata(
  callEntry: StreamEntry,
  resultEntry: StreamEntry | undefined,
): Record<string, unknown> {
  const call = streamContentRecord(callEntry);
  const callId = streamToolCallId(callEntry);
  if (call?.skipped === true) {
    return {
      status: "not_attempted",
      outcome: "skipped",
      ...(callId === null ? {} : { call_id: callId }),
      ...(typeof call.skip_reason === "string" ? { skip_reason: call.skip_reason } : {}),
    };
  }
  const result = resultEntry === undefined ? null : streamContentRecord(resultEntry);
  return {
    status: "attempted",
    outcome: result?.ok === true ? "succeeded" : result?.ok === false ? "failed" : "unknown",
    ...outboundDeliveryFields(result),
    ...(callId === null ? {} : { call_id: callId }),
    ...(resultEntry === undefined ? {} : { tool_result_stream_id: resultEntry.id }),
  };
}

function provenanceEpisodeIds(provenance: Provenance | null): EpisodeId[] {
  if (provenance === null) {
    return [];
  }

  if (provenance.kind === "episodes") {
    return provenance.episode_ids;
  }

  if (provenance.kind === "online_reflector") {
    return provenance.evidence_episode_ids;
  }

  return [];
}

function provenanceStreamIds(provenance: Provenance | null): StreamEntryId[] {
  if (provenance === null) {
    return [];
  }

  return provenance.kind === "online_reflector" ? provenance.evidence_stream_entry_ids : [];
}

export class AutobiographicalRecallService {
  constructor(private readonly options: AutobiographicalRecallServiceOptions) {}

  async recall(input: AutobiographicalRecallInput): Promise<AutobiographicalRecallResult | null> {
    if (!shouldRecallAutobiographicalEvidence(input)) {
      return null;
    }

    const nowMs = this.options.clock.now();
    const window = resolveAutobiographicalRecallWindow(input.temporalCue, nowMs);

    if (window.startMs > window.endMs) {
      return { window, evidence: [] };
    }

    const sourceCap = boundedCap(
      this.options.sourceCap,
      DEFAULT_AUTOBIOGRAPHICAL_RECALL_SOURCE_CAP,
    );
    const totalCap = boundedCap(this.options.totalCap, DEFAULT_AUTOBIOGRAPHICAL_RECALL_TOTAL_CAP);
    const items: AutobiographicalRecallEvidenceItem[] = [];
    const groupSelections = new Map<string, GroupSelection>();
    const recordGroupSelection: RecordGroupSelection = (selection) => {
      const current = groupSelections.get(selection.groupId);

      if (current === undefined) {
        groupSelections.set(selection.groupId, {
          candidateCount: selection.candidateCount,
          selectedCount: selection.selectedCount,
          ...(selection.candidateScope === undefined
            ? {}
            : { candidateScope: selection.candidateScope }),
        });
        return;
      }

      groupSelections.set(selection.groupId, {
        candidateCount: mergeCandidateCounts(current.candidateCount, selection.candidateCount),
        selectedCount: current.selectedCount + selection.selectedCount,
        ...(current.candidateScope === undefined && selection.candidateScope === undefined
          ? {}
          : { candidateScope: current.candidateScope ?? selection.candidateScope }),
      });
    };
    const addItem = (item: AddItemInput): void => {
      if (!withinWindow(item.occurredAt, window)) {
        return;
      }

      items.push({
        ...item,
        relativeAge: formatRelativeAge(item.occurredAt, nowMs),
      });
    };

    await this.collectActivity({ window, sourceCap, addItem, recordGroupSelection });
    this.collectSelfDecisions({ window, sourceCap, addItem, recordGroupSelection });
    await this.collectStreamEvents({ window, sourceCap, addItem, recordGroupSelection });
    await this.collectEpisodes({ window, sourceCap, addItem, recordGroupSelection });
    this.collectObservedEvents({ window, sourceCap, addItem, recordGroupSelection });
    this.collectOpenQuestions({ window, sourceCap, addItem, recordGroupSelection });
    this.collectGoals({ window, sourceCap, addItem, recordGroupSelection });
    this.collectAutobiographicalPeriods({ window, sourceCap, addItem, recordGroupSelection });
    this.collectActions({ window, sourceCap, addItem, recordGroupSelection });

    const rankedCandidates = items.sort(
      (left, right) =>
        right.score - left.score ||
        right.occurredAt - left.occurredAt ||
        left.kind.localeCompare(right.kind) ||
        left.id.localeCompare(right.id),
    );
    const ranked = rankedCandidates.slice(0, totalCap);
    const renderedCountsByGroup = new Map<string, number>();

    for (const item of ranked) {
      renderedCountsByGroup.set(item.groupId, (renderedCountsByGroup.get(item.groupId) ?? 0) + 1);
    }

    const annotatedGroups = new Set<string>();
    const totalCapBit = rankedCandidates.length > ranked.length;
    // Keep cap facts sparse: the first ranked item that survives for a group carries its counts.
    const evidence = ranked.map((item, index) => {
      const groupSelection = groupSelections.get(item.groupId);
      // A saturated fetch is itself a cap fact: its eligible count is only a floor even when
      // that floor happens to equal the number selected from the sampled rows.
      const sourceGroupCapBit =
        groupSelection !== undefined &&
        (groupSelection.candidateCount.precision === "lower_bound" ||
          groupSelection.candidateCount.count > groupSelection.selectedCount);
      const shouldAnnotateGroup = sourceGroupCapBit && !annotatedGroups.has(item.groupId);

      if (shouldAnnotateGroup) {
        annotatedGroups.add(item.groupId);
      }

      if (!shouldAnnotateGroup && (!totalCapBit || index !== 0)) {
        return item;
      }

      return {
        ...item,
        capMetadata: {
          ...(shouldAnnotateGroup && groupSelection !== undefined
            ? {
                sourceGroup: {
                  renderedCount: renderedCountsByGroup.get(item.groupId) ?? 0,
                  ...publicCandidateCount(groupSelection.candidateCount),
                  ...(groupSelection.candidateScope === undefined
                    ? {}
                    : { candidateScope: groupSelection.candidateScope }),
                },
              }
            : {}),
          ...(totalCapBit && index === 0
            ? {
                total: {
                  candidateCount: rankedCandidates.length,
                  renderedCount: ranked.length,
                  candidateScope: "post_source_caps" as const,
                },
              }
            : {}),
        },
      };
    });

    return {
      window,
      evidence,
    };
  }

  private async collectActivity(input: {
    window: AutobiographicalRecallWindow;
    sourceCap: number;
    addItem: (item: AddItemInput) => void;
    recordGroupSelection: RecordGroupSelection;
  }): Promise<void> {
    const repository = this.options.activityRepository;
    const fetchLimit = input.sourceCap + 1;
    const events =
      repository !== undefined && typeof repository.listRecentGlobalEvents === "function"
        ? repository.listRecentGlobalEvents({
            sinceMs: input.window.startMs,
            untilMs: input.window.endMs,
            limit: fetchLimit,
          })
        : [];
    const selected = selectWindowEligibleCandidatesWithinSourceCap(events, input.sourceCap);

    recordGroupedSelection({
      candidates: events,
      selected,
      candidateCount: boundedFetchCandidateCount({
        fetchedCount: events.length,
        eligibleCount: events.length,
        fetchLimit,
      }),
      groupIdFor: (event) => `activity:${sourceLabelForActivity(event)}`,
      recordGroupSelection: input.recordGroupSelection,
    });

    for (const event of selected) {
      const sourceLabel = sourceLabelForActivity(event);
      const groupId = `activity:${sourceLabel}`;
      input.addItem({
        id: `activity:${event.id}`,
        kind: "activity",
        groupId,
        groupLabel: stableGroupLabel(groupId),
        occurredAt: event.occurredAt,
        score: 0.55 + recencyScore(event.occurredAt, input.window) * 0.25,
        text: [
          `activity_kind=${event.kind}`,
          `source_type=${event.sessionSourceType}`,
          `audience_role=${event.sessionAudienceRole}`,
          `session_label=${sanitizePromptText(event.sessionLabel, 160)}`,
          `participant=${sanitizePromptText(event.participantLabel, 160)}`,
        ].join(" "),
        disclosureLabel: activityDisclosureLabel(event),
        sourceStreamEntryIds: event.sourceStreamEntryIds,
        sourceEpisodeIds: [],
        metadata: {
          event_kind: event.kind,
          session_source_type: event.sessionSourceType,
          session_audience_role: event.sessionAudienceRole,
          audience_entity_id: event.audienceEntityId,
          participant_entity_ids: [...event.participantEntityIds],
        },
      });
    }
  }

  private collectSelfDecisions(input: {
    window: AutobiographicalRecallWindow;
    sourceCap: number;
    addItem: (item: AddItemInput) => void;
    recordGroupSelection: RecordGroupSelection;
  }): void {
    const repository = this.options.selfDecisionRepository;
    const fetchLimit = input.sourceCap + 1;
    const fetchedEvents =
      repository !== undefined && typeof repository.listRecentAutonomousSelfPrivate === "function"
        ? repository.listRecentAutonomousSelfPrivate({
            sinceMs: input.window.startMs,
            limit: fetchLimit,
          })
        : [];
    const events = fetchedEvents.filter((item) => item.occurredAt <= input.window.endMs);
    const selected = selectWindowEligibleCandidatesWithinSourceCap(events, input.sourceCap);
    const candidateCount =
      repository !== undefined &&
      typeof repository.countAutonomousSelfPrivateDecisions === "function"
        ? {
            count: repository.countAutonomousSelfPrivateDecisions({
              sinceMs: input.window.startMs,
              untilMs: input.window.endMs,
            }),
            precision: "exact" as const,
          }
        : boundedFetchCandidateCount({
            fetchedCount: fetchedEvents.length,
            eligibleCount: events.length,
            fetchLimit,
          });

    recordGroupedSelection({
      candidates: events,
      selected,
      candidateCount,
      groupIdFor: () => "self_decisions",
      recordGroupSelection: input.recordGroupSelection,
    });

    for (const event of selected) {
      input.addItem({
        id: `self_decision:${event.occurredAt}:${event.triggerName}`,
        kind: "self_decision",
        groupId: "self_decisions",
        groupLabel: stableGroupLabel("self_decisions"),
        occurredAt: event.occurredAt,
        score: 0.7 + recencyScore(event.occurredAt, input.window) * 0.2,
        text: [
          `trigger_type=${event.triggerType}`,
          `trigger_name=${sanitizePromptText(event.triggerName, 160)}`,
          `decision=${sanitizePromptText(event.decisionSummary)}`,
          event.decisionRationale === null
            ? null
            : `rationale=${sanitizePromptText(event.decisionRationale)}`,
        ]
          .filter((part): part is string => part !== null)
          .join(" "),
        disclosureLabel: selfPrivateMemoryDisclosureLabel(),
        sourceStreamEntryIds: event.sourceStreamEntryIds,
        sourceEpisodeIds: [],
        metadata: {
          trigger_name: event.triggerName,
          trigger_type: event.triggerType,
        },
      });
    }
  }

  private async collectStreamEvents(input: {
    window: AutobiographicalRecallWindow;
    sourceCap: number;
    addItem: (item: AddItemInput) => void;
    recordGroupSelection: RecordGroupSelection;
  }): Promise<void> {
    if (
      this.options.sessionsRepository === undefined ||
      this.options.createStreamReader === undefined
    ) {
      return;
    }

    const sessionLimit = boundedCap(
      this.options.sessionCap,
      DEFAULT_AUTOBIOGRAPHICAL_RECALL_SESSION_CAP,
    );
    const sessionFetchLimit = sessionLimit + 1;
    const fetchedSessions = this.options.sessionsRepository.list({
      activeSince: input.window.startMs,
      limit: sessionFetchLimit,
    });
    const sessions = fetchedSessions.slice(0, sessionLimit);
    const candidates: StreamEventCandidate[] = [];
    const toolResults = new Map<string, StreamEntry>();

    for (const session of sessions) {
      for await (const entry of this.options.createStreamReader(session.session_id).iterate({
        sinceTs: input.window.startMs,
        untilTs: input.window.endMs,
        kinds: AUTOBIOGRAPHICAL_STREAM_KINDS,
      })) {
        if (entry.kind === "tool_result") {
          const callId = streamToolCallId(entry);
          if (callId !== null) {
            toolResults.set(streamToolResultKey(session.session_id, callId), entry);
          }
          continue;
        }
        const kind = streamItemKind(entry);

        if (kind === null) {
          continue;
        }

        candidates.push({ entry, kind, session });
      }
    }

    const sortedCandidates = candidates.sort(
      (left, right) => right.entry.timestamp - left.entry.timestamp,
    );
    const selected = selectWindowEligibleCandidatesWithinSourceCap(
      sortedCandidates,
      input.sourceCap,
    );

    recordGroupedSelection({
      candidates: sortedCandidates,
      selected,
      candidateCount: {
        count: sortedCandidates.length,
        precision: fetchedSessions.length < sessionFetchLimit ? "exact" : "lower_bound",
      },
      groupIdFor: (candidate) => streamGroupId(candidate.kind),
      candidateScope: "scanned_sessions",
      recordGroupSelection: input.recordGroupSelection,
    });

    for (const candidate of selected) {
      const { entry, kind, session } = candidate;
      const groupId = streamGroupId(kind);
      const callId = kind === "outbound_attempt" ? streamToolCallId(entry) : null;
      const toolResult =
        callId === null
          ? undefined
          : toolResults.get(streamToolResultKey(session.session_id, callId));
      input.addItem({
        id: `stream:${entry.id}`,
        kind,
        groupId,
        groupLabel: stableGroupLabel(groupId),
        occurredAt: entry.timestamp,
        score: 0.58 + recencyScore(entry.timestamp, input.window) * 0.18,
        text: [
          `stream_kind=${entry.kind}`,
          `session_source_type=${session.source_type}`,
          `audience_role=${session.audience_role}`,
          `content=${sanitizePromptText(entry.content)}`,
        ].join(" "),
        disclosureLabel: streamDisclosureLabel(entry),
        sourceStreamEntryIds: toolResult === undefined ? [entry.id] : [entry.id, toolResult.id],
        sourceEpisodeIds: [],
        metadata: {
          stream_kind: entry.kind,
          session_source_type: session.source_type,
          session_audience_role: session.audience_role,
          ...(kind === "outbound_attempt" ? outboundAttemptMetadata(entry, toolResult) : {}),
        },
      });
    }
  }

  private async collectEpisodes(input: {
    window: AutobiographicalRecallWindow;
    sourceCap: number;
    addItem: (item: AddItemInput) => void;
    recordGroupSelection: RecordGroupSelection;
  }): Promise<void> {
    const repository = this.options.episodicRepository;
    const fetchLimit = Math.max(input.sourceCap * 3, input.sourceCap + 1);
    const fetchedCandidates =
      repository !== undefined && typeof repository.listRecentForCognition === "function"
        ? await repository.listRecentForCognition({
            limit: fetchLimit,
          })
        : [];
    const candidates = fetchedCandidates
      .filter((item) => item.episode.start_time <= input.window.endMs)
      .filter((item) => withinWindow(item.episode.end_time, input.window));
    const selected = selectWindowEligibleCandidatesWithinSourceCap(candidates, input.sourceCap);

    recordGroupedSelection({
      candidates,
      selected,
      candidateCount: boundedFetchCandidateCount({
        fetchedCount: fetchedCandidates.length,
        eligibleCount: candidates.length,
        fetchLimit,
      }),
      groupIdFor: () => "episodes",
      recordGroupSelection: input.recordGroupSelection,
    });

    for (const candidate of selected) {
      input.addItem({
        id: `episode:${candidate.episode.id}`,
        kind: "episode",
        groupId: "episodes",
        groupLabel: stableGroupLabel("episodes"),
        occurredAt: candidate.episode.end_time,
        score:
          0.5 +
          recencyScore(candidate.episode.end_time, input.window) * 0.18 +
          candidate.episode.significance * 0.18,
        text: [
          `title=${sanitizePromptText(candidate.episode.title, 160)}`,
          `narrative=${sanitizePromptText(candidate.episode.narrative)}`,
          `tags=${candidate.episode.tags.join(",") || "none"}`,
        ].join(" "),
        disclosureLabel: memoryDisclosureLabelFromEpisodeAccess(candidate.episode),
        sourceStreamEntryIds: candidate.episode.source_stream_ids,
        sourceEpisodeIds: [candidate.episode.id],
        metadata: {
          episode_id: candidate.episode.id,
          significance: candidate.episode.significance,
          tags: [...candidate.episode.tags],
        },
      });
    }
  }

  private collectObservedEvents(input: {
    window: AutobiographicalRecallWindow;
    sourceCap: number;
    addItem: (item: AddItemInput) => void;
    recordGroupSelection: RecordGroupSelection;
  }): void {
    if (this.options.observedEventRepository === undefined) {
      return;
    }

    const fetchLimit = input.sourceCap + 1;
    const fetchedByDisclosureClass = OBSERVED_EVENT_DISCLOSURE_CLASSES.map((disclosureClass) =>
      this.options.observedEventRepository!.listRecentGlobal({
        disclosureClass,
        sinceMs: input.window.startMs,
        limit: fetchLimit,
      }),
    );
    const events = fetchedByDisclosureClass
      .flat()
      .filter((event) => event.lastSeenAt <= input.window.endMs)
      .sort((left, right) => right.lastSeenAt - left.lastSeenAt);
    const selected = selectWindowEligibleCandidatesWithinSourceCap(events, input.sourceCap);

    recordGroupedSelection({
      candidates: events,
      selected,
      candidateCount: {
        count: events.length,
        precision: fetchedByDisclosureClass.every((items) => items.length < fetchLimit)
          ? "exact"
          : "lower_bound",
      },
      groupIdFor: () => "observed_social_events",
      recordGroupSelection: input.recordGroupSelection,
    });

    for (const event of selected) {
      input.addItem({
        id: `observed_event:${event.id}`,
        kind: "observed_social_event",
        groupId: "observed_social_events",
        groupLabel: stableGroupLabel("observed_social_events"),
        occurredAt: event.lastSeenAt,
        score: 0.54 + recencyScore(event.lastSeenAt, input.window) * 0.2,
        text: [
          `stance=${event.stance}`,
          `taint=${event.taint}`,
          `belief_effect=${event.beliefEffect}`,
          `recurrence_count=${event.recurrenceCount}`,
          `interaction=${sanitizePromptText(event.interactionText)}`,
        ].join(" "),
        disclosureLabel: observedEventMemoryDisclosureLabel(event),
        sourceStreamEntryIds: event.sourceStreamEntryIds,
        sourceEpisodeIds: [],
        metadata: {
          observed_event_id: event.id,
          stance: event.stance,
          taint: event.taint,
          belief_effect: event.beliefEffect,
          recurrence_count: event.recurrenceCount,
        },
      });
    }
  }

  private collectOpenQuestions(input: {
    window: AutobiographicalRecallWindow;
    sourceCap: number;
    addItem: (item: AddItemInput) => void;
    recordGroupSelection: RecordGroupSelection;
  }): void {
    const fetchLimit = Math.max(input.sourceCap * 4, input.sourceCap + 1);
    const fetchedQuestions =
      this.options.openQuestionsRepository?.list({
        limit: fetchLimit,
      }) ?? [];
    const questions = fetchedQuestions.filter((item) =>
      withinWindow(item.last_touched, input.window),
    );
    const selected = selectWindowEligibleCandidatesWithinSourceCap(questions, input.sourceCap);

    recordGroupedSelection({
      candidates: questions,
      selected,
      candidateCount: boundedFetchCandidateCount({
        fetchedCount: fetchedQuestions.length,
        eligibleCount: questions.length,
        fetchLimit,
      }),
      groupIdFor: () => "open_questions",
      recordGroupSelection: input.recordGroupSelection,
    });

    for (const question of selected) {
      input.addItem({
        id: `open_question:${question.id}`,
        kind: "open_question",
        groupId: "open_questions",
        groupLabel: stableGroupLabel("open_questions"),
        occurredAt: question.last_touched,
        score:
          0.48 + question.urgency * 0.18 + recencyScore(question.last_touched, input.window) * 0.18,
        text: [
          `status=${question.status}`,
          `source=${question.source}`,
          `urgency=${question.urgency.toFixed(2)}`,
          `question=${sanitizePromptText(question.question)}`,
        ].join(" "),
        disclosureLabel: openQuestionMemoryDisclosureLabel(question),
        sourceStreamEntryIds: [
          ...question.resolution_evidence_stream_entry_ids,
          ...provenanceStreamIds(question.provenance),
        ],
        sourceEpisodeIds: [
          ...question.related_episode_ids,
          ...question.resolution_evidence_episode_ids,
          ...provenanceEpisodeIds(question.provenance),
        ],
        metadata: {
          open_question_id: question.id,
          status: question.status,
          source: question.source,
          urgency: question.urgency,
        },
      });
    }
  }

  private collectGoals(input: {
    window: AutobiographicalRecallWindow;
    sourceCap: number;
    addItem: (item: AddItemInput) => void;
    recordGroupSelection: RecordGroupSelection;
  }): void {
    const goals = flattenGoals(this.options.goalsRepository?.list({}) ?? []).filter((goal) =>
      withinWindow(goal.last_progress_ts ?? goal.created_at, input.window),
    );
    const selected = selectWindowEligibleCandidatesWithinSourceCap(goals, input.sourceCap);

    recordGroupedSelection({
      candidates: goals,
      selected,
      candidateCount: { count: goals.length, precision: "exact" },
      groupIdFor: () => "goals",
      recordGroupSelection: input.recordGroupSelection,
    });

    for (const goal of selected) {
      const occurredAt = goal.last_progress_ts ?? goal.created_at;
      input.addItem({
        id: `goal:${goal.id}`,
        kind: "goal",
        groupId: "goals",
        groupLabel: stableGroupLabel("goals"),
        occurredAt,
        score: 0.5 + goal.priority * 0.16 + recencyScore(occurredAt, input.window) * 0.16,
        text: [
          `status=${goal.status}`,
          `priority=${goal.priority.toFixed(2)}`,
          `counterparty_entity_id=${goal.counterparty_entity_id ?? "none"}`,
          `description=${sanitizePromptText(goal.description)}`,
          goal.terminal_condition === null
            ? null
            : `terminal_condition=${sanitizePromptText(goal.terminal_condition)}`,
          goal.progress_notes === null
            ? null
            : `progress=${sanitizeAppendedPromptText(goal.progress_notes)}`,
        ]
          .filter((part): part is string => part !== null)
          .join(" "),
        disclosureLabel: goalMemoryDisclosureLabel(goal),
        sourceStreamEntryIds: goal.source_stream_entry_ids ?? [],
        sourceEpisodeIds: provenanceEpisodeIds(goal.provenance),
        metadata: {
          goal_id: goal.id,
          status: goal.status,
          priority: goal.priority,
          counterparty_entity_id: goal.counterparty_entity_id ?? null,
          counterparty_semantics:
            "participant the responsibility runs toward; not owner or audience",
        },
      });
    }
  }

  private collectAutobiographicalPeriods(input: {
    window: AutobiographicalRecallWindow;
    sourceCap: number;
    addItem: (item: AddItemInput) => void;
    recordGroupSelection: RecordGroupSelection;
  }): void {
    const fetchLimit = input.sourceCap + 1;
    const fetchedPeriods =
      this.options.autobiographicalRepository?.listPeriods({
        fromTs: input.window.startMs,
        toTs: input.window.endMs,
        limit: fetchLimit,
      }) ?? [];
    const periods = fetchedPeriods.filter((period) =>
      withinWindow(period.last_updated, input.window),
    );
    const selected = selectWindowEligibleCandidatesWithinSourceCap(periods, input.sourceCap);

    recordGroupedSelection({
      candidates: periods,
      selected,
      candidateCount: boundedFetchCandidateCount({
        fetchedCount: fetchedPeriods.length,
        eligibleCount: periods.length,
        fetchLimit,
      }),
      groupIdFor: () => "autobiographical_periods",
      recordGroupSelection: input.recordGroupSelection,
    });

    for (const period of selected) {
      input.addItem({
        id: `autobiographical_period:${period.id}`,
        kind: "autobiographical_period",
        groupId: "autobiographical_periods",
        groupLabel: stableGroupLabel("autobiographical_periods"),
        occurredAt: period.last_updated,
        score: 0.45 + recencyScore(period.last_updated, input.window) * 0.14,
        text: [
          `label=${sanitizePromptText(period.label, 160)}`,
          `narrative=${sanitizePromptText(period.narrative)}`,
          `themes=${period.themes.join(",") || "none"}`,
        ].join(" "),
        disclosureLabel: selfPrivateMemoryDisclosureLabel(),
        sourceStreamEntryIds: [],
        sourceEpisodeIds: period.key_episode_ids,
        metadata: periodMetadata(period),
      });
    }
  }

  private collectActions(input: {
    window: AutobiographicalRecallWindow;
    sourceCap: number;
    addItem: (item: AddItemInput) => void;
    recordGroupSelection: RecordGroupSelection;
  }): void {
    const fetchLimit = Math.max(input.sourceCap * 3, input.sourceCap + 1);
    const fetchedActions =
      this.options.actionRepository?.list({
        actor: "borg",
        limit: fetchLimit,
      }) ?? [];
    const actions = fetchedActions.filter((action) =>
      withinWindow(actionTimestamp(action), input.window),
    );
    const selected = selectWindowEligibleCandidatesWithinSourceCap(actions, input.sourceCap);

    recordGroupedSelection({
      candidates: actions,
      selected,
      candidateCount: boundedFetchCandidateCount({
        fetchedCount: fetchedActions.length,
        eligibleCount: actions.length,
        fetchLimit,
      }),
      groupIdFor: () => "goals",
      recordGroupSelection: input.recordGroupSelection,
    });

    for (const action of selected) {
      const occurredAt = actionTimestamp(action);
      input.addItem({
        id: `action:${action.id}`,
        kind: "action",
        groupId: "goals",
        groupLabel: stableGroupLabel("goals"),
        occurredAt,
        score: 0.42 + recencyScore(occurredAt, input.window) * 0.14,
        text: [
          `action_state=${action.state}`,
          `description=${sanitizePromptText(action.description)}`,
        ].join(" "),
        disclosureLabel: actionMemoryDisclosureLabel(action),
        sourceStreamEntryIds: action.provenance_stream_entry_ids,
        sourceEpisodeIds: action.provenance_episode_ids,
        metadata: {
          action_id: action.id,
          state: action.state,
          goal_id: action.goal_id,
          open_question_id: action.open_question_id,
        },
      });
    }
  }
}

function periodMetadata(period: AutobiographicalPeriod): Record<string, unknown> {
  return {
    autobiographical_period_id: period.id,
    label: period.label,
    start_ts: period.start_ts,
    end_ts: period.end_ts,
    themes: [...period.themes],
  };
}
