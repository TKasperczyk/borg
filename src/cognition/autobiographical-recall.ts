import {
  actionMemoryDisclosureLabel,
  openQuestionMemoryDisclosureLabel,
  goalMemoryDisclosureLabel,
  observedEventMemoryDisclosureLabel,
} from "./disclosure-labels.js";
import type { PerceptionResult, TemporalCue } from "./types.js";
import type {
  ActivityAutobiographicalSourceEvent,
  ActivityRepository,
} from "../memory/activity/index.js";
import type { ActionRecord, ActionRepository } from "../memory/actions/index.js";
import type { EpisodicRepository } from "../memory/episodic/index.js";
import type {
  ObservedEventRepository,
} from "../memory/observed-events/index.js";
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

const DEFAULT_AUTOBIOGRAPHICAL_RECALL_WINDOW_MS = 7 * 24 * 60 * 60_000;
const DEFAULT_AUTOBIOGRAPHICAL_RECALL_SESSION_CAP = 24;
const DEFAULT_AUTOBIOGRAPHICAL_RECALL_SOURCE_CAP = 10;
const DEFAULT_AUTOBIOGRAPHICAL_RECALL_TOTAL_CAP = 48;

const AUTOBIOGRAPHICAL_STREAM_KINDS = [
  "thought",
  "agent_suppressed",
  "agent_observed",
  "tool_call",
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
};

export type AutobiographicalRecallResult = {
  window: AutobiographicalRecallWindow;
  evidence: readonly AutobiographicalRecallEvidenceItem[];
};

export type AutobiographicalRecallServiceOptions = {
  clock: Clock;
  activityRepository?: Partial<Pick<ActivityRepository, "listRecentGlobalEvents">>;
  selfDecisionRepository?: Pick<SelfDecisionRepository, "listRecentAutonomousSelfPrivate">;
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

type AddItemInput = Omit<AutobiographicalRecallEvidenceItem, "relativeAge">;

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

function boundedCap(value: number | undefined, fallback: number): number {
  return Math.max(1, Math.floor(value ?? fallback));
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
    const addItem = (item: AddItemInput): void => {
      if (!withinWindow(item.occurredAt, window)) {
        return;
      }

      items.push({
        ...item,
        relativeAge: formatRelativeAge(item.occurredAt, nowMs),
      });
    };

    await this.collectActivity({ window, sourceCap, addItem });
    this.collectSelfDecisions({ window, sourceCap, addItem });
    await this.collectStreamEvents({ window, sourceCap, addItem });
    await this.collectEpisodes({ window, sourceCap, addItem });
    this.collectObservedEvents({ window, sourceCap, addItem });
    this.collectOpenQuestions({ window, sourceCap, addItem });
    this.collectGoals({ window, sourceCap, addItem });
    this.collectAutobiographicalPeriods({ window, sourceCap, addItem });
    this.collectActions({ window, sourceCap, addItem });

    const ranked = items
      .sort(
        (left, right) =>
          right.score - left.score ||
          right.occurredAt - left.occurredAt ||
          left.kind.localeCompare(right.kind) ||
          left.id.localeCompare(right.id),
      )
      .slice(0, totalCap);

    return {
      window,
      evidence: ranked,
    };
  }

  private async collectActivity(input: {
    window: AutobiographicalRecallWindow;
    sourceCap: number;
    addItem: (item: AddItemInput) => void;
  }): Promise<void> {
    const repository = this.options.activityRepository;
    const events =
      repository !== undefined && typeof repository.listRecentGlobalEvents === "function"
        ? repository.listRecentGlobalEvents({
            sinceMs: input.window.startMs,
            untilMs: input.window.endMs,
            limit: input.sourceCap,
          })
        : [];

    for (const event of events) {
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
  }): void {
    const events =
      this.options.selfDecisionRepository?.listRecentAutonomousSelfPrivate({
        sinceMs: input.window.startMs,
        limit: input.sourceCap,
      }) ?? [];

    for (const event of events.filter((item) => item.occurredAt <= input.window.endMs)) {
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
  }): Promise<void> {
    if (
      this.options.sessionsRepository === undefined ||
      this.options.createStreamReader === undefined
    ) {
      return;
    }

    const sessions = this.options.sessionsRepository.list({
      activeSince: input.window.startMs,
      limit: boundedCap(this.options.sessionCap, DEFAULT_AUTOBIOGRAPHICAL_RECALL_SESSION_CAP),
    });
    const candidates: StreamEventCandidate[] = [];

    for (const session of sessions) {
      for await (const entry of this.options.createStreamReader(session.session_id).iterate({
        sinceTs: input.window.startMs,
        untilTs: input.window.endMs,
        kinds: AUTOBIOGRAPHICAL_STREAM_KINDS,
      })) {
        const kind = streamItemKind(entry);

        if (kind === null) {
          continue;
        }

        candidates.push({ entry, kind, session });
      }
    }

    for (const candidate of candidates
      .sort((left, right) => right.entry.timestamp - left.entry.timestamp)
      .slice(0, input.sourceCap)) {
      const { entry, kind, session } = candidate;
      const groupId = streamGroupId(kind);
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
        sourceStreamEntryIds: [entry.id],
        sourceEpisodeIds: [],
        metadata: {
          stream_kind: entry.kind,
          session_source_type: session.source_type,
          session_audience_role: session.audience_role,
        },
      });
    }
  }

  private async collectEpisodes(input: {
    window: AutobiographicalRecallWindow;
    sourceCap: number;
    addItem: (item: AddItemInput) => void;
  }): Promise<void> {
    const repository = this.options.episodicRepository;
    const candidates =
      repository !== undefined && typeof repository.listRecentForCognition === "function"
        ? await repository.listRecentForCognition({
            limit: Math.max(input.sourceCap * 3, input.sourceCap),
          })
        : [];

    for (const candidate of candidates
      .filter((item) => item.episode.end_time >= input.window.startMs)
      .filter((item) => item.episode.start_time <= input.window.endMs)
      .slice(0, input.sourceCap)) {
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
  }): void {
    if (this.options.observedEventRepository === undefined) {
      return;
    }

    const events = OBSERVED_EVENT_DISCLOSURE_CLASSES.flatMap((disclosureClass) =>
      this.options.observedEventRepository!.listRecentGlobal({
        disclosureClass,
        sinceMs: input.window.startMs,
        limit: input.sourceCap,
      }),
    )
      .filter((event) => event.lastSeenAt <= input.window.endMs)
      .sort((left, right) => right.lastSeenAt - left.lastSeenAt)
      .slice(0, input.sourceCap);

    for (const event of events) {
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
  }): void {
    const questions =
      this.options.openQuestionsRepository?.list({
        limit: Math.max(input.sourceCap * 4, input.sourceCap),
      }) ?? [];

    for (const question of questions
      .filter((item) => withinWindow(item.last_touched, input.window))
      .slice(0, input.sourceCap)) {
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
  }): void {
    const goals = flattenGoals(this.options.goalsRepository?.list({}) ?? [])
      .filter((goal) => withinWindow(goal.last_progress_ts ?? goal.created_at, input.window))
      .slice(0, input.sourceCap);

    for (const goal of goals) {
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
          `description=${sanitizePromptText(goal.description)}`,
          goal.progress_notes === null
            ? null
            : `progress=${sanitizePromptText(goal.progress_notes)}`,
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
        },
      });
    }
  }

  private collectAutobiographicalPeriods(input: {
    window: AutobiographicalRecallWindow;
    sourceCap: number;
    addItem: (item: AddItemInput) => void;
  }): void {
    const periods =
      this.options.autobiographicalRepository?.listPeriods({
        fromTs: input.window.startMs,
        toTs: input.window.endMs,
        limit: input.sourceCap,
      }) ?? [];

    for (const period of periods) {
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
  }): void {
    const actions =
      this.options.actionRepository
        ?.list({
          actor: "borg",
          limit: Math.max(input.sourceCap * 3, input.sourceCap),
        })
        .filter((action) => withinWindow(actionTimestamp(action), input.window))
        .slice(0, input.sourceCap) ?? [];

    for (const action of actions) {
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
