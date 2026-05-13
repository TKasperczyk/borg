import {
  ACTION_STATES,
  type ActionDescriptionSimilarityPair,
  type ActionRecord,
  type ActionRepository,
  type ActionState,
} from "../../memory/actions/index.js";
import type { CommitmentRecord, CommitmentRepository } from "../../memory/commitments/index.js";
import type { EntityRepository } from "../../memory/commitments/index.js";
import type {
  RelationalSlot,
  RelationalSlotRepository,
} from "../../memory/relational-slots/index.js";
import type {
  OpenQuestion,
  OpenQuestionsRepository,
  OpenQuestionStatus,
  GoalRecord,
  GoalsRepository,
} from "../../memory/self/index.js";
import type { ReviewQueueItem } from "../../memory/semantic/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type { EvidenceItem, RetrievedEpisode, RetrievedSemantic } from "../../retrieval/index.js";
import {
  activeSessionTranscriptEntries,
  isQuarantinedUserEntryMarker,
  loadSessionStreamEntries,
  type StreamEntry,
  type StreamReader,
  type TranscriptStreamEntry,
} from "../../stream/index.js";
import { estimatePromptTokens, stringifyPromptContent } from "../../util/token-estimate.js";
import type { EntityId, EpisodeId, SessionId, StreamEntryId } from "../../util/ids.js";
import type { FrameAnomalyClassification } from "../frame-anomaly/index.js";
import type { ActiveParticipant } from "../participants.js";
import { resolveSpeakerDisplayName, type SpeakerEntityRepository } from "../speaker-tags.js";
import {
  EVIDENCE_LEDGER_SECTION_DEFINITIONS,
  type EvidenceLedger,
  type EvidenceLedgerActor,
  type EvidenceLedgerEntry,
  type EvidenceLedgerSection,
  type EvidenceLedgerSectionId,
  type EvidenceLedgerSessionScope,
  type EvidenceLedgerSourceType,
  type EvidenceLedgerTaint,
  type EvidenceLedgerTraceSummary,
} from "./types.js";

const CURRENT_USER_TRUST_RANK = 100;
const TRANSCRIPT_TRUST_RANK = 95;
const COMMITMENT_TRUST_RANK = 82;
const DISCOURSE_TRUST_RANK = 80;
const QUARANTINE_TRUST_RANK = 78;
const ACTION_TRUST_RANK = 72;
const SLOT_TRUST_RANK = 70;
const RAW_STREAM_TRUST_RANK = 68;
const EPISODE_TRUST_RANK = 52;
const SEMANTIC_TRUST_RANK = 42;
const OPEN_QUESTION_TRUST_RANK = 38;
const WARM_RECALL_TRUST_RANK = 34;
const PRIOR_SESSION_TRUST_RANK_CAP = 30;

const RELATIONAL_SLOT_LEDGER_LIMIT = 64;
export const DEFAULT_ACTION_THREAD_RENDER_LIMIT = 12;
export const DEFAULT_ACTION_THREAD_SIMILARITY_THRESHOLD = 0.85;
export const DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT = 256;
const OLDER_ACTION_THREAD_SAMPLE_LIMIT = 4;
const OLDER_ACTION_THREAD_SAMPLE_MAX_CHARS = 80;
const TRANSCRIPT_RAW_TAIL_MIN_ENTRIES = 8;
const TRANSCRIPT_RAW_TAIL_BUDGET_FRACTION = 0.6;
const LIFECYCLE_OPEN_QUESTION_STATUSES = [
  "resolved",
  "abandoned",
] as const satisfies readonly OpenQuestionStatus[];

type ActionLedgerRepository = Pick<ActionRepository, "list"> &
  Partial<Pick<ActionRepository, "findSimilarDescriptionPairs">>;
type CommitmentLedgerRepository = Pick<CommitmentRepository, "list">;
type GoalLedgerRepository = Pick<GoalsRepository, "list">;

export type EvidenceLedgerBuilderOptions = {
  createStreamReader: (sessionId: SessionId) => StreamReader;
  relationalSlotRepository: Pick<RelationalSlotRepository, "list">;
  actionRepository: ActionLedgerRepository;
  commitmentRepository?: CommitmentLedgerRepository;
  goalsRepository?: GoalLedgerRepository;
  openQuestionsRepository?: Pick<OpenQuestionsRepository, "findByHandles">;
  currentSessionTranscriptTokenBudget: number;
  actionThreadRenderLimit?: number;
  actionThreadSimilarityThreshold?: number;
  actionThreadSourceRecordLimit?: number;
  entityRepository?: Pick<EntityRepository, "get">;
};

export type EvidenceLedgerBuildInput = {
  sessionId: SessionId;
  turnId?: string;
  audienceEntityId: EntityId | null;
  currentUserMessage: string;
  currentUserEntry?: StreamEntry;
  workingMemory: WorkingMemory;
  applicableCommitments: readonly CommitmentRecord[];
  retrievedEvidence: readonly EvidenceItem[];
  retrievedEpisodes: readonly RetrievedEpisode[];
  retrievedSemantic?: RetrievedSemantic | null;
  openQuestions: readonly OpenQuestion[];
  pendingCorrections: readonly ReviewQueueItem[];
  frameAnomaly?: FrameAnomalyClassification | null;
  activeParticipants?: readonly ActiveParticipant[];
};

type SectionBucket = {
  entries: EvidenceLedgerEntry[];
  seenEntryIds: Set<string>;
};

type SectionBuckets = Map<EvidenceLedgerSectionId, SectionBucket>;

type ScopeResolver = {
  currentSessionId: SessionId;
  streamEntriesById: ReadonlyMap<string, StreamEntry>;
  streamOrderById: ReadonlyMap<string, number>;
  episodeScopesById: ReadonlyMap<string, EvidenceLedgerSessionScope>;
  episodeSourceStreamIdsById: ReadonlyMap<string, readonly string[]>;
};

type ActionThread = {
  id: string;
  records: ActionRecord[];
  origin: ActionRecord;
  current: ActionRecord;
  scope: EvidenceLedgerSessionScope;
};

type TranscriptCompactionResult = {
  entries: EvidenceLedgerEntry[];
  rawStreamIds: Set<string>;
  compacted: boolean;
  originalTokenEstimate: number;
  compactedEntryCount: number;
  rawPreservedUserEntryCount: number;
};

function createSectionBuckets(): SectionBuckets {
  return new Map(
    EVIDENCE_LEDGER_SECTION_DEFINITIONS.map((section) => [
      section.id,
      {
        entries: [],
        seenEntryIds: new Set<string>(),
      },
    ]),
  );
}

function sectionBucket(
  sections: SectionBuckets,
  sectionId: EvidenceLedgerSectionId,
): SectionBucket {
  const bucket = sections.get(sectionId);

  if (bucket !== undefined) {
    return bucket;
  }

  const next: SectionBucket = {
    entries: [],
    seenEntryIds: new Set<string>(),
  };
  sections.set(sectionId, next);
  return next;
}

function sectionEntries(
  sections: SectionBuckets,
  sectionId: EvidenceLedgerSectionId,
): EvidenceLedgerEntry[] {
  return sectionBucket(sections, sectionId).entries;
}

function finalSections(sections: SectionBuckets): EvidenceLedgerSection[] {
  return EVIDENCE_LEDGER_SECTION_DEFINITIONS.map((definition) => ({
    id: definition.id,
    label: definition.label,
    entries: sectionEntries(sections, definition.id),
  }));
}

function addEntry(
  sections: SectionBuckets,
  sectionId: EvidenceLedgerSectionId,
  entry: EvidenceLedgerEntry,
): void {
  const targetSectionId =
    entry.session_scope === "prior_session" && sectionId !== "prior_session_memory"
      ? "prior_session_memory"
      : sectionId;
  const targetBucket = sectionBucket(sections, targetSectionId);

  if (targetBucket.seenEntryIds.has(entry.id)) {
    return;
  }

  targetBucket.seenEntryIds.add(entry.id);
  targetBucket.entries.push(entry);
}

function cappedTrustRank(entry: EvidenceLedgerEntry): EvidenceLedgerEntry {
  if (entry.session_scope !== "prior_session") {
    return entry;
  }

  return {
    ...entry,
    trust_rank: Math.min(entry.trust_rank, PRIOR_SESSION_TRUST_RANK_CAP),
  };
}

function actorForStreamEntry(entry: Pick<StreamEntry, "kind">): EvidenceLedgerActor {
  if (entry.kind === "user_msg") {
    return "user";
  }

  if (
    entry.kind === "agent_msg" ||
    entry.kind === "agent_suppressed" ||
    entry.kind === "agent_observed"
  ) {
    return "assistant";
  }

  return "system";
}

function transcriptState(entry: TranscriptStreamEntry): string | undefined {
  if (entry.kind === "agent_suppressed") {
    return "suppressed";
  }

  if (entry.kind === "agent_observed") {
    return "observed";
  }

  return undefined;
}

function streamPersistenceClass(entry: Pick<StreamEntry, "persistence_class">) {
  return entry.persistence_class === undefined
    ? {}
    : { persistence_class: entry.persistence_class };
}

function speakerStateMetadata(
  entityRepository: SpeakerEntityRepository | undefined,
  senderEntityId: EntityId | null | undefined,
): Record<string, unknown> | undefined {
  if (senderEntityId === null || senderEntityId === undefined) {
    return undefined;
  }

  const displayName = resolveSpeakerDisplayName(entityRepository, senderEntityId);

  return {
    sender_entity_id: senderEntityId,
    ...(displayName === null ? {} : { sender_display_name: displayName }),
  };
}

function replyTargetStateMetadata(
  entry: TranscriptStreamEntry,
  entityRepository: SpeakerEntityRepository | undefined,
): Record<string, unknown> | undefined {
  if (entry.kind !== "agent_msg") {
    return speakerStateMetadata(entityRepository, entry.sender_entity_id);
  }

  const replyTargetEntityId = entry.reply_target_entity_id ?? null;

  if (replyTargetEntityId === null) {
    return undefined;
  }

  const displayName = resolveSpeakerDisplayName(entityRepository, replyTargetEntityId);

  return {
    reply_target_kind: "entity",
    reply_target_entity_id: replyTargetEntityId,
    ...(displayName === null ? {} : { reply_target_display_name: displayName }),
  };
}

function optionalStateMetadata(
  stateMetadata: Record<string, unknown> | undefined,
): Pick<EvidenceLedgerEntry, "state_metadata"> {
  return stateMetadata === undefined ? {} : { state_metadata: stateMetadata };
}

function rawStreamActor(
  streamEntryIds: readonly StreamEntryId[] | readonly string[] | undefined,
  resolver: ScopeResolver,
): EvidenceLedgerActor {
  const actors = new Set<EvidenceLedgerActor>();

  for (const streamEntryId of streamEntryIds ?? []) {
    const entry = resolver.streamEntriesById.get(streamEntryId);

    if (entry !== undefined) {
      actors.add(actorForStreamEntry(entry));
    }
  }

  return actors.size === 1 ? ([...actors][0] ?? "memory") : "memory";
}

function estimateTranscriptTokens(entries: readonly TranscriptStreamEntry[]): number {
  if (entries.length === 0) {
    return 0;
  }

  return estimatePromptTokens(
    entries.map((entry) => stringifyPromptContent(entry.content)).join("\n"),
  );
}

function estimateTranscriptEntryTokens(entry: TranscriptStreamEntry): number {
  return estimatePromptTokens(stringifyPromptContent(entry.content));
}

function transcriptRawEntry(
  entry: TranscriptStreamEntry,
  resolver: ScopeResolver,
  entityRepository: SpeakerEntityRepository | undefined,
): EvidenceLedgerEntry {
  const stateMetadata = replyTargetStateMetadata(entry, entityRepository);

  return {
    id: `current_session_stream:${entry.id}`,
    source_type: "current_session_stream",
    session_scope: "current_session",
    actor: actorForStreamEntry(entry),
    trust_rank: TRANSCRIPT_TRUST_RANK,
    text: stringifyPromptContent(entry.content),
    stream_index: resolver.streamOrderById.get(entry.id),
    state: transcriptState(entry),
    ...optionalStateMetadata(stateMetadata),
    taint: "none",
    ...streamPersistenceClass(entry),
  };
}

function compactedTranscriptRunEntry(
  entries: readonly TranscriptStreamEntry[],
  resolver: ScopeResolver,
): EvidenceLedgerEntry {
  const first = entries[0] as TranscriptStreamEntry;
  const last = entries[entries.length - 1] as TranscriptStreamEntry;
  const streamIds = entries.map((entry) => entry.id).join(", ");
  const firstIndex = resolver.streamOrderById.get(first.id);
  const lastIndex = resolver.streamOrderById.get(last.id);
  const indexRange =
    firstIndex === undefined || lastIndex === undefined ? "unknown" : `${firstIndex}..${lastIndex}`;

  return {
    id: `current_session_compacted:${first.id}`,
    source_type: "system_metadata",
    session_scope: "current_session",
    actor: "system",
    trust_rank: TRANSCRIPT_TRUST_RANK,
    text: `Earlier assistant/system transcript entries compacted: entries=${entries.length}, stream_indexes=${indexRange}, stream_ids=${streamIds}.`,
    stream_index: firstIndex,
    state: "compacted",
    taint: "none",
  };
}

function compactedCurrentUserTranscriptEntry(
  entry: TranscriptStreamEntry,
  resolver: ScopeResolver,
): EvidenceLedgerEntry {
  return {
    id: `current_session_compacted_current_user:${entry.id}`,
    source_type: "system_metadata",
    session_scope: "current_session",
    actor: "system",
    trust_rank: TRANSCRIPT_TRUST_RANK,
    text: `Current user transcript duplicate compacted; full text is rendered in section 1 as current_user_message:${entry.id}.`,
    stream_index: resolver.streamOrderById.get(entry.id),
    state: "compacted",
    taint: "none",
  };
}

function rawTailStreamIds(
  entries: readonly TranscriptStreamEntry[],
  budget: number,
  currentUserEntryId: string | undefined,
): Set<string> {
  const tailBudget = Math.max(1, Math.floor(budget * TRANSCRIPT_RAW_TAIL_BUDGET_FRACTION));
  const ids = new Set<string>();
  let tokens = 0;

  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index];

    if (entry === undefined || entry.id === currentUserEntryId) {
      continue;
    }

    const entryTokens = estimateTranscriptEntryTokens(entry);

    if (ids.size >= TRANSCRIPT_RAW_TAIL_MIN_ENTRIES && tokens + entryTokens > tailBudget) {
      break;
    }

    ids.add(entry.id);
    tokens += entryTokens;
  }

  return ids;
}

function shouldKeepRawCompactedTranscriptEntry(
  entry: TranscriptStreamEntry,
  tailIds: ReadonlySet<string>,
  currentUserEntryId: string | undefined,
): boolean {
  if (entry.id === currentUserEntryId) {
    return false;
  }

  return (
    tailIds.has(entry.id) ||
    entry.kind === "user_msg" ||
    entry.persistence_class === "assistant_self_report"
  );
}

function compactTranscriptEntries(input: {
  entries: readonly TranscriptStreamEntry[];
  budget: number;
  currentUserEntryId?: string;
  resolver: ScopeResolver;
  entityRepository?: SpeakerEntityRepository;
}): TranscriptCompactionResult {
  const transcriptTokens = estimateTranscriptTokens(input.entries);

  if (transcriptTokens <= input.budget) {
    return {
      entries: input.entries.map((entry) =>
        transcriptRawEntry(entry, input.resolver, input.entityRepository),
      ),
      rawStreamIds: new Set(input.entries.map((entry) => entry.id)),
      compacted: false,
      originalTokenEstimate: transcriptTokens,
      compactedEntryCount: 0,
      rawPreservedUserEntryCount: input.entries.filter((entry) => entry.kind === "user_msg").length,
    };
  }

  const tailIds = rawTailStreamIds(input.entries, input.budget, input.currentUserEntryId);
  const renderedEntries: EvidenceLedgerEntry[] = [];
  const rawStreamIds = new Set<string>();
  let compactedRun: TranscriptStreamEntry[] = [];
  let compactedEntryCount = 0;
  let rawPreservedUserEntryCount = 0;

  const flushCompactedRun = () => {
    if (compactedRun.length === 0) {
      return;
    }

    renderedEntries.push(compactedTranscriptRunEntry(compactedRun, input.resolver));
    compactedEntryCount += compactedRun.length;
    compactedRun = [];
  };

  for (const entry of input.entries) {
    if (shouldKeepRawCompactedTranscriptEntry(entry, tailIds, input.currentUserEntryId)) {
      flushCompactedRun();
      renderedEntries.push(transcriptRawEntry(entry, input.resolver, input.entityRepository));
      rawStreamIds.add(entry.id);
      if (entry.kind === "user_msg") {
        rawPreservedUserEntryCount += 1;
      }
      continue;
    }

    if (entry.id === input.currentUserEntryId) {
      flushCompactedRun();
      renderedEntries.push(compactedCurrentUserTranscriptEntry(entry, input.resolver));
      compactedEntryCount += 1;
      continue;
    }

    compactedRun.push(entry);
  }

  flushCompactedRun();

  return {
    entries: renderedEntries,
    rawStreamIds,
    compacted: true,
    originalTokenEstimate: transcriptTokens,
    compactedEntryCount,
    rawPreservedUserEntryCount,
  };
}

function scopeFromStreamIds(
  streamEntryIds: readonly StreamEntryId[] | readonly string[] | undefined,
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  if (streamEntryIds === undefined || streamEntryIds.length === 0) {
    return "global";
  }

  let currentSessionCount = 0;
  let priorSessionCount = 0;

  for (const streamEntryId of streamEntryIds) {
    const entry = resolver.streamEntriesById.get(streamEntryId);

    if (entry === undefined) {
      return "global";
    }

    if (entry.session_id === resolver.currentSessionId) {
      currentSessionCount += 1;
    } else {
      priorSessionCount += 1;
    }
  }

  if (currentSessionCount === streamEntryIds.length) {
    return "current_session";
  }

  if (priorSessionCount === streamEntryIds.length) {
    return "prior_session";
  }

  return "global";
}

function streamIndexFromSingleCurrentSessionStreamId(
  streamEntryIds: readonly StreamEntryId[] | readonly string[] | undefined,
  resolver: ScopeResolver,
): number | undefined {
  if (streamEntryIds === undefined || streamEntryIds.length !== 1) {
    return undefined;
  }

  const streamEntryId = streamEntryIds[0];

  if (streamEntryId === undefined) {
    return undefined;
  }

  return resolver.streamOrderById.get(streamEntryId);
}

function scopeFromStreamEntries(
  entries: readonly StreamEntry[],
  currentSessionId: SessionId,
): EvidenceLedgerSessionScope {
  if (entries.length === 0) {
    return "global";
  }

  if (entries.every((entry) => entry.session_id === currentSessionId)) {
    return "current_session";
  }

  return entries.every((entry) => entry.session_id !== currentSessionId)
    ? "prior_session"
    : "global";
}

function scopeFromEpisodeIds(
  episodeIds: readonly EpisodeId[] | readonly string[] | undefined,
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  if (episodeIds === undefined || episodeIds.length === 0) {
    return "global";
  }

  let sawPriorSession = false;

  for (const episodeId of episodeIds) {
    const scope = resolver.episodeScopesById.get(episodeId);

    if (scope === "current_session") {
      return "current_session";
    }

    if (scope === "prior_session") {
      sawPriorSession = true;
    }
  }

  return sawPriorSession ? "prior_session" : "global";
}

function streamIdsFromEpisodeIds(
  episodeIds: readonly EpisodeId[] | readonly string[] | undefined,
  resolver: ScopeResolver,
): string[] {
  if (episodeIds === undefined) {
    return [];
  }

  return episodeIds.flatMap((episodeId) => [
    ...(resolver.episodeSourceStreamIdsById.get(episodeId) ?? []),
  ]);
}

function persistenceClassFromProvenance(
  input: {
    streamEntryIds?: readonly StreamEntryId[] | readonly string[];
    episodeIds?: readonly EpisodeId[] | readonly string[];
  },
  resolver: ScopeResolver,
) {
  return persistenceClassFromStreamIds(
    [...(input.streamEntryIds ?? []), ...streamIdsFromEpisodeIds(input.episodeIds, resolver)],
    resolver,
  );
}

function combineScopes(scopes: readonly EvidenceLedgerSessionScope[]): EvidenceLedgerSessionScope {
  if (scopes.some((scope) => scope === "current_session")) {
    return "current_session";
  }

  if (scopes.some((scope) => scope === "prior_session")) {
    return "prior_session";
  }

  return "global";
}

function slotTaint(slot: RelationalSlot): EvidenceLedgerTaint {
  if (slot.state === "quarantined") {
    return "quarantined";
  }

  if (slot.state === "contested") {
    return "contested";
  }

  return "none";
}

function semanticTaint(input: {
  underReview?: unknown;
  validTo?: number | null;
  invalidatedAt?: number | null;
}): EvidenceLedgerTaint {
  if (
    input.underReview !== undefined ||
    (input.validTo !== undefined && input.validTo !== null) ||
    (input.invalidatedAt !== undefined && input.invalidatedAt !== null)
  ) {
    return "contested";
  }

  return "none";
}

function semanticNodeStateMetadata(node: {
  id?: string;
  source_episode_ids?: readonly string[];
  partial_source_visibility?: boolean;
  source_visibility_fraction?: number;
}): Record<string, unknown> | undefined {
  const metadata: Record<string, unknown> = {
    ...(node.id === undefined ? {} : { node_id: node.id }),
    ...(node.source_episode_ids === undefined || node.source_episode_ids.length === 0
      ? {}
      : { source_episode_ids: [...node.source_episode_ids] }),
  };

  if (node.partial_source_visibility === true) {
    metadata.partial_source_visibility = true;
    if (node.source_visibility_fraction !== undefined) {
      metadata.source_visibility_fraction = node.source_visibility_fraction;
    }
  }

  return Object.keys(metadata).length === 0 ? undefined : metadata;
}

function commitmentScope(
  commitment: CommitmentRecord,
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  return scopeFromStreamIds(commitment.source_stream_entry_ids ?? [], resolver);
}

function actionScope(action: ActionRecord, resolver: ScopeResolver): EvidenceLedgerSessionScope {
  return combineScopes([
    scopeFromStreamIds(action.provenance_stream_entry_ids, resolver),
    scopeFromEpisodeIds(action.provenance_episode_ids, resolver),
  ]);
}

function slotScope(slot: RelationalSlot, resolver: ScopeResolver): EvidenceLedgerSessionScope {
  return scopeFromStreamIds(
    [
      ...slot.evidence_stream_entry_ids,
      ...slot.contradicted_by_stream_entry_ids,
      ...slot.alternate_values.flatMap((alternate) => alternate.evidence_stream_entry_ids),
    ],
    resolver,
  );
}

function participantForSlot(
  slot: RelationalSlot,
  participants: readonly ActiveParticipant[] | undefined,
): ActiveParticipant | undefined {
  return participants?.find((participant) => participant.entityId === slot.subject_entity_id);
}

function slotSubjectStateMetadata(
  slot: RelationalSlot,
  participant: ActiveParticipant | undefined,
  participantCount: number,
): Record<string, unknown> | undefined {
  if (participant === undefined || participantCount <= 1) {
    return undefined;
  }

  return {
    subject_entity_id: slot.subject_entity_id,
    subject_display_name: participant.displayName ?? slot.subject_entity_id,
    subject_role: participant.role,
  };
}

function openQuestionScope(
  question: OpenQuestion,
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  return combineScopes([
    scopeFromStreamIds(openQuestionStreamEntryIds(question), resolver),
    scopeFromEpisodeIds(openQuestionEpisodeIds(question), resolver),
  ]);
}

function openQuestionProvenanceEpisodeIds(question: OpenQuestion): readonly string[] {
  if (question.provenance?.kind === "episodes") {
    return question.provenance.episode_ids;
  }

  if (question.provenance?.kind === "online_reflector") {
    return question.provenance.evidence_episode_ids;
  }

  return [];
}

function openQuestionProvenanceStreamEntryIds(question: OpenQuestion): readonly string[] {
  return question.provenance?.kind === "online_reflector"
    ? question.provenance.evidence_stream_entry_ids
    : [];
}

function openQuestionStreamEntryIds(question: OpenQuestion): readonly string[] {
  return [
    ...question.resolution_evidence_stream_entry_ids,
    ...openQuestionProvenanceStreamEntryIds(question),
  ];
}

function openQuestionEpisodeIds(question: OpenQuestion): readonly string[] {
  return [
    ...question.related_episode_ids,
    ...question.resolution_evidence_episode_ids,
    ...openQuestionProvenanceEpisodeIds(question),
  ];
}

function relevantOpenQuestionStreamIds(
  input: EvidenceLedgerBuildInput,
  resolver: ScopeResolver,
): Set<string> {
  const streamIds = new Set<string>();

  for (const entryId of resolver.streamEntriesById.keys()) {
    streamIds.add(entryId);
  }

  if (input.currentUserEntry !== undefined) {
    streamIds.add(input.currentUserEntry.id);
  }

  for (const item of input.retrievedEvidence) {
    for (const streamId of item.provenance?.streamIds ?? []) {
      streamIds.add(streamId);
    }
  }

  for (const result of input.retrievedEpisodes) {
    for (const streamId of result.episode.source_stream_ids) {
      streamIds.add(streamId);
    }

    for (const entry of result.citationChain) {
      streamIds.add(entry.id);
    }
  }

  return streamIds;
}

function relevantOpenQuestionEpisodeIds(input: EvidenceLedgerBuildInput): Set<string> {
  const episodeIds = new Set<string>();

  for (const item of input.retrievedEvidence) {
    if (item.provenance?.episodeId !== undefined) {
      episodeIds.add(item.provenance.episodeId);
    }

    if (item.provenance?.parentEpisodeId !== undefined) {
      episodeIds.add(item.provenance.parentEpisodeId);
    }
  }

  for (const result of input.retrievedEpisodes) {
    episodeIds.add(result.episode.id);
  }

  return episodeIds;
}

function openQuestionStateMetadata(question: OpenQuestion): Record<string, unknown> | undefined {
  if (question.status === "resolved") {
    return {
      resolution_note: question.resolution_note,
      resolved_at: question.resolved_at,
      resolution_evidence_episode_ids: question.resolution_evidence_episode_ids,
      resolution_evidence_stream_entry_ids: question.resolution_evidence_stream_entry_ids,
    };
  }

  if (question.status === "abandoned") {
    return {
      abandoned_reason: question.abandoned_reason,
      abandoned_at: question.abandoned_at,
    };
  }

  return undefined;
}

function reviewQueueScope(
  item: ReviewQueueItem,
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  return scopeFromStreamIds(reviewQueueStreamIds(item), resolver);
}

function reviewQueueStreamIds(item: ReviewQueueItem): string[] {
  return Object.values(item.refs).flatMap((value) => {
    if (Array.isArray(value)) {
      return value.filter((candidate): candidate is string => typeof candidate === "string");
    }

    return typeof value === "string" ? [value] : [];
  });
}

function rawStreamSourceType(scope: EvidenceLedgerSessionScope): EvidenceLedgerSourceType {
  if (scope === "current_session") {
    return "current_session_stream";
  }

  if (scope === "prior_session") {
    return "prior_session_stream";
  }

  return "system_metadata";
}

function evidenceItemSourceType(
  item: EvidenceItem,
  scope: EvidenceLedgerSessionScope,
): EvidenceLedgerSourceType {
  if (item.provenance?.streamIds !== undefined && item.provenance.streamIds.length > 0) {
    return rawStreamSourceType(scope);
  }

  if (item.provenance?.episodeId !== undefined || item.source === "episode") {
    return "episode";
  }

  if (item.provenance?.nodeId !== undefined || item.source === "semantic_node") {
    return "semantic_node";
  }

  if (item.provenance?.edgeId !== undefined || item.source === "semantic_edge") {
    return "semantic_edge";
  }

  if (item.provenance?.commitmentId !== undefined || item.source === "commitment") {
    return "commitment";
  }

  return "system_metadata";
}

function evidenceItemScope(
  item: EvidenceItem,
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  return combineScopes([
    scopeFromStreamIds(item.provenance?.streamIds, resolver),
    scopeFromEpisodeIds(
      [
        ...(item.provenance?.episodeId === undefined ? [] : [item.provenance.episodeId]),
        ...(item.source_episode_ids ?? []),
      ],
      resolver,
    ),
  ]);
}

function evidenceItemState(item: EvidenceItem): string {
  const parts = [
    `score=${item.score.toFixed(2)}`,
    `intent=${item.recallIntentId}`,
    item.matchedTerms.length === 0 ? null : `terms=${item.matchedTerms.join(", ")}`,
    item.source_episode_ids === undefined || item.source_episode_ids.length === 0
      ? null
      : `sources=${item.source_episode_ids.slice(0, 3).join(", ")}`,
    item.partial_source_visibility === true ? "partial_sources=true" : null,
    item.source_visibility_fraction === undefined
      ? null
      : `visible_fraction=${item.source_visibility_fraction.toFixed(2)}`,
  ].filter((part): part is string => part !== null);

  return parts.join(" ");
}

function evidenceItemProvenanceMetadata(item: EvidenceItem): Record<string, unknown> | undefined {
  const provenance = item.provenance;

  if (provenance === undefined) {
    return undefined;
  }

  return {
    ...(provenance.episodeId === undefined ? {} : { episode_id: provenance.episodeId }),
    ...(provenance.parentEpisodeId === undefined
      ? {}
      : { parent_episode_id: provenance.parentEpisodeId }),
    ...(provenance.nodeId === undefined ? {} : { node_id: provenance.nodeId }),
    ...(provenance.edgeId === undefined ? {} : { edge_id: provenance.edgeId }),
    ...(provenance.commitmentId === undefined ? {} : { commitment_id: provenance.commitmentId }),
    ...(provenance.openQuestionId === undefined
      ? {}
      : { open_question_id: provenance.openQuestionId }),
    ...(provenance.streamIds === undefined || provenance.streamIds.length === 0
      ? {}
      : { stream_ids: provenance.streamIds }),
  };
}

function persistenceClassFromStreamIds(
  streamEntryIds: readonly StreamEntryId[] | readonly string[] | undefined,
  resolver: ScopeResolver,
) {
  if (streamEntryIds === undefined) {
    return {};
  }

  return streamEntryIds.some(
    (streamEntryId) =>
      resolver.streamEntriesById.get(streamEntryId)?.persistence_class === "assistant_self_report",
  )
    ? { persistence_class: "assistant_self_report" as const }
    : {};
}

function buildEpisodeScopeMap(
  retrievedEpisodes: readonly RetrievedEpisode[],
  resolverBase: Omit<ScopeResolver, "episodeScopesById" | "episodeSourceStreamIdsById">,
): Map<string, EvidenceLedgerSessionScope> {
  const episodeScopes = new Map<string, EvidenceLedgerSessionScope>();

  for (const result of retrievedEpisodes) {
    const citationScope = scopeFromStreamEntries(
      result.citationChain,
      resolverBase.currentSessionId,
    );
    const sourceScope =
      citationScope === "global"
        ? scopeFromStreamIds(result.episode.source_stream_ids, {
            ...resolverBase,
            episodeScopesById: new Map(),
            episodeSourceStreamIdsById: new Map(),
          })
        : citationScope;

    episodeScopes.set(result.episode.id, sourceScope);
  }

  return episodeScopes;
}

function buildEpisodeSourceStreamIdMap(
  retrievedEpisodes: readonly RetrievedEpisode[],
): Map<string, readonly string[]> {
  const episodeSourceStreamIds = new Map<string, readonly string[]>();

  for (const result of retrievedEpisodes) {
    episodeSourceStreamIds.set(result.episode.id, result.episode.source_stream_ids);
  }

  return episodeSourceStreamIds;
}

function normalizePositiveInteger(value: number | undefined, fallback: number): number {
  return value === undefined ? fallback : Math.max(1, Math.floor(value));
}

function normalizeUnitInterval(value: number | undefined, fallback: number): number {
  if (value === undefined || !Number.isFinite(value)) {
    return fallback;
  }

  return Math.max(0, Math.min(1, value));
}

function listVisibleActions(
  actionRepository: ActionLedgerRepository,
  audienceEntityId: EntityId | null,
  activeParticipants: readonly ActiveParticipant[] | undefined,
  limit: number,
): ActionRecord[] {
  const records: ActionRecord[] = [...actionRepository.list({ audienceEntityId: null, limit })];
  const activeParticipantIds = new Set(
    (activeParticipants ?? []).map((participant) => participant.entityId),
  );

  if (audienceEntityId !== null) {
    records.push(...actionRepository.list({ audienceEntityId, limit }));
  }

  for (const participant of activeParticipants ?? []) {
    records.push(
      ...actionRepository
        .list({ actor: participant.entityId })
        .filter((action) =>
          isActionVisibleToSession(action, audienceEntityId, activeParticipantIds),
        ),
    );
    records.push(...actionRepository.list({ audienceEntityId: participant.entityId, limit }));
  }

  return [...new Map(records.map((record) => [record.id, record])).values()]
    .sort((left, right) => right.updated_at - left.updated_at || left.id.localeCompare(right.id))
    .slice(0, limit);
}

function actionActorDisplay(
  actor: ActionRecord["actor"],
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): string {
  if (actor === "borg") {
    return "assistant";
  }

  if (actor === "user") {
    return "user";
  }

  return entityRepository?.get(actor)?.canonical_name ?? "participant";
}

function scopedCommitmentsForEntity(
  commitments: readonly CommitmentRecord[],
  entityId: EntityId,
): CommitmentRecord[] {
  return commitments.filter(
    (commitment) =>
      commitment.made_to_entity === entityId ||
      commitment.restricted_audience === entityId ||
      commitment.about_entity === entityId ||
      commitment.committed_by_entity_id === entityId,
  );
}

function scopedGoalsForEntity(goals: readonly GoalRecord[], entityId: EntityId): GoalRecord[] {
  return goals.filter(
    (goal) => goal.audience_entity_id === entityId || goal.owner_entity_id === entityId,
  );
}

function dedupeCommitments(records: readonly CommitmentRecord[]): CommitmentRecord[] {
  return [...new Map(records.map((record) => [record.id, record])).values()].sort(
    (left, right) => right.priority - left.priority || left.created_at - right.created_at,
  );
}

function dedupeGoals(records: readonly GoalRecord[]): GoalRecord[] {
  return [...new Map(records.map((record) => [record.id, record])).values()].sort(
    (left, right) => right.priority - left.priority || left.created_at - right.created_at,
  );
}

function dedupeActions(records: readonly ActionRecord[]): ActionRecord[] {
  return [...new Map(records.map((record) => [record.id, record])).values()].sort(
    (left, right) => right.updated_at - left.updated_at || left.id.localeCompare(right.id),
  );
}

function visibleAudienceEntityIds(
  audienceEntityId: EntityId | null,
  activeParticipants: readonly ActiveParticipant[] | undefined,
): ReadonlySet<EntityId> {
  const ids = new Set((activeParticipants ?? []).map((participant) => participant.entityId));

  if (audienceEntityId !== null) {
    ids.add(audienceEntityId);
  }

  return ids;
}

function audienceIsVisibleToSession(
  scopedAudienceEntityId: EntityId | null | undefined,
  currentAudienceEntityId: EntityId | null,
  activeParticipantIds: ReadonlySet<EntityId>,
): boolean {
  if (scopedAudienceEntityId === null || scopedAudienceEntityId === undefined) {
    return true;
  }

  return (
    scopedAudienceEntityId === currentAudienceEntityId ||
    activeParticipantIds.has(scopedAudienceEntityId)
  );
}

function isActionVisibleToSession(
  action: ActionRecord,
  audienceEntityId: EntityId | null,
  activeParticipantIds: ReadonlySet<EntityId>,
): boolean {
  return audienceIsVisibleToSession(
    action.audience_entity_id,
    audienceEntityId,
    activeParticipantIds,
  );
}

function isCommitmentVisibleToSession(
  commitment: CommitmentRecord,
  audienceEntityId: EntityId | null,
  activeParticipantIds: ReadonlySet<EntityId>,
): boolean {
  if (commitment.restricted_audience !== null) {
    return audienceIsVisibleToSession(
      commitment.restricted_audience,
      audienceEntityId,
      activeParticipantIds,
    );
  }

  return audienceIsVisibleToSession(
    commitment.made_to_entity,
    audienceEntityId,
    activeParticipantIds,
  );
}

function isGoalVisibleToSession(
  goal: GoalRecord,
  audienceEntityId: EntityId | null,
  activeParticipantIds: ReadonlySet<EntityId>,
): boolean {
  return audienceIsVisibleToSession(
    goal.audience_entity_id,
    audienceEntityId,
    activeParticipantIds,
  );
}

function entityIdPointsAtPerson(
  entityId: EntityId | null | undefined,
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): boolean {
  return (
    entityId !== null &&
    entityId !== undefined &&
    entityRepository?.get(entityId)?.kind === "person"
  );
}

function actionBelongsToGroupChannel(
  action: ActionRecord,
  audienceEntityId: EntityId,
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): boolean {
  if (action.actor === "user" || action.audience_entity_id !== audienceEntityId) {
    return false;
  }

  return action.actor === "borg" || !entityIdPointsAtPerson(action.actor, entityRepository);
}

function commitmentBelongsToGroupChannel(
  commitment: CommitmentRecord,
  audienceEntityId: EntityId,
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): boolean {
  if (entityIdPointsAtPerson(commitment.committed_by_entity_id ?? null, entityRepository)) {
    return false;
  }

  return scopedCommitmentsForEntity([commitment], audienceEntityId).length > 0;
}

function goalBelongsToGroupChannel(
  goal: GoalRecord,
  audienceEntityId: EntityId,
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): boolean {
  if (entityIdPointsAtPerson(goal.owner_entity_id ?? null, entityRepository)) {
    return false;
  }

  return scopedGoalsForEntity([goal], audienceEntityId).length > 0;
}

function findParent(parents: Map<string, string>, id: string): string {
  const parent = parents.get(id);

  if (parent === undefined || parent === id) {
    parents.set(id, id);
    return id;
  }

  const root = findParent(parents, parent);
  parents.set(id, root);
  return root;
}

function unionParents(parents: Map<string, string>, leftId: string, rightId: string): void {
  const leftRoot = findParent(parents, leftId);
  const rightRoot = findParent(parents, rightId);

  if (leftRoot === rightRoot) {
    return;
  }

  const root = leftRoot < rightRoot ? leftRoot : rightRoot;
  const child = root === leftRoot ? rightRoot : leftRoot;
  parents.set(child, root);
}

function actionTimestampForState(action: ActionRecord): number {
  switch (action.state) {
    case "considering":
      return action.considering_at ?? action.updated_at;
    case "committed_to_do":
      return action.committed_at ?? action.updated_at;
    case "scheduled":
      return action.scheduled_at ?? action.updated_at;
    case "completed":
      return action.completed_at ?? action.updated_at;
    case "not_done":
      return action.not_done_at ?? action.updated_at;
    case "unknown":
      return action.unknown_at ?? action.updated_at;
  }
}

function combineActionScopes(
  records: readonly ActionRecord[],
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  return combineScopes(records.map((record) => actionScope(record, resolver)));
}

function selectThreadOrigin(records: readonly ActionRecord[]): ActionRecord {
  return [...records].sort(
    (left, right) => left.created_at - right.created_at || left.id.localeCompare(right.id),
  )[0] as ActionRecord;
}

function selectThreadCurrent(records: readonly ActionRecord[]): ActionRecord {
  return [...records].sort(
    (left, right) => right.updated_at - left.updated_at || left.id.localeCompare(right.id),
  )[0] as ActionRecord;
}

function canThreadActions(left: ActionRecord, right: ActionRecord): boolean {
  return (
    left.goal_id !== null &&
    right.goal_id !== null &&
    left.goal_id === right.goal_id &&
    left.actor === right.actor
  );
}

function sameThreadablePair(
  pair: ActionDescriptionSimilarityPair,
  actionsById: ReadonlyMap<string, ActionRecord>,
  threshold: number,
): [ActionRecord, ActionRecord] | null {
  if (pair.similarity < threshold) {
    return null;
  }

  const left = actionsById.get(pair.leftId);
  const right = actionsById.get(pair.rightId);

  if (left === undefined || right === undefined || !canThreadActions(left, right)) {
    return null;
  }

  return [left, right];
}

async function buildActionThreads(input: {
  records: readonly ActionRecord[];
  repository: ActionLedgerRepository;
  resolver: ScopeResolver;
  similarityThreshold: number;
}): Promise<ActionThread[]> {
  const parents = new Map<string, string>();
  const actionsById = new Map(input.records.map((record) => [record.id, record]));

  for (const record of input.records) {
    parents.set(record.id, record.id);
  }

  const pairs =
    input.repository.findSimilarDescriptionPairs === undefined
      ? []
      : await input.repository.findSimilarDescriptionPairs(
          input.records.filter((record) => record.goal_id !== null),
          input.similarityThreshold,
        );

  for (const pair of pairs) {
    const records = sameThreadablePair(pair, actionsById, input.similarityThreshold);

    if (records === null) {
      continue;
    }

    unionParents(parents, records[0].id, records[1].id);
  }

  const groups = new Map<string, ActionRecord[]>();

  for (const record of input.records) {
    const root = findParent(parents, record.id);
    groups.set(root, [...(groups.get(root) ?? []), record]);
  }

  return [...groups.entries()]
    .map(([id, records]) => {
      const origin = selectThreadOrigin(records);
      const current = selectThreadCurrent(records);

      return {
        id,
        records: [...records].sort(
          (left, right) => left.updated_at - right.updated_at || left.id.localeCompare(right.id),
        ),
        origin,
        current,
        scope: combineActionScopes(records, input.resolver),
      };
    })
    .sort(
      (left, right) =>
        right.current.updated_at - left.current.updated_at ||
        left.current.id.localeCompare(right.current.id),
    );
}

function renderActionThreadText(
  thread: ActionThread,
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): string {
  const currentAt = new Date(actionTimestampForState(thread.current)).toISOString();
  const actor = actionActorDisplay(thread.current.actor, entityRepository);
  const lines = [
    `actor: ${actor}`,
    `originating_intent: ${thread.origin.description}`,
    `transitions: ${thread.records.length}, current: ${thread.current.state} at ${currentAt}`,
  ];

  if (thread.current.id !== thread.origin.id) {
    lines.push(`current_intent: ${thread.current.description}`);
  }

  return lines.join("\n");
}

function actionThreadStateMetadata(
  thread: ActionThread,
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): Record<string, unknown> {
  return {
    record_ids: thread.records.map((record) => record.id),
    transitions: thread.records.length,
    current_action_id: thread.current.id,
    current_updated_at: thread.current.updated_at,
    current_actor: actionActorDisplay(thread.current.actor, entityRepository),
    goal_id: thread.current.goal_id,
    open_question_id: thread.current.open_question_id,
  };
}

function actionThreadState(thread: ActionThread): ActionState {
  return thread.current.state;
}

function truncateOlderActionThreadSample(text: string): string {
  if (text.length <= OLDER_ACTION_THREAD_SAMPLE_MAX_CHARS) {
    return text;
  }

  return `${text.slice(0, OLDER_ACTION_THREAD_SAMPLE_MAX_CHARS - 3)}...`;
}

function renderOlderActionThreadsSummary(olderThreads: readonly ActionThread[]): string {
  const olderRecordCount = olderThreads.reduce((count, thread) => count + thread.records.length, 0);
  const stateCounts = new Map<ActionState, number>(ACTION_STATES.map((state) => [state, 0]));

  for (const thread of olderThreads) {
    stateCounts.set(thread.current.state, (stateCounts.get(thread.current.state) ?? 0) + 1);
  }

  const stateSummary = ACTION_STATES.map((state) => {
    const count = stateCounts.get(state) ?? 0;
    return count > 0 ? `${state}=${count}` : null;
  })
    .filter((entry): entry is string => entry !== null)
    .join(" ");
  const samples = olderThreads
    .slice(0, OLDER_ACTION_THREAD_SAMPLE_LIMIT)
    .map(
      (thread) =>
        `${thread.current.state}: ${JSON.stringify(
          truncateOlderActionThreadSample(thread.current.description),
        )}`,
    )
    .join(" | ");

  return `Older action threads omitted from this section: threads=${olderThreads.length}, records=${olderRecordCount}, states=${stateSummary}, recent_samples=${samples}.`;
}

function entryFlatText(section: EvidenceLedgerSection, entry: EvidenceLedgerEntry): string {
  return [
    section.label,
    entry.id,
    entry.source_type,
    entry.session_scope,
    entry.actor,
    String(entry.trust_rank),
    entry.stream_index === undefined ? "" : String(entry.stream_index),
    entry.state ?? "",
    entry.taint ?? "",
    entry.persistence_class ?? "",
    entry.via_retrieval === true ? "via_retrieval" : "",
    entry.value ?? "",
    entry.text ?? "",
  ].join("\n");
}

function estimateSectionTokens(section: EvidenceLedgerSection): number {
  const text = section.entries.map((entry) => entryFlatText(section, entry)).join("\n");
  return text.length === 0 ? 0 : estimatePromptTokens(text);
}

function estimateLedgerTokens(sections: readonly EvidenceLedgerSection[]): number {
  const text = sections
    .flatMap((section) => section.entries.map((entry) => entryFlatText(section, entry)))
    .join("\n");

  return text.length === 0 ? 0 : estimatePromptTokens(text);
}

function currentUserMessageStateMetadata(
  input: EvidenceLedgerBuildInput,
  entityRepository: SpeakerEntityRepository | undefined,
): Record<string, unknown> | undefined {
  return speakerStateMetadata(entityRepository, input.currentUserEntry?.sender_entity_id);
}

export function summarizeEvidenceLedgerTrace(ledger: EvidenceLedger): EvidenceLedgerTraceSummary {
  // Sprint 8d.3: per-section token accounting. v36 mean input tokens were
  // ~113k -- without per-section breakdown there is no way to attribute
  // that load to specific bands. Surfacing it in the trace lets us tell
  // ledger-side bloat (transcript, action records) apart from
  // retrieval-side bloat (semantic walks, episodes).
  const estimatedTokensBySection = Object.fromEntries(
    ledger.sections.map((section) => [section.id, estimateSectionTokens(section)]),
  ) as Record<EvidenceLedgerSectionId, number>;

  return {
    entryCountsBySection: Object.fromEntries(
      ledger.sections.map((section) => [section.id, section.entries.length]),
    ) as Record<EvidenceLedgerSectionId, number>,
    estimatedTokensBySection,
    transcriptIncluded: ledger.transcriptIncluded,
    transcriptCompacted: ledger.transcriptCompacted,
    transcriptOmittedReason: ledger.transcriptOmittedReason,
    originalTranscriptTokenEstimate: ledger.originalTranscriptTokenEstimate,
    compactedTranscriptTokenEstimate: estimatedTokensBySection.current_session_transcript,
    compactedEntryCount: ledger.compactedTranscriptEntryCount,
    rawPreservedUserEntryCount: ledger.rawPreservedUserTranscriptEntryCount,
    totalEstimatedTokens: ledger.estimatedTokens,
  };
}

export class EvidenceLedgerBuilder {
  constructor(private readonly options: EvidenceLedgerBuilderOptions) {}

  async build(input: EvidenceLedgerBuildInput): Promise<EvidenceLedger> {
    const streamEntries = await loadSessionStreamEntries(
      this.options.createStreamReader(input.sessionId),
    );
    const streamEntriesById = new Map<string, StreamEntry>();
    const streamOrderById = new Map<string, number>();

    for (const [index, entry] of streamEntries.entries()) {
      streamOrderById.set(entry.id, index);
    }

    if (
      input.currentUserEntry !== undefined &&
      input.currentUserEntry.session_id === input.sessionId &&
      !streamOrderById.has(input.currentUserEntry.id)
    ) {
      streamOrderById.set(input.currentUserEntry.id, streamOrderById.size);
    }

    for (const entry of [
      ...streamEntries,
      ...input.retrievedEpisodes.flatMap((result) => result.citationChain),
    ]) {
      streamEntriesById.set(entry.id, entry);
    }

    const resolverBase = {
      currentSessionId: input.sessionId,
      streamEntriesById,
      streamOrderById,
    };
    const episodeScopesById = buildEpisodeScopeMap(input.retrievedEpisodes, resolverBase);
    const episodeSourceStreamIdsById = buildEpisodeSourceStreamIdMap(input.retrievedEpisodes);
    const resolver: ScopeResolver = {
      ...resolverBase,
      episodeScopesById,
      episodeSourceStreamIdsById,
    };
    const transcriptEntries = activeSessionTranscriptEntries(streamEntries);
    const transcript = compactTranscriptEntries({
      entries: transcriptEntries,
      budget: this.options.currentSessionTranscriptTokenBudget,
      currentUserEntryId: input.currentUserEntry?.id,
      resolver,
      entityRepository: this.options.entityRepository,
    });
    const sections = createSectionBuckets();

    this.addCurrentUserMessage(sections, input, resolver);
    this.addTranscript(sections, transcript.entries);
    this.addCommitmentsAndConstraints(sections, input, resolver);
    this.addDiscourseState(sections, input, resolver);
    this.addContradictionsAndQuarantines(sections, input, streamEntries, resolver);
    await this.addActionStates(sections, input, resolver);
    this.addGroupChannelMemory(sections, input, resolver);
    this.addRelationalSlots(sections, resolver, input.audienceEntityId, input.activeParticipants);
    // Sprint 8d.6.3: stream IDs covered by the current_session_transcript
    // section don't need to be re-rendered as retrieved_raw_stream_evidence.
    // The same underlying entry's text was duplicated across both sections
    // (~25k tokens on heavy v37 turns) because dedupe only matched on
    // rendered ledger entry IDs, not provenance stream IDs.
    const transcriptStreamIds = transcript.rawStreamIds;
    this.addRetrievedRawStreamEvidence(sections, input, resolver, transcriptStreamIds);
    this.addRetrievedStructuredEvidence(sections, input, resolver);
    this.addEpisodes(sections, input, resolver);
    this.addSemanticGraph(sections, input, resolver);
    this.addOpenQuestions(sections, input, resolver);

    const orderedSections = finalSections(sections);

    return {
      sections: orderedSections,
      transcriptIncluded: true,
      transcriptCompacted: transcript.compacted,
      originalTranscriptTokenEstimate: transcript.originalTokenEstimate,
      compactedTranscriptEntryCount: transcript.compactedEntryCount,
      rawPreservedUserTranscriptEntryCount: transcript.rawPreservedUserEntryCount,
      estimatedTokens: estimateLedgerTokens(orderedSections),
    };
  }

  private addCurrentUserMessage(
    sections: SectionBuckets,
    input: EvidenceLedgerBuildInput,
    resolver: ScopeResolver,
  ): void {
    const stateMetadata = currentUserMessageStateMetadata(input, this.options.entityRepository);

    addEntry(sections, "current_user_message", {
      id: `current_user_message:${input.currentUserEntry?.id ?? input.turnId ?? "unpersisted"}`,
      source_type: "current_user_message",
      session_scope: "current_session",
      actor: "user",
      trust_rank: CURRENT_USER_TRUST_RANK,
      text: input.currentUserMessage,
      ...optionalStateMetadata(stateMetadata),
      stream_index:
        input.currentUserEntry === undefined
          ? undefined
          : resolver.streamOrderById.get(input.currentUserEntry.id),
      taint:
        input.frameAnomaly?.status === "ok" && input.frameAnomaly.kind !== "normal"
          ? "quarantined"
          : "none",
    });
  }

  private addTranscript(sections: SectionBuckets, entries: readonly EvidenceLedgerEntry[]): void {
    for (const entry of entries) {
      addEntry(sections, "current_session_transcript", entry);
    }
  }

  private addCommitmentsAndConstraints(
    sections: SectionBuckets,
    input: EvidenceLedgerBuildInput,
    resolver: ScopeResolver,
  ): void {
    for (const commitment of input.applicableCommitments) {
      addEntry(
        sections,
        "commitments_and_constraints",
        cappedTrustRank({
          id: `commitment:${commitment.id}`,
          source_type: "commitment",
          session_scope: commitmentScope(commitment, resolver),
          actor: "memory",
          trust_rank: COMMITMENT_TRUST_RANK,
          text: commitment.directive,
          value: commitment.directive_family,
          state:
            commitment.revoked_at !== null
              ? "revoked"
              : commitment.expired_at !== null
                ? "expired"
                : "active",
          taint: "none",
          ...persistenceClassFromProvenance(
            { streamEntryIds: commitment.source_stream_entry_ids ?? [] },
            resolver,
          ),
        }),
      );
    }

    const stopState = input.workingMemory.discourse_state?.stop_until_substantive_content;

    if (stopState !== undefined && stopState !== null) {
      addEntry(sections, "commitments_and_constraints", {
        id: "discourse_constraint:stop_until_substantive_content",
        source_type: "system_metadata",
        session_scope: "current_session",
        actor: "system",
        trust_rank: DISCOURSE_TRUST_RANK,
        text: stopState.reason,
        value: stopState.provenance,
        state: "active",
        taint: "none",
      });
    }
  }

  private addDiscourseState(
    sections: SectionBuckets,
    input: EvidenceLedgerBuildInput,
    resolver: ScopeResolver,
  ): void {
    const discourseState = input.workingMemory.discourse_state;

    addEntry(sections, "closure_discourse_state", {
      id: "discourse_state:working_memory",
      source_type: "system_metadata",
      session_scope: "current_session",
      actor: "system",
      trust_rank: DISCOURSE_TRUST_RANK,
      text: `mode=${input.workingMemory.mode}; turn_counter=${input.workingMemory.turn_counter}`,
      state: input.workingMemory.mode ?? undefined,
      taint: "none",
    });

    if (discourseState?.closure_loop !== undefined && discourseState.closure_loop !== null) {
      addEntry(sections, "closure_discourse_state", {
        id: "discourse_state:closure_loop",
        source_type: "system_metadata",
        session_scope: "current_session",
        actor: "system",
        trust_rank: DISCOURSE_TRUST_RANK,
        text: discourseState.closure_loop.reason,
        value: discourseState.closure_loop.source_stream_entry_ids.join(", "),
        state: discourseState.closure_loop.status,
        taint: "none",
        ...persistenceClassFromProvenance(
          { streamEntryIds: discourseState.closure_loop.source_stream_entry_ids },
          resolver,
        ),
      });
    }
  }

  private addContradictionsAndQuarantines(
    sections: SectionBuckets,
    input: EvidenceLedgerBuildInput,
    streamEntries: readonly StreamEntry[],
    resolver: ScopeResolver,
  ): void {
    if (input.frameAnomaly?.status === "ok") {
      addEntry(sections, "contradictions_quarantines", {
        id: `frame_anomaly:${input.frameAnomaly.kind}`,
        source_type: "system_metadata",
        session_scope: "current_session",
        actor: "system",
        trust_rank: QUARANTINE_TRUST_RANK,
        text: input.frameAnomaly.rationale,
        value: input.frameAnomaly.kind,
        state: "quarantined",
        taint: "quarantined",
      });
    }

    for (const entry of streamEntries) {
      if (!isQuarantinedUserEntryMarker(entry)) {
        continue;
      }

      addEntry(sections, "contradictions_quarantines", {
        id: `stream_quarantine:${entry.id}`,
        source_type: "system_metadata",
        session_scope: "current_session",
        actor: "system",
        trust_rank: QUARANTINE_TRUST_RANK,
        text: stringifyPromptContent(entry.content),
        stream_index: resolver.streamOrderById.get(entry.id),
        state: "quarantined",
        taint: "quarantined",
      });
    }

    for (const correction of input.pendingCorrections) {
      addEntry(
        sections,
        "contradictions_quarantines",
        cappedTrustRank({
          id: `review_queue:${correction.id}`,
          source_type: "system_metadata",
          session_scope: reviewQueueScope(correction, resolver),
          actor: "system",
          trust_rank: QUARANTINE_TRUST_RANK,
          text: correction.reason,
          value: correction.kind,
          state: correction.resolved_at === null ? "open" : "resolved",
          taint: "contested",
          ...persistenceClassFromProvenance(
            { streamEntryIds: reviewQueueStreamIds(correction) },
            resolver,
          ),
        }),
      );
    }

    const contradictionCount = input.retrievedSemantic?.contradiction_hits.length ?? 0;

    if (contradictionCount > 0) {
      addEntry(sections, "contradictions_quarantines", {
        id: "semantic_contradictions:retrieved",
        source_type: "system_metadata",
        session_scope: "global",
        actor: "memory",
        trust_rank: QUARANTINE_TRUST_RANK,
        text: `Retrieved semantic contradiction hits: ${contradictionCount}`,
        state: "present",
        taint: "contested",
      });
    }
  }

  private async addActionStates(
    sections: SectionBuckets,
    input: EvidenceLedgerBuildInput,
    resolver: ScopeResolver,
  ): Promise<void> {
    const sourceRecordLimit = normalizePositiveInteger(
      this.options.actionThreadSourceRecordLimit,
      DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
    );
    const renderLimit = normalizePositiveInteger(
      this.options.actionThreadRenderLimit,
      DEFAULT_ACTION_THREAD_RENDER_LIMIT,
    );
    const similarityThreshold = normalizeUnitInterval(
      this.options.actionThreadSimilarityThreshold,
      DEFAULT_ACTION_THREAD_SIMILARITY_THRESHOLD,
    );
    const visibleActions = listVisibleActions(
      this.options.actionRepository,
      input.audienceEntityId,
      input.activeParticipants,
      sourceRecordLimit,
    );
    const threads = await buildActionThreads({
      records: visibleActions,
      repository: this.options.actionRepository,
      resolver,
      similarityThreshold,
    });
    const renderedThreads = threads.slice(0, renderLimit);

    for (const thread of renderedThreads) {
      addEntry(
        sections,
        "action_states",
        cappedTrustRank({
          id: `action_thread:${thread.id}`,
          source_type: "action_record",
          session_scope: thread.scope,
          actor: thread.current.actor === "borg" ? "assistant" : "user",
          trust_rank: ACTION_TRUST_RANK,
          text: renderActionThreadText(thread, this.options.entityRepository),
          value: actionActorDisplay(thread.current.actor, this.options.entityRepository),
          state: actionThreadState(thread),
          state_metadata: actionThreadStateMetadata(thread, this.options.entityRepository),
          taint: "none",
          ...persistenceClassFromProvenance(
            {
              streamEntryIds: thread.records.flatMap(
                (record) => record.provenance_stream_entry_ids,
              ),
              episodeIds: thread.records.flatMap((record) => record.provenance_episode_ids),
            },
            resolver,
          ),
        }),
      );
    }

    if (threads.length <= renderLimit) {
      return;
    }

    const olderThreads = threads.slice(renderLimit);

    addEntry(sections, "action_states", {
      id: "action_threads:older_summary",
      source_type: "system_metadata",
      session_scope: "global",
      actor: "system",
      trust_rank: ACTION_TRUST_RANK,
      text: renderOlderActionThreadsSummary(olderThreads),
      value: "older_action_threads",
      state: "omitted",
      taint: "none",
    });
  }

  private addGroupChannelMemory(
    sections: SectionBuckets,
    input: EvidenceLedgerBuildInput,
    resolver: ScopeResolver,
  ): void {
    const audienceEntityId = input.audienceEntityId;

    if (audienceEntityId === null) {
      return;
    }

    const audienceEntity = this.options.entityRepository?.get(audienceEntityId);

    if (audienceEntity?.kind !== "group") {
      return;
    }

    const displayName = audienceEntity.canonical_name;

    addEntry(sections, "group_channel_memory", {
      id: `group_channel:${audienceEntityId}`,
      source_type: "system_metadata",
      session_scope: "global",
      actor: "system",
      trust_rank: SLOT_TRUST_RANK,
      text: `Group/channel memory for ${displayName}. These entries belong to the channel, not to any active participant.`,
      value: displayName,
      state: "group_channel",
      taint: "none",
    });

    for (const slot of this.options.relationalSlotRepository
      .list({
        subjectEntityId: audienceEntityId,
        states: ["established", "contested", "quarantined"],
        limit: RELATIONAL_SLOT_LEDGER_LIMIT,
      })
      .slice(0, RELATIONAL_SLOT_LEDGER_LIMIT)) {
      addEntry(
        sections,
        "group_channel_memory",
        cappedTrustRank({
          id: `group_relational_slot:${slot.id}`,
          source_type: "relational_slot",
          session_scope: slotScope(slot, resolver),
          actor: "memory",
          trust_rank: SLOT_TRUST_RANK,
          text:
            slot.alternate_values.length === 0
              ? undefined
              : `alternate_values=${slot.alternate_values.map((alternate) => alternate.value).join(", ")}`,
          value: `${slot.slot_key}=${slot.value}`,
          state: slot.state,
          state_metadata: {
            subject_display_name: displayName,
            subject_role: "audience",
          },
          taint: slotTaint(slot),
          ...persistenceClassFromProvenance(
            {
              streamEntryIds: [
                ...slot.evidence_stream_entry_ids,
                ...slot.contradicted_by_stream_entry_ids,
                ...slot.alternate_values.flatMap(
                  (alternate) => alternate.evidence_stream_entry_ids,
                ),
              ],
            },
            resolver,
          ),
        }),
      );
    }

    const scopedCommitments = scopedCommitmentsForEntity(
      this.options.commitmentRepository?.list({
        activeOnly: true,
        audience: audienceEntityId,
      }) ?? input.applicableCommitments,
      audienceEntityId,
    ).filter((commitment) =>
      commitmentBelongsToGroupChannel(commitment, audienceEntityId, this.options.entityRepository),
    );

    for (const commitment of scopedCommitments) {
      addEntry(
        sections,
        "group_channel_memory",
        cappedTrustRank({
          id: `group_commitment:${commitment.id}`,
          source_type: "commitment",
          session_scope: commitmentScope(commitment, resolver),
          actor: "memory",
          trust_rank: COMMITMENT_TRUST_RANK,
          text: commitment.directive,
          value: commitment.directive_family,
          state: "active",
          taint: "none",
          ...persistenceClassFromProvenance(
            { streamEntryIds: commitment.source_stream_entry_ids ?? [] },
            resolver,
          ),
        }),
      );
    }

    const scopedGoals = scopedGoalsForEntity(
      this.options.goalsRepository?.list({
        status: "active",
        visibleToAudienceEntityId: audienceEntityId,
      }) ?? [],
      audienceEntityId,
    ).filter((goal) =>
      goalBelongsToGroupChannel(goal, audienceEntityId, this.options.entityRepository),
    );

    for (const goal of scopedGoals) {
      addEntry(sections, "group_channel_memory", {
        id: `group_goal:${goal.id}`,
        source_type: "system_metadata",
        session_scope: scopeFromStreamIds(goal.source_stream_entry_ids ?? [], resolver),
        actor: "memory",
        trust_rank: OPEN_QUESTION_TRUST_RANK,
        text: goal.description,
        value: "goal",
        state: goal.status,
        taint: "none",
      });
    }

    for (const action of dedupeActions([
      ...this.options.actionRepository
        .list({
          audienceEntityId,
          limit: DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
        })
        .filter((record) =>
          actionBelongsToGroupChannel(record, audienceEntityId, this.options.entityRepository),
        ),
      ...this.options.actionRepository
        .list({
          actor: audienceEntityId,
          limit: DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
        })
        .filter((record) =>
          actionBelongsToGroupChannel(record, audienceEntityId, this.options.entityRepository),
        ),
    ]).slice(0, DEFAULT_ACTION_THREAD_RENDER_LIMIT)) {
      addEntry(
        sections,
        "group_channel_memory",
        cappedTrustRank({
          id: `group_action:${action.id}`,
          source_type: "action_record",
          session_scope: actionScope(action, resolver),
          actor: action.actor === "borg" ? "assistant" : "user",
          trust_rank: ACTION_TRUST_RANK,
          text: action.description,
          value: actionActorDisplay(action.actor, this.options.entityRepository),
          state: action.state,
          taint: "none",
          ...persistenceClassFromProvenance(
            {
              streamEntryIds: action.provenance_stream_entry_ids,
              episodeIds: action.provenance_episode_ids,
            },
            resolver,
          ),
        }),
      );
    }
  }

  private addRelationalSlots(
    sections: SectionBuckets,
    resolver: ScopeResolver,
    audienceEntityId: EntityId | null,
    activeParticipants: readonly ActiveParticipant[] | undefined,
  ): void {
    const activeParticipantIds = visibleAudienceEntityIds(audienceEntityId, activeParticipants);
    const slots =
      activeParticipants === undefined || activeParticipants.length === 0
        ? this.options.relationalSlotRepository.list({
            states: ["established", "contested", "quarantined"],
            limit: RELATIONAL_SLOT_LEDGER_LIMIT,
          })
        : activeParticipants.flatMap((participant) =>
            this.options.relationalSlotRepository.list({
              subjectEntityId: participant.entityId,
              states: ["established", "contested", "quarantined"],
              limit: RELATIONAL_SLOT_LEDGER_LIMIT,
            }),
          );
    const cappedSlots = slots.slice(0, RELATIONAL_SLOT_LEDGER_LIMIT);

    for (const slot of cappedSlots) {
      const participant = participantForSlot(slot, activeParticipants);
      addEntry(
        sections,
        "relational_slots",
        cappedTrustRank({
          id: `relational_slot:${slot.id}`,
          source_type: "relational_slot",
          session_scope: slotScope(slot, resolver),
          actor: "memory",
          trust_rank: SLOT_TRUST_RANK,
          text:
            slot.alternate_values.length === 0
              ? undefined
              : `alternate_values=${slot.alternate_values.map((alternate) => alternate.value).join(", ")}`,
          value: `${slot.slot_key}=${slot.value}`,
          state: slot.state,
          ...optionalStateMetadata(
            slotSubjectStateMetadata(slot, participant, activeParticipants?.length ?? 0),
          ),
          taint: slotTaint(slot),
          ...persistenceClassFromProvenance(
            {
              streamEntryIds: [
                ...slot.evidence_stream_entry_ids,
                ...slot.contradicted_by_stream_entry_ids,
                ...slot.alternate_values.flatMap(
                  (alternate) => alternate.evidence_stream_entry_ids,
                ),
              ],
            },
            resolver,
          ),
        }),
      );
    }

    for (const participant of activeParticipants ?? []) {
      const participantCommitments =
        this.options.commitmentRepository === undefined
          ? []
          : scopedCommitmentsForEntity(
              dedupeCommitments([
                ...this.options.commitmentRepository.list({
                  activeOnly: true,
                  committedByEntity: participant.entityId,
                }),
                ...this.options.commitmentRepository.list({
                  activeOnly: true,
                  audience: participant.entityId,
                }),
              ]),
              participant.entityId,
            ).filter((commitment) =>
              isCommitmentVisibleToSession(commitment, audienceEntityId, activeParticipantIds),
            );

      for (const commitment of participantCommitments) {
        addEntry(
          sections,
          "relational_slots",
          cappedTrustRank({
            id: `participant_commitment:${participant.entityId}:${commitment.id}`,
            source_type: "commitment",
            session_scope: commitmentScope(commitment, resolver),
            actor: "memory",
            trust_rank: COMMITMENT_TRUST_RANK,
            text: commitment.directive,
            value: `${participant.displayName ?? "participant"}:${commitment.directive_family}`,
            state: "active",
            state_metadata: {
              subject_display_name: participant.displayName ?? "participant",
              subject_role: participant.role,
            },
            taint: "none",
            ...persistenceClassFromProvenance(
              { streamEntryIds: commitment.source_stream_entry_ids ?? [] },
              resolver,
            ),
          }),
        );
      }

      const participantGoals =
        this.options.goalsRepository === undefined
          ? []
          : scopedGoalsForEntity(
              dedupeGoals([
                ...this.options.goalsRepository.list({
                  status: "active",
                  ownerEntityId: participant.entityId,
                }),
                ...this.options.goalsRepository.list({
                  status: "active",
                  visibleToAudienceEntityId: participant.entityId,
                }),
              ]),
              participant.entityId,
            ).filter((goal) =>
              isGoalVisibleToSession(goal, audienceEntityId, activeParticipantIds),
            );

      for (const goal of participantGoals) {
        addEntry(sections, "relational_slots", {
          id: `participant_goal:${participant.entityId}:${goal.id}`,
          source_type: "system_metadata",
          session_scope: scopeFromStreamIds(goal.source_stream_entry_ids ?? [], resolver),
          actor: "memory",
          trust_rank: OPEN_QUESTION_TRUST_RANK,
          text: goal.description,
          value: `${participant.displayName ?? "participant"}:goal`,
          state: goal.status,
          state_metadata: {
            subject_display_name: participant.displayName ?? "participant",
            subject_role: participant.role,
          },
          taint: "none",
        });
      }
    }
  }

  private addRetrievedRawStreamEvidence(
    sections: SectionBuckets,
    input: EvidenceLedgerBuildInput,
    resolver: ScopeResolver,
    transcriptStreamIds: ReadonlySet<string>,
  ): void {
    for (const item of input.retrievedEvidence) {
      if (item.source !== "raw_stream" && item.source !== "recent_raw_stream") {
        continue;
      }

      const itemStreamIds = item.provenance?.streamIds ?? [];

      // If every stream ID this retrieval item points to is already in the
      // current_session_transcript section, skip emitting the duplicate
      // retrieved_raw_stream_evidence row. The transcript renders the same
      // underlying content with higher trust rank.
      if (itemStreamIds.length > 0 && itemStreamIds.every((id) => transcriptStreamIds.has(id))) {
        continue;
      }

      const scope = scopeFromStreamIds(itemStreamIds, resolver);
      const streamIndex = streamIndexFromSingleCurrentSessionStreamId(itemStreamIds, resolver);
      addEntry(
        sections,
        "retrieved_raw_stream_evidence",
        cappedTrustRank({
          id: `retrieved_stream:${item.id}`,
          source_type: rawStreamSourceType(scope),
          session_scope: scope,
          actor: rawStreamActor(itemStreamIds, resolver),
          trust_rank: RAW_STREAM_TRUST_RANK,
          text: item.text,
          value: item.source,
          ...(streamIndex === undefined ? {} : { stream_index: streamIndex }),
          state: `score=${item.score.toFixed(2)}`,
          state_metadata:
            itemStreamIds.length === 0 ? undefined : { stream_ids: [...itemStreamIds] },
          taint: "none",
          via_retrieval: true,
          ...persistenceClassFromStreamIds(itemStreamIds, resolver),
        }),
      );
    }
  }

  private addRetrievedStructuredEvidence(
    sections: SectionBuckets,
    input: EvidenceLedgerBuildInput,
    resolver: ScopeResolver,
  ): void {
    for (const item of input.retrievedEvidence) {
      if (item.source === "raw_stream" || item.source === "recent_raw_stream") {
        continue;
      }

      const scope = evidenceItemScope(item, resolver);
      const entry = cappedTrustRank({
        id: `retrieved_evidence:${item.id}`,
        source_type: evidenceItemSourceType(item, scope),
        session_scope: scope,
        actor: "memory" as const,
        trust_rank: item.source === "warm_recall" ? WARM_RECALL_TRUST_RANK : RAW_STREAM_TRUST_RANK,
        text: item.text,
        value: item.source,
        state: evidenceItemState(item),
        state_metadata: evidenceItemProvenanceMetadata(item),
        taint: "none" as const,
        via_retrieval: true,
        ...persistenceClassFromProvenance(
          {
            streamEntryIds: item.provenance?.streamIds ?? [],
            episodeIds: [
              ...(item.provenance?.episodeId === undefined ? [] : [item.provenance.episodeId]),
              ...(item.source_episode_ids ?? []),
            ],
          },
          resolver,
        ),
      });

      if (item.source === "commitment") {
        addEntry(sections, "commitments_and_constraints", {
          ...entry,
          trust_rank: COMMITMENT_TRUST_RANK,
        });
        continue;
      }

      if (item.source === "working_state") {
        addEntry(sections, "closure_discourse_state", {
          ...entry,
          actor: "system",
          trust_rank: DISCOURSE_TRUST_RANK,
        });
        continue;
      }

      if (item.source === "open_question") {
        addEntry(sections, "open_questions", {
          ...entry,
          trust_rank: OPEN_QUESTION_TRUST_RANK,
        });
        continue;
      }

      addEntry(sections, "retrieved_memory_evidence", entry);
    }
  }

  private addEpisodes(
    sections: SectionBuckets,
    input: EvidenceLedgerBuildInput,
    resolver: ScopeResolver,
  ): void {
    for (const result of input.retrievedEpisodes) {
      const scope =
        resolver.episodeScopesById.get(result.episode.id) ??
        scopeFromStreamIds(result.episode.source_stream_ids, resolver);
      addEntry(
        sections,
        "episodes",
        cappedTrustRank({
          id: `episode:${result.episode.id}`,
          source_type: "episode",
          session_scope: scope,
          actor: "memory",
          trust_rank: EPISODE_TRUST_RANK,
          text: result.episode.narrative,
          value: result.episode.title,
          state: `confidence=${result.episode.confidence.toFixed(2)} score=${result.score.toFixed(2)}`,
          state_metadata: {
            episode_id: result.episode.id,
            source_stream_ids: [...result.episode.source_stream_ids],
          },
          taint: "none",
          ...persistenceClassFromProvenance(
            { streamEntryIds: result.episode.source_stream_ids },
            resolver,
          ),
        }),
      );
    }
  }

  private addSemanticGraph(
    sections: SectionBuckets,
    input: EvidenceLedgerBuildInput,
    resolver: ScopeResolver,
  ): void {
    const semantic = input.retrievedSemantic;

    if (semantic === null || semantic === undefined) {
      return;
    }

    for (const node of semantic.matched_nodes) {
      addEntry(
        sections,
        "semantic_graph",
        cappedTrustRank({
          id: `semantic_node:${node.id}`,
          source_type: "semantic_node",
          session_scope: scopeFromEpisodeIds(node.source_episode_ids, resolver),
          actor: "memory",
          trust_rank: SEMANTIC_TRUST_RANK,
          text: node.description,
          value: node.label,
          state: node.under_review === undefined ? node.kind : `under_review:${node.kind}`,
          state_metadata: semanticNodeStateMetadata(node),
          taint: semanticTaint({ underReview: node.under_review }),
          ...persistenceClassFromProvenance({ episodeIds: node.source_episode_ids }, resolver),
        }),
      );
    }

    for (const hit of [
      ...semantic.support_hits,
      ...semantic.causal_hits,
      ...semantic.contradiction_hits,
      ...semantic.category_hits,
    ]) {
      addEntry(
        sections,
        "semantic_graph",
        cappedTrustRank({
          id: `semantic_node:${hit.node.id}`,
          source_type: "semantic_node",
          session_scope: scopeFromEpisodeIds(hit.node.source_episode_ids, resolver),
          actor: "memory",
          trust_rank: SEMANTIC_TRUST_RANK,
          text: hit.node.description,
          value: hit.node.label,
          state:
            hit.node.under_review === undefined ? hit.node.kind : `under_review:${hit.node.kind}`,
          state_metadata: semanticNodeStateMetadata(hit.node),
          taint: semanticTaint({ underReview: hit.node.under_review }),
          ...persistenceClassFromProvenance({ episodeIds: hit.node.source_episode_ids }, resolver),
        }),
      );

      for (const edge of hit.edgePath) {
        addEntry(
          sections,
          "semantic_graph",
          cappedTrustRank({
            id: `semantic_edge:${edge.id}`,
            source_type: "semantic_edge",
            session_scope: scopeFromEpisodeIds(edge.evidence_episode_ids, resolver),
            actor: "memory",
            trust_rank: SEMANTIC_TRUST_RANK,
            text: `${edge.from_node_id} ${edge.relation} ${edge.to_node_id}`,
            value: edge.relation,
            state: edge.valid_to === null ? "valid" : "closed",
            state_metadata: {
              edge_id: edge.id,
              evidence_episode_ids: [...edge.evidence_episode_ids],
            },
            taint: semanticTaint({
              validTo: edge.valid_to,
              invalidatedAt: edge.invalidated_at,
            }),
            ...persistenceClassFromProvenance({ episodeIds: edge.evidence_episode_ids }, resolver),
          }),
        );
      }
    }
  }

  private addOpenQuestions(
    sections: SectionBuckets,
    input: EvidenceLedgerBuildInput,
    resolver: ScopeResolver,
  ): void {
    const questionsById = new Map(input.openQuestions.map((question) => [question.id, question]));

    if (this.options.openQuestionsRepository !== undefined) {
      const streamIds = relevantOpenQuestionStreamIds(input, resolver);
      const episodeIds = relevantOpenQuestionEpisodeIds(input);

      for (const question of this.options.openQuestionsRepository.findByHandles({
        streamEntryIds: [...streamIds],
        episodeIds: [...episodeIds],
        statuses: LIFECYCLE_OPEN_QUESTION_STATUSES,
        visibleToAudienceEntityId: input.audienceEntityId,
      })) {
        questionsById.set(question.id, question);
      }
    }

    for (const question of questionsById.values()) {
      addEntry(
        sections,
        "open_questions",
        cappedTrustRank({
          id: `open_question:${question.id}`,
          source_type: "system_metadata",
          session_scope: openQuestionScope(question, resolver),
          actor: "memory",
          trust_rank: OPEN_QUESTION_TRUST_RANK,
          text: question.question,
          value: question.source,
          state: question.status,
          state_metadata: openQuestionStateMetadata(question),
          taint: "none",
          ...persistenceClassFromProvenance(
            {
              streamEntryIds: openQuestionStreamEntryIds(question),
              episodeIds: openQuestionEpisodeIds(question),
            },
            resolver,
          ),
        }),
      );
    }
  }
}
