import type { ActionRecord, ActionRepository } from "../../memory/actions/index.js";
import type { CommitmentRecord } from "../../memory/commitments/index.js";
import type {
  RelationalSlot,
  RelationalSlotRepository,
} from "../../memory/relational-slots/index.js";
import type {
  OpenQuestion,
  OpenQuestionsRepository,
  OpenQuestionStatus,
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
const PRIOR_SESSION_TRUST_RANK_CAP = 30;

const RELATIONAL_SLOT_LEDGER_LIMIT = 64;
const ACTION_LEDGER_LIMIT = 64;
const LIFECYCLE_OPEN_QUESTION_STATUSES = ["resolved", "abandoned"] as const satisfies readonly
  OpenQuestionStatus[];

export type EvidenceLedgerBuilderOptions = {
  createStreamReader: (sessionId: SessionId) => StreamReader;
  relationalSlotRepository: Pick<RelationalSlotRepository, "list">;
  actionRepository: Pick<ActionRepository, "list">;
  openQuestionsRepository?: Pick<OpenQuestionsRepository, "findByHandles">;
  currentSessionTranscriptTokenBudget: number;
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

  if (entry.kind === "agent_msg" || entry.kind === "agent_suppressed") {
    return "assistant";
  }

  return "system";
}

function transcriptState(entry: TranscriptStreamEntry): string | undefined {
  return entry.kind === "agent_suppressed" ? "suppressed" : undefined;
}

function streamPersistenceClass(entry: Pick<StreamEntry, "persistence_class">) {
  return entry.persistence_class === undefined
    ? {}
    : { persistence_class: entry.persistence_class };
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

function persistenceClassFromStreamIds(
  streamEntryIds: readonly StreamEntryId[] | readonly string[] | undefined,
  resolver: ScopeResolver,
) {
  if (streamEntryIds === undefined) {
    return {};
  }

  return streamEntryIds.some(
    (streamEntryId) =>
      resolver.streamEntriesById.get(streamEntryId)?.persistence_class ===
      "assistant_self_report",
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

function listVisibleActions(
  actionRepository: Pick<ActionRepository, "list">,
  audienceEntityId: EntityId | null,
): ActionRecord[] {
  const records =
    audienceEntityId === null
      ? actionRepository.list({ audienceEntityId: null, limit: ACTION_LEDGER_LIMIT })
      : [
          ...actionRepository.list({ audienceEntityId: null, limit: ACTION_LEDGER_LIMIT }),
          ...actionRepository.list({ audienceEntityId, limit: ACTION_LEDGER_LIMIT }),
        ];

  return records
    .sort((left, right) => right.updated_at - left.updated_at || left.id.localeCompare(right.id))
    .slice(0, ACTION_LEDGER_LIMIT);
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

export function summarizeEvidenceLedgerTrace(ledger: EvidenceLedger) {
  // Sprint 8d.3: per-section token accounting. v36 mean input tokens were
  // ~113k -- without per-section breakdown there is no way to attribute
  // that load to specific bands. Surfacing it in the trace lets us tell
  // ledger-side bloat (transcript, action records) apart from
  // retrieval-side bloat (semantic walks, episodes).
  return {
    entryCountsBySection: Object.fromEntries(
      ledger.sections.map((section) => [section.id, section.entries.length]),
    ) as Record<EvidenceLedgerSectionId, number>,
    estimatedTokensBySection: Object.fromEntries(
      ledger.sections.map((section) => [section.id, estimateSectionTokens(section)]),
    ) as Record<EvidenceLedgerSectionId, number>,
    transcriptIncluded: ledger.transcriptIncluded,
    transcriptOmittedReason: ledger.transcriptOmittedReason,
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
    const transcriptTokens = estimateTranscriptTokens(transcriptEntries);
    const transcriptIncluded = transcriptTokens <= this.options.currentSessionTranscriptTokenBudget;
    const sections = createSectionBuckets();

    this.addCurrentUserMessage(sections, input, resolver);
    this.addTranscript(sections, transcriptEntries, transcriptIncluded, resolver);
    this.addCommitmentsAndConstraints(sections, input, resolver);
    this.addDiscourseState(sections, input, resolver);
    this.addContradictionsAndQuarantines(sections, input, streamEntries, resolver);
    this.addActionStates(sections, input, resolver);
    this.addRelationalSlots(sections, resolver);
    this.addRetrievedRawStreamEvidence(sections, input, resolver);
    this.addEpisodes(sections, input, resolver);
    this.addSemanticGraph(sections, input, resolver);
    this.addOpenQuestions(sections, input, resolver);

    const orderedSections = finalSections(sections);

    return {
      sections: orderedSections,
      transcriptIncluded,
      ...(transcriptIncluded ? {} : { transcriptOmittedReason: "over_budget" as const }),
      estimatedTokens: estimateLedgerTokens(orderedSections),
    };
  }

  private addCurrentUserMessage(
    sections: SectionBuckets,
    input: EvidenceLedgerBuildInput,
    resolver: ScopeResolver,
  ): void {
    addEntry(sections, "current_user_message", {
      id: `current_user_message:${input.currentUserEntry?.id ?? input.turnId ?? "unpersisted"}`,
      source_type: "current_user_message",
      session_scope: "current_session",
      actor: "user",
      trust_rank: CURRENT_USER_TRUST_RANK,
      text: input.currentUserMessage,
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

  private addTranscript(
    sections: SectionBuckets,
    transcriptEntries: readonly TranscriptStreamEntry[],
    transcriptIncluded: boolean,
    resolver: ScopeResolver,
  ): void {
    if (!transcriptIncluded) {
      return;
    }

    for (const entry of transcriptEntries) {
      addEntry(sections, "current_session_transcript", {
        id: `current_session_stream:${entry.id}`,
        source_type: "current_session_stream",
        session_scope: "current_session",
        actor: actorForStreamEntry(entry),
        trust_rank: TRANSCRIPT_TRUST_RANK,
        text: stringifyPromptContent(entry.content),
        stream_index: resolver.streamOrderById.get(entry.id),
        state: transcriptState(entry),
        taint: "none",
        ...streamPersistenceClass(entry),
      });
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

  private addActionStates(
    sections: SectionBuckets,
    input: EvidenceLedgerBuildInput,
    resolver: ScopeResolver,
  ): void {
    for (const action of listVisibleActions(
      this.options.actionRepository,
      input.audienceEntityId,
    )) {
      addEntry(
        sections,
        "action_states",
        cappedTrustRank({
          id: `action_record:${action.id}`,
          source_type: "action_record",
          session_scope: actionScope(action, resolver),
          actor: action.actor === "borg" ? "assistant" : "user",
          trust_rank: ACTION_TRUST_RANK,
          text: action.description,
          value: action.actor,
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

  private addRelationalSlots(sections: SectionBuckets, resolver: ScopeResolver): void {
    const slots = this.options.relationalSlotRepository.list({
      states: ["established", "contested", "quarantined"],
      limit: RELATIONAL_SLOT_LEDGER_LIMIT,
    });

    for (const slot of slots) {
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
  }

  private addRetrievedRawStreamEvidence(
    sections: SectionBuckets,
    input: EvidenceLedgerBuildInput,
    resolver: ScopeResolver,
  ): void {
    for (const item of input.retrievedEvidence) {
      if (item.source !== "raw_stream" && item.source !== "recent_raw_stream") {
        continue;
      }

      const scope = scopeFromStreamIds(item.provenance?.streamIds ?? [], resolver);
      const streamIndex = streamIndexFromSingleCurrentSessionStreamId(
        item.provenance?.streamIds,
        resolver,
      );
      addEntry(
        sections,
        "retrieved_raw_stream_evidence",
        cappedTrustRank({
          id: `retrieved_stream:${item.id}`,
          source_type: rawStreamSourceType(scope),
          session_scope: scope,
          actor: rawStreamActor(item.provenance?.streamIds, resolver),
          trust_rank: RAW_STREAM_TRUST_RANK,
          text: item.text,
          value: item.source,
          ...(streamIndex === undefined ? {} : { stream_index: streamIndex }),
          state: `score=${item.score.toFixed(2)}`,
          taint: "none",
          via_retrieval: true,
          ...persistenceClassFromStreamIds(item.provenance?.streamIds, resolver),
        }),
      );
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
            taint: semanticTaint({
              validTo: edge.valid_to,
              invalidatedAt: edge.invalidated_at,
            }),
            ...persistenceClassFromProvenance(
              { episodeIds: edge.evidence_episode_ids },
              resolver,
            ),
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
