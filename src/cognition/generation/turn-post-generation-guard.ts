import type { LLMClient } from "../../llm/index.js";
import type { PostGenerationGuardMode } from "../../config/index.js";
import type { ActionRecord, ActionRepository } from "../../memory/actions/index.js";
import type { CommitmentRecord } from "../../memory/commitments/index.js";
import type {
  RelationalSlot,
  RelationalSlotRepository,
} from "../../memory/relational-slots/index.js";
import type {
  ClosureLoopState,
  ClosurePressureHistoryEntry,
  RecentSuppressionEntry,
} from "../../memory/working/index.js";
import type { RetrievedEpisode } from "../../retrieval/index.js";
import {
  activeSessionTranscriptEntries,
  type StreamEntry,
  type StreamReader,
} from "../../stream/index.js";
import type { Clock } from "../../util/clock.js";
import type { EntityId, SessionId } from "../../util/ids.js";
import { listActionCandidatesForCognition } from "../evidence-ledger/action-threads.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { ClosureLoopDialogueAct } from "./closure-loop.js";
import { ClosurePressureGuard } from "./closure-pressure-guard.js";
import type { PendingTurnEmission } from "./types.js";

const COMPLETED_ACTION_LIMIT = 8;
const RELATIONAL_SLOT_GUARD_LIMIT = 64;
const INTERNAL_IDENTIFIER_RECENT_STREAM_MAX_ENTRIES = 512;
const INTERNAL_IDENTIFIER_RECENT_STREAM_MAX_BYTES = 4 * 1024 * 1024;
const INTERNAL_IDENTIFIER_EXACT_PATTERN =
  /^(?:strm|sess|ep|goal|val|trt|abp|grw|oq|semn|seme|cmt|ent|act|rslot|skl|procevi|run|exstep)_[a-z0-9]{16}$|^autonomy_wake_[a-f0-9]{16}$|^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

type TurnPostGenerationGuardEmission = Extract<
  PendingTurnEmission,
  { kind: "message" | "suppressed" }
>;
type TurnPostGenerationGuardMessage = Extract<PendingTurnEmission, { kind: "message" }>;

export type TurnPostGenerationGuardRunnerOptions = {
  auditModel: string;
  closurePressureMode: PostGenerationGuardMode;
  createStreamReader: (sessionId: SessionId) => StreamReader;
  actionRepository: Pick<ActionRepository, "list">;
  relationalSlotRepository: Pick<RelationalSlotRepository, "list">;
  clock: Clock;
  tracer: TurnTracer;
};

export type RunTurnPostGenerationGuardInput = {
  llmClient: LLMClient;
  turnId: string;
  response: string;
  sessionId: SessionId;
  persistedUserEntry?: StreamEntry;
  persistedUserEntries?: readonly StreamEntry[];
  retrievedEpisodes: readonly RetrievedEpisode[];
  activeCommitments: readonly CommitmentRecord[];
  closureLoop: ClosureLoopState | null;
  closurePressureHistory?: readonly ClosurePressureHistoryEntry[];
  recentSuppressions?: readonly RecentSuppressionEntry[];
  currentUserClosureKind?: ClosureLoopDialogueAct | null;
  currentTurn?: number;
  audienceEntityId: EntityId | null;
  knownInternalIdentifiers?: readonly string[];
};

function addInternalIdentifier(identifiers: Set<string>, value: string | null | undefined): void {
  if (value !== undefined && value !== null && INTERNAL_IDENTIFIER_EXACT_PATTERN.test(value)) {
    identifiers.add(value);
  }
}

function addInternalIdentifiers(
  identifiers: Set<string>,
  values: readonly (string | null | undefined)[],
): void {
  for (const value of values) {
    addInternalIdentifier(identifiers, value);
  }
}

function collectInternalIdentifiers(input: {
  turnId: string;
  sessionId: SessionId;
  persistedUserEntry?: StreamEntry;
  persistedUserEntries?: readonly StreamEntry[];
  currentSessionStreamEntries: readonly StreamEntry[];
  retrievedEpisodes: readonly RetrievedEpisode[];
  activeCommitments: readonly CommitmentRecord[];
  closurePressureHistory: readonly ClosurePressureHistoryEntry[];
  recentSuppressions: readonly RecentSuppressionEntry[];
  relationalSlots: readonly RelationalSlot[];
  recentCompletedActions: readonly ActionRecord[];
  audienceEntityId: EntityId | null;
  knownInternalIdentifiers: readonly string[];
}): string[] {
  const identifiers = new Set<string>();

  addInternalIdentifiers(identifiers, input.knownInternalIdentifiers);
  addInternalIdentifier(identifiers, input.turnId);
  addInternalIdentifier(identifiers, input.sessionId);
  addInternalIdentifier(identifiers, input.persistedUserEntry?.id);
  addInternalIdentifier(identifiers, input.persistedUserEntry?.session_id);
  for (const entry of input.persistedUserEntries ?? []) {
    addInternalIdentifier(identifiers, entry.id);
    addInternalIdentifier(identifiers, entry.session_id);
    addInternalIdentifier(identifiers, entry.sender_entity_id);
  }
  addInternalIdentifier(identifiers, input.audienceEntityId);
  for (const entry of input.currentSessionStreamEntries) {
    addInternalIdentifier(identifiers, entry.id);
    addInternalIdentifier(identifiers, entry.session_id);
    addInternalIdentifier(identifiers, entry.sender_entity_id);
    addInternalIdentifier(identifiers, entry.reply_target_entity_id);
  }

  for (const result of input.retrievedEpisodes) {
    addInternalIdentifier(identifiers, result.episode.id);
    addInternalIdentifier(identifiers, result.episode.audience_entity_id ?? null);
    addInternalIdentifiers(identifiers, result.episode.origin_audience_entity_ids ?? []);
    if (result.disclosureLabel !== undefined) {
      addInternalIdentifiers(identifiers, result.disclosureLabel.originAudienceEntityIds);
      addInternalIdentifiers(identifiers, result.disclosureLabel.privateToEntityIds);
      addInternalIdentifiers(identifiers, result.disclosureLabel.publicToEntityIds);
    }
    addInternalIdentifiers(identifiers, result.episode.source_stream_ids);
    addInternalIdentifiers(identifiers, result.episode.lineage.derived_from);
    addInternalIdentifiers(identifiers, result.episode.lineage.supersedes);
    addInternalIdentifiers(
      identifiers,
      result.citationChain.flatMap((entry) => [entry.id, entry.session_id]),
    );
  }

  for (const commitment of input.activeCommitments) {
    addInternalIdentifier(identifiers, commitment.id);
    addInternalIdentifier(identifiers, commitment.made_to_entity);
    addInternalIdentifier(identifiers, commitment.restricted_audience);
    addInternalIdentifier(identifiers, commitment.about_entity);
    addInternalIdentifier(identifiers, commitment.superseded_by);
    addInternalIdentifiers(identifiers, commitment.source_stream_entry_ids ?? []);
  }

  for (const entry of input.closurePressureHistory) {
    addInternalIdentifier(identifiers, entry.turn_id);
  }

  for (const entry of input.recentSuppressions) {
    addInternalIdentifier(identifiers, entry.turn_id);
  }

  for (const slot of input.relationalSlots) {
    addInternalIdentifier(identifiers, slot.id);
    addInternalIdentifier(identifiers, slot.subject_entity_id);
    addInternalIdentifiers(identifiers, slot.evidence_stream_entry_ids);
    addInternalIdentifiers(identifiers, slot.contradicted_by_stream_entry_ids);
    for (const alternate of slot.alternate_values) {
      addInternalIdentifiers(identifiers, alternate.evidence_stream_entry_ids);
    }
  }

  for (const action of input.recentCompletedActions) {
    addInternalIdentifier(identifiers, action.id);
    addInternalIdentifier(identifiers, action.actor);
    addInternalIdentifier(identifiers, action.audience_entity_id);
    addInternalIdentifiers(identifiers, action.provenance_episode_ids);
    addInternalIdentifiers(identifiers, action.provenance_stream_entry_ids);
  }

  return [...identifiers].sort();
}

function leakedInternalIdentifiers(
  response: string,
  knownIdentifiers: readonly string[],
): string[] {
  const leaked = new Set<string>();

  for (const identifier of knownIdentifiers) {
    if (response.includes(identifier)) {
      leaked.add(identifier);
    }
  }

  return [...leaked].sort();
}

function applyInternalIdentifierGuard(input: {
  turnId: string;
  sessionId?: SessionId;
  emission: TurnPostGenerationGuardMessage;
  knownIdentifiers: readonly string[];
  tracer: TurnTracer;
}): TurnPostGenerationGuardEmission {
  const leakedIdentifiers = leakedInternalIdentifiers(
    input.emission.content,
    input.knownIdentifiers,
  );

  if (leakedIdentifiers.length === 0) {
    return input.emission;
  }

  if (input.tracer.enabled) {
    input.tracer.emit("internal_identifier_guard.completed", {
      turnId: input.turnId,
      ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
      verdict: "suppressed",
      leaked_identifiers: leakedIdentifiers,
    });
  }

  return {
    kind: "suppressed",
    reason: "internal_identifier_leak",
    ...(input.emission.closure_pressure_history_reason === undefined
      ? {}
      : { closure_pressure_history_reason: input.emission.closure_pressure_history_reason }),
  };
}

export class TurnPostGenerationGuardRunner {
  constructor(private readonly options: TurnPostGenerationGuardRunnerOptions) {}

  async run(input: RunTurnPostGenerationGuardInput): Promise<TurnPostGenerationGuardEmission> {
    const currentSessionStreamEntries = await this.loadStreamEntries(input.sessionId);
    const relationalSlots = this.options.relationalSlotRepository.list({
      limit: RELATIONAL_SLOT_GUARD_LIMIT,
    });
    const recentCompletedActions = this.listRecentCompletedActionsForCognition(
      input.audienceEntityId,
    );
    const recentCompletedActionsForInternalIdentifierGuard =
      this.listRecentCompletedActionsForInternalIdentifierGuard(recentCompletedActions);

    const closureGuard = new ClosurePressureGuard({
      llmClient: input.llmClient,
      auditModel: this.options.auditModel,
      mode: this.options.closurePressureMode,
      tracer: this.options.tracer,
    });
    const closureResult = await closureGuard.run({
      turnId: input.turnId,
      sessionId: input.sessionId,
      response: input.response,
      activeCommitments: input.activeCommitments,
      closureLoop: input.closureLoop,
      closurePressureHistory: input.closurePressureHistory,
      currentUserClosureKind: input.currentUserClosureKind,
      currentTurn: input.currentTurn,
      nowMs: this.options.clock.now(),
    });

    if (closureResult.emission.kind === "suppressed") {
      return closureResult.emission;
    }

    return applyInternalIdentifierGuard({
      turnId: input.turnId,
      sessionId: input.sessionId,
      emission: closureResult.emission,
      knownIdentifiers: collectInternalIdentifiers({
        turnId: input.turnId,
        sessionId: input.sessionId,
        persistedUserEntry: input.persistedUserEntry,
        persistedUserEntries: input.persistedUserEntries,
        currentSessionStreamEntries,
        retrievedEpisodes: input.retrievedEpisodes,
        activeCommitments: input.activeCommitments,
        closurePressureHistory: input.closurePressureHistory ?? [],
        recentSuppressions: input.recentSuppressions ?? [],
        relationalSlots,
        recentCompletedActions: recentCompletedActionsForInternalIdentifierGuard,
        audienceEntityId: input.audienceEntityId,
        knownInternalIdentifiers: input.knownInternalIdentifiers ?? [],
      }),
      tracer: this.options.tracer,
    });
  }

  listRecentCompletedActionsForCognition(audienceEntityId: EntityId | null): ActionRecord[] {
    return listActionCandidatesForCognition({
      actionRepository: this.options.actionRepository,
      audienceEntityId,
      state: "completed",
      limit: COMPLETED_ACTION_LIMIT,
    }).map((candidate) => candidate.record);
  }

  private listRecentCompletedActionsForInternalIdentifierGuard(
    modelContextActions: readonly ActionRecord[],
  ): ActionRecord[] {
    return [...modelContextActions];
  }

  private async loadStreamEntries(sessionId: SessionId): Promise<StreamEntry[]> {
    const reader = this.options.createStreamReader(sessionId);
    const entries = new Map<string, StreamEntry>();

    for (const entry of activeSessionTranscriptEntries(
      reader.scanReverse({
        maxEntries: INTERNAL_IDENTIFIER_RECENT_STREAM_MAX_ENTRIES,
        maxBytes: INTERNAL_IDENTIFIER_RECENT_STREAM_MAX_BYTES,
      }).entries,
    )) {
      entries.set(entry.id, entry);
    }

    return [...entries.values()];
  }
}
