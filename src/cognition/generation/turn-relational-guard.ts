import type { LLMClient } from "../../llm/index.js";
import type { PostGenerationGuardMode, RelationalClaimGuardMode } from "../../config/index.js";
import type { ActionRecord, ActionRepository } from "../../memory/actions/index.js";
import type { CommitmentRecord, CommitmentRepository } from "../../memory/commitments/index.js";
import type {
  RelationalSlot,
  RelationalSlotRepository,
} from "../../memory/relational-slots/index.js";
import type { ClosureLoopState, ClosurePressureHistoryEntry } from "../../memory/working/index.js";
import type { RetrievedEpisode } from "../../retrieval/index.js";
import {
  loadActiveSessionTranscriptEntries,
  type StreamEntry,
  type StreamReader,
} from "../../stream/index.js";
import type { Clock } from "../../util/clock.js";
import type { EntityId, SessionId } from "../../util/ids.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { ClosureLoopDialogueAct } from "./closure-loop.js";
import { ClosurePressureGuard } from "./closure-pressure-guard.js";
import type { PendingTurnEmission } from "./types.js";
import {
  RelationalClaimGuard,
  actionRecordToRelationalGuardEvidence,
  commitmentToRelationalGuardEvidence,
  correctivePreferencesFromCommitments,
  relationalSlotToRelationalGuardEvidence,
  retrievedEpisodeToRelationalGuardEvidence,
  streamEntryToRelationalGuardEvidence,
  type RelationalGuardCurrentUserMessage,
  type RelationalGuardStreamEvidence,
} from "./relational-guard.js";

const COMPLETED_ACTION_LIMIT = 8;
const RELATIONAL_SLOT_GUARD_LIMIT = 64;
const INTERNAL_IDENTIFIER_EXACT_PATTERN =
  /^(?:strm|sess|ep|goal|val|trt|abp|grw|oq|semn|seme|cmt|ent|act|rslot|skl|procevi|run|exstep)_[a-z0-9]{16}$|^autonomy_wake_[a-f0-9]{16}$/;
const INTERNAL_IDENTIFIER_SCAN_PATTERN =
  /(?:strm|sess|ep|goal|val|trt|abp|grw|oq|semn|seme|cmt|ent|act|rslot|skl|procevi|run|exstep)_[a-z0-9]{16}|autonomy_wake_[a-f0-9]{16}/g;

export type TurnRelationalGuardRunnerOptions = {
  auditModel: string;
  rewriteModel: string;
  relationalClaimMode: RelationalClaimGuardMode;
  closurePressureMode: PostGenerationGuardMode;
  createStreamReader: (sessionId: SessionId) => StreamReader;
  actionRepository: Pick<ActionRepository, "list">;
  commitmentRepository: Pick<CommitmentRepository, "findByEvidenceStreamEntryId">;
  relationalSlotRepository: Pick<RelationalSlotRepository, "list">;
  clock: Clock;
  tracer: TurnTracer;
};

export type RunTurnRelationalGuardInput = {
  llmClient: LLMClient;
  turnId: string;
  response: string;
  userMessage: string;
  sessionId: SessionId;
  persistedUserEntry?: StreamEntry;
  retrievedEpisodes: readonly RetrievedEpisode[];
  activeCommitments: readonly CommitmentRecord[];
  closureLoop: ClosureLoopState | null;
  closurePressureHistory?: readonly ClosurePressureHistoryEntry[];
  currentUserClosureKind?: ClosureLoopDialogueAct | null;
  currentTurn?: number;
  audienceEntityId: EntityId | null;
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
  sessionId: SessionId;
  persistedUserEntry?: StreamEntry;
  currentSessionStreamEntries: readonly RelationalGuardStreamEvidence[];
  retrievedEpisodes: readonly RetrievedEpisode[];
  activeCommitments: readonly CommitmentRecord[];
  relationalSlots: readonly RelationalSlot[];
  recentCompletedActions: readonly ActionRecord[];
  audienceEntityId: EntityId | null;
}): string[] {
  const identifiers = new Set<string>();

  addInternalIdentifier(identifiers, input.sessionId);
  addInternalIdentifier(identifiers, input.persistedUserEntry?.id);
  addInternalIdentifier(identifiers, input.persistedUserEntry?.session_id);
  addInternalIdentifier(identifiers, input.audienceEntityId);
  addInternalIdentifiers(
    identifiers,
    input.currentSessionStreamEntries.map((entry) => entry.entry_id),
  );

  for (const result of input.retrievedEpisodes) {
    addInternalIdentifier(identifiers, result.episode.id);
    addInternalIdentifier(identifiers, result.episode.audience_entity_id ?? null);
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

  for (const match of response.matchAll(INTERNAL_IDENTIFIER_SCAN_PATTERN)) {
    leaked.add(match[0]!);
  }

  return [...leaked].sort();
}

function applyInternalIdentifierGuard(input: {
  turnId: string;
  response: string;
  knownIdentifiers: readonly string[];
  tracer: TurnTracer;
}): PendingTurnEmission {
  const leakedIdentifiers = leakedInternalIdentifiers(input.response, input.knownIdentifiers);

  if (leakedIdentifiers.length === 0) {
    return {
      kind: "message",
      content: input.response,
    };
  }

  if (input.tracer.enabled) {
    input.tracer.emit("internal_identifier_guard", {
      turnId: input.turnId,
      verdict: "suppressed",
      leaked_identifiers: leakedIdentifiers,
    });
  }

  return {
    kind: "suppressed",
    reason: "internal_identifier_leak",
  };
}

export class TurnRelationalGuardRunner {
  constructor(private readonly options: TurnRelationalGuardRunnerOptions) {}

  async run(input: RunTurnRelationalGuardInput): Promise<PendingTurnEmission> {
    const currentUserMessage: RelationalGuardCurrentUserMessage | null =
      input.persistedUserEntry === undefined
        ? null
        : {
            text: input.userMessage,
            stream_entry_id: input.persistedUserEntry.id,
            ts: input.persistedUserEntry.timestamp,
          };
    const currentSessionStreamEntries = await this.loadStreamEvidence(input.sessionId);
    const relationalSlots = this.options.relationalSlotRepository.list({
      limit: RELATIONAL_SLOT_GUARD_LIMIT,
    });
    const recentCompletedActions = this.listRecentCompletedActions(input.audienceEntityId);
    const guard = new RelationalClaimGuard({
      llmClient: input.llmClient,
      auditModel: this.options.auditModel,
      rewriteModel: this.options.rewriteModel,
      mode: this.options.relationalClaimMode,
      tracer: this.options.tracer,
      hasCorrectivePreferenceEvidence: (entryId) =>
        this.options.commitmentRepository.findByEvidenceStreamEntryId(entryId),
    });
    const result = await guard.run({
      turnId: input.turnId,
      response: input.response,
      currentSessionId: input.sessionId,
      currentTurnTs: input.persistedUserEntry?.timestamp ?? this.options.clock.now(),
      evidence: {
        current_user_message: currentUserMessage,
        current_session_stream_entries: currentSessionStreamEntries,
        retrieved_episodes: input.retrievedEpisodes.map(retrievedEpisodeToRelationalGuardEvidence),
        active_commitments: input.activeCommitments.map(commitmentToRelationalGuardEvidence),
        corrective_preferences: correctivePreferencesFromCommitments(input.activeCommitments),
        relational_slots: relationalSlots.map(relationalSlotToRelationalGuardEvidence),
        recent_completed_actions: recentCompletedActions.map(actionRecordToRelationalGuardEvidence),
      },
    });

    if (result.emission.kind === "suppressed") {
      return result.emission;
    }

    const closureGuard = new ClosurePressureGuard({
      llmClient: input.llmClient,
      auditModel: this.options.auditModel,
      rewriteModel: this.options.rewriteModel,
      mode: this.options.closurePressureMode,
      tracer: this.options.tracer,
    });
    const closureResult = await closureGuard.run({
      turnId: input.turnId,
      response: result.emission.content,
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
      response: closureResult.emission.content,
      knownIdentifiers: collectInternalIdentifiers({
        sessionId: input.sessionId,
        persistedUserEntry: input.persistedUserEntry,
        currentSessionStreamEntries,
        retrievedEpisodes: input.retrievedEpisodes,
        activeCommitments: input.activeCommitments,
        relationalSlots,
        recentCompletedActions,
        audienceEntityId: input.audienceEntityId,
      }),
      tracer: this.options.tracer,
    });
  }

  listRecentCompletedActions(audienceEntityId: EntityId | null): ActionRecord[] {
    const visibleActions =
      audienceEntityId === null
        ? this.options.actionRepository.list({
            state: "completed",
            audienceEntityId: null,
            limit: COMPLETED_ACTION_LIMIT,
          })
        : [
            ...this.options.actionRepository.list({
              state: "completed",
              audienceEntityId: null,
              limit: COMPLETED_ACTION_LIMIT,
            }),
            ...this.options.actionRepository.list({
              state: "completed",
              audienceEntityId,
              limit: COMPLETED_ACTION_LIMIT,
            }),
          ];

    return visibleActions
      .sort((left, right) => right.updated_at - left.updated_at || left.id.localeCompare(right.id))
      .slice(0, COMPLETED_ACTION_LIMIT);
  }

  private async loadStreamEvidence(sessionId: SessionId): Promise<RelationalGuardStreamEvidence[]> {
    const reader = this.options.createStreamReader(sessionId);
    const entries = new Map<string, RelationalGuardStreamEvidence>();

    for (const entry of await loadActiveSessionTranscriptEntries(reader)) {
      const evidence = streamEntryToRelationalGuardEvidence(entry);

      if (evidence !== null) {
        entries.set(evidence.entry_id, evidence);
      }
    }

    return [...entries.values()];
  }
}
