import type { LLMClient } from "../../llm/index.js";
import type { ActionRecord, ActionRepository } from "../../memory/actions/index.js";
import type { CommitmentRecord, CommitmentRepository } from "../../memory/commitments/index.js";
import type { RelationalSlotRepository } from "../../memory/relational-slots/index.js";
import type { RetrievedEpisode } from "../../retrieval/index.js";
import {
  filterActiveStreamEntries,
  type StreamEntry,
  type StreamReader,
} from "../../stream/index.js";
import type { Clock } from "../../util/clock.js";
import type { EntityId, SessionId, StreamEntryId } from "../../util/ids.js";
import type { TurnTracer } from "../tracing/tracer.js";
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

export type TurnRelationalGuardRunnerOptions = {
  auditModel: string;
  rewriteModel: string;
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
  audienceEntityId: EntityId | null;
};

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
    const guard = new RelationalClaimGuard({
      llmClient: input.llmClient,
      auditModel: this.options.auditModel,
      rewriteModel: this.options.rewriteModel,
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
        current_session_stream_entries: await this.loadStreamEvidence(input.sessionId),
        retrieved_episodes: input.retrievedEpisodes.map(retrievedEpisodeToRelationalGuardEvidence),
        active_commitments: input.activeCommitments.map(commitmentToRelationalGuardEvidence),
        corrective_preferences: correctivePreferencesFromCommitments(input.activeCommitments),
        relational_slots: this.options.relationalSlotRepository
          .list({ limit: RELATIONAL_SLOT_GUARD_LIMIT })
          .map(relationalSlotToRelationalGuardEvidence),
        recent_completed_actions: this.listRecentCompletedActions(input.audienceEntityId).map(
          actionRecordToRelationalGuardEvidence,
        ),
        trusted_runtime_evidence: [],
      },
    });

    return result.emission;
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
    const streamEntries: StreamEntry[] = [];
    const entries = new Map<StreamEntryId, RelationalGuardStreamEvidence>();

    for await (const entry of reader.iterate()) {
      streamEntries.push(entry);
    }

    for (const entry of filterActiveStreamEntries(streamEntries)) {
      const evidence = streamEntryToRelationalGuardEvidence(entry);

      if (evidence !== null) {
        entries.set(evidence.entry_id, evidence);
      }
    }

    return [...entries.values()];
  }
}
