import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_CONFIG } from "../../../config/index.js";
import { FakeLLMClient } from "../../../llm/test-support/fake-client.js";
import { sharedStateMigrations } from "../../../memory/decision-artifacts/index.js";
import { SharedStateRepository } from "../../../memory/decision-artifacts/repository.js";
import { openDatabase } from "../../../storage/sqlite/index.js";
import { FixedClock } from "../../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createActionId,
  createEntityId,
  createStreamEntryId,
  type EntityId,
} from "../../../util/ids.js";
import type { ActionRecord } from "../../../memory/actions/index.js";
import type { StreamEntry, StreamReader } from "../../../stream/index.js";
import type { PerceptionResult } from "../../types.js";
import { SHARED_STATE_TOOL_NAME } from "../../shared-state/schema.js";
import { compileSharedStateArtifactForEvidenceLedger } from "./retrieval-phase.js";
import type { TurnPhaseCoordinatorOptions } from "./types.js";

describe("compileSharedStateArtifactForEvidenceLedger", () => {
  const cleanup: Array<() => void> = [];

  afterEach(() => {
    while (cleanup.length > 0) {
      cleanup.pop()?.();
    }
  });

  it("uses the global turn counter for shared-state action canonicalization", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(10_000);
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const actionId = createActionId();
    const streamEntryId = createStreamEntryId();
    const currentUserContent = "The clinic callback follow-up is locked.";
    const currentUserEntry = {
      id: streamEntryId,
      kind: "user_msg",
      content: currentUserContent,
      timestamp: 10_000,
      session_id: DEFAULT_SESSION_ID,
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const action = {
      id: actionId,
      description: "Follow up with the clinic",
      actor: "user",
      audience_entity_id: audienceEntityId,
      state: "committed_to_do",
      updated_at: 9_000,
      session_scope: null,
      scheduled_at: null,
      last_referenced_turn_counter: 2,
      last_referenced_turn_global: null,
    } as ActionRecord;
    const update = vi.fn();
    const llmClient = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 12,
          output_tokens: 8,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_shared_state",
              name: SHARED_STATE_TOOL_NAME,
              input: {
                operations: [
                  {
                    type: "add",
                    state_key: "decision.route",
                    kind: "locked",
                    text: "The clinic callback follow-up is locked.",
                    owner_entity_id: audienceEntityId,
                    source_stream_entry_ids: [streamEntryId],
                    canonicalizes: {
                      action_ids: [actionId],
                    },
                  },
                ],
              },
            },
          ],
        },
      ],
    });
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: {
                enabled: false,
              },
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () => llmClient,
      clock,
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        resolve: () => selfEntityId,
      },
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [action],
        get: () => action,
        update,
      },
      goalsRepository: {
        list: () => [],
      },
      commitmentRepository: {
        list: () => [],
      },
      openQuestionsRepository: {
        list: () => [],
      },
      createStreamReader: () =>
        ({
          async *iterate() {
            yield currentUserEntry;
          },
        }) as StreamReader,
    } as unknown as TurnPhaseCoordinatorOptions;

    await compileSharedStateArtifactForEvidenceLedger({
      options,
      input: {
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-global-canonicalization",
        audienceEntityId,
        currentUserMessage: currentUserContent,
        currentUserEntry,
        globalTurnCounter: 42,
        workingMemory: {
          turn_counter: 3,
        } as never,
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [],
        participantRoster: null,
        isUserTurn: true,
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [
          {
            id: "current_user_message",
            label: "1. Current User Message",
            entries: [
              {
                id: `current_user_message:${streamEntryId}`,
                source_type: "current_user_message",
                session_scope: "current_session",
                actor: "user",
                trust_rank: 0,
                text: currentUserContent,
              },
            ],
          },
        ],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "Action candidate: Follow up with the clinic.",
    });

    expect(update).toHaveBeenCalledWith(
      actionId,
      expect.objectContaining({
        last_referenced_turn_counter: 42,
        last_referenced_turn_global: 42,
      }),
      { skipSideEffects: true },
    );
  });
});
