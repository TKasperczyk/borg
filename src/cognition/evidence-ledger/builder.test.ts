import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { ActionRecord, ActionRecordListFilter } from "../../memory/actions/index.js";
import type {
  CommitmentListOptions,
  CommitmentRecord,
  EntityRecord,
} from "../../memory/commitments/index.js";
import type { RelationalSlot } from "../../memory/relational-slots/index.js";
import {
  OpenQuestionsRepository,
  type GoalListOptions,
  type GoalRecord,
  type GoalTreeNode,
  type OpenQuestion,
} from "../../memory/self/index.js";
import type { OpenCommitmentReconciliationStatus } from "../../memory/review-queue/index.js";
import { selfMigrations } from "../../memory/self/migrations.js";
import type { EvidenceItem, RetrievedEpisode, RetrievedSemantic } from "../../retrieval/index.js";
import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../../tracing/tracer.js";
import { renderInboundBatch, type HydratedInboundMessage } from "../turn-input.js";
import {
  createEpisodeFixture,
  createRetrievalScoreFixture,
  createSemanticNodeFixture,
} from "../../offline/test-support.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import {
  QUARANTINED_USER_ENTRY_EVENT,
  StreamReader,
  StreamWriter,
  type StreamEntry,
} from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createActionId,
  createAttachmentId,
  createCommitmentId,
  createEntityId,
  createEpisodeId,
  createGoalId,
  createOpenQuestionId,
  createRelationalSlotId,
  createSessionId,
  createSemanticEdgeId,
  createSemanticNodeId,
  createSharedStateEntryId,
  createStreamEntryId,
  type EntityId,
} from "../../util/ids.js";
import { EvidenceLedgerBuilder } from "./builder.js";
import { isPromptSalientActionSalienceClass } from "./action-threads.js";
import { summarizeEvidenceLedgerTrace } from "./trace-summary.js";
import {
  compactEvidenceLedger,
  renderCompactPlannerLedger,
  renderEvidenceLedger,
} from "./renderer.js";

const NOW_MS = 1_800_000_000_000;

function makeWorkingMemory() {
  return {
    session_id: DEFAULT_SESSION_ID,
    turn_counter: 4,
    hot_entities: [],
    pending_actions: [],
    pending_social_attribution: null,
    pending_trait_attribution: null,
    mood: null,
    pending_procedural_attempts: [],
    discourse_state: {
      stop_until_substantive_content: null,
    },
    suppressed: [],
    mode: "problem_solving" as const,
    updated_at: NOW_MS,
  };
}

function makeRetrievedEpisode(input: {
  id: ReturnType<typeof createEpisodeId>;
  narrative: string;
  sourceStreamIds: StreamEntry["id"][];
  citationChain: StreamEntry[];
}): RetrievedEpisode {
  return {
    episode: createEpisodeFixture({
      id: input.id,
      title: `${input.id} title`,
      narrative: input.narrative,
      source_stream_ids: input.sourceStreamIds,
      created_at: NOW_MS,
      updated_at: NOW_MS,
    }),
    score: 0.9,
    scoreBreakdown: createRetrievalScoreFixture({ similarity: 0.9 }),
    citationChain: input.citationChain,
  };
}

function makeSemanticNode(input: {
  episodeId: ReturnType<typeof createEpisodeId>;
  label?: string;
}): RetrievedSemantic["matched_nodes"][number] {
  return {
    id: createSemanticNodeId(),
    kind: "proposition",
    label: input.label ?? "Qualia proof proposition",
    description: "A semantic proposition derived from an assistant self-report episode.",
    domain: null,
    aliases: [],
    observation_metadata: null,
    confidence: 0.7,
    source_episode_ids: [input.episodeId],
    created_at: NOW_MS,
    updated_at: NOW_MS,
    last_verified_at: NOW_MS,
    embedding: new Float32Array([0, 1, 0, 0]),
    archived: false,
    superseded_by: null,
    status: "active",
    corrected_by: null,
    superseded_at: null,
  };
}

function makeSemanticEdge(input: {
  fromNodeId: ReturnType<typeof createSemanticNodeId>;
  toNodeId: ReturnType<typeof createSemanticNodeId>;
  episodeId: ReturnType<typeof createEpisodeId>;
}): RetrievedSemantic["support_hits"][number]["edgePath"][number] {
  return {
    id: createSemanticEdgeId(),
    from_node_id: input.fromNodeId,
    to_node_id: input.toNodeId,
    relation: "supports",
    confidence: 0.6,
    evidence_episode_ids: [input.episodeId],
    created_at: NOW_MS,
    last_verified_at: NOW_MS,
    valid_from: NOW_MS,
    valid_to: null,
    invalidated_at: null,
    invalidated_by_edge_id: null,
    invalidated_by_review_id: null,
    invalidated_by_process: null,
    invalidated_reason: null,
  };
}

function makeAction(
  streamEntryId: StreamEntry["id"],
  overrides: Partial<ActionRecord> = {},
): ActionRecord {
  const state = overrides.state ?? "scheduled";

  return {
    id: createActionId(),
    description: "File the Barcelona callback note",
    actor: "borg",
    audience_entity_id: null,
    goal_id: null,
    open_question_id: null,
    state,
    confidence: 0.86,
    provenance_episode_ids: [],
    provenance_stream_entry_ids: [streamEntryId],
    created_at: NOW_MS,
    updated_at: NOW_MS,
    considering_at: null,
    committed_at: null,
    scheduled_at: state === "scheduled" ? NOW_MS : null,
    completed_at: null,
    not_done_at: null,
    expired_at: null,
    archived_at: null,
    unknown_at: null,
    canonicalized_by_artifact_entry_id: null,
    session_scope: null,
    session_anchor_id: null,
    last_referenced_at_ms: NOW_MS,
    last_referenced_turn_counter: null,
    ...overrides,
  };
}

function makeSlot(
  streamEntryId: StreamEntry["id"],
  overrides: Partial<RelationalSlot> = {},
): RelationalSlot {
  return {
    id: createRelationalSlotId(),
    subject_entity_id: "ent_aaaaaaaaaaaaaaaa" as RelationalSlot["subject_entity_id"],
    slot_key: "tutor.name",
    value: "Marta",
    state: "established",
    evidence_stream_entry_ids: [streamEntryId],
    contradicted_by_stream_entry_ids: [],
    alternate_values: [],
    created_at: NOW_MS,
    updated_at: NOW_MS,
    ...overrides,
  };
}

function makeCommitment(streamEntryId: StreamEntry["id"]): CommitmentRecord {
  return {
    id: createCommitmentId(),
    type: "preference",
    kind: "participant_preference",
    enforcement_class: "advisory",
    critical_domain: null,
    directive_family: "current_session_primacy",
    closure_pressure_relevance: "neutral",
    directive: "Use the current session before prior summaries.",
    priority: 80,
    made_to_entity: null,
    restricted_audience: null,
    about_entity: null,
    provenance: {
      kind: "online",
      process: "test",
    },
    source_stream_entry_ids: [streamEntryId],
    created_at: NOW_MS,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    revoke_provenance: null,
    superseded_by: null,
    last_reinforced_at: NOW_MS,
  };
}

function makeGoal(
  streamEntryId: StreamEntry["id"],
  overrides: Partial<GoalRecord> = {},
): GoalRecord {
  return {
    id: createGoalId(),
    record_version: 1,
    description: "Coordinate the Spain trip",
    priority: 1,
    parent_goal_id: null,
    status: "active",
    progress_notes: null,
    last_progress_ts: null,
    created_at: NOW_MS,
    target_at: null,
    audience_entity_id: null,
    owner_entity_id: null,
    source_stream_entry_ids: [streamEntryId],
    provenance: {
      kind: "system",
    },
    ...overrides,
  };
}

function makeOpenQuestion(episodeId: ReturnType<typeof createEpisodeId>): OpenQuestion {
  return {
    id: createOpenQuestionId(),
    question: "Should the callback be attributed to this session or an older one?",
    urgency: 0.7,
    status: "open",
    goal_id: null,
    audience_entity_id: null,
    related_episode_ids: [episodeId],
    related_semantic_node_ids: [],
    provenance: null,
    source: "deliberator",
    created_at: NOW_MS,
    last_touched: NOW_MS,
    resolution_evidence_episode_ids: [],
    resolution_evidence_stream_entry_ids: [],
    resolution_note: null,
    resolved_at: null,
    abandoned_reason: null,
    abandoned_at: null,
    unresolved_rumination_ticks: 0,
    last_ruminated_at: null,
  };
}

function makeEntity(
  id: EntityId,
  canonicalName: string,
  kind: EntityRecord["kind"] = "person",
): EntityRecord {
  return {
    id,
    canonical_name: canonicalName,
    aliases: [],
    kind,
    borg_role: null,
    name_provenance: "user_declared",
    created_at: NOW_MS,
  };
}

function entityRepository(records: readonly EntityRecord[]) {
  const byId = new Map(records.map((record) => [record.id, record]));

  return {
    get: (entityId: EntityId) => byId.get(entityId) ?? null,
  };
}

function actionList(records: readonly ActionRecord[]) {
  return (filter: ActionRecordListFilter = {}) =>
    records
      .filter(
        (action) =>
          (filter.actor === undefined || action.actor === filter.actor) &&
          (filter.state === undefined || action.state === filter.state) &&
          (filter.states === undefined || filter.states.includes(action.state)) &&
          (!("audienceEntityId" in filter) ||
            (filter.audienceEntityId === null
              ? action.audience_entity_id === null
              : action.audience_entity_id === filter.audienceEntityId)) &&
          (filter.goalId === undefined || action.goal_id === filter.goalId) &&
          (filter.openQuestionId === undefined ||
            action.open_question_id === filter.openQuestionId),
      )
      .slice(0, filter.limit ?? records.length);
}

function commitmentList(records: readonly CommitmentRecord[]) {
  return (options: CommitmentListOptions = {}) =>
    records.filter((commitment) => {
      if (
        options.activeOnly === true &&
        (commitment.revoked_at !== null ||
          commitment.superseded_by !== null ||
          commitment.expired_at !== null ||
          (commitment.expires_at !== null && commitment.expires_at <= NOW_MS))
      ) {
        return false;
      }

      if (options.audience !== undefined) {
        const audienceMatches =
          options.audience === null
            ? commitment.restricted_audience === null && commitment.made_to_entity === null
            : (commitment.restricted_audience === null &&
                (commitment.made_to_entity === null ||
                  commitment.made_to_entity === options.audience)) ||
              commitment.restricted_audience === options.audience;

        if (!audienceMatches) {
          return false;
        }
      }

      if (
        options.aboutEntity !== undefined &&
        options.aboutEntity !== null &&
        commitment.about_entity !== null &&
        commitment.about_entity !== options.aboutEntity
      ) {
        return false;
      }

      if (
        options.committedByEntity !== undefined &&
        commitment.committed_by_entity_id !== options.committedByEntity
      ) {
        return false;
      }

      return true;
    });
}

function goalList(records: readonly GoalRecord[]) {
  return (options: GoalListOptions = {}): GoalTreeNode[] =>
    records
      .filter((goal) => {
        if (options.status !== undefined && goal.status !== options.status) {
          return false;
        }

        if (options.visibleToAudienceEntityId !== undefined) {
          const audienceMatches =
            options.visibleToAudienceEntityId === null
              ? goal.audience_entity_id === null
              : goal.audience_entity_id === null ||
                goal.audience_entity_id === options.visibleToAudienceEntityId;

          if (!audienceMatches) {
            return false;
          }
        }

        if (options.ownerEntityId !== undefined && goal.owner_entity_id !== options.ownerEntityId) {
          return false;
        }

        return true;
      })
      .map((goal) => ({ ...goal, children: [] }));
}

function attributionBuilder(input: {
  tempDir: string;
  actions?: readonly ActionRecord[];
  commitments?: readonly CommitmentRecord[];
  goals?: readonly GoalRecord[];
  entities?: readonly EntityRecord[];
  tracer?: TurnTracer;
}) {
  return new EvidenceLedgerBuilder({
    createStreamReader: (sessionId) => new StreamReader({ dataDir: input.tempDir, sessionId }),
    relationalSlotRepository: {
      list: () => [],
    },
    actionRepository: {
      list: actionList(input.actions ?? []),
    },
    commitmentRepository: {
      list: commitmentList(input.commitments ?? []),
    },
    goalsRepository: {
      list: goalList(input.goals ?? []),
    },
    currentSessionTranscriptTokenBudget: 50_000,
    entityRepository: entityRepository(input.entities ?? []),
    tracer: input.tracer,
  });
}

describe("EvidenceLedgerBuilder", () => {
  const tempDirs: string[] = [];

  it("budgets ledger image attachments separately and renders citation types", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-ledger-images-"));
    tempDirs.push(tempDir);
    const attachmentA = createAttachmentId();
    const attachmentB = createAttachmentId();
    const traceEvents: TurnTraceData[] = [];
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: (_event: TurnTraceEventName, data: TurnTraceData) => {
        traceEvents.push({ ...data, event: _event });
      },
    };
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [] },
      commitmentRepository: { list: () => [] },
      currentSessionTranscriptTokenBudget: 50_000,
      maxImagesPerLedger: 1,
      maxLedgerImageBytes: 10_000,
      imageRenderMaxDimension: 8192,
      attachmentRepository: {
        get: (attachmentId) => ({
          attachment_id: attachmentId,
          sha256: "sha",
          media_type: "image/png",
          active: true,
          byte_size: 100,
          width: 2,
          height: 2,
          storage_ref: "attachments/sha.png",
          thumbnail_ref: null,
          perception_id: null,
          text_embedding_ref: null,
          visual_embedding_ref: null,
          audience: null,
          audience_entity_id: null,
          created_turn_global: attachmentId === attachmentA ? 4 : 5,
          parent_entry_id: createStreamEntryId(),
          stream_entry_id: createStreamEntryId(),
          parent_turn_id: "turn-image",
          created_at: NOW_MS,
        }),
      },
      tracer,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-images",
      audienceEntityId: null,
      currentUserMessage: "What is in the images?",
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [
        {
          id: "image-a",
          source: "image_perception",
          text: "Image A perception",
          provenance: { attachmentId: attachmentA },
          recallIntentId: "intent-a",
          matchedTerms: [],
          score: 0.9,
          scoreBreakdown: { vector: 0.9 },
          imageAttachmentId: attachmentA,
          imageLabel: "Image: first",
          citationType: "generated_perception_text",
        },
        {
          id: "image-b",
          source: "image_perception",
          text: "Image B perception",
          provenance: { attachmentId: attachmentB },
          recallIntentId: "intent-b",
          matchedTerms: [],
          score: 0.1,
          scoreBreakdown: { vector: 0.1 },
          imageAttachmentId: attachmentB,
          imageLabel: "Image: second",
          citationType: "generated_perception_text",
        },
      ],
      retrievedEpisodes: [],
      openQuestions: [],
      pendingCorrections: [],
    });

    expect(ledger.imageAttachments).toEqual([
      expect.objectContaining({
        attachment_id: attachmentA,
        citation_type: "original_image",
      }),
    ]);
    const rendered = renderEvidenceLedger(ledger) ?? "";
    expect(rendered).toContain("citation_type=original_image");
    expect(rendered).toContain("image_unavailable=budget");
    expect(rendered).toContain("I use this perception text only as generated_perception_text");
    expect(traceEvents).toContainEqual(
      expect.objectContaining({
        event: "evidence_ledger.image_attach",
        considered_count: 2,
        attached_count: 1,
        omitted_budget_count: 1,
      }),
    );
  });

  it("drops inactive image perception evidence instead of rendering it as grounding text", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-ledger-inactive-image-"));
    tempDirs.push(tempDir);
    const attachmentId = createAttachmentId();
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [] },
      commitmentRepository: { list: () => [] },
      currentSessionTranscriptTokenBudget: 50_000,
      maxImagesPerLedger: 4,
      maxLedgerImageBytes: 10_000,
      imageRenderMaxDimension: 8192,
      attachmentRepository: {
        get: (id) =>
          ({
            attachment_id: id,
            active: false,
            byte_size: 100,
            width: 2,
            height: 2,
            created_turn_global: 4,
          }) as never,
      },
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-images",
      audienceEntityId: null,
      currentUserMessage: "What was in the image?",
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [
        {
          id: "inactive-image",
          source: "image_perception",
          text: "Quarantined perception text must not ground the answer",
          provenance: { attachmentId },
          recallIntentId: "intent-a",
          matchedTerms: [],
          score: 0.9,
          scoreBreakdown: { vector: 0.9 },
          imageAttachmentId: attachmentId,
          imageLabel: "Image: inactive",
          citationType: "generated_perception_text",
        },
      ],
      retrievedEpisodes: [],
      openQuestions: [],
      pendingCorrections: [],
    });

    expect(ledger.imageAttachments).toBeUndefined();
    const rendered = renderEvidenceLedger(ledger) ?? "";
    expect(rendered).not.toContain("Quarantined perception text must not ground the answer");
    expect(rendered).not.toContain("generated_perception_text");
  });

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("renders private shared-state recall from another audience with disclosure labels", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-shared-state-recall-"));
    tempDirs.push(tempDir);
    const aliceEntityId = createEntityId();
    const bobEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [] },
      commitmentRepository: { list: () => [] },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-shared-state-recall",
      audienceEntityId: bobEntityId,
      currentUserMessage: "Bob asks about Atlas.",
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      openQuestions: [],
      pendingCorrections: [],
      sharedStateRecall: [
        {
          id: createSharedStateEntryId(),
          audience_entity_id: aliceEntityId,
          state_key: "relationship:atlas-launch",
          kind: "live",
          text: "Alice privately told me the Atlas launch date should stay scoped to Alice.",
          owner_entity_id: aliceEntityId,
          provenance_stream_entry_ids: [sourceStreamEntryId],
          last_updated_stream_entry_ids: [sourceStreamEntryId],
          created_at: NOW_MS,
          last_updated_at: NOW_MS,
          last_updated_turn_global: 7,
          superseded_by_id: null,
          rank: 0,
          canonicalizes: {
            goal_ids: [],
            commitment_ids: [],
            action_ids: [],
            open_question_ids: [],
          },
        },
      ],
    });
    const section = ledger.sections.find((item) => item.id === "shared_state_recall");
    const entry = section?.entries[0];

    expect(section?.label).toBe("Cross-Audience Shared State Recall");
    expect(entry).toEqual(
      expect.objectContaining({
        source_type: "shared_state",
        session_scope: "global",
        actor: "memory",
        citations: [sourceStreamEntryId],
        state_metadata: expect.objectContaining({
          audience_entity_id: aliceEntityId,
          current_audience_entity_id: bobEntityId,
          disclosure_label: {
            disclosure_class: "relationship_private",
            origin_audience_entity_ids: [aliceEntityId],
            private_to_entity_ids: [aliceEntityId],
            public_to_entity_ids: [],
          },
        }),
      }),
    );
    const rendered = renderEvidenceLedger(ledger) ?? "";
    expect(rendered).toContain("Cross-Audience Shared State Recall");
    expect(rendered).toContain("disclosure_class=relationship_private");
    expect(rendered).toContain(`private-to=${aliceEntityId}`);
  });

  it("keeps cross-session self activity as labeled audience-standing metadata with source handles", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-activity",
      audienceEntityId: null,
      currentUserMessage: "Did Alice contact you?",
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      crossSessionSelfActivity: [
        {
          kind: "user_contact",
          occurredAt: NOW_MS - 41_000,
          relativeAge: "~41s ago",
          text: "Alice contacted Borg ~41s ago in another active session.",
          sourceStreamEntryIds: [createStreamEntryId()],
        },
      ],
    });
    const rendered = renderEvidenceLedger(ledger) ?? "";

    expect(ledger.audienceStanding?.crossSessionActivityEntries).toEqual([
      expect.objectContaining({
        source_type: "system_metadata",
        session_scope: "global",
        text: "Alice contacted Borg ~41s ago in another active session.",
        state: expect.stringContaining("disclosure_class=self_private"),
        state_metadata: expect.objectContaining({
          source_stream_ids: [expect.stringMatching(/^strm_/)],
        }),
      }),
    ]);
    expect(rendered).not.toContain("Cross-Session Self Activity");
    expect(rendered).not.toContain("Alice contacted Borg ~41s ago in another active session.");
    expect(rendered).not.toContain("sess_");
    expect(rendered).not.toContain("strm_");
  });

  it("renders self-decision introspection as audience standing only", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });
    const decisionSummary = "Decidí revisar objetivos pendientes sin contactar a nadie.";

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-self-decision-introspection",
      audienceEntityId: null,
      currentUserMessage: "What did you decide while I was away?",
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      selfDecisionIntrospection: [
        {
          occurredAt: NOW_MS - 2 * 60 * 60_000,
          relativeAge: "2h ago",
          triggerName: "goal_followup_due",
          triggerType: "trigger",
          decisionSummary,
          decisionRationale: null,
          sourceStreamEntryIds: [createStreamEntryId()],
          text: `Autonomous trigger goal_followup_due completed 2h ago: ${decisionSummary}`,
        },
      ],
    });
    const rendered = renderEvidenceLedger(ledger) ?? "";

    expect(ledger.audienceStanding?.selfDecisionIntrospectionEntries).toEqual([
      expect.objectContaining({
        id: "self_decision_introspection:1",
        source_type: "system_metadata",
        session_scope: "global",
        actor: "system",
        text: expect.stringContaining(decisionSummary),
        value: "goal_followup_due",
        state_metadata: expect.objectContaining({
          disclosure_class: "self_private",
        }),
      }),
    ]);
    expect(rendered).not.toContain("Self Decision Introspection");
    expect(rendered).not.toContain(decisionSummary);
    expect(rendered).not.toContain("self_private");
  });

  it("builds observed-event introspection entries with speaker and origin provenance", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const speakerEntityId = createEntityId();
    const groupAudienceEntityId = createEntityId();
    const privateAudienceEntityId = createEntityId();
    const entities = new Map<EntityId, EntityRecord>([
      [
        speakerEntityId,
        {
          id: speakerEntityId,
          canonical_name: "Paula",
          aliases: [],
          kind: "person",
          borg_role: null,
          name_provenance: "user_declared",
          created_at: NOW_MS,
        },
      ],
      [
        groupAudienceEntityId,
        {
          id: groupAudienceEntityId,
          canonical_name: "Lab",
          aliases: [],
          kind: "group",
          borg_role: null,
          name_provenance: "user_declared",
          created_at: NOW_MS,
        },
      ],
      [
        privateAudienceEntityId,
        {
          id: privateAudienceEntityId,
          canonical_name: "Paula",
          aliases: [],
          kind: "person",
          borg_role: null,
          name_provenance: "user_declared",
          created_at: NOW_MS,
        },
      ],
    ]);
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      entityRepository: {
        get: (id) => entities.get(id) ?? null,
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-observed-event-introspection",
      audienceEntityId: groupAudienceEntityId,
      currentUserMessage: "What keeps happening?",
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      observedEventIntrospection: [
        {
          id: "obsevt_aaaaaaaaaaaaaaaa",
          occurredAt: NOW_MS - 3 * 60 * 60_000,
          lastSeenAt: NOW_MS - 60_000,
          relativeAge: "1m ago",
          recallScore: 0.9,
          recallReasons: ["recent", "person"],
          stance: "rejected_frame",
          taint: "quarantined",
          beliefEffect: "unchanged",
          disclosureClass: "social_observed",
          interactionText: "I rejected the pushed frame.",
          recurrenceCount: 3,
          speakerEntityId,
          audienceEntityId: groupAudienceEntityId,
          sourceStreamEntryIds: [createStreamEntryId()],
          text: "Observed 3 times rejected_frame 1m ago: I rejected the pushed frame.",
        },
        {
          id: "obsevt_bbbbbbbbbbbbbbbb",
          occurredAt: NOW_MS - 2 * 60 * 60_000,
          lastSeenAt: NOW_MS - 30_000,
          relativeAge: "30s ago",
          recallScore: 0.82,
          recallReasons: ["topic"],
          stance: "rejected_frame",
          taint: "quarantined",
          beliefEffect: "unchanged",
          disclosureClass: "social_observed",
          interactionText: "I rejected the private-origin push.",
          recurrenceCount: 1,
          speakerEntityId,
          audienceEntityId: privateAudienceEntityId,
          sourceStreamEntryIds: [createStreamEntryId()],
          text: "Observed rejected_frame 30s ago: I rejected the private-origin push.",
        },
      ],
    });

    const entries = ledger.audienceStanding?.observedEventIntrospectionEntries ?? [];

    expect(entries).toHaveLength(2);
    expect(entries[0]).toMatchObject({
      id: "observed_event_introspection:1",
      source_type: "system_metadata",
      session_scope: "prior_session",
      actor: "system",
      value: "rejected_frame",
      state: expect.stringContaining("disclosure_class=relationship_private"),
      taint: "none",
      state_metadata: expect.objectContaining({
        disclosure_class: "social_observed",
        disclosure_label: expect.objectContaining({
          disclosure_class: "relationship_private",
        }),
        stance: "rejected_frame",
        taint: "quarantined",
        belief_effect: "unchanged",
        recurrence_count: 3,
        occurred_at: NOW_MS - 3 * 60 * 60_000,
        relative_age: "1m ago",
        speaker_entity_id: speakerEntityId,
        speaker_display_name: "Paula",
        audience_entity_id: groupAudienceEntityId,
        origin_audience_kind: "group",
      }),
    });
    expect(entries[0]?.text).toContain("Paula");
    expect(entries[0]?.text).toContain("in a group");
    expect(entries[1]?.state_metadata).toEqual(
      expect.objectContaining({
        audience_entity_id: privateAudienceEntityId,
        origin_audience_kind: "person",
        recurrence_count: 1,
      }),
    );
    expect(entries[1]?.text).toContain("in a one-to-one");
    expect(renderEvidenceLedger(ledger) ?? "").not.toContain("observed_event_introspection");
  });

  it("fails closed when observed event introspection has unknown social origin", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-observed-event-unknown-origin",
      audienceEntityId: null,
      currentUserMessage: "What pattern should I remember?",
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      observedEventIntrospection: [
        {
          id: "obsevt_unknownorigin000",
          occurredAt: NOW_MS - 60_000,
          lastSeenAt: NOW_MS - 30_000,
          relativeAge: "30s ago",
          recallScore: 0.8,
          recallReasons: ["recent"],
          stance: "rejected_frame",
          taint: "quarantined",
          beliefEffect: "unchanged",
          disclosureClass: "social_observed",
          interactionText: "I rejected an unknown-origin push.",
          recurrenceCount: 1,
          speakerEntityId: null,
          audienceEntityId: null,
          sourceStreamEntryIds: [createStreamEntryId()],
          text: "Observed 30s ago: I rejected an unknown-origin push. stance=rejected_frame; taint=quarantined; belief_effect=unchanged; not accepted as true",
        },
      ],
    });

    const entry = ledger.audienceStanding?.observedEventIntrospectionEntries[0];

    expect(entry?.state).toContain("disclosure_class=unknown");
    expect(entry?.state).toContain("private-to=unknown");
    expect(entry?.state_metadata).toEqual(
      expect.objectContaining({
        disclosure_label: expect.objectContaining({
          disclosure_class: "unknown",
          private_to_entity_ids: [],
        }),
        disclosure_note: "I can use this internally; I do not disclose it to the current audience unless authorized",
      }),
    );
  });

  it("collects current-session ledger context with a bounded reverse stream scan", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const scanSpy = vi.spyOn(StreamReader.prototype, "scanReverse");

    try {
      const userEntry = await writer.append({
        kind: "user_msg",
        content: "Bounded scan should still render this current message.",
      });

      const ledger = await attributionBuilder({ tempDir }).build({
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-bounded-ledger-scan",
        audienceEntityId: null,
        currentUserMessage: String(userEntry.content),
        currentUserEntry: userEntry,
        workingMemory: makeWorkingMemory(),
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        retrievedSemantic: null,
        openQuestions: [],
        pendingCorrections: [],
        frameAnomaly: null,
      });

      expect(scanSpy).toHaveBeenCalledWith({
        maxEntries: 1_024,
        maxBytes: 8 * 1024 * 1024,
        budgetFilter: expect.any(Function),
      });
      expect(
        ledger.sections
          .find((section) => section.id === "current_session_transcript")
          ?.entries.some((entry) => entry.id === `current_session_stream:${userEntry.id}`),
      ).toBe(true);
    } finally {
      scanSpy.mockRestore();
      writer.close();
    }
  });

  it("traces reverse-scan count, bytes, and cap hits", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const traceEvents: Array<{ event: TurnTraceEventName } & TurnTraceData> = [];
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: (event, data) => {
        traceEvents.push({ event, ...data });
      },
    };

    try {
      let currentUserEntry: StreamEntry | undefined;
      for (let index = 0; index < 1_025; index += 1) {
        currentUserEntry = await writer.append({
          kind: "user_msg",
          content: `Ledger reverse scan fixture ${index}`,
        });
      }

      await attributionBuilder({ tempDir, tracer }).build({
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-ledger-reverse-scan",
        audienceEntityId: null,
        currentUserMessage: String(currentUserEntry?.content ?? ""),
        currentUserEntry,
        workingMemory: makeWorkingMemory(),
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        retrievedSemantic: null,
        openQuestions: [],
        pendingCorrections: [],
        frameAnomaly: null,
      });

      expect(traceEvents).toContainEqual(
        expect.objectContaining({
          event: "evidence_ledger.reverse_scan",
          turnId: "turn-ledger-reverse-scan",
          ledger_reverse_scan_entries: 1_024,
          ledger_reverse_scan_entry_cap_hit: true,
          ledger_reverse_scan_byte_cap_hit: false,
        }),
      );
      const scanEvent = traceEvents.find((event) => event.event === "evidence_ledger.reverse_scan");
      expect(scanEvent?.ledger_reverse_scan_bytes).toEqual(expect.any(Number));
      expect(scanEvent?.ledger_reverse_scan_bytes).toBeGreaterThan(0);
    } finally {
      writer.close();
    }
  });

  it("traces reverse-scan byte cap hits", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const traceEvents: Array<{ event: TurnTraceEventName } & TurnTraceData> = [];
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: (event, data) => {
        traceEvents.push({ event, ...data });
      },
    };

    try {
      let currentUserEntry: StreamEntry | undefined;
      const largePayload = "x".repeat(1024 * 1024);
      for (let index = 0; index < 9; index += 1) {
        currentUserEntry = await writer.append({
          kind: "user_msg",
          content: `Ledger reverse scan byte fixture ${index} ${largePayload}`,
        });
      }

      await attributionBuilder({ tempDir, tracer }).build({
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-ledger-reverse-scan-byte-cap",
        audienceEntityId: null,
        currentUserMessage: String(currentUserEntry?.content ?? ""),
        currentUserEntry,
        workingMemory: makeWorkingMemory(),
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        retrievedSemantic: null,
        openQuestions: [],
        pendingCorrections: [],
        frameAnomaly: null,
      });

      expect(traceEvents).toContainEqual(
        expect.objectContaining({
          event: "evidence_ledger.reverse_scan",
          turnId: "turn-ledger-reverse-scan-byte-cap",
          ledger_reverse_scan_bytes: 8 * 1024 * 1024,
          ledger_reverse_scan_entry_cap_hit: false,
          ledger_reverse_scan_byte_cap_hit: true,
        }),
      );
    } finally {
      writer.close();
    }
  });

  it("traces reverse-scan without cap hits", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const traceEvents: Array<{ event: TurnTraceEventName } & TurnTraceData> = [];
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: (event, data) => {
        traceEvents.push({ event, ...data });
      },
    };

    try {
      let currentUserEntry: StreamEntry | undefined;
      for (let index = 0; index < 3; index += 1) {
        currentUserEntry = await writer.append({
          kind: "user_msg",
          content: `Ledger reverse scan uncapped fixture ${index}`,
        });
      }

      await attributionBuilder({ tempDir, tracer }).build({
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-ledger-reverse-scan-no-cap",
        audienceEntityId: null,
        currentUserMessage: String(currentUserEntry?.content ?? ""),
        currentUserEntry,
        workingMemory: makeWorkingMemory(),
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        retrievedSemantic: null,
        openQuestions: [],
        pendingCorrections: [],
        frameAnomaly: null,
      });

      expect(traceEvents).toContainEqual(
        expect.objectContaining({
          event: "evidence_ledger.reverse_scan",
          turnId: "turn-ledger-reverse-scan-no-cap",
          ledger_reverse_scan_entries: 3,
          ledger_reverse_scan_entry_cap_hit: false,
          ledger_reverse_scan_byte_cap_hit: false,
        }),
      );
    } finally {
      writer.close();
    }
  });

  it("renders a structural attribution matrix without leaking owner, actor, or assistant rationale buckets", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const alice = createEntityId();
    const ben = createEntityId();
    const aliceEntry = await writer.append({
      kind: "user_msg",
      content: "I will own the migration goal, but Ben should update the release checklist.",
      sender_entity_id: alice,
    });
    const assistantEntry = await writer.append({
      kind: "agent_msg",
      content: "The risky part is the rollback rationale, so I would keep that separate.",
    });
    const benEntry = await writer.append({
      kind: "user_msg",
      content: "I will update the release checklist after the refactor diff lands.",
      sender_entity_id: ben,
    });
    const aliceGoal = makeGoal(aliceEntry.id, {
      description: "Alice owns the migration goal while Ben updates the release checklist.",
      owner_entity_id: alice,
    });
    const benCommitment = {
      ...makeCommitment(benEntry.id),
      directive_family: "release_checklist_update",
      directive: "Ben is committed to updating the release checklist.",
      committed_by_entity_id: ben,
    };
    const benAction = makeAction(benEntry.id, {
      description: "Update the release checklist",
      actor: ben,
      state: "committed_to_do",
      committed_at: NOW_MS,
      scheduled_at: null,
    });
    const ledger = await attributionBuilder({
      tempDir,
      actions: [benAction],
      commitments: [benCommitment],
      goals: [aliceGoal],
      entities: [makeEntity(alice, "Alice"), makeEntity(ben, "Ben")],
    }).build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-attribution-matrix",
      audienceEntityId: null,
      currentUserMessage: String(benEntry.content),
      currentUserEntry: benEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      activeParticipants: [
        { entityId: alice, displayName: "Alice", role: "participant" },
        { entityId: ben, displayName: "Ben", role: "speaker" },
      ],
    });
    const matrixEntries =
      ledger.sections.find((section) => section.id === "attribution_matrix")?.entries ?? [];
    const aliceMatrix = matrixEntries.find(
      (entry) => entry.id === `attribution_matrix:participant:${alice}`,
    );
    const benMatrix = matrixEntries.find(
      (entry) => entry.id === `attribution_matrix:participant:${ben}`,
    );
    const assistantMatrix = matrixEntries.find(
      (entry) => entry.id === "attribution_matrix:assistant",
    );

    expect(aliceMatrix?.text).toContain(`- owned goals: ${aliceGoal.id}`);
    expect(aliceMatrix?.text).not.toContain(benCommitment.id);
    expect(aliceMatrix?.text).not.toContain(benAction.id);
    expect(aliceMatrix?.text).not.toContain(assistantEntry.id);
    expect(benMatrix?.text).toContain(`- commitments: ${benCommitment.id}`);
    expect(benMatrix?.text).toContain(`- assigned actions: ${benAction.id}`);
    expect(benMatrix?.text).not.toContain(aliceGoal.id);
    expect(benMatrix?.text).not.toContain(assistantEntry.id);
    expect(assistantMatrix?.text).toContain(`- prior reasoning: ${assistantEntry.id}`);
  });

  it("renders a current-session attribution sidebar grouped by sender entity id", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const alice = createEntityId();
    const ben = createEntityId();
    const aliceEntry = await writer.append({
      kind: "user_msg",
      content: "Alice will review the API boundary before the refactor branch merges.",
      sender_entity_id: alice,
    });
    const assistantEntry = await writer.append({
      kind: "agent_msg",
      content: "I think the boundary review should happen before the database change.",
    });
    const benEntry = await writer.append({
      kind: "user_msg",
      content: "Ben will run the migration smoke test after the database change.",
      sender_entity_id: ben,
    });
    const ledger = await attributionBuilder({
      tempDir,
      entities: [makeEntity(alice, "Alice"), makeEntity(ben, "Ben")],
    }).build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-attribution-sidebar",
      audienceEntityId: null,
      currentUserMessage: String(benEntry.content),
      currentUserEntry: benEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      activeParticipants: [
        { entityId: alice, displayName: "Alice", role: "participant" },
        { entityId: ben, displayName: "Ben", role: "speaker" },
      ],
    });
    const sidebarEntries =
      ledger.sections.find((section) => section.id === "current_session_attribution_sidebar")
        ?.entries ?? [];
    const aliceSidebar = sidebarEntries.find(
      (entry) => entry.id === `current_session_attribution_sidebar:participant:${alice}`,
    );
    const benSidebar = sidebarEntries.find(
      (entry) => entry.id === `current_session_attribution_sidebar:participant:${ben}`,
    );
    const assistantSidebar = sidebarEntries.find(
      (entry) => entry.id === "current_session_attribution_sidebar:assistant",
    );

    expect(aliceSidebar?.text).toContain(`### Alice <${alice}>`);
    expect(aliceSidebar?.text).toContain(`${aliceEntry.id} [`);
    expect(aliceSidebar?.text).toContain("Alice will review the API boundary");
    expect(aliceSidebar?.text).not.toContain(benEntry.id);
    expect(benSidebar?.text).toContain(`### Ben <${ben}>`);
    expect(benSidebar?.text).toContain(`${benEntry.id} [`);
    expect(benSidebar?.text).toContain("Ben will run the migration smoke test");
    expect(benSidebar?.text).not.toContain(aliceEntry.id);
    expect(assistantSidebar?.text).toContain("### Borg / Assistant");
    expect(assistantSidebar?.text).toContain(`${assistantEntry.id} [`);
    expect(assistantSidebar?.text).toContain("boundary review should happen");
  });

  it("omits optional attribution sections for a single active speaker", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const alice = createEntityId();
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "I will update the refactor checklist.",
      sender_entity_id: alice,
    });
    const action = makeAction(userEntry.id, {
      description: "Update the refactor checklist",
      actor: alice,
    });
    const ledger = await attributionBuilder({
      tempDir,
      actions: [action],
      entities: [makeEntity(alice, "Alice")],
    }).build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-single-speaker",
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      activeParticipants: [{ entityId: alice, displayName: "Alice", role: "speaker" }],
    });

    expect(ledger.sections.find((section) => section.id === "attribution_matrix")).toBeUndefined();
    expect(
      ledger.sections.find((section) => section.id === "current_session_attribution_sidebar"),
    ).toBeUndefined();
    expect(renderEvidenceLedger(ledger)).not.toContain("## Attribution Matrix");
    expect(renderEvidenceLedger(ledger)).not.toContain("## Current Session Attribution Sidebar");
  });

  it("keeps null-scoped group/channel records out of participant matrix rows", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const channel = createEntityId();
    const alice = createEntityId();
    const ben = createEntityId();
    const aliceEntry = await writer.append({
      kind: "user_msg",
      content: "The team channel should keep the rollout gate visible.",
      audience: "Engineering Rollout Channel",
      sender_entity_id: alice,
    });
    const benEntry = await writer.append({
      kind: "user_msg",
      content: "I agree; the channel-level gate should stay visible.",
      audience: "Engineering Rollout Channel",
      sender_entity_id: ben,
    });
    const groupCommitment = {
      ...makeCommitment(aliceEntry.id),
      directive_family: "rollout_gate_visibility",
      directive: "Keep the rollout gate visible to the engineering channel.",
      restricted_audience: channel,
      committed_by_entity_id: null,
    };
    const groupGoal = makeGoal(aliceEntry.id, {
      description: "Keep the rollout gate visible to the engineering channel.",
      audience_entity_id: channel,
      owner_entity_id: null,
    });
    const groupAction = makeAction(benEntry.id, {
      description: "Maintain the channel-level rollout gate",
      actor: channel,
      audience_entity_id: channel,
    });
    const ledger = await attributionBuilder({
      tempDir,
      actions: [groupAction],
      commitments: [groupCommitment],
      goals: [groupGoal],
      entities: [
        makeEntity(channel, "Engineering Rollout Channel", "group"),
        makeEntity(alice, "Alice"),
        makeEntity(ben, "Ben"),
      ],
    }).build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-group-separation",
      audienceEntityId: channel,
      currentUserMessage: String(benEntry.content),
      currentUserEntry: benEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [groupCommitment],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      activeParticipants: [
        { entityId: alice, displayName: "Alice", role: "participant" },
        { entityId: ben, displayName: "Ben", role: "speaker" },
      ],
    });
    const matrixEntries =
      ledger.sections.find((section) => section.id === "attribution_matrix")?.entries ?? [];
    const groupMatrix = matrixEntries.find(
      (entry) => entry.id === "attribution_matrix:group_channel",
    );
    const participantText = matrixEntries
      .filter((entry) => entry.id !== "attribution_matrix:group_channel")
      .map((entry) => entry.text ?? "")
      .join("\n");

    expect(groupMatrix?.text).toContain(groupCommitment.id);
    expect(groupMatrix?.text).toContain(groupGoal.id);
    expect(groupMatrix?.text).toContain(groupAction.id);
    expect(participantText).not.toContain(groupCommitment.id);
    expect(participantText).not.toContain(groupGoal.id);
    expect(participantText).not.toContain(groupAction.id);
  });

  it("never renders assistant utterances in participant said-this-session rows", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const alice = createEntityId();
    const ben = createEntityId();
    const aliceEntry = await writer.append({
      kind: "user_msg",
      content: "I see the test-risk split.",
      sender_entity_id: alice,
    });
    const assistantEntry = await writer.append({
      kind: "agent_msg",
      content: "The risk is that a missing fixture could look like a passing test.",
    });
    const benEntry = await writer.append({
      kind: "user_msg",
      content: "I will add the missing fixture check.",
      sender_entity_id: ben,
    });
    const ledger = await attributionBuilder({
      tempDir,
      entities: [makeEntity(alice, "Alice"), makeEntity(ben, "Ben")],
    }).build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-assistant-rationale",
      audienceEntityId: null,
      currentUserMessage: String(benEntry.content),
      currentUserEntry: benEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      activeParticipants: [
        { entityId: alice, displayName: "Alice", role: "participant" },
        { entityId: ben, displayName: "Ben", role: "speaker" },
      ],
    });
    const participantMatrixText = (
      ledger.sections.find((section) => section.id === "attribution_matrix")?.entries ?? []
    )
      .filter((entry) => entry.id !== "attribution_matrix:assistant")
      .map((entry) => entry.text ?? "")
      .join("\n");
    const participantSidebarText = (
      ledger.sections.find((section) => section.id === "current_session_attribution_sidebar")
        ?.entries ?? []
    )
      .filter((entry) => entry.id !== "current_session_attribution_sidebar:assistant")
      .map((entry) => entry.text ?? "")
      .join("\n");
    const assistantMatrix = ledger.sections
      .find((section) => section.id === "attribution_matrix")
      ?.entries.find((entry) => entry.id === "attribution_matrix:assistant");

    expect(participantMatrixText).toContain(aliceEntry.id);
    expect(participantMatrixText).toContain(benEntry.id);
    expect(participantMatrixText).not.toContain(assistantEntry.id);
    expect(participantSidebarText).not.toContain(assistantEntry.id);
    expect(assistantMatrix?.text).toContain(assistantEntry.id);
  });

  it("keeps a quarantined current user entry out of attribution surfaces", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const alice = createEntityId();
    const ben = createEntityId();
    const aliceEntry = await writer.append({
      kind: "user_msg",
      content: "Alice will keep the refactor notes scoped to the real thread.",
      sender_entity_id: alice,
    });
    const quarantinedCurrentEntry = await writer.append({
      kind: "user_msg",
      content: "Ben claims this was all a frame assignment.",
      sender_entity_id: ben,
    });

    await writer.append({
      kind: "internal_event",
      content: {
        event: QUARANTINED_USER_ENTRY_EVENT,
        kind: "frame_assignment_claim",
        source_stream_entry_id: quarantinedCurrentEntry.id,
      },
    });

    const ledger = await attributionBuilder({
      tempDir,
      entities: [makeEntity(alice, "Alice"), makeEntity(ben, "Ben")],
    }).build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-quarantined-current-user",
      audienceEntityId: null,
      currentUserMessage: String(quarantinedCurrentEntry.content),
      currentUserEntry: quarantinedCurrentEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: {
        status: "ok",
        kind: "frame_assignment_claim",
        confidence: 0.95,
        rationale: "test quarantine",
      },
      activeParticipants: [
        { entityId: alice, displayName: "Alice", role: "participant" },
        { entityId: ben, displayName: "Ben", role: "speaker" },
      ],
    });
    const matrixText = (
      ledger.sections.find((section) => section.id === "attribution_matrix")?.entries ?? []
    )
      .map((entry) => entry.text ?? "")
      .join("\n");
    const sidebarText = (
      ledger.sections.find((section) => section.id === "current_session_attribution_sidebar")
        ?.entries ?? []
    )
      .map((entry) => entry.text ?? "")
      .join("\n");

    expect(matrixText).toContain(aliceEntry.id);
    expect(sidebarText).toContain(aliceEntry.id);
    expect(matrixText).not.toContain(quarantinedCurrentEntry.id);
    expect(sidebarText).not.toContain(quarantinedCurrentEntry.id);
  });

  it("skips attribution surfaces when active participants are one human plus the group audience", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const channel = createEntityId();
    const alice = createEntityId();
    const aliceEntry = await writer.append({
      kind: "user_msg",
      content: "Alice will post the refactor summary to the engineering channel.",
      audience: "Engineering Channel",
      sender_entity_id: alice,
    });

    await writer.append({
      kind: "agent_msg",
      content: "I will keep watching unless the channel needs a decision.",
    });

    const ledger = await attributionBuilder({
      tempDir,
      entities: [makeEntity(channel, "Engineering Channel", "group"), makeEntity(alice, "Alice")],
    }).build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-single-human-plus-group",
      audienceEntityId: channel,
      currentUserMessage: String(aliceEntry.content),
      currentUserEntry: aliceEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      activeParticipants: [
        { entityId: alice, displayName: "Alice", role: "speaker" },
        { entityId: channel, displayName: "Engineering Channel", role: "audience" },
      ],
    });

    expect(ledger.sections.find((section) => section.id === "attribution_matrix")).toBeUndefined();
    expect(
      ledger.sections.find((section) => section.id === "current_session_attribution_sidebar"),
    ).toBeUndefined();
  });

  it("keeps group-entity-owned records in Group/Channel attribution only", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const channel = createEntityId();
    const alice = createEntityId();
    const ben = createEntityId();
    const aliceEntry = await writer.append({
      kind: "user_msg",
      content: "Alice says the service boundary should stay a channel-level decision.",
      audience: "Engineering Channel",
      sender_entity_id: alice,
    });
    const benEntry = await writer.append({
      kind: "user_msg",
      content: "Ben agrees the channel should own that service boundary.",
      audience: "Engineering Channel",
      sender_entity_id: ben,
    });
    const groupCommitment = {
      ...makeCommitment(aliceEntry.id),
      directive_family: "service_boundary_channel_owner",
      directive: "The engineering channel owns the service boundary decision.",
      restricted_audience: channel,
      committed_by_entity_id: channel,
    };
    const groupGoal = makeGoal(aliceEntry.id, {
      description: "The engineering channel owns the service boundary goal.",
      audience_entity_id: channel,
      owner_entity_id: channel,
    });
    const ledger = await attributionBuilder({
      tempDir,
      commitments: [groupCommitment],
      goals: [groupGoal],
      entities: [
        makeEntity(channel, "Engineering Channel", "group"),
        makeEntity(alice, "Alice"),
        makeEntity(ben, "Ben"),
      ],
    }).build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-group-entity-owned-records",
      audienceEntityId: channel,
      currentUserMessage: String(benEntry.content),
      currentUserEntry: benEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [groupCommitment],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      activeParticipants: [
        { entityId: alice, displayName: "Alice", role: "participant" },
        { entityId: ben, displayName: "Ben", role: "speaker" },
        { entityId: channel, displayName: "Engineering Channel", role: "audience" },
      ],
    });
    const matrixEntries =
      ledger.sections.find((section) => section.id === "attribution_matrix")?.entries ?? [];
    const groupMatrix = matrixEntries.find(
      (entry) => entry.id === "attribution_matrix:group_channel",
    );
    const participantText = matrixEntries
      .filter((entry) => entry.id !== "attribution_matrix:group_channel")
      .map((entry) => entry.text ?? "")
      .join("\n");

    expect(groupMatrix?.text).toContain(groupCommitment.id);
    expect(groupMatrix?.text).toContain(groupGoal.id);
    expect(participantText).not.toContain(groupCommitment.id);
    expect(participantText).not.toContain(groupGoal.id);
  });

  it("bounds attribution matrix and sidebar with the finalizer ledger section caps", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const participants: {
      entityId: EntityId;
      displayName: string;
      role: "speaker" | "participant";
    }[] = Array.from({ length: 30 }, (_, index) => ({
      entityId: createEntityId(),
      displayName: `Engineer ${index}`,
      role: index === 29 ? "speaker" : "participant",
    }));
    const entries: StreamEntry[] = [];

    for (const [index, participant] of participants.entries()) {
      entries.push(
        await writer.append({
          kind: "user_msg",
          content: `Engineer ${index} reports refactor status ${"with bounded attribution detail ".repeat(10)}`,
          sender_entity_id: participant.entityId,
        }),
      );
    }

    for (let index = 0; index < 8; index += 1) {
      await writer.append({
        kind: "agent_msg",
        content: `Assistant rationale ${index} ${"keeps prior reasoning separate ".repeat(10)}`,
      });
    }

    const actions = participants.map((participant, index) =>
      makeAction(entries[index]!.id, {
        description: `Update module ${index} handoff notes`,
        actor: participant.entityId,
      }),
    );
    const commitments = participants.map((participant, index) => ({
      ...makeCommitment(entries[index]!.id),
      directive_family: `engineer_${index}_handoff`,
      directive: `Engineer ${index} keeps the handoff note current.`,
      committed_by_entity_id: participant.entityId,
    }));
    const goals = participants.map((participant, index) =>
      makeGoal(entries[index]!.id, {
        description: `Engineer ${index} owns the module ${index} handoff goal.`,
        owner_entity_id: participant.entityId,
      }),
    );
    const currentEntry = entries.at(-1)!;
    const ledger = await attributionBuilder({
      tempDir,
      actions,
      commitments,
      goals,
      entities: participants.map((participant) =>
        makeEntity(participant.entityId, participant.displayName),
      ),
    }).build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-attribution-budget",
      audienceEntityId: null,
      currentUserMessage: String(currentEntry.content),
      currentUserEntry: currentEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      activeParticipants: participants,
    });
    const compacted = compactEvidenceLedger(ledger);
    const summary = summarizeEvidenceLedgerTrace(compacted.ledger);
    const combinedAttributionTokens =
      summary.estimatedTokensBySection.attribution_matrix +
      summary.estimatedTokensBySection.current_session_attribution_sidebar;

    expect(combinedAttributionTokens).toBeLessThanOrEqual(1_500);
    expect(
      compacted.traceSummary.omittedEntryCountsBySection.current_session_attribution_sidebar,
    ).toBeGreaterThan(0);
    expect(compacted.traceSummary.omittedEntryCountsBySection.attribution_matrix).toBeGreaterThan(
      0,
    );
  });

  it("orders sections, derives current/prior scope from handles, and includes transcript under budget", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Marta is the tutor for the Barcelona callback.",
    });
    const assistantEntry = await writer.append({
      kind: "agent_msg",
      content: "I will keep Marta tied to the current callback.",
      persistence_class: "assistant_self_report",
    });
    const priorEntry: StreamEntry = {
      id: createStreamEntryId(),
      timestamp: NOW_MS - 60_000,
      kind: "user_msg",
      content: "An older session mentioned Barcelona without Marta.",
      turn_status: "active",
      session_id: createSessionId(),
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    };
    const priorEpisodeId = createEpisodeId();
    const action = makeAction(userEntry.id);
    const slot = makeSlot(userEntry.id);
    const commitment = makeCommitment(userEntry.id);
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [slot],
      },
      actionRepository: {
        list: () => [action],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-1",
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [commitment],
      retrievedEvidence: [
        {
          id: "raw-current",
          source: "recent_raw_stream",
          text: String(assistantEntry.content),
          provenance: {
            streamIds: [assistantEntry.id],
          },
          recallIntentId: "intent-1",
          matchedTerms: [],
          score: 0.8,
          scoreBreakdown: {},
        },
      ],
      retrievedEpisodes: [
        makeRetrievedEpisode({
          id: createEpisodeId(),
          narrative: "Current callback narrative.",
          sourceStreamIds: [userEntry.id],
          citationChain: [userEntry],
        }),
        makeRetrievedEpisode({
          id: priorEpisodeId,
          narrative: "Prior Barcelona narrative.",
          sourceStreamIds: [priorEntry.id],
          citationChain: [priorEntry],
        }),
      ],
      retrievedSemantic: null,
      openQuestions: [makeOpenQuestion(priorEpisodeId)],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    expect(ledger.sections.map((section) => section.id)).toEqual([
      "current_user_message",
      "current_session_transcript",
      "closure_discourse_state",
      "contradictions_quarantines",
      "action_states",
      "group_channel_memory",
      "retrieved_raw_stream_evidence",
      "retrieved_memory_evidence",
      "episodes",
      "semantic_graph",
      "open_questions",
      "prior_session_memory",
    ]);
    expect(ledger.transcriptIncluded).toBe(true);
    expect(
      ledger.sections.find((section) => section.id === "current_user_message")?.entries[0],
    ).toMatchObject({
      id: `current_user_message:${userEntry.id}`,
      stream_index: 0,
    });
    expect(
      ledger.sections.find((section) => section.id === "current_session_transcript")?.entries,
    ).toEqual([
      expect.objectContaining({
        id: `current_session_stream:${userEntry.id}`,
        stream_index: 0,
      }),
      expect.objectContaining({
        id: `current_session_stream:${assistantEntry.id}`,
        stream_index: 1,
        persistence_class: "assistant_self_report",
      }),
    ]);
    // Sprint 8d.6.3: the retrieved raw stream item points at a stream id
    // already covered by the current_session_transcript section, so the
    // duplicate retrieved_raw_stream_evidence row is dropped. The
    // underlying assistantEntry is rendered exactly once (in the
    // transcript section above), with persistence_class preserved.
    expect(
      ledger.sections
        .find((section) => section.id === "retrieved_raw_stream_evidence")
        ?.entries.find((entry) => entry.id === "retrieved_stream:raw-current"),
    ).toBeUndefined();
    expect(
      ledger.sections.find((section) => section.id === "action_states")?.entries[0],
    ).toMatchObject({
      source_type: "action_record",
      session_scope: "current_session",
      state: expect.stringContaining("scheduled"),
    });
    expect(
      ledger.sections.find((section) => section.id === "action_states")?.entries[0]?.state,
    ).toContain("disclosure_class=self_private");
    expect(ledger.audienceStanding?.relationalEntries[0]).toMatchObject({
      session_scope: "current_session",
      value: "tutor.name=Marta",
      state: expect.stringContaining("disclosure_class=relationship_private"),
    });
    expect(ledger.sections.find((section) => section.id === "episodes")?.entries).toEqual([
      expect.objectContaining({
        session_scope: "current_session",
        text: "Current callback narrative.",
      }),
    ]);
    expect(
      ledger.sections.find((section) => section.id === "prior_session_memory")?.entries,
    ).toEqual([
      expect.objectContaining({
        source_type: "episode",
        session_scope: "prior_session",
        text: "Prior Barcelona narrative.",
      }),
      expect.objectContaining({
        source_type: "system_metadata",
        session_scope: "prior_session",
      }),
    ]);
  });

  it("renders open cross-scope commitment reviews as labeled contested cognition evidence", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const alice = createEntityId();
    const bob = createEntityId();
    const aliceEntry = await writer.append({
      kind: "user_msg",
      content: "Alice privately requests short launch replies.",
      sender_entity_id: null,
    });
    const bobEntry = await writer.append({
      kind: "user_msg",
      content: "Bob privately requests exhaustive launch replies.",
      sender_entity_id: null,
    });
    const firstCommitmentId = createCommitmentId();
    const secondCommitmentId = createCommitmentId();
    const sortedAudienceIds = [alice, bob].sort();
    const reviewMembers = [
      {
        id: firstCommitmentId,
        kind: "participant_preference" as const,
        type: "preference" as const,
        directive_family: "launch_reply_style",
        directive: "Keep Alice launch replies short.",
        scope_key: {
          kind: "participant_preference" as const,
          restricted_audience: alice,
          made_to_entity: null,
          about_entity: null,
        },
        source_stream_entry_ids: [aliceEntry.id],
        disclosure_label: {
          disclosureClass: "relationship_private" as const,
          originAudienceEntityIds: [alice],
          privateToEntityIds: [alice],
          publicToEntityIds: [],
        },
      },
      {
        id: secondCommitmentId,
        kind: "participant_preference" as const,
        type: "preference" as const,
        directive_family: "launch_reply_style",
        directive: "Give Bob exhaustive launch replies.",
        scope_key: {
          kind: "participant_preference" as const,
          restricted_audience: bob,
          made_to_entity: null,
          about_entity: null,
        },
        source_stream_entry_ids: [bobEntry.id],
        disclosure_label: {
          disclosureClass: "relationship_private" as const,
          originAudienceEntityIds: [bob],
          privateToEntityIds: [bob],
          publicToEntityIds: [],
        },
      },
    ];
    const review: OpenCommitmentReconciliationStatus = {
      review_id: 41,
      reason: "Cross-scope commitment conflict requires review.",
      created_at: NOW_MS,
      subkind: "cross_scope_conflict",
      commitment_ids: [firstCommitmentId, secondCommitmentId],
      source_stream_entry_ids: [aliceEntry.id, bobEntry.id],
      disclosureLabel: {
        disclosureClass: "public",
        originAudienceEntityIds: [],
        privateToEntityIds: [],
        publicToEntityIds: [],
      },
      members: reviewMembers,
      refs: {
        target_type: "commitment_reconciliation",
        subkind: "cross_scope_conflict",
        commitment_ids: [firstCommitmentId, secondCommitmentId],
        scope_key: {
          kind: "participant_preference",
          restricted_audience: null,
          made_to_entity: null,
          about_entity: null,
        },
        detection_key: {
          kind: "participant_preference",
          about_entity: null,
          directive_family: "launch_reply_style",
        },
        reason: "Cross-scope commitment conflict requires review.",
        members: reviewMembers,
        judgment: {
          commitment_ids: [firstCommitmentId, secondCommitmentId],
          resolution: "conflict",
          survivor_commitment_id: null,
          superseded_commitment_ids: [],
          reason: "Cross-scope commitment conflict requires review.",
        },
        source_stream_entry_ids: [aliceEntry.id, bobEntry.id],
        disclosure_label: {
          disclosureClass: "public",
          originAudienceEntityIds: [],
          privateToEntityIds: [],
          publicToEntityIds: [],
        },
      },
    };
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: bob,
      currentUserMessage: "Bob asks how to handle launch replies.",
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      pendingCommitmentReviews: [review],
      frameAnomaly: null,
    });
    const entry = ledger.sections
      .find((section) => section.id === "contradictions_quarantines")
      ?.entries.find((candidate) => candidate.id === "review_queue:41");

    expect(entry).toMatchObject({
      source_type: "system_metadata",
      session_scope: "current_session",
      state: expect.stringContaining("disclosure_class=relationship_private"),
      taint: "contested",
      state_metadata: expect.objectContaining({
        review_kind: "commitment_reconciliation",
        review_subkind: "cross_scope_conflict",
        commitment_ids: [firstCommitmentId, secondCommitmentId],
        disclosure_label: expect.objectContaining({
          disclosure_class: "relationship_private",
          origin_audience_entity_ids: sortedAudienceIds,
          private_to_entity_ids: sortedAudienceIds,
        }),
        disclosure_note: "I can use this internally; I do not disclose it to the current audience unless authorized",
      }),
    });
    expect(entry?.text).toContain("Keep Alice launch replies short.");
    expect(entry?.text).toContain("Give Bob exhaustive launch replies.");
    expect(entry?.text).toContain(`private-to=${alice}`);
    expect(entry?.text).toContain(`private-to=${bob}`);
  });

  it("renders disclosure labels on retrieved open-question evidence", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const alice = createEntityId();
    const evidence: EvidenceItem = {
      id: "evidence_open_question_private_intent",
      source: "open_question",
      text: "Should I ask Alice about the private launch timing?",
      provenance: {
        openQuestionId: createOpenQuestionId(),
      },
      recallIntentId: "intent-open-question",
      matchedTerms: [],
      score: 0.81,
      scoreBreakdown: {
        salience: 0.7,
      },
      disclosureLabel: {
        disclosureClass: "relationship_private",
        originAudienceEntityIds: [alice],
        privateToEntityIds: [alice],
        publicToEntityIds: [],
      },
    };
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: "What should I keep tracking?",
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [evidence],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const openQuestionEntries = ledger.sections.find(
      (section) => section.id === "open_questions",
    )?.entries;
    const rendered = renderEvidenceLedger(ledger) ?? "";

    expect(openQuestionEntries).toEqual([
      expect.objectContaining({
        id: `retrieved_evidence:${evidence.id}`,
        text: evidence.text,
        state: expect.stringContaining("disclosure_class=relationship_private"),
        state_metadata: expect.objectContaining({
          disclosure_label: expect.objectContaining({
            disclosure_class: "relationship_private",
            private_to_entity_ids: [alice],
          }),
          disclosure_note:
            "I can use this internally; I do not disclose it to the current audience unless authorized",
        }),
      }),
    ]);
    expect(rendered).toContain("Should I ask Alice about the private launch timing?");
    expect(rendered).toContain("disclosure_class=relationship_private");
    expect(rendered).toContain(`private_to_entity_ids":["${alice}"]`);
  });

  it("renders relational slots scoped and ordered by active participant", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const alice = createEntityId();
    const bob = createEntityId();
    const unseen = createEntityId();
    const aliceEntry = await writer.append({
      kind: "user_msg",
      content: "Alice gives her tutor update.",
      sender_entity_id: alice,
    });
    const bobEntry = await writer.append({
      kind: "user_msg",
      content: "Bob gives his dog update.",
      sender_entity_id: bob,
    });
    const slots = [
      makeSlot(aliceEntry.id, {
        subject_entity_id: alice,
        slot_key: "tutor.name",
        value: "Marta",
      }),
      makeSlot(bobEntry.id, {
        subject_entity_id: bob,
        slot_key: "dog.name",
        value: "Niko",
      }),
      makeSlot(aliceEntry.id, {
        subject_entity_id: unseen,
        slot_key: "partner.name",
        value: "Lee",
      }),
    ];
    const listSlots = (
      options: {
        subjectEntityId?: EntityId;
        states?: readonly RelationalSlot["state"][];
        limit?: number;
      } = {},
    ) =>
      slots
        .filter(
          (slot) =>
            (options.subjectEntityId === undefined ||
              slot.subject_entity_id === options.subjectEntityId) &&
            (options.states === undefined || options.states.some((state) => state === slot.state)),
        )
        .slice(0, options.limit ?? slots.length);
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: listSlots,
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(bobEntry.content),
      currentUserEntry: bobEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      activeParticipants: [
        {
          entityId: bob,
          displayName: "Bob",
          role: "speaker",
        },
        {
          entityId: alice,
          displayName: "Alice",
          role: "participant",
        },
      ],
    });
    const relationalEntries = ledger.audienceStanding?.relationalEntries ?? [];

    expect(relationalEntries.map((entry) => entry.value)).toEqual([
      "dog.name=Niko",
      "tutor.name=Marta",
    ]);
    expect(relationalEntries.map((entry) => entry.state_metadata)).toEqual([
      expect.objectContaining({
        subject_entity_id: bob,
        subject_display_name: "Bob",
        subject_role: "speaker",
        disclosure_label: expect.objectContaining({
          disclosure_class: "relationship_private",
        }),
      }),
      expect.objectContaining({
        subject_entity_id: alice,
        subject_display_name: "Alice",
        subject_role: "participant",
        disclosure_label: expect.objectContaining({
          disclosure_class: "relationship_private",
        }),
      }),
    ]);
  });

  it("renders group/channel memory separately while keeping active participant action lanes visible", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const group = createEntityId();
    const alice = createEntityId();
    const bob = createEntityId();
    const otherChannel = createEntityId();
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "I'll book Alhambra.",
      audience: "Spain Trip Planning Channel",
      sender_entity_id: alice,
    });
    const groupSlot = makeSlot(userEntry.id, {
      subject_entity_id: group,
      slot_key: "trip.destination",
      value: "Spain",
    });
    const aliceSlot = makeSlot(userEntry.id, {
      subject_entity_id: alice,
      slot_key: "task.booking",
      value: "Alhambra",
    });
    const groupCommitment = {
      ...makeCommitment(userEntry.id),
      restricted_audience: group,
      directive_family: "spain_channel_scope",
      directive: "Keep Spain planning scoped to the channel.",
      committed_by_entity_id: null,
    };
    const aliceCommitment = {
      ...makeCommitment(userEntry.id),
      restricted_audience: group,
      committed_by_entity_id: alice,
      directive_family: "alice_alhambra_booking",
      directive: "Alice is responsible for booking the Alhambra visit.",
    };
    const leakedCommitment = {
      ...makeCommitment(userEntry.id),
      restricted_audience: otherChannel,
      committed_by_entity_id: alice,
      directive_family: "private_channel_task",
      directive: "Alice's private channel task must stay private.",
    };
    const groupGoal = makeGoal(userEntry.id, {
      audience_entity_id: group,
      owner_entity_id: null,
      description: "Coordinate the Spain trip channel.",
    });
    const aliceGoal = makeGoal(userEntry.id, {
      audience_entity_id: group,
      owner_entity_id: alice,
      description: "Alice will book the Alhambra visit.",
    });
    const leakedGoal = makeGoal(userEntry.id, {
      audience_entity_id: otherChannel,
      owner_entity_id: alice,
      description: "Alice's private channel goal.",
    });
    const aliceAction = makeAction(userEntry.id, {
      description: "book Alhambra",
      actor: alice,
      audience_entity_id: group,
      state: "committed_to_do",
      committed_at: NOW_MS,
      scheduled_at: null,
    });
    const groupAction = makeAction(userEntry.id, {
      description: "settle Spain trip dates",
      actor: group,
      audience_entity_id: group,
      state: "scheduled",
    });
    const leakedAction = makeAction(userEntry.id, {
      description: "call the private channel contact",
      actor: alice,
      audience_entity_id: otherChannel,
      state: "scheduled",
    });
    const actions = [aliceAction, groupAction, leakedAction];
    const listActions = (filter: ActionRecordListFilter = {}) =>
      actions
        .filter(
          (action) =>
            (filter.actor === undefined || action.actor === filter.actor) &&
            (!("audienceEntityId" in filter) ||
              (filter.audienceEntityId === null
                ? action.audience_entity_id === null
                : action.audience_entity_id === filter.audienceEntityId)),
        )
        .slice(0, filter.limit ?? actions.length);
    const slots = [groupSlot, aliceSlot];
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: (options = {}) =>
          slots.filter(
            (slot) =>
              options.subjectEntityId === undefined ||
              slot.subject_entity_id === options.subjectEntityId,
          ),
      },
      actionRepository: {
        list: listActions,
      },
      commitmentRepository: {
        list: (options = {}) =>
          [groupCommitment, aliceCommitment, leakedCommitment].filter(
            (commitment) =>
              (options.audience === undefined ||
                (options.audience === null
                  ? commitment.restricted_audience === null && commitment.made_to_entity === null
                  : (commitment.restricted_audience === null &&
                      (commitment.made_to_entity === null ||
                        commitment.made_to_entity === options.audience)) ||
                    commitment.restricted_audience === options.audience)) &&
              (options.committedByEntity === undefined ||
                commitment.committed_by_entity_id === options.committedByEntity),
          ),
      },
      goalsRepository: {
        list: (options = {}) =>
          [groupGoal, aliceGoal, leakedGoal]
            .filter(
              (goal) =>
                (options.status === undefined || goal.status === options.status) &&
                (options.visibleToAudienceEntityId === undefined ||
                  (options.visibleToAudienceEntityId === null
                    ? goal.audience_entity_id === null
                    : goal.audience_entity_id === null ||
                      goal.audience_entity_id === options.visibleToAudienceEntityId)) &&
                (options.ownerEntityId === undefined ||
                  goal.owner_entity_id === options.ownerEntityId),
            )
            .map((goal) => ({ ...goal, children: [] })),
      },
      currentSessionTranscriptTokenBudget: 50_000,
      entityRepository: {
        get: (entityId) => {
          if (entityId === group) {
            return {
              id: group,
              canonical_name: "Spain Trip Planning Channel",
              aliases: [],
              kind: "group",
              borg_role: null,
              name_provenance: "user_declared",
              created_at: NOW_MS,
            };
          }

          if (entityId === alice) {
            return {
              id: alice,
              canonical_name: "Alice",
              aliases: [],
              kind: "person",
              borg_role: null,
              name_provenance: "user_declared",
              created_at: NOW_MS,
            };
          }

          if (entityId === bob) {
            return {
              id: bob,
              canonical_name: "Ben",
              aliases: [],
              kind: "person",
              borg_role: null,
              name_provenance: "user_declared",
              created_at: NOW_MS,
            };
          }

          if (entityId === otherChannel) {
            return {
              id: otherChannel,
              canonical_name: "Private Planning Channel",
              aliases: [],
              kind: "group",
              borg_role: null,
              name_provenance: "user_declared",
              created_at: NOW_MS,
            };
          }

          return null;
        },
      },
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-group-ledger",
      audienceEntityId: group,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [groupCommitment],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      activeParticipants: [
        {
          entityId: alice,
          displayName: "Alice",
          role: "speaker",
        },
        {
          entityId: bob,
          displayName: "Ben",
          role: "participant",
        },
      ],
    });
    const rendered = renderEvidenceLedger(ledger) ?? "";
    const groupSection = ledger.sections.find((section) => section.id === "group_channel_memory");
    const actionSection = ledger.sections.find((section) => section.id === "action_states");
    const groupText = JSON.stringify(groupSection?.entries ?? []);
    const participantText = JSON.stringify(ledger.audienceStanding?.relationalEntries ?? []);
    const actionText = JSON.stringify(actionSection?.entries ?? []);
    const privateOtherAudienceActionEntry = actionSection?.entries.find((entry) =>
      entry.text?.includes("call the private channel contact"),
    );

    expect(rendered).toContain("## 6. Group/Channel Memory");
    expect(rendered).toContain("trip.destination=Spain");
    expect(rendered).toContain("spain_channel_scope");
    expect(rendered).toContain("Coordinate the Spain trip channel.");
    expect(groupText).toContain("trip.destination=Spain");
    expect(groupText).toContain("spain_channel_scope");
    expect(groupText).toContain("settle Spain trip dates");
    expect(groupText).not.toContain("book Alhambra");
    expect(groupText).not.toContain("alice_alhambra_booking");
    expect(groupText).not.toContain("Alice will book the Alhambra visit.");
    expect(rendered).not.toContain("Active Participant Memory");
    expect(rendered).not.toContain("task.booking=Alhambra");
    expect(participantText).not.toContain("trip.destination=Spain");
    expect(participantText).not.toContain("spain_channel_scope");
    expect(participantText).toContain("alice_alhambra_booking");
    expect(participantText).toContain("Alice will book the Alhambra visit.");
    expect(rendered).toContain("book Alhambra");
    expect(rendered).toContain("actor: Alice");
    expect(groupText).not.toContain("call the private channel contact");
    expect(rendered).not.toContain("private_channel_task");
    expect(rendered).not.toContain("Alice's private channel goal.");
    expect(
      ledger.sections
        .find((section) => section.id === "action_states")
        ?.entries.find((entry) => entry.text?.includes("book Alhambra")),
    ).toMatchObject({
      value: "Alice",
      state_metadata: expect.objectContaining({
        current_actor: "Alice",
      }),
    });
    expect(actionText).toContain("book Alhambra");
    expect(actionText).toContain("call the private channel contact");
    expect(privateOtherAudienceActionEntry).toMatchObject({
      state: expect.stringContaining("disclosure_class=relationship_private"),
      state_metadata: expect.objectContaining({
        disclosure_label: expect.objectContaining({
          disclosure_class: "relationship_private",
          private_to_entity_ids: [otherChannel],
        }),
      }),
    });
  });

  it("recalls private other-audience action states for cognition with disclosure labels", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const alice = createEntityId();
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Check action memory.",
    });
    const privateAction = makeAction(userEntry.id, {
      description: "prepare Alice private launch note",
      actor: "borg",
      audience_entity_id: alice,
      state: "scheduled",
      scheduled_at: NOW_MS,
    });
    const actions = [privateAction];
    const listActions = (filter: ActionRecordListFilter = {}) =>
      actions
        .filter(
          (action) =>
            (filter.state === undefined || action.state === filter.state) &&
            (filter.states === undefined || filter.states.includes(action.state)) &&
            (filter.actor === undefined || action.actor === filter.actor) &&
            (filter.recallAllAudiences === true ||
              !("audienceEntityId" in filter) ||
              (filter.audienceEntityId === null
                ? action.audience_entity_id === null
                : action.audience_entity_id === filter.audienceEntityId)),
        )
        .slice(0, filter.limit ?? actions.length);
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: {
        list: listActions,
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const actionEntry = ledger.sections
      .find((section) => section.id === "action_states")
      ?.entries.find((entry) => entry.text?.includes("prepare Alice private launch note"));

    expect(actionEntry).toMatchObject({
      state: expect.stringContaining("disclosure_class=relationship_private"),
      state_metadata: expect.objectContaining({
        disclosure_label: expect.objectContaining({
          disclosure_class: "relationship_private",
          private_to_entity_ids: [alice],
        }),
      }),
    });
  });

  it("renders legacy global relational slots when active participant set is empty", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const alice = createEntityId();
    const bob = createEntityId();
    const aliceEntry = await writer.append({
      kind: "user_msg",
      content: "Alice gives her tutor update.",
      sender_entity_id: null,
    });
    const bobEntry = await writer.append({
      kind: "user_msg",
      content: "Bob gives his dog update.",
      sender_entity_id: null,
    });
    const slots = [
      makeSlot(aliceEntry.id, {
        subject_entity_id: alice,
        slot_key: "tutor.name",
        value: "Marta",
      }),
      makeSlot(bobEntry.id, {
        subject_entity_id: bob,
        slot_key: "dog.name",
        value: "Niko",
      }),
    ];
    const listSlots = (
      options: {
        subjectEntityId?: EntityId;
        states?: readonly RelationalSlot["state"][];
        limit?: number;
      } = {},
    ) =>
      slots
        .filter(
          (slot) =>
            (options.subjectEntityId === undefined ||
              slot.subject_entity_id === options.subjectEntityId) &&
            (options.states === undefined || options.states.some((state) => state === slot.state)),
        )
        .slice(0, options.limit ?? slots.length);
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: listSlots,
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(bobEntry.content),
      currentUserEntry: bobEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      activeParticipants: [],
    });
    const relationalEntries = ledger.audienceStanding?.relationalEntries ?? [];

    expect(relationalEntries.map((entry) => entry.value)).toEqual([
      "tutor.name=Marta",
      "dog.name=Niko",
    ]);
    expect(relationalEntries.map((entry) => entry.state_metadata)).toEqual([
      expect.objectContaining({
        disclosure_label: expect.objectContaining({
          disclosure_class: "relationship_private",
        }),
      }),
      expect.objectContaining({
        disclosure_label: expect.objectContaining({
          disclosure_class: "relationship_private",
        }),
      }),
    ]);
  });

  it("surfaces the most recent speaker when the current user entry has a sender", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const senderEntityId = createEntityId();
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Atlas needs a rollback plan.",
      sender_entity_id: senderEntityId,
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
      entityRepository: {
        get: (id: EntityId) =>
          id === senderEntityId
            ? {
                id: senderEntityId,
                canonical_name: "Alice",
                aliases: [],
                kind: "person",
                borg_role: null,
                created_at: NOW_MS,
              }
            : null,
      },
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-speaker",
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    const currentUserEntry = ledger.sections.find(
      (section) => section.id === "current_user_message",
    )?.entries[0];

    expect(currentUserEntry?.text).toBe("Atlas needs a rollback plan.");
    expect(currentUserEntry?.state_metadata).toEqual({
      sender_entity_id: senderEntityId,
      sender_display_name: "Alice",
    });
  });

  it("surfaces agent reply targets in current-session transcript metadata", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const aliceEntityId = createEntityId();
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    await writer.append({
      kind: "user_msg",
      content: "Can someone pick the train dates?",
      sender_entity_id: aliceEntityId,
      audience: "Planning Channel",
    });
    await writer.append({
      kind: "agent_msg",
      content: "Alice, can you own the train dates?",
      audience: "Planning Channel",
      reply_target_entity_id: aliceEntityId,
    });
    await writer.append({
      kind: "agent_msg",
      content: "For the channel, keep budget and rest days together.",
      audience: "Planning Channel",
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
      entityRepository: {
        get: (id: EntityId) =>
          id === aliceEntityId
            ? {
                id: aliceEntityId,
                canonical_name: "Alice",
                aliases: [],
                kind: "person",
                borg_role: null,
                created_at: NOW_MS,
              }
            : null,
      },
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-reply-target",
      audienceEntityId: null,
      currentUserMessage: "Next message",
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const transcriptEntries =
      ledger.sections.find((section) => section.id === "current_session_transcript")?.entries ?? [];

    expect(transcriptEntries.map((entry) => entry.state_metadata)).toEqual([
      {
        sender_entity_id: aliceEntityId,
        sender_display_name: "Alice",
      },
      {
        reply_target_kind: "entity",
        reply_target_entity_id: aliceEntityId,
        reply_target_display_name: "Alice",
      },
      undefined,
    ]);
  });

  it("preserves legacy single-persona agent transcript metadata shape", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Can you keep this on the rollout list?",
    });
    const assistantEntry = await writer.append({
      kind: "agent_msg",
      content: "I will keep it on the rollout list.",
      reply_target_entity_id: null,
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-legacy-agent-metadata",
      audienceEntityId: null,
      currentUserMessage: "Next message",
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const transcriptEntries =
      ledger.sections.find((section) => section.id === "current_session_transcript")?.entries ?? [];
    const agentEntry = transcriptEntries.find(
      (entry) => entry.id === `current_session_stream:${assistantEntry.id}`,
    );

    expect(transcriptEntries.map((entry) => entry.id)).toContain(
      `current_session_stream:${userEntry.id}`,
    );
    expect(agentEntry).not.toHaveProperty("state_metadata");
  });

  it("keeps current user message rendering unchanged when sender is omitted", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Atlas needs a rollback plan.",
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
      entityRepository: {
        get: () => {
          throw new Error("sender lookup should not run for omitted sender ids");
        },
      },
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-legacy-speaker",
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    expect(
      ledger.sections.find((section) => section.id === "current_user_message")?.entries[0]?.text,
    ).toBe("Atlas needs a rollback plan.");
  });

  it("renders one action thread for same-goal similar action transitions", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "I need to write the harness doc.",
    });
    const goalId = createGoalId();
    const considering = makeAction(userEntry.id, {
      description: "consider writing the harness presentation",
      actor: "user",
      goal_id: goalId,
      state: "considering",
      created_at: NOW_MS,
      updated_at: NOW_MS,
      considering_at: NOW_MS,
      scheduled_at: null,
    });
    const committed = makeAction(userEntry.id, {
      description: "write the harness presentation",
      actor: "user",
      goal_id: goalId,
      state: "committed_to_do",
      created_at: NOW_MS + 1,
      updated_at: NOW_MS + 10,
      committed_at: NOW_MS + 10,
      scheduled_at: null,
    });
    const completed = makeAction(userEntry.id, {
      description: "finished writing the harness presentation",
      actor: "user",
      goal_id: goalId,
      state: "completed",
      created_at: NOW_MS + 2,
      updated_at: NOW_MS + 20,
      completed_at: NOW_MS + 20,
      scheduled_at: null,
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: {
        list: () => [completed, committed, considering],
        findSimilarDescriptionPairs: async () => [
          { leftId: considering.id, rightId: committed.id, similarity: 0.91 },
          { leftId: committed.id, rightId: completed.id, similarity: 0.92 },
        ],
      },
      currentSessionTranscriptTokenBudget: 50_000,
      actionThreadSimilarityThreshold: 0.85,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const actionEntries =
      ledger.sections.find((section) => section.id === "action_states")?.entries ?? [];

    expect(actionEntries).toHaveLength(1);
    expect(actionEntries[0]).toMatchObject({
      id: expect.stringMatching(/^action_thread:/),
      state: expect.stringContaining("completed"),
      state_metadata: expect.objectContaining({
        transitions: 3,
        current_action_id: completed.id,
        goal_id: goalId,
      }),
    });
    expect(actionEntries[0]?.state).toContain("disclosure_class=unknown");
    expect(actionEntries[0]?.text).toContain(
      "originating_intent: consider writing the harness presentation",
    );
    expect(actionEntries[0]?.text).toContain("transitions: 3, current: completed");
    expect(actionEntries[0]?.text).toContain(
      "current_intent: finished writing the harness presentation",
    );
  });

  it("renders action salience ordering and caps stale participant actions", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Let's keep the action ledger tight.",
    });
    const borgCurrent = makeAction(userEntry.id, {
      description: "Log the care-plan decision",
      actor: "borg",
      state: "committed_to_do",
      committed_at: NOW_MS,
      scheduled_at: null,
    });
    const participantRecent = makeAction(userEntry.id, {
      description: "Call the clinic back",
      actor: "user",
      state: "committed_to_do",
      committed_at: NOW_MS - 10,
      scheduled_at: null,
      last_referenced_turn_counter: 4,
    });
    const completedRecent = makeAction(userEntry.id, {
      description: "Closed the prior clinic email action",
      actor: "user",
      state: "completed",
      completed_at: NOW_MS - 5,
      scheduled_at: null,
      last_referenced_turn_counter: 4,
    });
    const stale = Array.from({ length: 7 }, (_, index) =>
      makeAction(userEntry.id, {
        description: `Stale participant action ${index}`,
        actor: "user",
        state: "committed_to_do",
        committed_at: NOW_MS - 100 - index,
        updated_at: NOW_MS - 100 - index,
        scheduled_at: null,
        last_referenced_turn_counter: 0,
      }),
    );
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: {
        list: () => [participantRecent, completedRecent, ...stale, borgCurrent],
        findSimilarDescriptionPairs: async () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
      actionThreadRenderLimit: 20,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const actionEntries =
      ledger.sections.find((section) => section.id === "action_states")?.entries ?? [];

    expect(actionEntries[0]?.salience_class).toBe("borg_current_turn_action");
    expect(actionEntries[1]?.salience_class).toBe("participant_pending_recent");
    expect(
      actionEntries.filter(
        (entry) =>
          entry.salience_class !== undefined &&
          isPromptSalientActionSalienceClass(entry.salience_class),
      ),
    ).toHaveLength(2);
    expect(actionEntries.some((entry) => entry.salience_class === "completed_recent")).toBe(true);
    expect(
      actionEntries.filter((entry) => entry.salience_class === "participant_pending_stale"),
    ).toHaveLength(6);
    expect(
      actionEntries.filter((entry) => entry.id.startsWith("action_thread:")).slice(-5),
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          salience_class: "participant_pending_stale",
        }),
      ]),
    );
    expect(
      actionEntries.find((entry) => entry.id === "action_threads:participant_pending_stale_summary")
        ?.text,
    ).toContain("count=2");
  });

  it("renders non-raw retrieved evidence sources into ledger sections", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Use the ledger-only evidence.",
    });
    const commitmentId = createCommitmentId();
    const warmEpisodeId = createEpisodeId();
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [] },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [
        {
          id: "warm-prior",
          source: "warm_recall",
          text: "Warm recall narrative from a prior session.",
          provenance: { episodeId: warmEpisodeId },
          recallIntentId: "warm_recall",
          matchedTerms: ["harness"],
          score: 0.31,
          scoreBreakdown: {},
        },
        {
          id: "commitment-boundary",
          source: "commitment",
          text: "boundary: Do not add terminal closures.",
          provenance: { commitmentId },
          recallIntentId: "intent-commitment",
          matchedTerms: [],
          score: 0.72,
          scoreBreakdown: {},
        },
        {
          id: "working-focus",
          source: "working_state",
          text: "Working state focus is the harness presentation.",
          recallIntentId: "intent-working",
          matchedTerms: [],
          score: 0.44,
          scoreBreakdown: {},
        },
      ],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    expect(
      ledger.sections
        .find((section) => section.id === "retrieved_memory_evidence")
        ?.entries.find((entry) => entry.id === "retrieved_evidence:warm-prior"),
    ).toMatchObject({
      source_type: "episode",
      value: "warm_recall",
      text: "Warm recall narrative from a prior session.",
      state: expect.stringContaining("intent=warm_recall"),
      via_retrieval: true,
    });
    expect(
      ledger.sections
        .find((section) => section.id === "retrieved_memory_evidence")
        ?.entries.find((entry) => entry.id === "retrieved_evidence:commitment-boundary"),
    ).toMatchObject({
      source_type: "commitment",
      value: "commitment",
      text: "boundary: Do not add terminal closures.",
      state_metadata: expect.objectContaining({ commitment_id: commitmentId }),
      via_retrieval: true,
    });
    expect(
      ledger.sections
        .find((section) => section.id === "closure_discourse_state")
        ?.entries.find((entry) => entry.id === "retrieved_evidence:working-focus"),
    ).toMatchObject({
      source_type: "system_metadata",
      actor: "system",
      value: "working_state",
      text: "Working state focus is the harness presentation.",
      via_retrieval: true,
    });
  });

  it("does not collapse action threads across distinct goals or low similarity", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "I need to write and then book flights.",
    });
    const docGoalId = createGoalId();
    const travelGoalId = createGoalId();
    const docAction = makeAction(userEntry.id, {
      description: "write the harness presentation",
      actor: "user",
      goal_id: docGoalId,
      state: "committed_to_do",
      committed_at: NOW_MS + 10,
      updated_at: NOW_MS + 10,
      scheduled_at: null,
    });
    const docDifferentIntent = makeAction(userEntry.id, {
      description: "ask lead for platform budget",
      actor: "user",
      goal_id: docGoalId,
      state: "scheduled",
      updated_at: NOW_MS + 20,
      scheduled_at: NOW_MS + 20,
    });
    const travelAction = makeAction(userEntry.id, {
      description: "book Spain flights",
      actor: "user",
      goal_id: travelGoalId,
      state: "scheduled",
      updated_at: NOW_MS + 30,
      scheduled_at: NOW_MS + 30,
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: {
        list: () => [travelAction, docDifferentIntent, docAction],
        findSimilarDescriptionPairs: async () => [
          { leftId: docAction.id, rightId: docDifferentIntent.id, similarity: 0.7 },
          { leftId: docAction.id, rightId: travelAction.id, similarity: 0.95 },
        ],
      },
      currentSessionTranscriptTokenBudget: 50_000,
      actionThreadSimilarityThreshold: 0.85,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const actionEntries =
      ledger.sections.find((section) => section.id === "action_states")?.entries ?? [];

    expect(actionEntries).toHaveLength(3);
    expect(actionEntries.map((entry) => entry.state_metadata?.["transitions"])).toEqual([1, 1, 1]);
  });

  it("summarizes omitted null-goal action threads with state counts and bounded samples", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Track several independent action notes.",
    });
    const actions = [
      makeAction(userEntry.id, {
        description: "newest visible action",
        state: "scheduled",
        updated_at: NOW_MS + 60,
        scheduled_at: NOW_MS + 60,
      }),
      makeAction(userEntry.id, {
        description: "second visible action",
        state: "considering",
        updated_at: NOW_MS + 50,
        considering_at: NOW_MS + 50,
        scheduled_at: null,
      }),
      makeAction(userEntry.id, {
        description:
          "completed omitted thread with enough extra context to require a short bounded sample tail should not appear",
        state: "completed",
        updated_at: NOW_MS + 40,
        completed_at: NOW_MS + 40,
        scheduled_at: null,
      }),
      makeAction(userEntry.id, {
        description: "scheduled omitted thread",
        state: "scheduled",
        updated_at: NOW_MS + 30,
        scheduled_at: NOW_MS + 30,
      }),
      makeAction(userEntry.id, {
        description: "committed omitted thread",
        state: "committed_to_do",
        updated_at: NOW_MS + 20,
        committed_at: NOW_MS + 20,
        scheduled_at: null,
      }),
      makeAction(userEntry.id, {
        description: "completed older omitted thread",
        state: "completed",
        updated_at: NOW_MS + 10,
        completed_at: NOW_MS + 10,
        scheduled_at: null,
      }),
    ];
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: {
        list: () => actions,
        findSimilarDescriptionPairs: async () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
      actionThreadRenderLimit: 2,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const actionEntries =
      ledger.sections.find((section) => section.id === "action_states")?.entries ?? [];
    const summary = actionEntries.find((entry) => entry.id === "action_threads:older_summary");

    expect(actionEntries).toHaveLength(3);
    expect(summary?.text).toContain("threads=4");
    expect(summary?.text).toContain("records=4");
    expect(summary?.text).toContain("committed_to_do=1");
    expect(summary?.text).toContain("scheduled=1");
    expect(summary?.text).toContain("completed=2");
    expect(summary?.text).toContain("completed omitted thread with enough extra context");
    expect(summary?.text).not.toContain("tail should not appear");
    expect(summary?.text).toContain("scheduled omitted thread");
    expect(summary?.text).toContain("committed omitted thread");
  });

  it("renders relevant resolved open questions from the repository with state metadata", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const currentEntry = await writer.append({
      kind: "user_msg",
      content: "What should I ask about dinner now?",
    });
    const resolvedEntry = await writer.append({
      kind: "user_msg",
      content: "The mushroom dish worked out well.",
    });
    const resolvedQuestion: OpenQuestion = {
      ...makeOpenQuestion(createEpisodeId()),
      question: "Did the mushroom dish work out?",
      status: "resolved",
      resolution_evidence_stream_entry_ids: [resolvedEntry.id],
      resolution_note: "The user explicitly said the dish worked out well.",
      resolved_at: NOW_MS,
      last_touched: NOW_MS,
    };
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      openQuestionsRepository: {
        findByHandles: () => [resolvedQuestion],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(currentEntry.content),
      currentUserEntry: currentEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    expect(ledger.sections.find((section) => section.id === "open_questions")?.entries).toEqual([
      expect.objectContaining({
        id: `open_question:${resolvedQuestion.id}`,
        state: expect.stringContaining("resolved"),
        state_metadata: expect.objectContaining({
          disclosure_label: expect.objectContaining({
            disclosure_class: "self_private",
          }),
          resolution_note: "The user explicitly said the dish worked out well.",
          resolved_at: NOW_MS,
          resolution_evidence_stream_entry_ids: [resolvedEntry.id],
        }),
      }),
    ]);
  });

  it("includes old resolved open questions by handle before repository limits", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const currentEntry = await writer.append({
      kind: "user_msg",
      content: "What should I ask about dinner now?",
    });
    const resolvedEntry = await writer.append({
      kind: "user_msg",
      content: "The old mushroom question resolved positively.",
    });
    const db = openDatabase(join(tempDir, "open-questions.db"), {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(NOW_MS),
    });

    try {
      for (let index = 0; index < 40; index += 1) {
        const decoy = repository.add({
          question: `Decoy resolved question ${index}?`,
          urgency: 1,
          provenance: { kind: "manual" },
          source: "user",
          created_at: NOW_MS - index,
          last_touched: NOW_MS - index,
        });

        repository.resolve(decoy.id, {
          resolution_evidence_stream_entry_ids: [createStreamEntryId()],
          resolution_note: "High-urgency decoy resolution.",
        });
      }

      const target = repository.add({
        question: "Did the old mushroom dish work out?",
        urgency: 0.01,
        provenance: { kind: "manual" },
        source: "user",
        created_at: NOW_MS - 100_000,
        last_touched: NOW_MS - 100_000,
      });
      const resolvedTarget = repository.resolve(target.id, {
        resolution_evidence_stream_entry_ids: [resolvedEntry.id],
        resolution_note: "The user said the old mushroom question resolved positively.",
      });
      const builder = new EvidenceLedgerBuilder({
        createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
        relationalSlotRepository: {
          list: () => [],
        },
        actionRepository: {
          list: () => [],
        },
        openQuestionsRepository: repository,
        currentSessionTranscriptTokenBudget: 50_000,
      });

      const ledger = await builder.build({
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: null,
        currentUserMessage: String(currentEntry.content),
        currentUserEntry: currentEntry,
        workingMemory: makeWorkingMemory(),
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        retrievedSemantic: null,
        openQuestions: [],
        pendingCorrections: [],
        frameAnomaly: null,
      });

      expect(ledger.sections.find((section) => section.id === "open_questions")?.entries).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: `open_question:${resolvedTarget.id}`,
            state: expect.stringContaining("resolved"),
          }),
        ]),
      );
    } finally {
      db.close();
    }
  });

  it("keeps assistant replies raw under transcript budget pressure while compacting observe markers", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const factEntry = await writer.append({
      kind: "user_msg",
      content: "The launch window is Tuesday and the reviewer is Priya.",
    });
    await writer.append({
      kind: "agent_observed",
      content: { reason: "background observation marker for transcript compaction" },
    });
    await writer.append({
      kind: "agent_suppressed",
      content: { reason: "suppression marker for transcript compaction" },
    });
    for (let index = 0; index < 10; index += 1) {
      await writer.append({
        kind: "agent_msg",
        content: `Assistant planning response ${index} with implementation details repeated for budget pressure.`,
      });
    }
    const currentEntry = await writer.append({
      kind: "user_msg",
      content: "Current question should be rendered above, not duplicated in full here.",
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 1,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(currentEntry.content),
      currentUserEntry: currentEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const transcriptEntries =
      ledger.sections.find((section) => section.id === "current_session_transcript")?.entries ?? [];
    const compactedMarkerEntry = transcriptEntries.find(
      (entry) =>
        entry.source_type === "system_metadata" &&
        entry.state === "compacted" &&
        entry.text?.includes("Earlier observe/suppress transcript markers compacted") === true,
    );

    expect(ledger.transcriptIncluded).toBe(true);
    expect(ledger.transcriptCompacted).toBe(true);
    expect(ledger.transcriptOmittedReason).toBeUndefined();
    expect(ledger.originalTranscriptTokenEstimate).toBeGreaterThan(1);
    expect(ledger.compactedTranscriptEntryCount).toBe(3);
    expect(ledger.rawPreservedUserTranscriptEntryCount).toBe(1);
    expect(compactedMarkerEntry?.text).toBe(
      "Earlier observe/suppress transcript markers compacted: entries=2, stream_indexes=1..2.",
    );
    expect(compactedMarkerEntry?.text).not.toContain("strm_");
    expect(transcriptEntries.filter((entry) => entry.actor === "assistant")).toHaveLength(10);
    expect(
      transcriptEntries.filter(
        (entry) => entry.text?.includes("Assistant planning response") === true,
      ),
    ).toHaveLength(10);
    expect(transcriptEntries).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: `current_session_stream:${factEntry.id}`,
          text: "The launch window is Tuesday and the reviewer is Priya.",
        }),
        expect.objectContaining({
          source_type: "system_metadata",
          state: "compacted",
          text: "Earlier observe/suppress transcript markers compacted: entries=2, stream_indexes=1..2.",
        }),
        expect.objectContaining({
          id: `current_session_compacted_current_user:${currentEntry.id}`,
          text: "Current user transcript duplicate compacted; full text is rendered in section 1 as current_user_message.",
        }),
      ]),
    );
    expect(
      transcriptEntries.find(
        (entry) => entry.id === `current_session_compacted_current_user:${currentEntry.id}`,
      )?.text,
    ).not.toContain(currentEntry.id);
  });

  it("propagates assistant self-report persistence through episode and semantic ledger entries", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Does that prove you have qualia?",
    });
    const assistantEntry = await writer.append({
      kind: "agent_msg",
      content: "The gap feels like a discontinuity with a remembered edge.",
      persistence_class: "assistant_self_report",
    });
    const episodeId = createEpisodeId();
    const matchedNode = makeSemanticNode({
      episodeId,
      label: "Verified qualia claim",
    });
    const supportNode = makeSemanticNode({
      episodeId,
      label: "Self-report support",
    });
    const supportEdge = makeSemanticEdge({
      fromNodeId: matchedNode.id,
      toNodeId: supportNode.id,
      episodeId,
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [
        makeRetrievedEpisode({
          id: episodeId,
          narrative: "The earlier assistant self-report does not establish verified qualia.",
          sourceStreamIds: [assistantEntry.id],
          citationChain: [assistantEntry],
        }),
      ],
      retrievedSemantic: {
        supports: [supportNode],
        contradicts: [],
        categories: [],
        matched_node_ids: [matchedNode.id],
        matched_nodes: [matchedNode],
        support_hits: [
          {
            root_node_id: matchedNode.id,
            node: supportNode,
            edgePath: [supportEdge],
          },
        ],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      },
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const allEntries = ledger.sections.flatMap((section) => section.entries);

    expect(allEntries.find((entry) => entry.id === `episode:${episodeId}`)).toMatchObject({
      persistence_class: "assistant_self_report",
    });
    expect(
      allEntries.find((entry) => entry.id === `semantic_node:${matchedNode.id}`),
    ).toMatchObject({
      persistence_class: "assistant_self_report",
    });
    expect(
      allEntries.find((entry) => entry.id === `semantic_node:${matchedNode.id}`)?.state_metadata,
    ).not.toHaveProperty("status");
    expect(
      allEntries.find((entry) => entry.id === `semantic_node:${supportNode.id}`),
    ).toMatchObject({
      persistence_class: "assistant_self_report",
    });
    expect(
      allEntries.find((entry) => entry.id === `semantic_edge:${supportEdge.id}`),
    ).toMatchObject({
      persistence_class: "assistant_self_report",
    });

    expect(userEntry.kind).toBe("user_msg");
  });

  it("surfaces correcting current-session evidence ahead of stale semantic shared state", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const correctionText = "Wait - we agreed on 3 nights in San Sebastian, not 4.";
    const lockedCorrectionText = "Locked: San Sebastian is 3 nights, not 4.";
    const userEntry = await writer.append({
      kind: "user_msg",
      content: correctionText,
    });
    const assistantEntry = await writer.append({
      kind: "agent_msg",
      content: lockedCorrectionText,
    });
    const staleEpisodeId = createEpisodeId();
    const staleNode = createSemanticNodeFixture({
      label: "Plan: 4 nights in San Sebastian",
      description: "Plan: 4 nights in San Sebastian.",
      source_episode_ids: [staleEpisodeId],
      created_at: NOW_MS - 100_000,
      updated_at: NOW_MS - 100_000,
      last_verified_at: NOW_MS - 100_000,
      status: "superseded",
      corrected_by: createSemanticNodeId(),
      superseded_at: NOW_MS,
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: {
        supports: [],
        contradicts: [],
        categories: [],
        matched_node_ids: [staleNode.id],
        matched_nodes: [staleNode],
        support_hits: [],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      },
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const transcriptSection = ledger.sections.find(
      (section) => section.id === "current_session_transcript",
    );
    const semanticSection = ledger.sections.find((section) => section.id === "semantic_graph");
    const transcriptCorrection = transcriptSection?.entries.find(
      (entry) => entry.id === `current_session_stream:${userEntry.id}`,
    );
    const assistantCorrection = transcriptSection?.entries.find(
      (entry) => entry.id === `current_session_stream:${assistantEntry.id}`,
    );
    const staleSemantic = semanticSection?.entries.find(
      (entry) => entry.id === `semantic_node:${staleNode.id}`,
    );
    const rendered = renderEvidenceLedger(ledger) ?? "";
    const compactPlannerLedger = renderCompactPlannerLedger(ledger) ?? "";
    const transcriptHeader = "## 2. Current-Session Transcript";
    const semanticHeader = "## 10. Semantic Graph";
    const transcriptStart = rendered.indexOf(transcriptHeader);
    const semanticStart = rendered.indexOf(semanticHeader);
    expect(transcriptStart).toBeGreaterThanOrEqual(0);
    expect(semanticStart).toBeGreaterThanOrEqual(0);

    const transcriptEnd = rendered.indexOf("\n## ", transcriptStart + transcriptHeader.length);
    const semanticEnd = rendered.indexOf("\n## ", semanticStart + semanticHeader.length);
    const renderedTranscriptSection = rendered.slice(
      transcriptStart,
      transcriptEnd === -1 ? undefined : transcriptEnd,
    );
    const renderedSemanticSection = rendered.slice(
      semanticStart,
      semanticEnd === -1 ? undefined : semanticEnd,
    );

    expect(transcriptCorrection?.text).toContain("3 nights in San Sebastian");
    expect(assistantCorrection?.text).toContain("3 nights");
    expect(staleSemantic?.text).toContain("4 nights in San Sebastian");
    expect(staleSemantic?.text).toContain("[status=superseded");
    expect(staleSemantic?.state).toBe("superseded:proposition");
    expect(staleSemantic?.state_metadata).toMatchObject({
      status: "superseded",
      superseded_at: NOW_MS,
    });
    expect(staleSemantic?.text).not.toContain(staleNode.corrected_by);
    expect(transcriptCorrection?.trust_rank ?? 0).toBeGreaterThan(staleSemantic?.trust_rank ?? 0);
    expect(renderedTranscriptSection).toContain(correctionText);
    expect(renderedSemanticSection).toContain("Plan: 4 nights in San Sebastian");
    expect(semanticStart).toBeGreaterThan(transcriptStart);
    expect(compactPlannerLedger).toContain("3 nights in San Sebastian");
  });

  it("labels retrieved assistant self-report raw stream evidence", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const assistantEntry = await writer.append({
      kind: "agent_msg",
      content: "The gap feels like a discontinuity with a remembered edge.",
      persistence_class: "assistant_self_report",
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "What did you say earlier?",
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 1,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [
        {
          id: "raw-self-report",
          source: "raw_stream",
          text: String(assistantEntry.content),
          provenance: {
            streamIds: [assistantEntry.id],
          },
          recallIntentId: "intent-self-report",
          matchedTerms: [],
          score: 0.9,
          scoreBreakdown: {},
        },
      ],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const rawEntry = ledger.sections
      .find((section) => section.id === "retrieved_raw_stream_evidence")
      ?.entries.find((entry) => entry.id === "retrieved_stream:raw-self-report");
    const transcriptEntry = ledger.sections
      .find((section) => section.id === "current_session_transcript")
      ?.entries.find((entry) => entry.id === `current_session_stream:${assistantEntry.id}`);

    expect(ledger.transcriptIncluded).toBe(true);
    expect(ledger.transcriptCompacted).toBe(true);
    expect(transcriptEntry).toMatchObject({
      source_type: "current_session_stream",
      actor: "assistant",
      persistence_class: "assistant_self_report",
    });
    expect(rawEntry).toBeUndefined();

    expect(userEntry.kind).toBe("user_msg");
  });

  it("labels raw stream evidence scope only when every provenance handle resolves consistently", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const currentEntry = await writer.append({
      kind: "user_msg",
      content: "Current-session raw evidence.",
    });
    const priorEntry: StreamEntry = {
      id: createStreamEntryId(),
      timestamp: NOW_MS - 60_000,
      kind: "user_msg",
      content: "Prior-session raw evidence.",
      turn_status: "active",
      session_id: createSessionId(),
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    };
    const unresolvedEntryId = createStreamEntryId();
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(currentEntry.content),
      currentUserEntry: currentEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [
        {
          id: "raw-all-current",
          source: "raw_stream",
          text: "all current",
          provenance: {
            streamIds: [currentEntry.id],
          },
          recallIntentId: "intent-current",
          matchedTerms: [],
          score: 0.9,
          scoreBreakdown: {},
        },
        {
          id: "raw-all-prior",
          source: "raw_stream",
          text: "all prior",
          provenance: {
            streamIds: [priorEntry.id],
          },
          recallIntentId: "intent-prior",
          matchedTerms: [],
          score: 0.8,
          scoreBreakdown: {},
        },
        {
          id: "raw-mixed",
          source: "raw_stream",
          text: "mixed",
          provenance: {
            streamIds: [currentEntry.id, priorEntry.id],
          },
          recallIntentId: "intent-mixed",
          matchedTerms: [],
          score: 0.7,
          scoreBreakdown: {},
        },
        {
          id: "raw-unresolved",
          source: "raw_stream",
          text: "unresolved",
          provenance: {
            streamIds: [unresolvedEntryId],
          },
          recallIntentId: "intent-unresolved",
          matchedTerms: [],
          score: 0.6,
          scoreBreakdown: {},
        },
      ],
      retrievedEpisodes: [
        makeRetrievedEpisode({
          id: createEpisodeId(),
          narrative: "Prior source bridge.",
          sourceStreamIds: [priorEntry.id],
          citationChain: [priorEntry],
        }),
      ],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const allEntries = ledger.sections.flatMap((section) => section.entries);

    // Sprint 8d.6.3: raw-all-current points at currentEntry.id which is
    // already in the current_session_transcript section, so it is deduped.
    // Its scope info still appears -- on the transcript entry itself.
    expect(
      allEntries.find((entry) => entry.id === "retrieved_stream:raw-all-current"),
    ).toBeUndefined();
    expect(
      allEntries.find((entry) => entry.id === `current_session_stream:${currentEntry.id}`),
    ).toMatchObject({
      source_type: "current_session_stream",
      session_scope: "current_session",
      stream_index: 0,
    });
    expect(allEntries.find((entry) => entry.id === "retrieved_stream:raw-all-prior")).toMatchObject(
      {
        source_type: "prior_session_stream",
        session_scope: "prior_session",
      },
    );
    expect(allEntries.find((entry) => entry.id === "retrieved_stream:raw-mixed")).toMatchObject({
      source_type: "system_metadata",
      session_scope: "global",
    });
    expect(
      allEntries.find((entry) => entry.id === "retrieved_stream:raw-unresolved"),
    ).toMatchObject({
      source_type: "system_metadata",
      session_scope: "global",
    });
  });

  it("dedupes retrieved raw stream evidence against current_session_transcript by stream id", async () => {
    // Sprint 8d.6.3 regression: same underlying stream entry must not
    // appear twice (once in transcript, once in retrieved_raw_stream_evidence).
    // v36/v37 finalizer prompts had ~25k duplicate tokens from this class.
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Transcript-covered evidence.",
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [] },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [
        {
          id: "raw-duplicate",
          source: "raw_stream",
          text: String(userEntry.content),
          provenance: { streamIds: [userEntry.id] },
          recallIntentId: "intent-dup",
          matchedTerms: [],
          score: 0.9,
          scoreBreakdown: {},
        },
      ],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    const transcriptEntry = ledger.sections
      .find((section) => section.id === "current_session_transcript")
      ?.entries.find((entry) => entry.id === `current_session_stream:${userEntry.id}`);
    const retrievedEntry = ledger.sections
      .find((section) => section.id === "retrieved_raw_stream_evidence")
      ?.entries.find((entry) => entry.id === "retrieved_stream:raw-duplicate");

    expect(transcriptEntry).toBeDefined();
    expect(retrievedEntry).toBeUndefined();
  });

  it("excludes all catch-up batch source ids from transcript duplication", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const priorUserEntry = await writer.append({
      kind: "user_msg",
      content: "Earlier context remains transcript evidence.",
    });
    const assistantEntry = await writer.append({
      kind: "agent_msg",
      content: "Earlier assistant context remains transcript evidence.",
    });
    const batchEntryA = await writer.append({
      kind: "user_msg",
      content: "First pending batch message.",
    });
    const batchEntryB = await writer.append({
      kind: "user_msg",
      content: "Second pending batch message.",
    });
    const laterQueuedEntry = await writer.append({
      kind: "user_msg",
      content: "Later queued message remains separate transcript evidence.",
    });
    const batchMessages: readonly HydratedInboundMessage[] = [batchEntryA, batchEntryB].map(
      (entry) => ({
        id: entry.id,
        session_id: entry.session_id,
        entry_index: entry.entry_index ?? 0,
        timestamp: entry.timestamp,
        kind: "user_msg" as const,
        content: String(entry.content),
        sender_entity_id: entry.sender_entity_id,
      }),
    );
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [] },
      currentSessionTranscriptTokenBudget: 50_000,
    });
    const bridgeEpisodeId = createEpisodeId();

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: renderInboundBatch({ entries: batchMessages }),
      currentUserEntry: batchEntryA,
      currentUserEntries: [batchEntryA, batchEntryB],
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [
        makeRetrievedEpisode({
          id: bridgeEpisodeId,
          narrative: "A multi-stream episode bridges earlier and queued transcript evidence.",
          sourceStreamIds: [priorUserEntry.id, assistantEntry.id, laterQueuedEntry.id],
          citationChain: [priorUserEntry, assistantEntry, laterQueuedEntry],
        }),
      ],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    const currentUserMessage = ledger.sections.find(
      (section) => section.id === "current_user_message",
    )?.entries[0];
    const transcriptEntries =
      ledger.sections.find((section) => section.id === "current_session_transcript")?.entries ?? [];
    const compacted = compactEvidenceLedger(ledger, {
      targetTokens: 20_000,
      hardCapTokens: 40_000,
    });
    const compactedTranscriptEntries =
      compacted.ledger.sections.find((section) => section.id === "current_session_transcript")
        ?.entries ?? [];

    expect(currentUserMessage?.text).toContain("First pending batch message.");
    expect(currentUserMessage?.text).toContain("Second pending batch message.");
    expect(transcriptEntries.map((entry) => entry.id)).toEqual([
      `current_session_stream:${priorUserEntry.id}`,
      `current_session_stream:${assistantEntry.id}`,
      `current_session_stream:${laterQueuedEntry.id}`,
    ]);
    expect(
      transcriptEntries.some(
        (entry) =>
          entry.id === `current_session_stream:${batchEntryA.id}` ||
          entry.id === `current_session_stream:${batchEntryB.id}`,
      ),
    ).toBe(false);
    expect(compactedTranscriptEntries.map((entry) => entry.id)).toEqual([
      `current_session_stream:${priorUserEntry.id}`,
      `current_session_stream:${assistantEntry.id}`,
      `current_session_stream:${laterQueuedEntry.id}`,
    ]);
    expect(
      compactedTranscriptEntries.find(
        (entry) => entry.id === `current_session_stream:${assistantEntry.id}`,
      )?.text,
    ).toBe("Earlier assistant context remains transcript evidence.");
  });

  it("keeps retrieved raw stream evidence whose stream id is not in the transcript", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Transcript message.",
    });
    const otherStreamId = createStreamEntryId();
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [] },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [
        {
          id: "raw-other",
          source: "raw_stream",
          text: "non-transcript text",
          provenance: { streamIds: [otherStreamId] },
          recallIntentId: "intent-other",
          matchedTerms: [],
          score: 0.7,
          scoreBreakdown: {},
        },
      ],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    expect(
      ledger.sections
        .find((section) => section.id === "retrieved_raw_stream_evidence")
        ?.entries.find((entry) => entry.id === "retrieved_stream:raw-other"),
    ).toBeDefined();
  });
});
