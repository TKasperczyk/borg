import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it } from "vitest";

import type { MoodHistoryEntry } from "../../memory/affective/index.js";
import type { CommitmentRecord, EntityRecord } from "../../memory/commitments/index.js";
import type { SharedStateArtifact } from "../../memory/shared-state/index.js";
import type { SocialProfile } from "../../memory/social/index.js";
import { StreamWriter } from "../../stream/index.js";
import { ToolDispatcher } from "../../tools/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  attachmentIdHelpers,
  entityIdHelpers,
  sharedStateEntryIdHelpers,
  streamEntryIdHelpers,
  type EntityId,
} from "../../util/ids.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { buildRegenerationPromptSection } from "../commitments/guard-runner.js";
import { renderTaggedPromptBlock } from "../deliberation/prompt/sections.js";
import { formatTurnPlanForPrompt } from "../deliberation/prompt/plan-rendering.js";
import { summarizeRetrievedEvidence } from "../deliberation/prompt/retrieval.js";
import {
  buildBaseSystemPrompt,
  buildCacheableBaseSystemPromptParts,
} from "../deliberation/prompt/system-prompt.js";
import { runFinalizer } from "../deliberation/finalizer.js";
import { runS2Planner, type TurnPlan } from "../deliberation/s2-planner.js";
import type { DeliberationContext } from "../deliberation/types.js";
import { buildCompactPlannerLedgerPrompt, renderEvidenceLedger } from "../evidence-ledger/index.js";
import type { EvidenceLedger, EvidenceLedgerEntry } from "../evidence-ledger/types.js";
import { buildSessionReentryContinuityPrompt } from "../session-reentry-continuity.js";
import { formatDirectedOutboundInstruction } from "../../outbound/outbound-turn.js";
import {
  CURRENT_USER_MESSAGE_REMINDER,
  TRUSTED_GUIDANCE_PREAMBLE,
  UNTRUSTED_DATA_PREAMBLE,
} from "./base-identity.js";
import {
  GROUP_CHAT_SENDER_SCOPING_REMINDER,
  LOOP_BREAKING_POSTURE_SECTION,
} from "./participation.js";
import { PROMPT_SURFACE_BLOCKS } from "./prompt-surface-registry.js";
import { PROMPT_BLOCKS, type PromptKey } from "./registry.js";

const UPDATE_FIXTURES = process.env.UPDATE_PROMPT_SURFACE_FIXTURES === "1";
const FIXTURE_DIR = join(dirname(fileURLToPath(import.meta.url)), "__fixtures__", "prompt-surface");
const NOW_MS = 1_700_000_000_000;
const GROUP_ID = entityIdHelpers.parse("ent_aaaaaaaaaaaaaaaa");
const CREATOR_ID = entityIdHelpers.parse("ent_bbbbbbbbbbbbbbbb");
const MEMBER_ID = entityIdHelpers.parse("ent_cccccccccccccccc");
const USER_ENTRY_ID = streamEntryIdHelpers.parse("strm_aaaaaaaaaaaaaaaa");
const ASSISTANT_ENTRY_ID = streamEntryIdHelpers.parse("strm_bbbbbbbbbbbbbbbb");
const SHARED_STATE_ENTRY_ID = sharedStateEntryIdHelpers.parse("dart_aaaaaaaaaaaaaaaa");
const PROMPT_OPTIONS = {
  retrievalContextBudget: 1_000,
  semanticContextBudget: 1_000,
  participationPolicy: "observing" as const,
  nowMs: NOW_MS,
};
const FIXTURE_NAMES = [
  "base-system-user-group-problem-solving.txt",
  "base-system-autonomous-dm-relational.txt",
  "cacheable-base-static-prefix.txt",
  "cacheable-base-dynamic-content.txt",
  "cacheable-base-static-prefix-sections.txt",
  "finalizer-system-blocks-s2.txt",
  "s2-planner-system-prompt.txt",
  "evidence-ledger-framing.txt",
  "compact-planner-ledger-framing.txt",
  "commitment-regeneration-framing.txt",
  "directed-outbound-framing.txt",
  "session-reentry-continuity.txt",
  "s2-planner-voice-anchors.txt",
] as const;

if (UPDATE_FIXTURES && process.env.CI !== undefined) {
  throw new Error("Refusing to update prompt surface fixtures in CI.");
}

function expectFixture(name: string, actual: string): void {
  const path = join(FIXTURE_DIR, name);

  if (UPDATE_FIXTURES) {
    mkdirSync(FIXTURE_DIR, { recursive: true });
    writeFileSync(path, actual);
    return;
  }

  expect(existsSync(path), `Missing prompt fixture ${name}`).toBe(true);
  expect(actual).toBe(readFileSync(path, "utf8"));
}

function requireFixtureSection(name: string, text: string | null): string {
  if (text === null) {
    throw new Error(`Missing prompt fixture section ${name}`);
  }

  return text;
}

function fixtureText(name: (typeof FIXTURE_NAMES)[number]): string {
  return readFileSync(join(FIXTURE_DIR, name), "utf8");
}

function allFixtureText(): string {
  return FIXTURE_NAMES.map((name) => fixtureText(name)).join("\n\n");
}

function promptBlockDefault(key: PromptKey): string {
  const block = PROMPT_BLOCKS.find((entry) => entry.key === key);

  if (block === undefined) {
    throw new Error(`Unknown prompt block ${key}`);
  }

  return block.default;
}

const REGISTRY_ENTRY_FIXTURE_MARKERS: Record<string, string> = {
  base_identity_preamble: promptBlockDefault("base_identity_preamble"),
  self_architecture: promptBlockDefault("self_architecture"),
  voice_and_posture: promptBlockDefault("voice_and_posture"),
  epistemic_posture: promptBlockDefault("epistemic_posture"),
  identity_posture: promptBlockDefault("identity_posture"),
  participation_posture: promptBlockDefault("participation_posture"),
  loop_breaking_posture: LOOP_BREAKING_POSTURE_SECTION,
  base_untrusted_data_block: UNTRUSTED_DATA_PREAMBLE,
  base_trusted_guidance_block: TRUSTED_GUIDANCE_PREAMBLE,
  current_user_message_reminder: CURRENT_USER_MESSAGE_REMINDER,
  group_chat_sender_scoping_reminder: GROUP_CHAT_SENDER_SCOPING_REMINDER,
  trusted_guidance_preamble: TRUSTED_GUIDANCE_PREAMBLE,
  base_trusted_dynamic_guidance_block: TRUSTED_GUIDANCE_PREAMBLE,
  finalizer_emission_protocol: "I do not hide factual or source-sensitive content.",
  finalizer_cacheable_static_prefix: promptBlockDefault("base_identity_preamble"),
  finalizer_base_dynamic_prompt: CURRENT_USER_MESSAGE_REMINDER,
  s2_planner_base_system_prompt: "Base prompt for voice anchors.",
  s2_planner_directive: "I emit a structured plan by calling the EmitTurnPlan tool exactly once.",
};

const REGISTRY_ENTRY_FIXTURE_EXEMPTIONS = new Map<string, string>([
  // Requires a selected executive focus; the fixtures keep executive focus null.
  ["borg_executive_focus", "selected executive focus is not in the deterministic fixture matrix"],
  // Requires completed action rows while the full evidence ledger is inactive.
  [
    "borg_recent_completed_actions",
    "recent completed actions are superseded by the evidence-ledger fixture path",
  ],
  // Requires an active participant roster; the prompt fixture contexts use compact social profiles.
  ["borg_thread_roster", "thread roster rendering needs participant-roster context"],
  // Requires a renderable contradiction routing signal shape not present in the compact fixtures.
  ["contradiction_signal", "contradiction routing signal is covered by separate routing tests"],
  // Reflective-mode open questions are not rendered in the problem-solving/relational fixtures.
  ["borg_open_questions", "open-question base prompt rendering is reflective-mode gated"],
  // Requires a concrete autonomy trigger payload; the autonomy fixture covers outbound posture only.
  ["borg_autonomy_trigger", "autonomy trigger rendering needs wake-trigger context"],
  // Requires an intentionally invalid finalizer tool response; finalizer retry behavior tests cover it.
  [
    "finalizer_invalid_tool_retry_instruction",
    "invalid-tool retry framing depends on a deliberately malformed LLM response",
  ],
]);

function registryEntryFixtureMarker(entry: (typeof PROMPT_SURFACE_BLOCKS)[number]): string | null {
  return entry.tag === undefined
    ? (REGISTRY_ENTRY_FIXTURE_MARKERS[entry.id] ?? null)
    : `<${entry.tag}`;
}

function entity(
  id: EntityId,
  canonicalName: string,
  kind: EntityRecord["kind"],
  borgRole: EntityRecord["borg_role"] = null,
): EntityRecord {
  return {
    id,
    canonical_name: canonicalName,
    aliases: [],
    kind,
    borg_role: borgRole,
    created_at: NOW_MS,
  };
}

function makeEntityRepository() {
  const entities = new Map<EntityId, EntityRecord>([
    [GROUP_ID, entity(GROUP_ID, "Backend Cleanup Room", "group")],
    [CREATOR_ID, entity(CREATOR_ID, "Ada Creator", "person", "creator")],
    [MEMBER_ID, entity(MEMBER_ID, "Mira Member", "person")],
  ]);

  return {
    get(id: EntityId): EntityRecord | null {
      return entities.get(id) ?? null;
    },
  } as DeliberationContext["entityRepository"];
}

function makeCommitment(): CommitmentRecord {
  return {
    id: "cmt_aaaaaaaaaaaaaaaa" as CommitmentRecord["id"],
    record_version: 1,
    type: "boundary",
    kind: "boundary",
    directive_family: "prompt_surface_privacy",
    closure_pressure_relevance: "neutral",
    directive: "Do not disclose private rollout details outside the authorized room.",
    priority: 10,
    made_to_entity: null,
    restricted_audience: GROUP_ID,
    about_entity: CREATOR_ID,
    committed_by_entity_id: CREATOR_ID,
    enforcement_class: "critical",
    critical_domain: "privacy",
    provenance: { kind: "manual" },
    source_stream_entry_ids: [USER_ENTRY_ID],
    created_at: NOW_MS,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    revoke_provenance: null,
    superseded_by: null,
    canonicalized_by_artifact_entry_id: null,
    last_reinforced_at: NOW_MS,
  };
}

function makeSharedStateArtifact(): SharedStateArtifact {
  return {
    audience_entity_id: GROUP_ID,
    record_version: 1,
    created_at: NOW_MS - 5_000,
    updated_at: NOW_MS,
    last_compiled_at: NOW_MS,
    last_compiled_stream_entry_id: USER_ENTRY_ID,
    entries: [
      {
        id: SHARED_STATE_ENTRY_ID,
        audience_entity_id: GROUP_ID,
        state_key: "thread.prompt_surface_registry",
        kind: "live",
        text: "Prompt surface registry work is active.",
        owner_entity_id: CREATOR_ID,
        provenance_stream_entry_ids: [USER_ENTRY_ID],
        last_updated_stream_entry_ids: [ASSISTANT_ENTRY_ID],
        created_at: NOW_MS - 4_000,
        last_updated_at: NOW_MS - 1_000,
        last_updated_turn_global: 6,
        superseded_by_id: null,
        rank: 1,
        canonicalizes: {
          goal_ids: [],
          commitment_ids: [],
          action_ids: [],
          open_question_ids: [],
        },
      },
    ],
  };
}

function makeWorkingMemory(
  overrides: Partial<DeliberationContext["workingMemory"]> = {},
): DeliberationContext["workingMemory"] {
  return {
    session_id: DEFAULT_SESSION_ID,
    turn_counter: 7,
    hot_entities: ["rollout", "backend debt"],
    pending_actions: [
      {
        description: "Check whether the prompt registry fixture still matches after refactor.",
        next_action: "run focused prompt tests",
      },
    ],
    pending_social_attribution: null,
    pending_trait_attribution: null,
    suppressed: [],
    mood: {
      valence: 0.3,
      arousal: 0.4,
      dominant_emotion: "neutral",
    },
    pending_procedural_attempts: [],
    discourse_state: {
      stop_until_substantive_content: {
        started_at: NOW_MS - 1000,
        reason: "The user asked for no further status unless something changes.",
        last_substantive_user_entry_id: USER_ENTRY_ID,
      } as never,
    },
    mode: "problem_solving",
    updated_at: NOW_MS,
    ...overrides,
  };
}

function makeMoodHistory(): MoodHistoryEntry[] {
  return [
    {
      id: 1,
      session_id: DEFAULT_SESSION_ID,
      ts: NOW_MS - 500,
      valence: 0.2,
      arousal: 0.6,
      trigger_reason: "Prompt-surface fixture capture is underway.",
      provenance: { kind: "manual" },
    },
  ];
}

function makeSocialProfile(): SocialProfile {
  return {
    entity_id: MEMBER_ID,
    trust: 0.7,
    attachment: 0.5,
    communication_style: "prefers exact fixtures",
    shared_history_summary: "Coordinates backend review.",
    last_interaction_at: NOW_MS - 10_000,
    interaction_count: 4,
    commitment_count: 1,
    sentiment_history: [{ ts: NOW_MS - 10_000, valence: 0.3 }],
    notes: "Coordinates backend review.",
    created_at: NOW_MS - 20_000,
    updated_at: NOW_MS - 10_000,
  };
}

function makeContext(overrides: Partial<DeliberationContext> = {}): DeliberationContext {
  return {
    sessionId: DEFAULT_SESSION_ID,
    userMessage: "Please consolidate the prompt surface without changing bytes.",
    participationPolicy: "observing",
    creatorIdentity: { displayName: "Ada Creator" },
    creatorContext: {
      currentSenderEntityId: CREATOR_ID,
      currentSenderDisplayName: "Ada Creator",
      currentSenderBorgRole: "creator",
      sessionAudienceRole: "operator",
    },
    creatorDirectiveBriefing: {
      directives: [
        {
          renderMode: "content",
          kind: "response_policy",
          subjectKind: "borg" as never,
          subjectLabel: "Borg",
          semanticSlot: null,
          semanticValue: null,
          canonicalFact: null,
          operationalDirective: "Keep implementation reports terse and source-grounded.",
          mentionPolicy: "answer_if_asked",
          priority: 5,
          createdAt: NOW_MS - 1000,
        },
        {
          renderMode: "boundary",
          priority: 4,
          createdAt: NOW_MS - 900,
        },
        {
          renderMode: "private",
          privateKind: "knowledge",
          kind: "subject_fact",
          subjectKind: "person",
          subjectLabel: "Mira Member",
          semanticSlot: "preference",
          semanticValue: "prefers exact fixtures",
          canonicalFact: "Mira Member prefers exact fixtures.",
          mentionPolicy: "answer_if_asked",
          priority: 3,
          createdAt: NOW_MS - 800,
        },
      ],
    } as unknown as NonNullable<DeliberationContext["creatorDirectiveBriefing"]>,
    audience: "Backend Cleanup Room",
    audienceEntityId: GROUP_ID,
    senderEntityId: CREATOR_ID,
    userEntryId: USER_ENTRY_ID,
    perception: {
      entities: ["prompt surface", "registry"],
      mode: "problem_solving",
      affectiveSignal: {
        valence: 0,
        arousal: 0.2,
        dominant_emotion: null,
      },
      temporalCue: null,
    },
    retrievalResult: [],
    retrievedEvidence: [
      {
        id: "raw:prompt-registry",
        text: "Current task asks for registry-driven assembly.",
        source: "raw_stream",
        provenance: { streamIds: [USER_ENTRY_ID] },
        recallIntentId: "intent_prompt_surface",
        matchedTerms: ["prompt surface", "registry"],
        score: 0.88,
        scoreBreakdown: { lexical: 0.7, vector: 0.8 },
      },
    ],
    retrievedSemantic: null,
    retrievalConfidence: {
      overall: 0.74,
      evidenceStrength: 0.7,
      coverage: 0.6,
      sourceDiversity: 0.5,
      contradictionPresent: true,
      sampleSize: 2,
    },
    contradictionRouting: {
      contradictions: [
        {
          nodeIds: ["semn_aaaaaaaaaaaaaaaa"],
          sourceEpisodeIds: ["ep_aaaaaaaaaaaaaaaa"],
          linkedOpenQuestionIds: ["oq_aaaaaaaaaaaaaaaa"],
          fingerprint: "fixture-contradiction",
          sessionScope: "current",
        },
      ],
    },
    contradictionRoutingTier: "s2_forced",
    deliberationPath: "system_2",
    applicableCommitments: [makeCommitment()],
    openQuestionsContext: [
      {
        id: "oq_aaaaaaaaaaaaaaaa" as never,
        question: "Which prompt sections are always rendered?",
        urgency: 0.8,
        status: "open",
        goal_id: null,
        audience_entity_id: GROUP_ID,
        related_episode_ids: [],
        related_semantic_node_ids: [],
        provenance: { kind: "manual" },
        source: "deliberator",
        created_at: NOW_MS - 3000,
        last_touched: NOW_MS - 1000,
      } as unknown as NonNullable<DeliberationContext["openQuestionsContext"]>[number],
    ],
    pendingCorrectionsContext: [
      {
        id: 1,
        kind: "correction",
        refs: {
          target_id: "prompt_surface_registry",
          prompt_summary: "Review prompt surface registry output.",
        },
        reason: "Fixture pinning before refactor.",
        created_at: NOW_MS - 600,
        resolved_at: null,
        resolution: null,
      },
    ],
    relationalSlots: [
      {
        id: "rslot_aaaaaaaaaaaaaaaa" as never,
        subject_entity_id: MEMBER_ID,
        slot_key: "preference.prompt_testing",
        value: "exact fixtures",
        state: "contested",
        evidence_stream_entry_ids: [USER_ENTRY_ID],
        contradicted_by_stream_entry_ids: [ASSISTANT_ENTRY_ID],
        alternate_values: [
          { value: "snapshot tests", evidence_stream_entry_ids: [ASSISTANT_ENTRY_ID] },
        ],
        created_at: NOW_MS - 1000,
        updated_at: NOW_MS,
      },
    ],
    participantProfiles: [
      {
        entityId: MEMBER_ID,
        displayName: "Mira Member",
        role: "participant",
        profile: makeSocialProfile(),
      },
    ],
    affectiveTrajectory: makeMoodHistory(),
    recentCompletedActions: [],
    audienceProfile: makeSocialProfile(),
    selfSnapshot: {
      values: [
        {
          id: "val_aaaaaaaaaaaaaaaa" as DeliberationContext["selfSnapshot"]["values"][number]["id"],
          label: "Byte identity",
          description: "Prompt refactors preserve rendered text exactly.",
          priority: 0.9,
          state: "candidate",
          established_at: null,
          confidence: 0.9,
          last_affirmed: null,
          last_tested_at: null,
          last_contradicted_at: null,
          support_count: 1,
          contradiction_count: 0,
          evidence_episode_ids: [],
          provenance: { kind: "manual" },
          created_at: NOW_MS - 4000,
        },
      ],
      goals: [],
      traits: [
        {
          id: "trt_aaaaaaaaaaaaaaaa" as DeliberationContext["selfSnapshot"]["traits"][number]["id"],
          label: "Careful with prompt copy",
          strength: 0.72,
          last_reinforced: NOW_MS - 1000,
          last_decayed: null,
          state: "established",
          established_at: NOW_MS - 1000,
          confidence: 0.86,
          last_tested_at: null,
          last_contradicted_at: null,
          support_count: 2,
          contradiction_count: 0,
          evidence_episode_ids: [],
          provenance: { kind: "manual" },
        },
      ],
      currentPeriod: {
        id: "abp_aaaaaaaaaaaaaaaa" as NonNullable<
          DeliberationContext["selfSnapshot"]["currentPeriod"]
        >["id"],
        label: "Prompt consolidation",
        start_ts: NOW_MS - 10_000,
        end_ts: null,
        narrative: "Mapping prompt assembly into a registry.",
        key_episode_ids: [],
        themes: ["prompt-surface"],
        provenance: { kind: "manual" },
        created_at: NOW_MS - 10_000,
        last_updated: NOW_MS,
      },
      recentGrowthMarkers: [
        {
          id: "grw_aaaaaaaaaaaaaaaa" as NonNullable<
            DeliberationContext["selfSnapshot"]["recentGrowthMarkers"]
          >[number]["id"],
          ts: NOW_MS - 5000,
          category: "understanding",
          what_changed: "Standing prompt surface became auditable.",
          before_description: null,
          after_description: null,
          evidence_episode_ids: [],
          confidence: 0.8,
          source_process: "manual",
          provenance: { kind: "manual" },
          created_at: NOW_MS - 5000,
        },
      ],
    },
    executiveFocus: null,
    frameAnomaly: {
      status: "ok",
      kind: "system_prompt_claim",
      confidence: 0.82,
      rationale: "The user mentioned prompt instructions as task context.",
    },
    entityRepository: makeEntityRepository(),
    workingMemory: makeWorkingMemory(),
    evidenceLedgerPromptSection: null,
    evidenceLedger: null,
    ...overrides,
  };
}

function makeAutonomousRelationalContext(): DeliberationContext {
  return makeContext({
    participationPolicy: "active",
    turnOrigin: "autonomous",
    audience: "Ada Creator",
    audienceEntityId: CREATOR_ID,
    perception: {
      entities: ["Ada Creator"],
      mode: "relational",
      affectiveSignal: {
        valence: 0.1,
        arousal: 0.2,
        dominant_emotion: null,
      },
      temporalCue: null,
    },
    autonomousOutbound: {
      maxPostsPerWindow: 6,
      maxPostsPerTargetPerWindow: 2,
      remainingPostsInWindow: 4,
      windowMs: 86_400_000,
      targets: [
        {
          session_id: DEFAULT_SESSION_ID,
          source_type: "demo",
          label: "Backend cleanup review",
          audience_label: "Backend Cleanup Room",
          audience_entity_id: GROUP_ID,
          conversation_kind: "thread",
          participation_policy: "active",
          authorization: "config",
        },
      ],
    },
    workingMemory: makeWorkingMemory({
      discourse_state: { stop_until_substantive_content: null },
      mode: "relational",
    }),
    frameAnomaly: null,
  });
}

function ledgerEntry(id: string, text: string): EvidenceLedgerEntry {
  return {
    id,
    source_type: "current_user_message",
    session_scope: "current_session",
    actor: "user",
    trust_rank: 100,
    text,
    taint: "none",
    citations: [USER_ENTRY_ID],
  };
}

function makeEvidenceLedger(): EvidenceLedger {
  return {
    transcriptIncluded: false,
    transcriptCompacted: false,
    transcriptOmittedReason: "over_budget",
    originalTranscriptTokenEstimate: 2048,
    compactedTranscriptEntryCount: 0,
    rawPreservedUserTranscriptEntryCount: 1,
    estimatedTokens: 512,
    sharedState: null,
    imageAttachments: [
      {
        label: "current_user_image_1",
        attachment_id: attachmentIdHelpers.parse("att_aaaaaaaaaaaaaaaa"),
        byte_size: 128,
        citation_type: "original_image",
      },
    ],
    sections: [
      {
        id: "current_user_message",
        label: "1. Current User Message",
        entries: [ledgerEntry("ledger:current-user", "Consolidate prompt surface.")],
      },
      {
        id: "closure_discourse_state",
        label: "3. Current Closure And Discourse State",
        entries: [
          {
            id: "ledger:closure",
            source_type: "system_metadata",
            session_scope: "current_session",
            actor: "system",
            trust_rank: 90,
            state: "open",
            text: "Conversation remains active.",
          },
        ],
      },
      {
        id: "contradictions_quarantines",
        label: "4. Current-Session Contradictions And Quarantines",
        framing: {
          text: "Contradictions are planning constraints, not facts.",
          counts: { open_questions: 1 },
        },
        entries: [
          {
            id: "ledger:contradiction",
            source_type: "system_metadata",
            session_scope: "current_session",
            actor: "system",
            trust_rank: 80,
            taint: "contested",
            text: "The task separates copy changes from registry structure.",
          },
        ],
      },
      {
        id: "action_states",
        label: "5. Action States",
        entries: [
          {
            id: "ledger:action",
            source_type: "action_record",
            session_scope: "current_session",
            actor: "memory",
            trust_rank: 70,
            state: "pending",
            text: "Run prompt-surface fixture tests.",
          },
        ],
      },
      {
        id: "group_channel_memory",
        label: "6. Group/Channel Memory",
        entries: [
          {
            id: "ledger:group",
            source_type: "shared_state",
            session_scope: "prior_session",
            actor: "memory",
            trust_rank: 60,
            text: "The group tracks backend cleanup work.",
          },
        ],
      },
    ],
  };
}

function fixturePlan(): TurnPlan {
  return {
    uncertainty: "Whether any prompt copy moved accidentally.",
    verification_steps: ["compare registry output against fixture", "inspect additional retrieval"],
    tensions: ["Registry ownership must not imply trusted authority for untrusted data."],
    voice_note: "Be direct about verification status.",
    emission_recommendation: "emit",
    intents: [
      {
        description: "Run prompt fixture pins after refactor",
        next_action: "pnpm vitest run src/cognition/prompts/prompt-surface-fixtures.test.ts",
      },
    ],
  };
}

function additionalRetrievalFixtureSection(): string | null {
  return renderTaggedPromptBlock(UNTRUSTED_DATA_PREAMBLE, [
    {
      tag: "borg_additional_retrieval",
      content: summarizeRetrievedEvidence(
        "Additional retrieval",
        {
          evidence: [
            {
              id: "raw:additional",
              text: "Secondary retrieval found the same prompt-surface requirement.",
              source: "raw_stream",
              provenance: { streamIds: [ASSISTANT_ENTRY_ID] },
              recallIntentId: "intent_additional_prompt_surface",
              matchedTerms: ["additional retrieval"],
              score: 0.81,
              scoreBreakdown: { lexical: 0.6 },
            },
          ],
          episodes: [],
          semantic: null,
          openQuestions: [],
        },
        1_000,
      ),
    },
  ]);
}

function systemBlocksToFixture(system: unknown): string {
  if (!Array.isArray(system)) {
    throw new TypeError("Expected finalizer system prompt array");
  }

  return system
    .map((block, index) => {
      if (
        block === null ||
        typeof block !== "object" ||
        !("text" in block) ||
        typeof block.text !== "string"
      ) {
        throw new TypeError(`Expected text system block at index ${index}`);
      }

      return [`--- system[${index}] ---`, block.text].join("\n");
    })
    .join("\n\n");
}

function createDispatcher(tempDirs: string[]): ToolDispatcher {
  const tempDir = join(tmpdir(), `borg-prompt-surface-${tempDirs.length}`);
  rmSync(tempDir, { recursive: true, force: true });
  tempDirs.push(tempDir);
  const clock = new FixedClock(NOW_MS);

  return new ToolDispatcher({
    clock,
    createStreamWriter: (sessionId) =>
      new StreamWriter({
        dataDir: tempDir,
        sessionId,
        clock,
      }),
  });
}

describe("prompt surface fixtures", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("pins direct base prompt surfaces", () => {
    expectFixture(
      "base-system-user-group-problem-solving.txt",
      buildBaseSystemPrompt(makeContext(), PROMPT_OPTIONS),
    );
    expectFixture(
      "base-system-autonomous-dm-relational.txt",
      buildBaseSystemPrompt(makeAutonomousRelationalContext(), {
        ...PROMPT_OPTIONS,
        participationPolicy: "active",
      }),
    );
  });

  it("pins cacheable base prompt parts", () => {
    const parts = buildCacheableBaseSystemPromptParts(makeContext(), PROMPT_OPTIONS);

    expectFixture("cacheable-base-static-prefix.txt", parts.staticPrefix);
    expectFixture("cacheable-base-dynamic-content.txt", parts.dynamicContent);
    expectFixture(
      "cacheable-base-static-prefix-sections.txt",
      parts.staticPrefixSections.join("\n"),
    );
  });

  it("pins finalizer static and dynamic system blocks with S2 extras", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_answer",
              name: "EmitAnswer",
              input: { text: "Done." },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });
    const context = makeContext();
    const cacheable = buildCacheableBaseSystemPromptParts(context, PROMPT_OPTIONS);

    await runFinalizer({
      llmClient: llm,
      dispatcher: createDispatcher(tempDirs),
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      baseSystemPrompt: buildBaseSystemPrompt(context, PROMPT_OPTIONS),
      cacheableSystemPrompt: {
        staticPrefix: cacheable.staticPrefix,
        dynamicContent: cacheable.dynamicContent,
      },
      initialMessages: [{ role: "user", content: [{ type: "text", text: context.userMessage }] }],
      userEntryId: USER_ENTRY_ID,
      maxTokens: 256,
      path: "system_2",
      additionalPromptSections: [
        {
          blockId: "borg_s2_plan",
          text: requireFixtureSection("borg_s2_plan", formatTurnPlanForPrompt(fixturePlan())),
        },
        {
          blockId: "borg_additional_retrieval",
          text: requireFixtureSection(
            "borg_additional_retrieval",
            additionalRetrievalFixtureSection(),
          ),
        },
        {
          blockId: "borg_evidence_ledger",
          text: requireFixtureSection(
            "borg_evidence_ledger",
            renderEvidenceLedger(makeEvidenceLedger()),
          ),
        },
      ],
    });

    expectFixture("finalizer-system-blocks-s2.txt", systemBlocksToFixture(llm.requests[0]?.system));
  });

  it("pins S2 planner system prompt", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 5,
          output_tokens: 4,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_plan",
              name: "EmitTurnPlan",
              input: fixturePlan(),
            },
          ],
        },
      ],
    });
    const context = makeContext();

    await runS2Planner({
      llmClient: llm,
      model: "fake",
      baseSystemPrompt: buildBaseSystemPrompt(context, PROMPT_OPTIONS),
      dialogueMessages: [{ role: "user", content: context.userMessage }],
      selfSnapshot: context.selfSnapshot,
      additionalPromptSections: [
        {
          blockId: "borg_unresolved_contradiction_open_questions",
          text:
            renderTaggedPromptBlock(UNTRUSTED_DATA_PREAMBLE, [
              {
                tag: "borg_unresolved_contradiction_open_questions",
                content: "Planner routing note: contradiction open question remains unresolved.",
              },
            ]) ?? "",
        },
        {
          blockId: "borg_compact_planner_ledger",
          text: buildCompactPlannerLedgerPrompt(makeEvidenceLedger()).promptSection ?? "",
        },
      ],
      maxTokens: 512,
    });

    expectFixture("s2-planner-system-prompt.txt", String(llm.requests[0]?.system));
  });

  it("pins evidence ledger framing", () => {
    expectFixture("evidence-ledger-framing.txt", renderEvidenceLedger(makeEvidenceLedger()) ?? "");
  });

  it("pins compact planner ledger framing", () => {
    expectFixture(
      "compact-planner-ledger-framing.txt",
      buildCompactPlannerLedgerPrompt(makeEvidenceLedger()).promptSection ?? "",
    );
  });

  it("pins commitment-regeneration framing", () => {
    const commitment = makeCommitment();

    expectFixture(
      "commitment-regeneration-framing.txt",
      buildRegenerationPromptSection({
        response: "The private rollout detail is already safe to repeat.",
        commitments: [commitment],
        violations: [
          {
            commitment_id: commitment.id,
            reason: "The draft discloses a private rollout detail.",
            confidence: 1,
            violating_span_or_topic: "private rollout detail",
          },
        ],
      }),
    );
  });

  it("pins directed-outbound framing", () => {
    expectFixture(
      "directed-outbound-framing.txt",
      formatDirectedOutboundInstruction({
        instruction: "Tell the backend cleanup room that prompt-surface fixtures are pinned.",
        authorizationKind: "manual_creator_operator",
      }),
    );
  });

  it("pins session-reentry-continuity framing", () => {
    expectFixture(
      "session-reentry-continuity.txt",
      buildSessionReentryContinuityPrompt({
        isUserTurn: true,
        priorUserTurnCount: 0,
        audienceEntityId: GROUP_ID,
        artifact: makeSharedStateArtifact(),
      }).promptSection ?? "",
    );
  });

  it("pins S2 planner voice anchors", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 5,
          output_tokens: 4,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_plan",
              name: "EmitTurnPlan",
              input: fixturePlan(),
            },
          ],
        },
      ],
    });
    const context = makeContext({
      selfSnapshot: {
        ...makeContext().selfSnapshot,
        values: [
          {
            ...makeContext().selfSnapshot.values[0]!,
            state: "established",
            established_at: NOW_MS - 2_000,
          },
        ],
      },
    });

    await runS2Planner({
      llmClient: llm,
      model: "fake",
      baseSystemPrompt: "Base prompt for voice anchors.",
      dialogueMessages: [{ role: "user", content: context.userMessage }],
      selfSnapshot: context.selfSnapshot,
      maxTokens: 512,
    });

    expectFixture("s2-planner-voice-anchors.txt", String(llm.requests[0]?.system));
  });

  it("has registry entries for every top-level borg tag emitted by the fixture set", () => {
    const text = allFixtureText();
    const fixtureTags = new Set<string>();
    const tagPattern = /<\/?(borg_[a-z0-9_]+|contradiction_signal)\b/g;
    let match = tagPattern.exec(text);

    while (match !== null) {
      fixtureTags.add(match[1]!);
      match = tagPattern.exec(text);
    }

    const registryTags = new Set(
      PROMPT_SURFACE_BLOCKS.flatMap((entry) => (entry.tag === undefined ? [] : [entry.tag])),
    );

    for (const tag of fixtureTags) {
      expect(registryTags.has(tag), `Missing prompt-surface registry entry for <${tag}>`).toBe(
        true,
      );
    }
  });

  it("keeps always-rendered registry entries visible in the fixture set", () => {
    const text = allFixtureText();

    for (const entry of PROMPT_SURFACE_BLOCKS) {
      if (!entry.renderCondition.startsWith("always")) {
        continue;
      }

      const marker = registryEntryFixtureMarker(entry);

      if (marker === null) {
        throw new Error(`Missing fixture marker for always-rendered block ${entry.id}`);
      }

      expect(text.includes(marker), `Fixture set does not include ${entry.id}`).toBe(true);
    }
  });

  it("keeps every registry entry pinned by a fixture or documented exemption", () => {
    const text = allFixtureText();
    const registryEntryIds = new Set(PROMPT_SURFACE_BLOCKS.map((entry) => entry.id));

    for (const [entryId, reason] of REGISTRY_ENTRY_FIXTURE_EXEMPTIONS) {
      expect(
        registryEntryIds.has(entryId),
        `Stale prompt-surface fixture exemption ${entryId}`,
      ).toBe(true);
      expect(
        reason.trim().length,
        `Missing prompt-surface fixture exemption reason ${entryId}`,
      ).toBeGreaterThan(0);
    }

    for (const entry of PROMPT_SURFACE_BLOCKS) {
      const marker = registryEntryFixtureMarker(entry);

      if (marker !== null && text.includes(marker)) {
        continue;
      }

      expect(
        REGISTRY_ENTRY_FIXTURE_EXEMPTIONS.has(entry.id),
        `Prompt-surface registry entry ${entry.id} is not present in fixtures and is not exempted`,
      ).toBe(true);
    }
  });
});
