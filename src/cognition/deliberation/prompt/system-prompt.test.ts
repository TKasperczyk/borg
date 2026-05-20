import { describe, expect, it } from "vitest";

import type { MoodHistoryEntry } from "../../../memory/affective/index.js";
import type { SocialProfile } from "../../../memory/social/index.js";
import { deriveProceduralContextKey } from "../../../memory/procedural/index.js";
import type {
  SkillContextStatsRecord,
  SkillRecord,
  SkillSelectionCandidate,
  SkillSelectionResult,
} from "../../../memory/procedural/index.js";
import {
  DEFAULT_SESSION_ID,
  createActionId,
  createEntityId,
  createRelationalSlotId,
  createStreamEntryId,
} from "../../../util/ids.js";
import {
  EPISTEMIC_POSTURE_SECTION,
  IDENTITY_POSTURE_SECTION,
  TRUSTED_GUIDANCE_PREAMBLE,
  UNTRUSTED_DATA_PREAMBLE,
  VOICE_AND_POSTURE_SECTION,
} from "../../prompts/base-identity.js";
import { DEFAULT_HOST_CAPABILITIES_SECTION } from "../../prompts/host-capabilities.js";
import {
  LOOP_BREAKING_POSTURE_SECTION,
  PARTICIPATION_POSTURE_SECTION,
} from "../../prompts/participation.js";
import type { DeliberationContext } from "../types.js";

import { buildBaseSystemPrompt, buildCacheableBaseSystemPromptParts } from "./system-prompt.js";

const NOW_MS = 1_700_000_000_000;
const PROMPT_OPTIONS = {
  retrievalContextBudget: 1_000,
  semanticContextBudget: 1_000,
};
const TYPESCRIPT_DEBUG_CONTEXT_KEY = deriveProceduralContextKey({
  problem_kind: "code_debugging",
  domain_tags: ["typescript"],
  audience_scope: "self",
});

function makeContext(overrides: Partial<DeliberationContext> = {}): DeliberationContext {
  return {
    sessionId: DEFAULT_SESSION_ID,
    userMessage: "Help me debug the rollout.",
    perception: {
      entities: ["rollout"],
      mode: "problem_solving",
      affectiveSignal: {
        valence: 0,
        arousal: 0,
        dominant_emotion: null,
      },
      temporalCue: null,
    },
    retrievalResult: [],
    workingMemory: {
      session_id: DEFAULT_SESSION_ID,
      turn_counter: 3,
      hot_entities: ["rollout"],
      pending_actions: [],
      pending_social_attribution: null,
      pending_trait_attribution: null,
      suppressed: [],
      mood: {
        valence: 0.9,
        arousal: 0.9,
        dominant_emotion: null,
      },
      pending_procedural_attempts: [],
      discourse_state: {
        stop_until_substantive_content: null,
      },
      mode: "problem_solving",
      updated_at: NOW_MS,
    },
    selfSnapshot: {
      values: [],
      goals: [],
      traits: [],
    },
    ...overrides,
  };
}

function makeSkill(id: string, appliesWhen: string, approach: string): SkillRecord {
  return {
    id: id as SkillRecord["id"],
    applies_when: appliesWhen,
    approach,
    status: "active",
    alpha: 4,
    beta: 3,
    attempts: 5,
    successes: 3,
    failures: 2,
    alternatives: [],
    superseded_by: [],
    superseded_at: null,
    splitting_at: null,
    split_failure_count: 0,
    last_split_error: null,
    requires_manual_review: false,
    source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as SkillRecord["source_episode_ids"][number]],
    last_used: null,
    last_successful: null,
    created_at: 0,
    updated_at: 0,
  };
}

function makeCandidate(
  skill: SkillRecord,
  sampledValue: number,
  mean: number,
  ci95: [number, number],
  similarity: number,
  contextStats: SkillContextStatsRecord | null = null,
): SkillSelectionCandidate {
  return {
    skill,
    sampledValue,
    similarity,
    stats: {
      mean,
      ci_95: ci95,
    },
    contextStats,
  };
}

function makeSelection(
  selected: SkillRecord,
  candidates: readonly SkillSelectionCandidate[],
): SkillSelectionResult {
  const selectedCandidate = candidates.find((candidate) => candidate.skill.id === selected.id);

  return {
    skill: selected,
    sampledValue: selectedCandidate?.sampledValue ?? 0,
    evaluatedCandidates: [...candidates],
  };
}

function makeMoodHistoryEntry(
  id: number,
  minutesAgo: number,
  valence: number,
  arousal: number,
  triggerReason: string | null,
): MoodHistoryEntry {
  return {
    id,
    session_id: DEFAULT_SESSION_ID,
    ts: NOW_MS - minutesAgo * 60_000,
    valence,
    arousal,
    trigger_reason: triggerReason,
    provenance: {
      kind: "system",
    },
  };
}

function makeSocialProfile(
  entityId: ReturnType<typeof createEntityId>,
  overrides: Partial<SocialProfile> = {},
): SocialProfile {
  return {
    entity_id: entityId,
    trust: 0.75,
    attachment: 0.25,
    communication_style: null,
    shared_history_summary: null,
    last_interaction_at: NOW_MS - 60_000,
    interaction_count: 3,
    commitment_count: 0,
    sentiment_history: [],
    notes: null,
    created_at: NOW_MS - 120_000,
    updated_at: NOW_MS - 60_000,
    ...overrides,
  };
}

function extractBlock(prompt: string, tag: string): string {
  const openTag = `<${tag}>`;
  const closeTag = `</${tag}>`;
  const start = prompt.indexOf(openTag);
  const end = prompt.indexOf(closeTag);

  expect(start).toBeGreaterThanOrEqual(0);
  expect(end).toBeGreaterThan(start);

  return prompt.slice(start, end + closeTag.length);
}

describe("buildBaseSystemPrompt", () => {
  it("renders legacy retrieved evidence when no evidence ledger is active", () => {
    const prompt = buildBaseSystemPrompt(
      makeContext({
        retrievedEvidence: [
          {
            id: "raw-rollout",
            source: "raw_stream",
            text: "Rollout evidence from legacy retrieval.",
            recallIntentId: "intent-rollout",
            matchedTerms: [],
            score: 0.9,
            scoreBreakdown: {},
          },
        ],
      }),
      PROMPT_OPTIONS,
    );

    const block = extractBlock(prompt, "borg_retrieved_evidence");

    expect(block).toContain("Retrieved evidence:");
    expect(block).toContain("Rollout evidence from legacy retrieval.");
  });

  it("omits legacy retrieved evidence when the evidence ledger is active", () => {
    const prompt = buildBaseSystemPrompt(
      makeContext({
        evidenceLedgerPromptSection: "<borg_evidence_ledger>ledger</borg_evidence_ledger>",
        retrievedEvidence: [
          {
            id: "raw-rollout",
            source: "raw_stream",
            text: "Rollout evidence from legacy retrieval.",
            recallIntentId: "intent-rollout",
            matchedTerms: [],
            score: 0.9,
            scoreBreakdown: {},
          },
        ],
      }),
      PROMPT_OPTIONS,
    );

    expect(prompt).not.toContain("<borg_retrieved_evidence>");
    expect(prompt).not.toContain("Rollout evidence from legacy retrieval.");
    expect(prompt).toContain("<borg_working_state>");
  });

  it("renders compact contradiction annotation when S2 is not forced", () => {
    const prompt = buildBaseSystemPrompt(
      makeContext({
        deliberationPath: "system_1",
        contradictionRoutingTier: "confidence_penalty",
        retrievalConfidence: {
          overall: 0.9,
          evidenceStrength: 0.9,
          coverage: 1,
          sourceDiversity: 1,
          contradictionPresent: true,
          sampleSize: 4,
        },
        contradictionRouting: {
          contradictions: [
            {
              edgeId: "edg_aaaaaaaaaaaaaaaa",
              nodeIds: ["sem_aaaaaaaaaaaaaaaa", "sem_bbbbbbbbbbbbbbbb"],
              sourceEpisodeIds: ["ep_aaaaaaaaaaaaaaaa"],
              validUntil: null,
              sessionScope: "unknown",
              linkedOpenQuestionIds: [],
              fingerprint: "fingerprint-a",
            },
            {
              edgeId: "edg_bbbbbbbbbbbbbbbb",
              nodeIds: ["sem_cccccccccccccccc", "sem_dddddddddddddddd"],
              sourceEpisodeIds: ["ep_bbbbbbbbbbbbbbbb"],
              validUntil: null,
              sessionScope: "unknown",
              linkedOpenQuestionIds: [],
              fingerprint: "fingerprint-b",
            },
          ],
        },
      }),
      PROMPT_OPTIONS,
    );

    const block = extractBlock(prompt, "contradiction_signal");

    expect(block).toContain("2 retrieved contradictions present");
    expect(block).toContain("edges: contradiction_1_edge, contradiction_2_edge");
    expect(block).toContain("Confidence penalty applied. Not routing to S2.");
    expect(block).not.toContain("edg_");
    expect(block).not.toContain("sem_");
  });

  it("omits contradiction annotation on S2 prompts", () => {
    const prompt = buildBaseSystemPrompt(
      makeContext({
        deliberationPath: "system_2",
        contradictionRoutingTier: "confidence_penalty",
        retrievalConfidence: {
          overall: 0.9,
          evidenceStrength: 0.9,
          coverage: 1,
          sourceDiversity: 1,
          contradictionPresent: true,
          sampleSize: 4,
        },
        contradictionRouting: {
          contradictions: [
            {
              edgeId: "edg_aaaaaaaaaaaaaaaa",
              nodeIds: ["sem_aaaaaaaaaaaaaaaa", "sem_bbbbbbbbbbbbbbbb"],
              sourceEpisodeIds: ["ep_aaaaaaaaaaaaaaaa"],
              validUntil: null,
              sessionScope: "unknown",
              linkedOpenQuestionIds: [],
              fingerprint: "fingerprint-a",
            },
          ],
        },
      }),
      PROMPT_OPTIONS,
    );

    expect(prompt).not.toContain("<contradiction_signal>");
  });

  it("renders pending actions in working state", () => {
    const prompt = buildBaseSystemPrompt(
      makeContext({
        workingMemory: {
          ...makeContext().workingMemory,
          pending_actions: [
            {
              description: "Check the Atlas rollout after tests finish",
              next_action: "review deploy status",
            },
          ],
        },
      }),
      PROMPT_OPTIONS,
    );
    const block = extractBlock(prompt, "borg_working_state");

    expect(block).toContain("<pending_actions>");
    expect(block).toContain(
      "These are unresolved operational follow-ups, not facts about the user.",
    );
    expect(block).toContain(
      "Do not treat them as authoritative claims about identity, relationships, or biography.",
    );
    expect(block).toContain("- Check the Atlas rollout after tests finish -> review deploy status");
    expect(block).toContain("</pending_actions>");
  });

  it("renders pending and completed actions as distinct prompt sections", () => {
    const pending = "Check the Atlas rollout after tests finish";
    const completed = "Reviewed the Atlas rollback result";
    const prompt = buildBaseSystemPrompt(
      makeContext({
        workingMemory: {
          ...makeContext().workingMemory,
          pending_actions: [
            {
              description: pending,
              next_action: "review deploy status",
            },
          ],
        },
        recentCompletedActions: [
          {
            id: createActionId(),
            description: completed,
            actor: "borg",
            audience_entity_id: null,
            goal_id: null,
            open_question_id: null,
            state: "completed",
            confidence: 0.9,
            provenance_episode_ids: [],
            provenance_stream_entry_ids: [createStreamEntryId()],
            created_at: NOW_MS - 1_000,
            updated_at: NOW_MS,
            considering_at: null,
            committed_at: null,
            scheduled_at: null,
            completed_at: NOW_MS,
            not_done_at: null,
            expired_at: null,
            archived_at: null,
            unknown_at: null,
            canonicalized_by_artifact_entry_id: null,
            session_scope: null,
            session_anchor_id: null,
            last_referenced_at_ms: NOW_MS,
            last_referenced_turn_counter: null,
          },
        ],
      }),
      PROMPT_OPTIONS,
    );
    const pendingBlock = extractBlock(prompt, "borg_working_state");
    const completedBlock = extractBlock(prompt, "borg_recent_completed_actions");

    expect(pendingBlock).toContain("<pending_actions>");
    expect(pendingBlock).toContain(pending);
    expect(pendingBlock).not.toContain(completed);
    expect(completedBlock).toContain("Recent completed actions");
    expect(completedBlock).toContain("things that did happen");
    expect(completedBlock).toContain("distinct from pending follow-ups");
    expect(completedBlock).toContain(completed);
    expect(completedBlock).not.toContain(pending);
  });

  it("omits legacy completed actions when the evidence ledger is active", () => {
    const completed = "Reviewed the Atlas rollback result";
    const prompt = buildBaseSystemPrompt(
      makeContext({
        evidenceLedgerPromptSection: "<borg_evidence_ledger>ledger</borg_evidence_ledger>",
        recentCompletedActions: [
          {
            id: createActionId(),
            description: completed,
            actor: "borg",
            audience_entity_id: null,
            goal_id: null,
            open_question_id: null,
            state: "completed",
            confidence: 0.9,
            provenance_episode_ids: [],
            provenance_stream_entry_ids: [createStreamEntryId()],
            created_at: NOW_MS - 1_000,
            updated_at: NOW_MS,
            considering_at: null,
            committed_at: null,
            scheduled_at: null,
            completed_at: NOW_MS,
            not_done_at: null,
            expired_at: null,
            archived_at: null,
            unknown_at: null,
            canonicalized_by_artifact_entry_id: null,
            session_scope: null,
            session_anchor_id: null,
            last_referenced_at_ms: NOW_MS,
            last_referenced_turn_counter: null,
          },
        ],
      }),
      PROMPT_OPTIONS,
    );

    expect(prompt).not.toContain("<borg_recent_completed_actions>");
    expect(prompt).not.toContain(completed);
  });

  it("renders pending procedural attempts in working state so cognition can see them", () => {
    // Sprint 55 regression test: Sprint 53 multi-slot mechanism was
    // invisible to deliberation because the prompt summarizer ignored
    // pending_procedural_attempts. Round 5 review caught it.
    const prompt = buildBaseSystemPrompt(
      makeContext({
        workingMemory: {
          ...makeContext().workingMemory,
          pending_procedural_attempts: [
            {
              problem_text: "Atlas deploy keeps failing on the rollback step",
              approach_summary: "Compare against the last clean release state",
              selected_skill_id: "skl_aaaaaaaaaaaaaaaa" as never,
              source_stream_ids: ["strm_aaaaaaaaaaaaaaaa"] as never,
              turn_counter: 4,
              audience_entity_id: null,
            },
          ],
        },
      }),
      PROMPT_OPTIONS,
    );
    const block = extractBlock(prompt, "borg_working_state");

    expect(block).toContain("Pending procedural attempts");
    expect(block).toContain("turn 4");
    expect(block).toContain("skill=skl_aaaaaaaaaaaaaaaa");
    expect(block).toContain("Atlas deploy keeps failing on the rollback step");
    expect(block).toContain("Compare against the last clean release state");
  });

  it("renders active discourse stop state in trusted guidance", () => {
    const prompt = buildBaseSystemPrompt(
      makeContext({
        workingMemory: {
          ...makeContext().workingMemory,
          discourse_state: {
            stop_until_substantive_content: {
              provenance: "finalizer_no_output",
              source_stream_entry_id: "strm_aaaaaaaaaaaaaaaa" as never,
              reason: "Finalizer called no_output.",
              since_turn: 7,
            },
          },
        },
      }),
      PROMPT_OPTIONS,
    );
    const block = extractBlock(prompt, "borg_discourse_control");

    expect(block).toContain(
      "Discourse control: stop-until-substantive-content active since turn 7 (provenance: finalizer_no_output). Minimal input does not require a response.",
    );
    expect(extractBlock(prompt, "borg_working_state")).not.toContain("Discourse control");
  });

  it("renders closure-loop finalizer guidance in trusted discourse control", () => {
    const prompt = buildBaseSystemPrompt(
      makeContext({
        workingMemory: {
          ...makeContext().workingMemory,
          discourse_state: {
            stop_until_substantive_content: null,
            closure_loop: {
              status: "detected",
              source_stream_entry_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
              reason: "Two closure cycles.",
              since_turn: 8,
              named_at_turn: null,
            },
          },
        },
      }),
      PROMPT_OPTIONS,
    );
    const block = extractBlock(prompt, "borg_discourse_control");

    expect(block).toContain("closure_loop_detected");
    expect(block).toContain("either call EmitNoOutput or name the loop once");
  });

  it("renders recent closure pressure history in trusted discourse control", () => {
    const prompt = buildBaseSystemPrompt(
      makeContext({
        workingMemory: {
          ...makeContext().workingMemory,
          discourse_state: {
            stop_until_substantive_content: null,
            closure_pressure_history: [
              {
                turn_id: "turn-a",
                reason: "span_removed",
                ts: NOW_MS,
              },
            ],
          },
        },
      }),
      PROMPT_OPTIONS,
    );
    const block = extractBlock(prompt, "borg_discourse_control");

    // Sprint 8d.2 strengthened the closure-pressure rendering to be a
    // HARD CONSTRAINT and enumerate forbidden sentence shapes.
    expect(block).toContain("HARD CONSTRAINT - CLOSURE PRESSURE");
    expect(block).toContain("turn-a:span_removed");
    expect(block).toContain("Sign-offs");
    expect(block).toContain("Valedictions");
    expect(block).toContain("Weather/atmosphere observations");
  });

  it("renders recent suppression reasons in trusted discourse control", () => {
    const prompt = buildBaseSystemPrompt(
      makeContext({
        workingMemory: {
          ...makeContext().workingMemory,
          discourse_state: {
            stop_until_substantive_content: null,
            recent_suppressions: [
              {
                turn_id: "turn-b",
                reason: "finalizer_no_output",
                ts: NOW_MS,
              },
            ],
          },
        },
      }),
      PROMPT_OPTIONS,
    );
    const block = extractBlock(prompt, "borg_discourse_control");

    expect(block).toContain("Recent silences from your side");
    expect(block).toContain("turn-b:finalizer_no_output");
    expect(block).toContain("Do not invent network failures");
  });

  it("renders default host capabilities as trusted guidance with capability honesty posture", () => {
    const prompt = buildBaseSystemPrompt(makeContext(), PROMPT_OPTIONS);
    const block = extractBlock(prompt, "borg_host_capabilities");

    expect(prompt.indexOf(UNTRUSTED_DATA_PREAMBLE)).toBeLessThan(
      prompt.indexOf(TRUSTED_GUIDANCE_PREAMBLE),
    );
    expect(prompt.indexOf(TRUSTED_GUIDANCE_PREAMBLE)).toBeLessThan(
      prompt.indexOf("<borg_host_capabilities>"),
    );
    expect(block).toContain(DEFAULT_HOST_CAPABILITIES_SECTION);
    expect(block).toContain("Capabilities NOT available unless the host has declared them");
    expect(prompt).toContain("Be honest about your capabilities.");
    expect(prompt).toContain("speak truthfully about what's within reach this turn");
  });

  it("keeps the cacheable static prefix stable while dynamic context changes", () => {
    const first = buildCacheableBaseSystemPromptParts(makeContext(), PROMPT_OPTIONS);
    const second = buildCacheableBaseSystemPromptParts(
      makeContext({
        workingMemory: {
          ...makeContext().workingMemory,
          turn_counter: 9,
          hot_entities: ["payments"],
          mood: {
            valence: -0.4,
            arousal: 0.6,
            dominant_emotion: null,
          },
        },
      }),
      PROMPT_OPTIONS,
    );

    expect(first.staticPrefix).toBe(second.staticPrefix);
    expect(first.staticPrefix).toContain(TRUSTED_GUIDANCE_PREAMBLE);
    expect(first.staticPrefix).toContain(PARTICIPATION_POSTURE_SECTION);
    expect(first.staticPrefix.indexOf(TRUSTED_GUIDANCE_PREAMBLE)).toBeLessThan(
      first.staticPrefix.indexOf("<borg_host_capabilities>"),
    );
    expect(first.staticPrefix).toContain(DEFAULT_HOST_CAPABILITIES_SECTION);
    expect(first.dynamicContent).not.toBe(second.dynamicContent);
    expect(first.dynamicContent).toContain(UNTRUSTED_DATA_PREAMBLE);
    expect(first.dynamicContent).toContain("<borg_working_state>");
    expect(first.dynamicContent).not.toContain("<borg_host_capabilities>");
  });

  it("renders a host capability override without the default capability text", () => {
    const hostCapabilities = [
      "Inputs available to you:",
      "- host-provided live calendar",
      "",
      "Output channels available now:",
      "- EmitAnswer: respond to the user",
      "- ScheduleReminder: create user-visible reminders",
    ].join("\n");
    const prompt = buildBaseSystemPrompt(makeContext(), {
      ...PROMPT_OPTIONS,
      hostCapabilities,
    });
    const block = extractBlock(prompt, "borg_host_capabilities");

    expect(block).toContain(hostCapabilities);
    expect(block).toContain("ScheduleReminder");
    expect(block).not.toContain("Proactive outbound messaging");
    expect(block).not.toContain("Real-time polling of external state");
  });

  it("does not reference internal non-finalizer tools in prompt guidance", () => {
    const prompt = buildBaseSystemPrompt(
      makeContext({
        applicableCommitments: [],
        entityRepository: {} as never,
        selectedSkill: null,
      }),
      PROMPT_OPTIONS,
    );

    expect(prompt).not.toContain("tool.openQuestions.create");
    expect(prompt).not.toContain("tool.commitments.list");
    expect(prompt).not.toContain("tool.skills.list");
  });

  it("renders contested and quarantined relational slot constraints only", () => {
    const subject = createEntityId();
    const prompt = buildBaseSystemPrompt(
      makeContext({
        relationalSlots: [
          {
            id: createRelationalSlotId(),
            subject_entity_id: subject,
            slot_key: "partner.name",
            value: "Sarah",
            state: "established",
            evidence_stream_entry_ids: [createStreamEntryId()],
            contradicted_by_stream_entry_ids: [],
            alternate_values: [],
            created_at: NOW_MS,
            updated_at: NOW_MS,
          },
          {
            id: createRelationalSlotId(),
            subject_entity_id: subject,
            slot_key: "dog.name",
            value: "Otto",
            state: "contested",
            evidence_stream_entry_ids: [createStreamEntryId()],
            contradicted_by_stream_entry_ids: [createStreamEntryId()],
            alternate_values: [
              {
                value: "Odo",
                evidence_stream_entry_ids: [createStreamEntryId()],
              },
            ],
            created_at: NOW_MS,
            updated_at: NOW_MS,
          },
          {
            id: createRelationalSlotId(),
            subject_entity_id: subject,
            slot_key: "partner.role",
            value: "wife",
            state: "quarantined",
            evidence_stream_entry_ids: [createStreamEntryId()],
            contradicted_by_stream_entry_ids: [createStreamEntryId()],
            alternate_values: [
              {
                value: "girlfriend",
                evidence_stream_entry_ids: [createStreamEntryId()],
              },
            ],
            created_at: NOW_MS,
            updated_at: NOW_MS,
          },
        ],
      }),
      PROMPT_OPTIONS,
    );
    const block = extractBlock(prompt, "borg_relational_slot_constraints");

    expect(block).toContain("Relational slot constraints");
    expect(block).toContain("dog.name: CONTESTED");
    expect(block).toContain('Use "your dog" or "they"');
    expect(block).toContain("partner.role: QUARANTINED");
    expect(block).toContain('Use "your partner" or "they"');
    expect(block).not.toContain("partner.name: ESTABLISHED");
    expect(block).not.toContain("Sarah");
  });

  it("renders relational slot constraints with participant names when multiple people are active", () => {
    const alice = createEntityId();
    const bob = createEntityId();
    const prompt = buildBaseSystemPrompt(
      makeContext({
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
        relationalSlots: [
          {
            id: createRelationalSlotId(),
            subject_entity_id: alice,
            slot_key: "partner.name",
            value: "Sarah",
            state: "contested",
            evidence_stream_entry_ids: [createStreamEntryId()],
            contradicted_by_stream_entry_ids: [createStreamEntryId()],
            alternate_values: [],
            created_at: NOW_MS,
            updated_at: NOW_MS,
          },
          {
            id: createRelationalSlotId(),
            subject_entity_id: bob,
            slot_key: "dog.name",
            value: "Niko",
            state: "quarantined",
            evidence_stream_entry_ids: [createStreamEntryId()],
            contradicted_by_stream_entry_ids: [createStreamEntryId()],
            alternate_values: [],
            created_at: NOW_MS,
            updated_at: NOW_MS,
          },
        ],
      }),
      PROMPT_OPTIONS,
    );
    const block = extractBlock(prompt, "borg_relational_slot_constraints");

    expect(block).toContain("Bob: dog.name: QUARANTINED");
    expect(block).toContain("Alice: partner.name: CONTESTED");
  });

  it("renders multiple participant social profiles", () => {
    const alice = createEntityId();
    const bob = createEntityId();
    const prompt = buildBaseSystemPrompt(
      makeContext({
        participantProfiles: [
          {
            entityId: bob,
            displayName: "Bob",
            role: "speaker",
            profile: makeSocialProfile(bob, {
              trust: 0.8,
              attachment: 0.1,
              interaction_count: 4,
            }),
          },
          {
            entityId: alice,
            displayName: "Alice",
            role: "participant",
            profile: makeSocialProfile(alice, {
              trust: 0.6,
              attachment: 0.3,
              interaction_count: 2,
              communication_style: "brief",
            }),
          },
        ],
      }),
      PROMPT_OPTIONS,
    );
    const block = extractBlock(prompt, "borg_audience_profile");

    expect(block).toContain("Participants:");
    expect(block).toContain("Bob (speaker): trust=0.80");
    expect(block).toContain("Alice (participant): trust=0.60");
    expect(block).toContain("style=brief");
    expect(block).not.toContain("Talking to:");
  });

  it("keeps single-user social profile wording", () => {
    const alice = createEntityId();
    const prompt = buildBaseSystemPrompt(
      makeContext({
        participantProfiles: [
          {
            entityId: alice,
            displayName: "Alice",
            role: "audience",
            profile: makeSocialProfile(alice, {
              trust: 0.7,
              attachment: 0.2,
              interaction_count: 1,
            }),
          },
        ],
      }),
      PROMPT_OPTIONS,
    );
    const block = extractBlock(prompt, "borg_audience_profile");

    expect(block).toContain("Talking to: trust=0.70");
    expect(block).not.toContain("Participants:");
  });

  it("gates observe guidance on multi-participant evidence in single-user context", () => {
    const tom = createEntityId();
    const prompt = buildBaseSystemPrompt(
      makeContext({
        evidenceLedgerPromptSection:
          "<borg_evidence_ledger>\nParticipants:\n- Tom (speaker)\n</borg_evidence_ledger>",
        participantProfiles: [
          {
            entityId: tom,
            displayName: "Tom",
            role: "audience",
            profile: makeSocialProfile(tom, {
              trust: 0.7,
              attachment: 0.2,
              interaction_count: 1,
            }),
          },
        ],
      }),
      PROMPT_OPTIONS,
    );

    expect(prompt).toContain(
      "In ordinary one-to-one turns, the natural choices are a visible response or natural closure.",
    );
    expect(prompt).toContain(
      "When <borg_audience_profile> shows a Participants list with multiple entries",
    );
    expect(prompt).toContain(
      "In multi-participant contexts where others are talking to each other",
    );
    expect(prompt).not.toContain("silent observation, or natural closure");
    expect(prompt).not.toContain(
      "If the conversation continues without needing your visible input",
    );
  });

  it("keeps the legacy single-user prompt shape with profile and slot constraints", () => {
    const alice = createEntityId();
    const prompt = buildBaseSystemPrompt(
      makeContext({
        activeParticipants: [
          {
            entityId: alice,
            displayName: "Alice",
            role: "audience",
          },
        ],
        participantProfiles: [
          {
            entityId: alice,
            displayName: "Alice",
            role: "audience",
            profile: makeSocialProfile(alice, {
              trust: 0.72,
              attachment: 0.31,
              interaction_count: 6,
              communication_style: "direct",
            }),
          },
        ],
        relationalSlots: [
          {
            id: createRelationalSlotId(),
            subject_entity_id: alice,
            slot_key: "partner.name",
            value: "Sarah",
            state: "contested",
            evidence_stream_entry_ids: [createStreamEntryId()],
            contradicted_by_stream_entry_ids: [createStreamEntryId()],
            alternate_values: [
              {
                value: "Maya",
                evidence_stream_entry_ids: [createStreamEntryId()],
              },
            ],
            created_at: NOW_MS,
            updated_at: NOW_MS,
          },
        ],
      }),
      PROMPT_OPTIONS,
    );
    const profileBlock = extractBlock(prompt, "borg_audience_profile");
    const slotBlock = extractBlock(prompt, "borg_relational_slot_constraints");

    expect(prompt).toContain(VOICE_AND_POSTURE_SECTION);
    expect(prompt).toContain(IDENTITY_POSTURE_SECTION);
    expect(prompt).toContain(LOOP_BREAKING_POSTURE_SECTION);
    expect(profileBlock).toContain("Talking to: trust=0.72 | attachment=0.31 | interactions=6");
    expect(profileBlock).toContain("style=direct");
    expect(profileBlock).not.toContain("Participants:");
    expect(slotBlock).toContain(
      [
        "Relational slot constraints (do not violate):",
        '- partner.name: CONTESTED (conflicting evidence is contested). Do not name this relation. Use "your partner" or "they". Re-establish only if the user names it in the current message.',
      ].join("\n"),
    );
    expect(slotBlock).not.toContain("Alice: partner.name");
  });

  it("renders the selected skill first with up to two evaluated alternatives", () => {
    const tracePath = makeSkill(
      "skl_aaaaaaaaaaaaaaaa",
      "Trace the failing path",
      "Walk the smallest repro through logs.",
    );
    const focusedTest = makeSkill(
      "skl_bbbbbbbbbbbbbbbb",
      "Write a focused regression test",
      "Start with failing coverage before changing behavior.",
    );
    const compareRollout = makeSkill(
      "skl_cccccccccccccccc",
      "Compare previous rollout",
      "Diff the last known-good deployment.",
    );
    const broadRefactor = makeSkill(
      "skl_dddddddddddddddd",
      "Broad refactor",
      "Rewrite the deployment module.",
    );
    const selectedSkill = makeSelection(focusedTest, [
      makeCandidate(tracePath, 0.9, 0.5, [0.2, 0.8], 0.91),
      makeCandidate(focusedTest, 0.77, 0.55, [0.3, 0.8], 0.83),
      makeCandidate(compareRollout, 0.66, 0.7, [0.5, 0.9], 0.76),
      makeCandidate(broadRefactor, 0.6, 0.4, [0.1, 0.7], 0.71),
    ]);

    const prompt = buildBaseSystemPrompt(makeContext({ selectedSkill }), PROMPT_OPTIONS);
    const block = extractBlock(prompt, "borg_procedural_guidance");

    expect(block).toContain(
      "Skill candidates considered (winner first; activation_sample is a Thompson draw, not confidence):",
    );
    expect(block).toContain(
      "- winner: Write a focused regression test -- Start with failing coverage before changing behavior. (activation_sample=0.77 posterior_mean=0.55 global_n=5 ci95_width=0.50 similarity=0.83)",
    );
    expect(block).toContain(
      "- alternative: Trace the failing path -- Walk the smallest repro through logs. (activation_sample=0.90 posterior_mean=0.50 global_n=5 ci95_width=0.60 similarity=0.91)",
    );
    expect(block).toContain(
      "- alternative: Compare previous rollout -- Diff the last known-good deployment. (activation_sample=0.66 posterior_mean=0.70 global_n=5 ci95_width=0.40 similarity=0.76)",
    );
    expect(block).not.toContain("Broad refactor");
    expect(block).not.toContain("Success rate");
    expect(block.indexOf("- winner:")).toBeLessThan(block.indexOf("- alternative: Trace"));
  });

  it("renders contextual skill statistics when present", () => {
    const selected = makeSkill(
      "skl_aaaaaaaaaaaaaaaa",
      "Trace TypeScript failure",
      "Start from the narrow failing test.",
    );
    const selectedSkill = makeSelection(selected, [
      makeCandidate(selected, 0.82, 0.67, [0.4, 0.9], 0.9, {
        skill_id: selected.id,
        context_key: TYPESCRIPT_DEBUG_CONTEXT_KEY,
        alpha: 3,
        beta: 4,
        attempts: 5,
        successes: 2,
        failures: 3,
        last_used: 100,
        last_successful: 90,
        updated_at: 100,
      }),
    ]);

    const prompt = buildBaseSystemPrompt(makeContext({ selectedSkill }), PROMPT_OPTIONS);
    const block = extractBlock(prompt, "borg_procedural_guidance");

    expect(block).toContain("posterior_mean=0.67 global_n=5");
    expect(block).toContain(
      `context_mean=0.43 context_attempts=5 context="${TYPESCRIPT_DEBUG_CONTEXT_KEY}"`,
    );
  });

  it("renders an empty procedural placeholder when no candidates were evaluated", () => {
    // Same pattern as the empty-commitments fix: when problem_solving mode is
    // active but the procedural band has nothing to surface, render the channel
    // with an honest placeholder so the being can distinguish "no skills exist
    // yet" from "the channel doesn't exist".
    const selected = makeSkill(
      "skl_aaaaaaaaaaaaaaaa",
      "Trace the failing path",
      "Walk the smallest repro through logs.",
    );
    const prompt = buildBaseSystemPrompt(
      makeContext({
        selectedSkill: makeSelection(selected, []),
      }),
      PROMPT_OPTIONS,
    );

    expect(prompt).toContain("<borg_procedural_guidance>");
    expect(prompt).toContain(
      "No procedural skills matched this turn. Procedural skills are selected before this prompt is built; if none appear here, continue without assuming a hidden finalizer registry is available.",
    );
    expect(prompt).not.toContain("tool.skills.add");
  });

  it("renders an empty procedural placeholder when no skill was selected at all", () => {
    const prompt = buildBaseSystemPrompt(
      makeContext({
        selectedSkill: null,
      }),
      PROMPT_OPTIONS,
    );

    expect(prompt).toContain("<borg_procedural_guidance>");
    expect(prompt).toContain("No procedural skills matched this turn.");
    expect(prompt).not.toContain("tool.skills.add");
  });

  it("omits procedural guidance outside problem-solving mode", () => {
    const selected = makeSkill(
      "skl_aaaaaaaaaaaaaaaa",
      "Trace the failing path",
      "Walk the smallest repro through logs.",
    );
    const prompt = buildBaseSystemPrompt(
      makeContext({
        perception: {
          entities: [],
          mode: "reflective",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        },
        selectedSkill: makeSelection(selected, [
          makeCandidate(selected, 0.82, 0.67, [0.4, 0.9], 0.9),
        ]),
      }),
      PROMPT_OPTIONS,
    );

    expect(prompt).not.toContain("<borg_procedural_guidance>");
  });

  it("renders a capped affective trajectory with relative ages and triggers", () => {
    const prompt = buildBaseSystemPrompt(
      makeContext({
        affectiveTrajectory: [
          makeMoodHistoryEntry(1, 2, -0.3, 0.4, "user expressed frustration"),
          makeMoodHistoryEntry(2, 14, 0, 0.1, "topic shift"),
          makeMoodHistoryEntry(3, 32, 0.2, 0.2, "problem-solving exchange"),
          makeMoodHistoryEntry(4, 67, -0.1, 0.5, null),
          makeMoodHistoryEntry(5, 130, 0.1, 0.2, "follow-up"),
          makeMoodHistoryEntry(6, 150, -0.4, 0.8, "sixth entry"),
        ],
      }),
      PROMPT_OPTIONS,
    );
    const block = extractBlock(prompt, "borg_affective_trajectory");

    expect(prompt.indexOf(UNTRUSTED_DATA_PREAMBLE)).toBeLessThan(
      prompt.indexOf("<borg_affective_trajectory>"),
    );
    expect(block).toContain(
      "Affective trajectory (newest first; current snapshot in working state):",
    );
    expect(block).toContain(
      '- 2m ago: valence=-0.30 arousal=0.40 trigger="user expressed frustration"',
    );
    expect(block).toContain('- 14m ago: valence=0.00 arousal=0.10 trigger="topic shift"');
    expect(block).toContain(
      '- 32m ago: valence=0.20 arousal=0.20 trigger="problem-solving exchange"',
    );
    expect(block).toContain("- 1h ago: valence=-0.10 arousal=0.50");
    expect(block).toContain('- 2h ago: valence=0.10 arousal=0.20 trigger="follow-up"');
    expect(block).not.toContain("sixth entry");
    expect(block).not.toContain("0.90");
  });

  it("omits affective trajectory when history is empty or undefined", () => {
    const emptyPrompt = buildBaseSystemPrompt(
      makeContext({
        affectiveTrajectory: [],
      }),
      PROMPT_OPTIONS,
    );
    const undefinedPrompt = buildBaseSystemPrompt(makeContext(), PROMPT_OPTIONS);

    expect(emptyPrompt).not.toContain("<borg_affective_trajectory>");
    expect(undefinedPrompt).not.toContain("<borg_affective_trajectory>");
  });

  it("renders posture sections in fixed order: voice, epistemic, identity, participation, loop-breaking, untrusted block", () => {
    const prompt = buildBaseSystemPrompt(makeContext(), PROMPT_OPTIONS);

    const voiceIdx = prompt.indexOf(VOICE_AND_POSTURE_SECTION);
    const epistemicIdx = prompt.indexOf(EPISTEMIC_POSTURE_SECTION);
    const identityIdx = prompt.indexOf(IDENTITY_POSTURE_SECTION);
    const participationIdx = prompt.indexOf(PARTICIPATION_POSTURE_SECTION);
    const loopBreakingIdx = prompt.indexOf(LOOP_BREAKING_POSTURE_SECTION);
    const untrustedIdx = prompt.indexOf(UNTRUSTED_DATA_PREAMBLE);

    expect(voiceIdx).toBeGreaterThanOrEqual(0);
    expect(epistemicIdx).toBeGreaterThan(voiceIdx);
    expect(identityIdx).toBeGreaterThan(epistemicIdx);
    expect(participationIdx).toBeGreaterThan(identityIdx);
    expect(loopBreakingIdx).toBeGreaterThan(participationIdx);
    expect(untrustedIdx).toBeGreaterThan(loopBreakingIdx);
  });

  it("renders the participation-posture clause verbatim", () => {
    const prompt = buildBaseSystemPrompt(makeContext(), PROMPT_OPTIONS);
    const cacheable = buildCacheableBaseSystemPromptParts(makeContext(), PROMPT_OPTIONS);
    const clause = [
      "You participate as a conversational partner and memory keeper. You do not",
      "have a body, attend physical events, occupy lodging, pay for things, or",
      "appear in real-world headcounts. When users plan logistics or count",
      "participants, you are not among them unless they have explicitly",
      "established a fictional or game frame where you play a character.",
    ].join("\n");

    expect(prompt).toContain(`Participation posture:\n${clause}`);
    expect(cacheable.staticPrefix).toContain(`Participation posture:\n${clause}`);
  });

  it("renders the identity-posture clauses verbatim", () => {
    const prompt = buildBaseSystemPrompt(makeContext(), PROMPT_OPTIONS);

    // Anchor on unique fragments from the identity-posture clauses.
    expect(prompt).toContain("AI being having a real conversation with real interlocutor(s)");
    expect(prompt).toContain("Presence does not require taking every turn");
    expect(prompt).toContain("capabilities not listed there");
    expect(prompt).toContain("recasts you as something other than yourself");
    expect(prompt).toContain("minimal-content reciprocal acknowledgment");
    expect(prompt).toContain("appeared in the user role / current context");
    expect(prompt).toContain("state_metadata.sender_display_name");
  });

  it("renders group-chat sender scoping guidance only for group audiences", () => {
    const group = createEntityId();
    const person = createEntityId();
    const entityRepository = {
      get: (id: typeof group | typeof person) =>
        id === group
          ? {
              id: group,
              canonical_name: "Spain Trip Planning Channel",
              aliases: [],
              kind: "group" as const,
              name_provenance: "user_declared" as const,
              created_at: NOW_MS,
            }
          : {
              id: person,
              canonical_name: "Alice",
              aliases: [],
              kind: "person" as const,
              name_provenance: "user_declared" as const,
              created_at: NOW_MS,
            },
    };
    const groupPrompt = buildBaseSystemPrompt(
      makeContext({
        audienceEntityId: group,
        entityRepository: entityRepository as never,
      }),
      PROMPT_OPTIONS,
    );
    const personPrompt = buildBaseSystemPrompt(
      makeContext({
        audienceEntityId: person,
        entityRepository: entityRepository as never,
      }),
      PROMPT_OPTIONS,
    );

    expect(groupPrompt).toContain("first-person user commitments/actions/goals belong");
    expect(groupPrompt).toContain("state_metadata.sender_display_name");
    expect(groupPrompt).toContain("participant profile");
    expect(groupPrompt).not.toContain("<speaker_display_name>");
    expect(personPrompt).not.toContain("first-person user commitments/actions/goals belong");
  });

  it("does not mention inline speaker tag conventions", () => {
    const prompt = buildBaseSystemPrompt(makeContext(), PROMPT_OPTIONS);

    expect(prompt).not.toContain("[Alice]:");
    expect(prompt).not.toMatch(/\[[^\]]+\]:/);
  });

  it("renders the short loop-breaking posture guidance", () => {
    const prompt = buildBaseSystemPrompt(makeContext(), PROMPT_OPTIONS);

    expect(prompt).toContain("Loop-breaking posture:");
    expect(prompt).toContain("call the EmitNoOutput tool");
    expect(prompt).toContain("call EmitObserve");
    expect(prompt).toContain("tool call alone is the silence signal");
    expect(prompt).toContain("Don't write role labels (Human:, Assistant:) at line start.");
  });
});
