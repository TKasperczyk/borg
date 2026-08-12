import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";

import { type LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import {
  createCommitmentId,
  createEntityId,
  createRelationalSlotId,
  createStreamEntryId,
} from "../../util/ids.js";
import { EXTRACTOR_MAX_TOKENS_DEFAULT } from "../prompts/constants.js";
import { CORRECTIVE_PREFERENCE_SYSTEM_PROMPT } from "../prompts/corrective-preference.js";
import type { RelationshipClaim } from "../../memory/common/relationship-claims.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import {
  CorrectivePreferenceExtractor,
  CorrectivePreferenceExtractorDegradedError,
} from "./corrective-preference-extractor.js";

function correctivePreferenceResponse(input: {
  classification: "corrective_preference" | "retire_commitment" | "none";
  type?: "preference" | "rule" | "boundary" | null;
  kind?: "audience_rule" | "participant_preference" | "boundary" | "process_norm" | null;
  enforcement_class?: "critical" | "advisory" | null;
  critical_domain?:
    | "privacy"
    | "audience_scope"
    | "safety"
    | "explicit_no_disclosure"
    | "internal_tool_hygiene"
    | null;
  directive?: string | null;
  directive_family?: string | null;
  closure_pressure_relevance?: "no_closure" | "neutral" | "closure_seeking" | null;
  priority?: number | null;
  reason?: string;
  confidence?: number;
  retiresCommitmentId?: ReturnType<typeof createCommitmentId> | null;
  relationship_claims?: RelationshipClaim[];
  slot_negations?: unknown[];
}): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_corrective_preference",
        name: "EmitCorrectivePreference",
        input: {
          classification: input.classification,
          type: input.type ?? null,
          kind:
            input.kind ??
            (input.classification === "corrective_preference" ? "participant_preference" : null),
          enforcement_class:
            input.enforcement_class ??
            (input.classification === "corrective_preference" ? "advisory" : null),
          critical_domain: input.critical_domain ?? null,
          directive: input.directive ?? null,
          directive_family: input.directive_family ?? null,
          closure_pressure_relevance:
            input.closure_pressure_relevance ??
            (input.classification === "corrective_preference" ? "neutral" : null),
          priority: input.priority ?? null,
          reason: input.reason ?? "Classification reason.",
          confidence: input.confidence ?? 0.9,
          supersedes_commitment_id: null,
          retires_commitment_id: input.retiresCommitmentId ?? null,
          relationship_claims: input.relationship_claims ?? [],
          slot_negations: input.slot_negations ?? [],
        },
      },
    ],
  };
}

function relationshipClaim(overrides: Partial<RelationshipClaim> = {}): RelationshipClaim {
  return {
    label_family: "kinship",
    subject_entity_id: null,
    object_entity_id: null,
    object_text: "relación familiar",
    requires_grounding: true,
    evidence_relational_slot_ids: [],
    evidence_stream_entry_ids: [],
    ...overrides,
  };
}

describe("CorrectivePreferenceExtractor", () => {
  it("emits a high-confidence corrective preference candidate", async () => {
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          directive: "Do not add ritual closing lines when the conversation is still open.",
          directive_family: "no_terminal_valediction",
          closure_pressure_relevance: "no_closure",
          priority: 8,
          reason: "The user corrected recurring future response behavior.",
          confidence: 0.9,
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
    });

    const result = await extractor.extract({
      userMessage: "You keep doing those closers. Stop that.",
      recentHistory: [],
      audienceEntityId: createEntityId(),
      activeCommitments: [],
    });

    expect(result).toMatchObject({
      type: "preference",
      kind: "participant_preference",
      enforcement_class: "advisory",
      critical_domain: null,
      directive: "Do not add ritual closing lines when the conversation is still open.",
      directive_family: "no_terminal_valediction",
      closure_pressure_relevance: "no_closure",
      priority: 8,
      reason: "The user corrected recurring future response behavior.",
      confidence: 0.9,
    });
    expect(llm.requests[0]?.model).toBe("haiku");
    expect(llm.requests[0]?.max_tokens).toBe(EXTRACTOR_MAX_TOKENS_DEFAULT);
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: "EmitCorrectivePreference",
    });
    expect(llm.requests[0]?.system).toEqual([
      {
        type: "text",
        text: CORRECTIVE_PREFERENCE_SYSTEM_PROMPT,
        cache_control: { type: "ephemeral", ttl: "5m" },
      },
    ]);
    expect(llm.requests[0]?.tools?.some((tool) => tool.cache_control !== undefined)).toBe(false);
  });

  it("emits a high-confidence retire-only commitment result", async () => {
    const commitmentId = createCommitmentId();
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "retire_commitment",
          reason: "The model judged the supplied active commitment resolved.",
          confidence: 0.91,
          retiresCommitmentId: commitmentId,
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
    });

    const result = await extractor.extractWithSlotNegations({
      userMessage: "That standing boundary is resolved now.",
      recentHistory: [],
      audienceEntityId: createEntityId(),
      activeCommitments: [
        {
          id: commitmentId,
          type: "preference",
          kind: "participant_preference",
          enforcement_class: "advisory",
          critical_domain: null,
          directive: "Keep the temporary discussion constraint active.",
          directive_family: "temporary_discussion_constraint",
          closure_pressure_relevance: "neutral",
          priority: 5,
        },
      ],
    });

    expect(result).toMatchObject({
      preference: null,
      retirement: {
        commitmentId,
        reason: "The model judged the supplied active commitment resolved.",
        confidence: 0.91,
      },
      slot_negations: [],
    });
  });

  it("carries relationship claims on corrective preference candidates", async () => {
    const streamEntryId = createStreamEntryId();
    const slotId = createRelationalSlotId();
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "corrective_preference",
          type: "rule",
          directive: "Use the parent constraint for future care-planning replies.",
          directive_family: "care_planning_parent_constraint",
          closure_pressure_relevance: "neutral",
          priority: 8,
          reason: "The user supplied durable relationship-grounded process guidance.",
          confidence: 0.9,
          relationship_claims: [
            relationshipClaim({
              evidence_relational_slot_ids: [slotId],
              evidence_stream_entry_ids: [streamEntryId],
            }),
          ],
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
    });

    await expect(
      extractor.extract({
        userMessage: "Keep that parent planning rule for next time.",
        currentUserStreamEntryId: streamEntryId,
        recentHistory: [],
        audienceEntityId: createEntityId(),
        activeCommitments: [],
      }),
    ).resolves.toMatchObject({
      directive: "Use the parent constraint for future care-planning replies.",
      relationship_claims: [
        expect.objectContaining({
          evidence_relational_slot_ids: [slotId],
          evidence_stream_entry_ids: [streamEntryId],
        }),
      ],
    });
  });

  it("returns enforcement class and critical domain from the existing extractor call", async () => {
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "corrective_preference",
          type: "boundary",
          kind: "audience_rule",
          enforcement_class: "critical",
          critical_domain: "audience_scope",
          directive: "Do not discuss the deployment incident in the public channel.",
          directive_family: "deployment_incident_channel_scope",
          closure_pressure_relevance: "neutral",
          priority: 10,
          reason: "The user set a durable audience-scoped disclosure boundary.",
          confidence: 0.92,
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
    });

    await expect(
      extractor.extract({
        userMessage: "Do not discuss the deployment incident in the public channel.",
        recentHistory: [],
        audienceEntityId: createEntityId(),
        activeCommitments: [],
      }),
    ).resolves.toMatchObject({
      type: "boundary",
      kind: "audience_rule",
      enforcement_class: "critical",
      critical_domain: "audience_scope",
    });
    expect(llm.requests).toHaveLength(1);
  });

  it("normalizes suspicious critical classifications before emitting a candidate", async () => {
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          kind: "participant_preference",
          enforcement_class: "critical",
          critical_domain: "internal_tool_hygiene",
          directive:
            "Surface durable decisions and held context in explicit language at natural wrap points.",
          directive_family: "surface_durable_decisions",
          closure_pressure_relevance: "neutral",
          priority: 8,
          reason: "The user gave durable process presentation guidance.",
          confidence: 0.91,
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
    });

    await expect(
      extractor.extract({
        userMessage:
          "From now on, surface durable decisions and held context explicitly at wrap points.",
        recentHistory: [],
        audienceEntityId: createEntityId(),
        activeCommitments: [],
      }),
    ).resolves.toMatchObject({
      type: "preference",
      kind: "participant_preference",
      enforcement_class: "advisory",
      critical_domain: null,
      directive_family: "surface_durable_decisions",
    });
  });

  it("traces classification downgrades without directive text", async () => {
    const emit = vi.fn();
    const tracer = {
      enabled: true,
      includePayloads: true,
      emit,
    } satisfies TurnTracer;
    const directive = "Hold working decisions as durable log entries.";
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "corrective_preference",
          type: "rule",
          kind: "process_norm",
          enforcement_class: "critical",
          critical_domain: "internal_tool_hygiene",
          directive,
          directive_family: "hold_working_decisions",
          closure_pressure_relevance: "neutral",
          priority: 8,
          reason: "The user gave durable state-management guidance.",
          confidence: 0.9,
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn-classification-downgrade",
    });

    await extractor.extract({
      userMessage: "Hold these working decisions as durable log entries.",
      recentHistory: [],
      audienceEntityId: createEntityId(),
      activeCommitments: [],
    });

    expect(emit).toHaveBeenCalledWith("commitment_classification.downgraded", {
      turnId: "turn-classification-downgrade",
      original_enforcement_class: "critical",
      original_critical_domain: "internal_tool_hygiene",
      new_enforcement_class: "advisory",
      new_critical_domain: null,
      reason: "process_norm_classified_critical",
      kind: "process_norm",
      type: "rule",
      directive_family: "hold_working_decisions",
    });
    const downgradePayload = emit.mock.calls.find(
      ([event]) => event === "commitment_classification.downgraded",
    )?.[1];
    expect(downgradePayload).not.toHaveProperty("directive");
    expect(JSON.stringify(downgradePayload)).not.toContain(directive);
  });

  it("traces corrective preference extractor LLM calls on success", async () => {
    const emit = vi.fn();
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit,
    } satisfies TurnTracer;
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          directive: "Do not add ritual closing lines when the conversation is still open.",
          directive_family: "no_terminal_valediction",
          closure_pressure_relevance: "no_closure",
          priority: 8,
          reason: "The user corrected recurring future response behavior.",
          confidence: 0.9,
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn-corrective-preference",
    });

    await expect(
      extractor.extract({
        userMessage: "You keep doing those closers. Stop that.",
        recentHistory: [],
        audienceEntityId: createEntityId(),
        activeCommitments: [],
      }),
    ).resolves.toMatchObject({
      type: "preference",
      kind: "participant_preference",
      directive: "Do not add ritual closing lines when the conversation is still open.",
      directive_family: "no_terminal_valediction",
      closure_pressure_relevance: "no_closure",
    });

    expect(emit).toHaveBeenCalledWith("llm_call.started", {
      turnId: "turn-corrective-preference",
      label: "corrective_preference_extractor",
      attempt: 1,
      schema_repair: false,
      model: "haiku",
      promptCharCount: expect.any(Number),
      toolSchemas: expect.any(Array),
    });
    expect(emit).toHaveBeenCalledWith("llm_call.completed", {
      turnId: "turn-corrective-preference",
      label: "corrective_preference_extractor",
      attempt: 1,
      schema_repair: false,
      responseShape: {
        textLength: 0,
        toolUseBlocks: [
          {
            id: "toolu_corrective_preference",
            name: "EmitCorrectivePreference",
          },
        ],
      },
      stopReason: "tool_use",
      usage: {
        inputTokens: 4,
        outputTokens: 2,
      },
    });
  });

  it("returns null for casual discussion", async () => {
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "none",
          reason: "The user is sharing a state, not correcting future behavior.",
          confidence: 0.95,
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
    });

    await expect(
      extractor.extract({
        userMessage: "I'm tired.",
        recentHistory: [],
        audienceEntityId: null,
        activeCommitments: [],
      }),
    ).resolves.toBeNull();
  });

  it("returns slot negations separately from durable corrective preferences", async () => {
    const subject = createEntityId();
    const streamEntryId = createStreamEntryId();
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "none",
          reason: "The user rejected a stored relational value, not future style.",
          confidence: 0.95,
          slot_negations: [
            {
              subject_entity_id: subject,
              slot_key: "partner.name",
              rejected_value: "Sarah",
              source_stream_entry_ids: [streamEntryId],
              confidence: 0.92,
            },
          ],
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
    });

    const result = await extractor.extractWithSlotNegations({
      userMessage: "Her name is not Sarah.",
      currentUserStreamEntryId: streamEntryId,
      recentHistory: [],
      audienceEntityId: null,
      activeCommitments: [],
      relationalSlots: [
        {
          subject_entity_id: subject,
          slot_key: "partner.name",
          value: "Sarah",
          state: "established",
          alternate_values: [],
        },
      ],
    });

    expect(result.preference).toBeNull();
    expect(result.slot_negations).toEqual([
      {
        subject_entity_id: subject,
        slot_key: "partner.name",
        rejected_value: "Sarah",
        source_stream_entry_ids: [streamEntryId],
        confidence: 0.92,
      },
    ]);
    const prompt = JSON.parse(String(llm.requests[0]?.messages[0]?.content ?? "{}")) as {
      relational_slots: Array<{
        disclosure?: string;
        disclosure_label?: { disclosure_class?: string; private_to_entity_ids?: string[] };
      }>;
    };
    expect(prompt.relational_slots[0]).toMatchObject({
      disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [subject],
      },
    });
    expect(prompt.relational_slots[0]?.disclosure).toContain(
      "disclosure_class=relationship_private",
    );
  });

  it("returns null for low-confidence corrective classifications", async () => {
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "corrective_preference",
          type: "rule",
          directive: "Adjust future response behavior.",
          directive_family: "adjust_future_response_behavior",
          closure_pressure_relevance: "neutral",
          priority: 5,
          reason: "The signal is ambiguous.",
          confidence: 0.5,
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
    });

    await expect(
      extractor.extract({
        userMessage: "Maybe don't do that.",
        recentHistory: [],
        audienceEntityId: null,
        activeCommitments: [],
      }),
    ).resolves.toBeNull();
  });

  it("returns null retirement for low-confidence retire classifications", async () => {
    const commitmentId = createCommitmentId();
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "retire_commitment",
          reason: "The model was uncertain about retiring the commitment.",
          confidence: 0.5,
          retiresCommitmentId: commitmentId,
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
    });

    await expect(
      extractor.extractWithSlotNegations({
        userMessage: "Maybe that old rule is not needed now.",
        recentHistory: [],
        audienceEntityId: null,
        activeCommitments: [],
      }),
    ).resolves.toMatchObject({
      preference: null,
      retirement: null,
      slot_negations: [],
    });
  });

  it("reports degraded extraction without throwing", async () => {
    const onDegraded = vi.fn();
    const extractor = new CorrectivePreferenceExtractor({
      onDegraded,
    });

    await expect(
      extractor.extract({
        userMessage: "Keep this behavior different later.",
        recentHistory: [],
        audienceEntityId: null,
        activeCommitments: [],
      }),
    ).resolves.toBeNull();
    expect(onDegraded).toHaveBeenCalledWith("llm_unavailable", undefined);
  });

  it("can surface degradation for strict sidecar retry semantics", async () => {
    const onDegraded = vi.fn();
    const extractor = new CorrectivePreferenceExtractor({
      onDegraded,
      throwOnDegraded: true,
    });

    await expect(
      extractor.extract({
        userMessage: "Never disclose this.",
        recentHistory: [],
        audienceEntityId: null,
        activeCommitments: [],
      }),
    ).rejects.toMatchObject({
      name: "CorrectivePreferenceExtractorDegradedError",
      reason: "llm_unavailable",
    } satisfies Partial<CorrectivePreferenceExtractorDegradedError>);
    expect(onDegraded).toHaveBeenCalledWith("llm_unavailable", undefined);
  });

  it("keeps the extractor free of semantic string-matching shortcuts", () => {
    const source = readFileSync(
      new URL("./corrective-preference-extractor.ts", import.meta.url),
      "utf8",
    );

    const forbiddenFragments = [
      [".", "includes", "("],
      [".", "index", "Of", "("],
      [".", "starts", "With", "("],
      [".", "ends", "With", "("],
      ["new ", "Set", "("],
      ["new ", "Reg", "Exp", "("],
      ["to", "Upper", "Case", "("],
    ];

    for (const fragment of forbiddenFragments) {
      expect(source).not.toContain(fragment.join(""));
    }
  });
});
