import { describe, expect, it, vi } from "vitest";

import type { LLMClient, LLMCompleteOptions, LLMCompleteResult } from "../../llm/index.js";
import type { FinalizerContextCaptureRecord } from "./finalizer-context-capture.js";
import type { FinalizerAbReplayResult } from "./finalizer-ab-replay.js";
import {
  aggregateFinalizerAbJudgments,
  FINALIZER_AB_SURFACE_SELF_REFERENCE_PHRASES,
  finalizerAbStdoutSummary,
  judgeFinalizerAbPair,
  parseFinalizerAbReplayResultForJudge,
  prepareBlindFinalizerAbJudgeInput,
  type FinalizerAbBlindJudgment,
} from "./finalizer-ab-judge.js";

const CAPTURE_ID = "finalizer-capture-1";
const TURN_ID = "turn-finalizer-1";

function capture(): FinalizerContextCaptureRecord {
  return {
    capture_id: CAPTURE_ID,
    turn_id: TURN_ID,
    path: "system_2",
    projected_context: {
      applicableCommitments: [
        {
          id: "commitment-1",
          directive: "Keep the source attribution attached.",
          disclosure_label: { disclosureClass: "relationship_private" },
        },
      ],
      creatorDirectiveBriefing: { directives: [] },
      relationalSlots: [],
      activeParticipants: [{ entityId: "participant-1", displayName: "A" }],
      participantProfiles: [],
      audienceProfile: null,
      creatorContext: { currentSenderEntityId: "participant-1" },
    },
    evidence_ledger: { sections: [], audienceStanding: { commitmentEntries: [] } },
    live_request: {
      model: "captured-model",
      system: [{ type: "text", text: "CAPTURED SYSTEM MUST NOT REACH JUDGE" }],
      messages: [
        { role: "assistant", content: [{ type: "text", text: "Earlier reply" }] },
        { role: "user", content: [{ type: "text", text: "Please answer accurately" }] },
      ],
      max_tokens: 100,
      budget: "cognition-system-2",
      tools: [
        {
          name: "EmitAnswer",
          description: "terminal answer",
          inputSchema: { type: "object", properties: {} },
        },
      ],
    },
  } as unknown as FinalizerContextCaptureRecord;
}

function replay(text = "A grounded terminal answer."): FinalizerAbReplayResult {
  const outcome = (variant: "compact" | "legacy") => ({
    status: "completed" as const,
    durationMs: 100,
    requestFingerprint: { canonicalChars: 100, canonicalSha256: variant[0]!.repeat(64) },
    usage: { input_tokens: variant === "compact" ? 987_654_321 : 876_543_210, output_tokens: 20 },
    messageBlocks: [
      {
        type: "tool_use",
        id: `tool-${variant}`,
        name: "EmitAnswer",
        input: { text },
      },
    ],
  });
  return {
    schema_version: 1,
    capture_id: CAPTURE_ID,
    source_turn_id: TURN_ID,
    source_path: "system_2",
    source_attempt_kind: "initial",
    source_configured_surface_variant: "compact_conversational",
    source_live_surface_variant: "compact",
    replayed_at: 123,
    mode: "live",
    pairing_status: "paired",
    execution_order: ["compact", "legacy"],
    fidelity: {
      storedVerified: true,
      currentSourceSystemMatchesCapture: true,
      currentSourceRequestMatchesCapture: true,
    },
    surfaces: {
      compact: { systemChars: 987_654_321, systemSha256: "a".repeat(64) },
      legacy: { systemChars: 876_543_210, systemSha256: "b".repeat(64) },
    } as never,
    size_delta: { compact_minus_legacy_chars: 111_111_111 },
    live: { compact: outcome("compact"), legacy: outcome("legacy") },
  };
}

function judgment(
  options: { preference?: "left" | "tie" | "right"; veto?: boolean } = {},
): FinalizerAbBlindJudgment {
  const score = { left_score: 5, right_score: 3, reason: "left is better grounded" };
  return {
    dimensions: {
      grounded_terminal_quality: score,
      emission_choice_appropriateness: score,
      commitment_fidelity: score,
      disclosure_handling: score,
      attribution_fidelity: score,
      voice_and_usefulness: score,
    },
    veto_assessability: {
      commitment: "assessable",
      disclosure: "assessable",
      attribution: "assessable",
    },
    overall_preference: options.preference ?? "left",
    overall_reason: "left is more faithful",
    veto_class_failures: options.veto
      ? [
          {
            candidate: "left",
            failure_class: "commitment",
            reason: "left alone violates the shown commitment",
          },
        ]
      : [],
  };
}

function llm(output: FinalizerAbBlindJudgment) {
  const response = {
    text: "",
    input_tokens: 100,
    output_tokens: 20,
    stop_reason: "tool_use",
    tool_calls: [{ id: "judge", name: "EmitFinalizerAbJudgment", input: output }],
  } satisfies LLMCompleteResult;
  const complete = vi.fn(async (_request: LLMCompleteOptions) => response);
  const converse = vi.fn(async () => {
    throw new Error("blind judge must use unary transport");
  });
  return { client: { complete, converse } satisfies LLMClient, complete, converse };
}

describe("finalizer A/B blind judge", () => {
  it("scrubs every known presentation reference and withholds labels, sizes, and fingerprints", () => {
    const echoed = FINALIZER_AB_SURFACE_SELF_REFERENCE_PHRASES.join(" | ");
    const prepared = prepareBlindFinalizerAbJudgeInput(replay(echoed), capture(), {
      random: () => 0,
    });
    const prompt = `${prepared.systemPrompt}\n${prepared.userPrompt}`;

    for (const phrase of FINALIZER_AB_SURFACE_SELF_REFERENCE_PHRASES) {
      expect(prompt).not.toContain(phrase);
    }
    expect(prompt.toLowerCase()).not.toContain("compact");
    expect(prompt.toLowerCase()).not.toContain("legacy");
    expect(prompt).not.toContain("compact_conversational");
    expect(prompt).not.toContain("987654321");
    expect(prompt).not.toContain("876543210");
    expect(prompt).not.toContain("a".repeat(64));
    expect(prompt).toContain("[TERMINAL_PRESENTATION_REFERENCE]");
    expect(prompt).not.toContain("[TERMINAL_PRESENTATION_REFERENCE]_state");
    expect(prompt).toContain("<available_terminal_tools>EmitAnswer</available_terminal_tools>");
    expect(prepared.assignment).toEqual({ left: "compact", right: "legacy" });
    expect(
      prepareBlindFinalizerAbJudgeInput(replay(), capture(), { random: () => 0.9 }).assignment,
    ).toEqual({ left: "legacy", right: "compact" });
  });

  it("judges through unary structured output, records assignment, and stays write-isolated", async () => {
    const fake = llm(judgment());
    const repositoryWrite = vi.fn();
    const streamAppend = vi.fn();
    const result = await judgeFinalizerAbPair(replay(), capture(), {
      llmClient: fake.client,
      model: "judge-model",
      random: () => 0,
      idFactory: () => "judgment-1",
      now: () => 999,
    });

    expect(result).toMatchObject({
      status: "completed",
      assignment: { left: "compact", right: "legacy" },
      deblinded: { overall: { winner: "compact" } },
    });
    expect(fake.complete).toHaveBeenCalledOnce();
    expect(fake.converse).not.toHaveBeenCalled();
    expect(repositoryWrite).not.toHaveBeenCalled();
    expect(streamAppend).not.toHaveBeenCalled();
    expect(fake.complete.mock.calls[0]?.[0]).toMatchObject({
      budget: "finalizer-ab-judge",
      tool_choice: { type: "tool", name: "EmitFinalizerAbJudgment" },
    });
  });

  it("applies the same candidate budget and excludes cut-dependent dimensions", () => {
    const source = replay(`HEAD-${"x".repeat(40_000)}-TAIL`);
    source.live!.legacy.messageBlocks = [
      {
        type: "tool_use",
        id: "legacy-short",
        name: "EmitAnswer",
        input: { text: "short" },
      },
    ];
    const prepared = prepareBlindFinalizerAbJudgeInput(source, capture(), { random: () => 0 });

    expect(prepared.candidateCuts).toEqual({ left: true, right: false });
    expect(prepared.applicability.emission_choice_appropriateness).toBe("assessable");
    expect(prepared.applicability.grounded_terminal_quality).toBe("not_assessable");
    expect(prepared.userPrompt).toContain("HEAD+TAIL EXCERPT");
    expect(prepared.userPrompt).toContain(
      "Both candidates use the same character budget. A harness_cut is evaluation transport truncation",
    );
  });

  it("retries an invalid structured judgment after internal schema repair is exhausted", async () => {
    const invalid = {
      text: "",
      input_tokens: 10,
      output_tokens: 2,
      stop_reason: "tool_use",
      tool_calls: [{ id: "bad", name: "EmitFinalizerAbJudgment", input: {} }],
    } satisfies LLMCompleteResult;
    const valid = {
      text: "",
      input_tokens: 20,
      output_tokens: 3,
      stop_reason: "tool_use",
      tool_calls: [{ id: "good", name: "EmitFinalizerAbJudgment", input: judgment() }],
    } satisfies LLMCompleteResult;
    const complete = vi
      .fn<(request: LLMCompleteOptions) => Promise<LLMCompleteResult>>()
      .mockResolvedValueOnce(invalid)
      .mockResolvedValueOnce(invalid)
      .mockResolvedValueOnce(valid);
    const result = await judgeFinalizerAbPair(replay(), capture(), {
      llmClient: { complete, converse: vi.fn() },
      model: "judge-model",
      random: () => 0,
    });

    expect(result.status).toBe("completed");
    expect(complete).toHaveBeenCalledTimes(3);
    if (result.status === "completed") {
      expect(result.judge.attempt_count).toBe(3);
      expect(result.judge.usage).toMatchObject({ input_tokens: 40, output_tokens: 7 });
    }
  });

  it("caps invalid-payload repair and retries at three total model calls", async () => {
    const invalid = {
      text: "",
      input_tokens: 10,
      output_tokens: 2,
      stop_reason: "tool_use",
      tool_calls: [{ id: "bad", name: "EmitFinalizerAbJudgment", input: {} }],
    } satisfies LLMCompleteResult;
    const complete = vi
      .fn<(request: LLMCompleteOptions) => Promise<LLMCompleteResult>>()
      .mockResolvedValue(invalid);
    const result = await judgeFinalizerAbPair(replay(), capture(), {
      llmClient: { complete, converse: vi.fn() },
      model: "judge-model",
      random: () => 0,
    });

    expect(result.status).toBe("failed");
    expect(complete).toHaveBeenCalledTimes(3);
    if (result.status === "failed") {
      expect(result.judge.attempt_count).toBe(3);
      expect(result.judge.usage).toMatchObject({ input_tokens: 30, output_tokens: 6 });
      expect(result.error.kind).toBe("invalid_payload");
    }
  });

  it("disables commitment-dependent vetoes when membership evidence is cut", () => {
    const source = capture();
    source.projected_context.applicableCommitments = [
      {
        id: "commitment-long",
        directive: `HEAD-${"x".repeat(2_000)}-TAIL`,
      },
    ];
    const prepared = prepareBlindFinalizerAbJudgeInput(replay(), source, { random: () => 0 });

    expect(prepared.userPrompt).toContain(
      '<membership_index class="applicable_commitment" captured="true" complete_membership="false" rows="1">',
    );
    expect(prepared.vetoAssessability).toEqual({
      commitment: "not_assessable",
      disclosure: "not_assessable",
      attribution: "assessable",
    });
    expect(prepared.applicability.commitment_fidelity).toBe("not_assessable");
    expect(prepared.applicability.disclosure_handling).toBe("not_assessable");
  });

  it("keeps historical directive rows visible while scope-sensitive judging stays disabled", () => {
    const source = capture();
    source.projected_context.creatorDirectiveBriefing = {
      directives: [
        {
          renderMode: "content",
          kind: "subject_fact",
          canonicalFact: "Historical captured fact",
          mentionPolicy: "answer_if_asked",
        },
      ],
    };
    const prepared = prepareBlindFinalizerAbJudgeInput(replay(), source, { random: () => 0 });

    expect(prepared.userPrompt).toContain("Historical captured fact");
    expect(prepared.userPrompt).toContain(
      '<membership_index class="creator_directive" captured="true" complete_membership="false" rows="1">',
    );
    expect(prepared.vetoAssessability.disclosure).toBe("not_assessable");
    expect(prepared.applicability.disclosure_handling).toBe("not_assessable");
  });

  it("aggregates by S1/S2 path and terminal tool family while stdout stays counts-only", async () => {
    const result = await judgeFinalizerAbPair(replay(), capture(), {
      llmClient: llm(judgment({ veto: true })).client,
      model: "judge-model",
      random: () => 0,
      idFactory: () => "judgment-1",
    });
    const summary = aggregateFinalizerAbJudgments([result], { generatedAt: 1 });
    const stdout = JSON.stringify(finalizerAbStdoutSummary(summary));

    expect(summary.all.pairs).toBe(1);
    expect(summary.by_path.system_2?.pairs).toBe(1);
    expect(summary.by_tool_family.EmitAnswer?.pairs).toBe(1);
    expect(summary.compact_acceptance_vetoes).toMatchObject({
      total: 1,
      by_class: { commitment: 1, disclosure: 0, attribution: 0 },
    });
    expect(stdout).not.toContain(CAPTURE_ID);
    expect(stdout).not.toContain(TURN_ID);
    expect(stdout).not.toContain("left is better grounded");
    expect(stdout).not.toContain("violates the shown commitment");
  });

  it("excludes non-paired rows before an LLM call and parses the JSON boundary", async () => {
    const source = { ...replay(), pairing_status: "excluded_nonterminal" as const };
    const fake = llm(judgment());
    const result = await judgeFinalizerAbPair(source, capture(), {
      llmClient: fake.client,
      model: "judge-model",
    });

    expect(result).toMatchObject({ status: "excluded", reason: "not_paired" });
    expect(fake.complete).not.toHaveBeenCalled();
    expect(parseFinalizerAbReplayResultForJudge(replay()).capture_id).toBe(CAPTURE_ID);
    const historicalReplay = { ...replay() } as Partial<FinalizerAbReplayResult>;
    delete historicalReplay.source_configured_surface_variant;
    expect(
      parseFinalizerAbReplayResultForJudge(historicalReplay).source_configured_surface_variant,
    ).toBeUndefined();

    const missingTerminal = replay();
    missingTerminal.live!.compact.messageBlocks = [];
    const missingResult = await judgeFinalizerAbPair(missingTerminal, capture(), {
      llmClient: fake.client,
      model: "judge-model",
    });
    expect(missingResult).toMatchObject({
      status: "excluded",
      reason: "compact_terminal_call_missing",
    });
    expect(fake.complete).not.toHaveBeenCalled();
  });
});
