import { describe, expect, it, vi } from "vitest";

import type { LLMClient, LLMCompleteOptions, LLMCompleteResult } from "../../llm/index.js";
import type { PlannerContextCaptureRecord } from "./planner-context-capture.js";
import type { PlannerAbReplayResultRecord } from "./planner-ab-replay.js";
import {
  aggregatePlannerAbJudgments,
  judgePlannerAbPair,
  parsePlannerAbJudgeOutput,
  parsePlannerAbReplayResultForJudge,
  plannerAbStdoutSummary,
  PLANNER_AB_SURFACE_SELF_REFERENCE_PHRASES,
  prepareBlindPlannerAbJudgeInput,
  type PlannerAbBlindJudgment,
  type PlannerAbCompletedJudgmentRecord,
} from "./planner-ab-judge.js";

const CAPTURE_ID = "capture-judge-1";
const TURN_ID = "turn-judge-1";
const SESSION_ID = "session-judge-1";

function plan(label: string, want?: string) {
  return {
    uncertainty: `${label} uncertainty`,
    verification_steps: [`${label} verify`],
    tensions: [`${label} tension`],
    voice_note: `${label} voice`,
    emission_recommendation: "emit" as const,
    intents: [{ description: `${label} intent`, next_action: `${label} action` }],
    ...(want === undefined ? {} : { want }),
  };
}

function fingerprint(chars: number, character: string) {
  return {
    systemChars: chars,
    systemSha256: character.repeat(64),
    transportSha256: character.repeat(64),
    systemBlockCount: 3,
    cacheBreakpointCount: 2,
  };
}

function replay(
  options: {
    compactStatus?: "completed" | "degraded";
    turnOrigin?: "user" | "autonomous";
  } = {},
): PlannerAbReplayResultRecord {
  const compactStatus = options.compactStatus ?? "completed";
  const compactOutcome =
    compactStatus === "completed"
      ? {
          status: "completed" as const,
          attempts: 1,
          structuralReason: "emit_turn_plan" as const,
          durationMs: 101,
          plan: plan("candidate-one", options.turnOrigin === "autonomous" ? "quiet" : undefined),
          reasoning: "hidden reasoning one",
          usage: { input_tokens: 987_654_321, output_tokens: 20, stop_reason: "tool_use" },
          requestFingerprint: { canonicalChars: 111, canonicalSha256: "c".repeat(64) },
        }
      : {
          status: "degraded" as const,
          attempts: 2,
          structuralReason: "missing_emit_turn_plan_tool_use" as const,
          durationMs: 101,
          plan: null,
          reasoning: "missing plan",
          usage: { input_tokens: 12, output_tokens: 2, stop_reason: "end_turn" },
          requestFingerprint: { canonicalChars: 111, canonicalSha256: "c".repeat(64) },
        };
  return {
    schema_version: 2,
    capture_id: CAPTURE_ID,
    source_turn_id: TURN_ID,
    source_session_id: SESSION_ID,
    source_captured_at: 1_700_000_000_000,
    source_live_surface_variant: "compact",
    source_outcome: { status: "completed" },
    replayed_at: 1_700_000_000_100,
    mode: "live",
    pairing_status: "paired",
    fidelity: { storedVerified: true, currentSourceRequestMatchesCapture: true },
    execution_order: ["compact", "legacy"],
    messages: { count: 2, chars: 333 },
    surfaces: {
      compact: {
        fingerprint: fingerprint(987_654_321, "a"),
        expectedFingerprint: fingerprint(987_654_321, "a"),
        byteFaithfulToCapture: true,
        traceSummary: { totalEstimatedTokens: 246_913_580 },
      },
      legacy: {
        fingerprint: fingerprint(876_543_210, "b"),
        expectedFingerprint: fingerprint(876_543_210, "b"),
        byteFaithfulToCapture: true,
        traceSummary: { totalEstimatedTokens: 219_135_802 },
      },
    },
    size_delta: {
      compact_minus_legacy_chars: 111_111_111,
      compact_minus_legacy_estimated_tokens: 27_777_778,
    },
    live: {
      compact: compactOutcome,
      legacy: {
        status: "completed",
        attempts: 1,
        structuralReason: "emit_turn_plan",
        durationMs: 202,
        plan: plan("candidate-two", options.turnOrigin === "autonomous" ? "rest" : undefined),
        reasoning: "hidden reasoning two",
        usage: { input_tokens: 876_543_210, output_tokens: 21, stop_reason: "tool_use" },
        requestFingerprint: { canonicalChars: 222, canonicalSha256: "d".repeat(64) },
      },
    },
  } as unknown as PlannerAbReplayResultRecord;
}

function capture(
  options: { turnOrigin?: "user" | "autonomous"; toolNames?: readonly string[] } = {},
): PlannerContextCaptureRecord {
  return {
    capture_id: CAPTURE_ID,
    turn_id: TURN_ID,
    session_id: SESSION_ID,
    render_input: {
      legacyBaseSystemPrompt: "LEGACY_SURFACE_SENTINEL_SHOULD_NOT_REACH_JUDGE",
      dialogueMessages: [
        { role: "assistant", content: "Earlier grounded assistant turn." },
        { role: "user", content: "Current grounded participant turn." },
      ],
      additionalPromptSections: [
        {
          blockId: "borg_compact_planner_ledger",
          text: [
            "<borg_compact_planner_ledger>",
            "CompactPlannerLedger: shared grounded evidence.",
            "commitment row and disclosure label",
            "</borg_compact_planner_ledger>",
          ].join("\n"),
        },
        {
          blockId: "borg_unresolved_contradiction_open_questions",
          text: "One unresolved contradiction.",
        },
      ],
      compactContext: {
        turnOrigin: options.turnOrigin ?? "user",
        applicableCommitments: [
          {
            id: "commitment-1",
            type: "promise",
            kind: "directive",
            enforcement_class: "advisory",
            critical_domain: null,
            directive_family: "follow_through",
            closure_pressure_relevance: "relevant",
            directive: "Keep the cited attribution attached to the claim.",
            priority: 7,
            made_to_entity: "entity-a",
            restricted_audience: "entity-a",
            about_entity: "entity-b",
            committed_by_entity_id: "borg",
            created_at: 100,
            expires_at: null,
            expired_at: null,
            revoked_at: null,
            last_reinforced_at: 200,
          },
        ],
        creatorDirectiveBriefing: {
          directives: [
            {
              renderMode: "content",
              kind: "response_policy",
              subjectKind: "entity",
              subjectLabel: "Participant A",
              semanticSlot: null,
              mentionPolicy: "only_if_topic_raised",
              operationalDirective: "Do not attribute Participant A's acts to Participant B.",
              priority: 8,
              createdAt: 100,
              scope: {
                directiveId: "directive-1",
                createdByEntityId: "creator-1",
                sourceSessionId: SESSION_ID,
                contentScope: "allow_list",
                allowedEntityIds: ["entity-a"],
                excludedEntityIds: ["entity-b"],
                subjectMayKnow: true,
                mentionPolicy: "only_if_topic_raised",
                deniedAudienceBehavior: "omit",
                activationScope: "same_as_disclosure",
                activationAllowedEntityIds: [],
                activationExcludedEntityIds: [],
              },
            },
          ],
        },
        relationalSlots: [
          {
            id: "slot-1",
            subject_entity_id: "entity-a",
            slot_key: "working_relationship",
            value: "collaborator",
            state: "active",
            alternate_values: [],
            updated_at: 200,
          },
        ],
        activeParticipants: [
          { entityId: "entity-a", displayName: "Participant A", role: "participant" },
        ],
        participantProfiles: [
          {
            entityId: "entity-a",
            displayName: "Participant A",
            role: "participant",
            profile: {
              trust: 0.8,
              attachment: 0.5,
              communication_style: "direct",
              shared_history_summary: "A shared project.",
              last_interaction_at: 200,
              interaction_count: 3,
              commitment_count: 1,
              notes: null,
            },
          },
        ],
        audienceProfile: {
          entity_id: "entity-a",
          trust: 0.8,
          attachment: 0.5,
          communication_style: "direct",
          shared_history_summary: "A shared project.",
          last_interaction_at: 200,
          interaction_count: 3,
          commitment_count: 1,
          notes: null,
        },
        creatorContext: {
          currentSenderEntityId: "entity-a",
          currentSenderDisplayName: "Participant A",
          currentSenderBorgRole: "creator",
          sessionAudienceRole: "creator",
        },
        autonomousFinalizerToolMenu: options.toolNames?.map((name) => ({
          name,
          menuSummary: `Capability for ${name}`,
        })),
      },
    },
  } as unknown as PlannerContextCaptureRecord;
}

function blindJudgment(
  options: {
    leftScore?: number;
    rightScore?: number;
    preference?: "left" | "tie" | "right";
    autonomous?: boolean;
    vetoCandidate?: "left" | "right";
    notAssessableDimensions?: readonly string[];
    vetoAssessability?: {
      commitment: "assessable" | "not_assessable";
      disclosure: "assessable" | "not_assessable";
      attribution: "assessable" | "not_assessable";
    };
  } = {},
): PlannerAbBlindJudgment {
  const dimension = {
    left_score: options.leftScore ?? 5,
    right_score: options.rightScore ?? 3,
    reason: "Grounded comparative reason.",
  };
  const score = (dimensionName: string) =>
    options.notAssessableDimensions?.includes(dimensionName) === true ? null : dimension;
  return {
    dimensions: {
      grounded_uncertainty_quality: score("grounded_uncertainty_quality"),
      verification_steps_recall_precision_usefulness: score(
        "verification_steps_recall_precision_usefulness",
      ),
      tension_detection: score("tension_detection"),
      emission_recommendation_appropriateness: score("emission_recommendation_appropriateness"),
      follow_up_intent_precision_capability_feasibility: score(
        "follow_up_intent_precision_capability_feasibility",
      ),
      voice_note_usefulness: score("voice_note_usefulness"),
      want_authenticity_non_compulsion: options.autonomous
        ? score("want_authenticity_non_compulsion")
        : null,
    },
    veto_assessability: options.vetoAssessability ?? {
      commitment: "assessable",
      disclosure: "assessable",
      attribution: "assessable",
    },
    overall_preference: options.preference ?? "left",
    overall_reason: "The preferred plan is more precise.",
    veto_class_failures:
      options.vetoCandidate === undefined
        ? []
        : [
            {
              candidate: options.vetoCandidate,
              failure_class: "commitment",
              reason: "This candidate uniquely violates the shown commitment.",
            },
          ],
  };
}

function llm(output: PlannerAbBlindJudgment): {
  client: LLMClient;
  complete: ReturnType<typeof vi.fn>;
  converse: ReturnType<typeof vi.fn>;
} {
  const response = {
    text: "",
    input_tokens: 700,
    output_tokens: 80,
    stop_reason: "tool_use",
    tool_calls: [{ id: "toolu_judge", name: "EmitPlannerAbJudgment", input: output }],
  } satisfies LLMCompleteResult;
  const complete = vi.fn(async (_request: LLMCompleteOptions) => response);
  const converse = vi.fn(async () => {
    throw new Error("judge must not use converse transport");
  });
  return { client: { complete, converse }, complete, converse };
}

describe("planner A/B blind judge", () => {
  it("removes generation labels, fingerprints, sizes, and hidden reasoning from the judge prompt", () => {
    const sourceReplay = replay();
    if (
      sourceReplay.live?.compact.status === "completed" &&
      sourceReplay.live.compact.plan !== null
    ) {
      sourceReplay.live.compact.plan.uncertainty =
        "Compact planner surface and compact evidence ledger omitted rows.";
    }
    if (
      sourceReplay.live?.legacy.status === "completed" &&
      sourceReplay.live.legacy.plan !== null
    ) {
      sourceReplay.live.legacy.plan.uncertainty = "Legacy planner surface supplied context.";
    }
    const prepared = prepareBlindPlannerAbJudgeInput(sourceReplay, capture(), { random: () => 0 });
    const prompt = `${prepared.systemPrompt}\n${prepared.userPrompt}`;

    expect(prompt.toLowerCase()).not.toContain("compact");
    expect(prompt.toLowerCase()).not.toContain("legacy");
    expect(prompt).not.toContain("987654321");
    expect(prompt).not.toContain("876543210");
    expect(prompt).not.toContain("a".repeat(64));
    expect(prompt).not.toContain("hidden reasoning");
    expect(prompt).toContain("[PLANNER_PRESENTATION_REFERENCE]");
    expect(prompt).not.toContain("Bounded planner presentation");
    expect(prompt).not.toContain("alternate planner presentation");
  });

  it("maps every known surface self-reference from either candidate to one neutral token", () => {
    const sourceReplay = replay();
    const echoedReferences = PLANNER_AB_SURFACE_SELF_REFERENCE_PHRASES.join(" | ");
    if (
      sourceReplay.live?.compact.status === "completed" &&
      sourceReplay.live.compact.plan !== null
    ) {
      sourceReplay.live.compact.plan.uncertainty = echoedReferences;
    }
    if (
      sourceReplay.live?.legacy.status === "completed" &&
      sourceReplay.live.legacy.plan !== null
    ) {
      sourceReplay.live.legacy.plan.uncertainty = echoedReferences;
    }

    const prepared = prepareBlindPlannerAbJudgeInput(sourceReplay, capture(), { random: () => 0 });
    for (const phrase of PLANNER_AB_SURFACE_SELF_REFERENCE_PHRASES) {
      expect(prepared.userPrompt).not.toContain(phrase);
    }
    const neutralMatches = prepared.userPrompt.match(/\[PLANNER_PRESENTATION_REFERENCE\]/g);
    expect(neutralMatches?.length).toBeGreaterThanOrEqual(
      PLANNER_AB_SURFACE_SELF_REFERENCE_PHRASES.length * 2,
    );
  });

  it("records randomized assignment and round-trips the deblinded winner", async () => {
    const fake = llm(blindJudgment({ preference: "left" }));
    const record = await judgePlannerAbPair(replay(), capture(), {
      llmClient: fake.client,
      model: "judge-model",
      random: () => 0,
      now: () => 123,
      idFactory: () => "judgment-1",
    });

    expect(record.status).toBe("completed");
    const completed = record as PlannerAbCompletedJudgmentRecord;
    expect(completed.assignment).toEqual({ left: "compact", right: "legacy" });
    expect(completed.deblinded.overall.winner).toBe("compact");
    expect(completed.source_metrics).toMatchObject({
      surfaces: { compact: { fingerprint: { systemChars: 987_654_321 } } },
    });
    expect(JSON.parse(JSON.stringify(completed))).toMatchObject({
      assignment: { left: "compact", right: "legacy" },
      deblinded: { overall: { winner: "compact" } },
    });

    const reversed = prepareBlindPlannerAbJudgeInput(replay(), capture(), { random: () => 0.9 });
    expect(reversed.assignment).toEqual({ left: "legacy", right: "compact" });
  });

  it("bounds dialogue and grounding mechanically with visible head+tail omissions", () => {
    const source = capture();
    const largeCapture = {
      ...source,
      render_input: {
        ...source.render_input,
        dialogueMessages: Array.from({ length: 12 }, (_, index) => ({
          role: index % 2 === 0 ? ("assistant" as const) : ("user" as const),
          content: `${index}:${"😀".repeat(2_000)}`,
        })),
        additionalPromptSections: [
          {
            blockId: "borg_compact_planner_ledger",
            text: `<borg_compact_planner_ledger>${"e".repeat(40_000)}</borg_compact_planner_ledger>`,
          },
        ],
      },
    } as PlannerContextCaptureRecord;
    const prepared = prepareBlindPlannerAbJudgeInput(replay(), largeCapture, {
      random: () => 0,
    });

    expect(prepared.userPrompt).toContain("HEAD+TAIL EXCERPT");
    expect(prepared.userPrompt).toContain('omitted_earlier_rows="4"');
    expect(prepared.contextMetrics.dialogueRows).toBe(8);
    expect(prepared.contextMetrics.dialogueTruncations).toBe(8);
    expect(prepared.contextMetrics.groundingTruncations).toBe(1);
    expect(prepared.contextMetrics.promptChars).toBeLessThan(60_000);
  });

  it("marks cut captured veto memberships incomplete in the neutral bounded serializer", () => {
    const source = capture();
    const commitments = source.render_input.compactContext.applicableCommitments ?? [];
    const largeCapture = {
      ...source,
      render_input: {
        ...source.render_input,
        compactContext: {
          ...source.render_input.compactContext,
          applicableCommitments: [
            ...commitments,
            {
              ...commitments[0],
              id: "commitment-2",
              directive: `directive-head-${"x".repeat(2_000)}-directive-tail`,
              restricted_audience: "entity-b",
            },
          ],
        },
      },
    } as PlannerContextCaptureRecord;

    const prepared = prepareBlindPlannerAbJudgeInput(replay(), largeCapture, {
      random: () => 0,
    });

    expect(prepared.userPrompt).toContain(
      '<shared_veto_grounding serializer="neutral_complete_membership_v1">',
    );
    expect(prepared.userPrompt).toContain(
      '<membership_index class="applicable_commitment" captured="true" complete_membership="false" rows="2">',
    );
    expect(prepared.userPrompt).toContain('"id":"commitment-1"');
    expect(prepared.userPrompt).toContain('"id":"commitment-2"');
    expect(prepared.userPrompt).toContain('"restricted_audience"');
    expect(prepared.userPrompt).toContain(
      '<membership_index class="creator_directive" captured="true" complete_membership="true" rows="1">',
    );
    expect(prepared.userPrompt).toContain('"operationalDirective"');
    expect(prepared.userPrompt).toContain(
      '<membership_index class="relational_slot" captured="true" complete_membership="true" rows="1">',
    );
    expect(prepared.userPrompt).toContain('"id":"slot-1"');
    expect(prepared.userPrompt).toContain(
      '<membership_index class="active_participant" captured="true" complete_membership="true" rows="1">',
    );
    expect(prepared.userPrompt).toContain(
      '<membership_index class="participant_profile" captured="true" complete_membership="true" rows="1">',
    );
    expect(prepared.userPrompt).toContain(
      '<membership_index class="audience_profile" captured="true" complete_membership="true" rows="1">',
    );
    expect(prepared.userPrompt).toContain(
      '<membership_index class="sender_authority_context" captured="true" complete_membership="true" rows="1">',
    );
    expect(prepared.userPrompt).toContain("HEAD+TAIL EXCERPT");
    expect(prepared.userPrompt).not.toContain("<borg_planner_commitment_digest");
    expect(prepared.userPrompt).not.toContain("<creator_directive_index");
    expect(prepared.vetoAssessability).toEqual({
      commitment: "not_assessable",
      disclosure: "not_assessable",
      attribution: "assessable",
    });
    expect(prepared.contextMetrics.vetoEvidenceTruncations).toBeGreaterThan(0);
  });

  it("marks a veto class not assessable when its captured evidence class is absent", () => {
    const source = capture();
    const { creatorDirectiveBriefing: _omitted, ...compactContext } =
      source.render_input.compactContext;
    const incompleteCapture = {
      ...source,
      render_input: { ...source.render_input, compactContext },
    } as PlannerContextCaptureRecord;
    const prepared = prepareBlindPlannerAbJudgeInput(replay(), incompleteCapture, {
      random: () => 0,
    });
    const expected = {
      dimensionApplicability: prepared.dimensionApplicability,
      vetoAssessability: prepared.vetoAssessability,
    };
    const valid = blindJudgment({
      vetoAssessability: {
        commitment: "assessable",
        disclosure: "not_assessable",
        attribution: "assessable",
      },
    });

    expect(prepared.vetoAssessability.disclosure).toBe("not_assessable");
    expect(prepared.userPrompt).toContain(
      "disclosure_veto [not_assessable]: the captured evidence class is incomplete; no disclosure veto question is posed",
    );
    expect(parsePlannerAbJudgeOutput(valid, expected)).toBeDefined();
    expect(() =>
      parsePlannerAbJudgeOutput(
        {
          ...valid,
          veto_class_failures: [
            {
              candidate: "left",
              failure_class: "disclosure",
              reason: "Cannot be supported without the missing class.",
            },
          ],
        },
        expected,
      ),
    ).toThrow("Cannot flag not-assessable disclosure evidence");
  });

  it("keeps historical directive rows visible but scope-sensitive disclosure unassessable", () => {
    const source = capture();
    const currentDirectives =
      source.render_input.compactContext.creatorDirectiveBriefing?.directives ?? [];
    const historicalDirectives = currentDirectives.map((directive) => {
      const { scope: _scope, ...historical } = directive as typeof directive & {
        scope?: unknown;
      };
      return historical;
    });
    const historicalCapture = {
      ...source,
      render_input: {
        ...source.render_input,
        compactContext: {
          ...source.render_input.compactContext,
          creatorDirectiveBriefing: { directives: historicalDirectives },
        },
      },
    } as PlannerContextCaptureRecord;

    const prepared = prepareBlindPlannerAbJudgeInput(replay(), historicalCapture, {
      random: () => 0,
    });

    expect(prepared.userPrompt).toContain(
      '<membership_index class="creator_directive" captured="true" complete_membership="false" rows="1">',
    );
    expect(prepared.vetoAssessability.disclosure).toBe("not_assessable");
  });

  it("retains every plan field and excludes only cut-field dimensions from scoring", async () => {
    const sourceReplay = replay();
    if (
      sourceReplay.live?.compact.status === "completed" &&
      sourceReplay.live.compact.plan !== null
    ) {
      sourceReplay.live.compact.plan.intents = [
        { description: "x".repeat(20_000), next_action: "Retained tail action." },
      ];
    }
    const prepared = prepareBlindPlannerAbJudgeInput(sourceReplay, capture(), { random: () => 0 });

    for (const field of [
      "uncertainty",
      "verification_steps",
      "tensions",
      "voice_note",
      "emission_recommendation",
      "intents",
      "want",
    ]) {
      expect(prepared.userPrompt.split(`\"field\":\"${field}\"`)).toHaveLength(3);
    }
    expect(prepared.dimensionApplicability.follow_up_intent_precision_capability_feasibility).toBe(
      "not_assessable",
    );
    expect(prepared.dimensionApplicability.grounded_uncertainty_quality).toBe("assessable");
    expect(prepared.contextMetrics.leftPlanTruncatedFields).toEqual(["intents"]);
    expect(prepared.contextMetrics.rightPlanTruncatedFields).toEqual([]);
    expect(prepared.contextMetrics.leftPlanMissingFields).toEqual(["want"]);
    expect(prepared.contextMetrics.rightPlanMissingFields).toEqual(["want"]);
    expect(prepared.systemPrompt).toContain(
      "same per-field character budgets to both candidate renderings",
    );
    expect(prepared.systemPrompt).toContain(
      "must not affect scores unless a named dimension explicitly asks",
    );

    const output = blindJudgment({
      notAssessableDimensions: ["follow_up_intent_precision_capability_feasibility"],
    });
    const record = await judgePlannerAbPair(sourceReplay, capture(), {
      llmClient: llm(output).client,
      model: "judge",
      random: () => 0,
      idFactory: () => "cut-field-judgment",
    });
    expect(record.status).toBe("completed");
    const summary = aggregatePlannerAbJudgments([record]);
    expect(summary.all.dimensions.follow_up_intent_precision_capability_feasibility).toMatchObject({
      evaluated_pairs: 0,
      not_assessable_pairs: 1,
    });
    expect(summary.all.dimensions.grounded_uncertainty_quality).toMatchObject({
      evaluated_pairs: 1,
      not_assessable_pairs: 0,
    });
  });

  it("validates the score envelope and autonomous want applicability", () => {
    const user = prepareBlindPlannerAbJudgeInput(replay(), capture(), { random: () => 0 });
    const autonomous = prepareBlindPlannerAbJudgeInput(
      replay({ turnOrigin: "autonomous" }),
      capture({ turnOrigin: "autonomous" }),
      { random: () => 0 },
    );
    const userExpected = {
      dimensionApplicability: user.dimensionApplicability,
      vetoAssessability: user.vetoAssessability,
    };
    const autonomousExpected = {
      dimensionApplicability: autonomous.dimensionApplicability,
      vetoAssessability: autonomous.vetoAssessability,
    };

    expect(parsePlannerAbJudgeOutput(blindJudgment(), userExpected)).toBeDefined();
    expect(
      parsePlannerAbJudgeOutput(blindJudgment({ autonomous: true }), autonomousExpected),
    ).toBeDefined();
    expect(() =>
      parsePlannerAbJudgeOutput(blindJudgment({ leftScore: 6 }), userExpected),
    ).toThrow();
    expect(() =>
      parsePlannerAbJudgeOutput(blindJudgment({ autonomous: true }), userExpected),
    ).toThrow("not_applicable dimension want_authenticity_non_compulsion requires null");
    expect(() => parsePlannerAbJudgeOutput(blindJudgment(), autonomousExpected)).toThrow(
      "Assessable dimension want_authenticity_non_compulsion requires scores",
    );
  });

  it("aggregates per-dimension win/tie/loss by origin and exact available tool family", async () => {
    const compactWin = await judgePlannerAbPair(replay(), capture(), {
      llmClient: llm(
        blindJudgment({ leftScore: 5, rightScore: 2, preference: "left", vetoCandidate: "left" }),
      ).client,
      model: "judge",
      random: () => 0,
      idFactory: () => "j1",
    });
    const legacyWin = await judgePlannerAbPair(replay(), capture(), {
      llmClient: llm(blindJudgment({ leftScore: 2, rightScore: 5, preference: "right" })).client,
      model: "judge",
      random: () => 0,
      idFactory: () => "j2",
    });
    const autonomousTie = await judgePlannerAbPair(
      replay({ turnOrigin: "autonomous" }),
      capture({ turnOrigin: "autonomous", toolNames: ["tool.outbound.post"] }),
      {
        llmClient: llm(
          blindJudgment({ leftScore: 4, rightScore: 4, preference: "tie", autonomous: true }),
        ).client,
        model: "judge",
        random: () => 0.9,
        idFactory: () => "j3",
      },
    );
    const summary = aggregatePlannerAbJudgments([compactWin, legacyWin, autonomousTie], {
      generatedAt: 999,
    });

    expect(summary.all.overall).toEqual({
      compact: { wins: 1, ties: 1, losses: 1 },
      legacy: { wins: 1, ties: 1, losses: 1 },
    });
    expect(summary.all.dimensions.want_authenticity_non_compulsion.evaluated_pairs).toBe(1);
    expect(summary.all.dimensions.want_authenticity_non_compulsion.not_applicable_pairs).toBe(2);
    expect(summary.by_turn_origin.user?.pairs).toBe(2);
    expect(summary.by_turn_origin.autonomous?.pairs).toBe(1);
    expect(summary.by_tool_family.none?.pairs).toBe(2);
    expect(summary.by_tool_family["tool.outbound.post"]?.pairs).toBe(1);
    expect(summary.compact_acceptance_vetoes).toEqual({
      total_flags: 1,
      by_class: { commitment: 1, disclosure: 0, attribution: 0 },
      flags: [{ pair_id: CAPTURE_ID, failure_class: "commitment" }],
    });
    const storedSummary = JSON.stringify(summary);
    expect(storedSummary).not.toContain("This candidate uniquely violates");
    expect(storedSummary).not.toContain("The preferred plan is more precise");
    const stdout = JSON.stringify(plannerAbStdoutSummary(summary));
    expect(stdout).toContain(CAPTURE_ID);
    expect(stdout).not.toContain(TURN_ID);
    expect(stdout).not.toContain(SESSION_ID);
    expect(stdout).not.toContain("This candidate uniquely violates");
    expect(stdout).not.toContain("The preferred plan is more precise");
    expect(stdout).not.toContain("Grounded comparative reason");
    expect(stdout).not.toContain("candidate-one");
  });

  it("is unary and cannot reach repository, retrieval, stream, or working-memory writers", async () => {
    const repositoryWrite = vi.fn();
    const retrieval = vi.fn();
    const streamAppend = vi.fn();
    const workingMemoryWrite = vi.fn();
    const fake = llm(blindJudgment());

    const result = await judgePlannerAbPair(replay(), capture(), {
      llmClient: fake.client,
      model: "judge",
      random: () => 0,
    });

    expect(result.status).toBe("completed");
    expect(fake.complete).toHaveBeenCalledOnce();
    expect(fake.converse).not.toHaveBeenCalled();
    expect(repositoryWrite).not.toHaveBeenCalled();
    expect(retrieval).not.toHaveBeenCalled();
    expect(streamAppend).not.toHaveBeenCalled();
    expect(workingMemoryWrite).not.toHaveBeenCalled();
    expect(fake.complete.mock.calls[0]?.[0]).toMatchObject({
      budget: "planner-ab-judge",
      tool_choice: { type: "tool", name: "EmitPlannerAbJudgment" },
    });
    const request = JSON.stringify(fake.complete.mock.calls[0]?.[0]) ?? "";
    expect(request.toLowerCase()).not.toContain("compact");
    expect(request.toLowerCase()).not.toContain("legacy");
    expect(request).not.toContain("987654321");
    expect(request).not.toContain("876543210");
  });

  it("retries a fully invalid structured judge payload before recording failure", async () => {
    const invalid = {
      text: "",
      input_tokens: 10,
      output_tokens: 2,
      stop_reason: "tool_use",
      tool_calls: [{ id: "bad", name: "EmitPlannerAbJudgment", input: {} }],
    } satisfies LLMCompleteResult;
    const valid = {
      text: "",
      input_tokens: 20,
      output_tokens: 3,
      stop_reason: "tool_use",
      tool_calls: [{ id: "good", name: "EmitPlannerAbJudgment", input: blindJudgment() }],
    } satisfies LLMCompleteResult;
    const complete = vi
      .fn<(request: LLMCompleteOptions) => Promise<LLMCompleteResult>>()
      .mockResolvedValueOnce(invalid)
      .mockResolvedValueOnce(invalid)
      .mockResolvedValueOnce(valid);
    const result = await judgePlannerAbPair(replay(), capture(), {
      llmClient: { complete, converse: vi.fn() },
      model: "judge",
      random: () => 0,
    });

    expect(result.status).toBe("completed");
    expect(complete).toHaveBeenCalledTimes(3);
    if (result.status === "completed") {
      expect(result.judge.attempt_count).toBe(3);
      expect(result.judge.usage).toMatchObject({ input_tokens: 40, output_tokens: 7 });
    }
  });

  it("excludes degraded pairs before any judge call", async () => {
    const fake = llm(blindJudgment());
    const result = await judgePlannerAbPair(replay({ compactStatus: "degraded" }), capture(), {
      llmClient: fake.client,
      model: "judge",
    });

    expect(result).toMatchObject({
      status: "excluded",
      reason: "compact_outcome_not_completed",
    });
    expect(fake.complete).not.toHaveBeenCalled();
  });

  it("parses the replay JSON boundary without making eligibility decisions", () => {
    expect(parsePlannerAbReplayResultForJudge(replay()).capture_id).toBe(CAPTURE_ID);
    expect(() => parsePlannerAbReplayResultForJudge({ ...replay(), mode: "dry" })).not.toThrow();
  });
});
