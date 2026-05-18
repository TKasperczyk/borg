import { describe, expect, it } from "vitest";

import { type LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { createStreamEntryId, type StreamEntryId } from "../../util/ids.js";
import { shouldSkipDecisionArtifactCompile } from "../lifecycle/turn-phase-coordinator.js";
import type { TurnTracer, TurnTraceData, TurnTraceEventName } from "../tracing/tracer.js";
import {
  CLOSURE_LOOP_CLASSIFIER_TOOL_NAME,
  ClosureLoopClassifier,
  assessDegradedClosureLoopFallback,
  assessClosureLoopClassification,
  type ClosureLoopClassifiedMessage,
  type ClosureLoopMessageForClassification,
} from "./closure-loop.js";

type TraceRecord = TurnTraceData & { event: TurnTraceEventName };

class TestTracer implements TurnTracer {
  readonly enabled = true;
  readonly records: TraceRecord[] = [];

  constructor(readonly includePayloads = true) {}

  emit(event: TurnTraceEventName, data: TurnTraceData): void {
    this.records.push({ event, ...data });
  }
}

function closureLoopResponse(
  messages: readonly unknown[],
  extra: Record<string, unknown> = {},
): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_closure_loop",
        name: CLOSURE_LOOP_CLASSIFIER_TOOL_NAME,
        input: {
          messages,
          confidence: 0.94,
          rationale: "The recent turns are repeated mutual closure beats.",
          ...extra,
        },
      },
    ],
  };
}

function message(input: {
  role: "user" | "assistant";
  content: string;
  ts: number;
}): ClosureLoopMessageForClassification {
  const streamEntryId = createStreamEntryId();

  return {
    message_ref: streamEntryId,
    role: input.role,
    content: input.content,
    stream_entry_id: streamEntryId,
    ts: input.ts,
  };
}

function classified(
  supplied: ClosureLoopMessageForClassification,
  act: ClosureLoopClassifiedMessage["act"],
  axes: Partial<
    Pick<
      ClosureLoopClassifiedMessage,
      "is_closure_shaped" | "has_substantive_content" | "has_substantive_state_delta"
    >
  > = {},
): ClosureLoopClassifiedMessage {
  const isClosureShaped =
    supplied.role === "user"
      ? act === "signoff"
      : act === "assistant_imperative_closer" ||
        act === "assistant_valediction" ||
        act === "minimal_acknowledgment";
  const hasSubstantiveContent =
    act === "substantive" ||
    act === "reopening_after_signoff" ||
    act === "meta_objection_to_closure";

  return {
    message_ref: supplied.message_ref,
    role: supplied.role,
    act,
    is_closure_shaped: axes.is_closure_shaped ?? isClosureShaped,
    has_substantive_content: axes.has_substantive_content ?? hasSubstantiveContent,
    has_substantive_state_delta: axes.has_substantive_state_delta ?? false,
  };
}

describe("ClosureLoopClassifier", () => {
  it("uses the recall-expansion model slot for dialogue-act classification", async () => {
    const supplied = [
      message({ role: "user", content: "going", ts: 1 }),
      message({ role: "assistant", content: "Go.", ts: 2 }),
    ];
    const llm = new FakeLLMClient({
      responses: [
        closureLoopResponse([
          classified(supplied[0]!, "signoff"),
          classified(supplied[1]!, "assistant_imperative_closer"),
        ]),
      ],
    });
    const classifier = new ClosureLoopClassifier({
      llmClient: llm,
      model: "test-recall",
    });

    const result = await classifier.classify({
      messages: supplied,
    });

    expect(result.degraded).toBe(false);
    expect(llm.requests[0]).toMatchObject({
      model: "test-recall",
      budget: "closure-loop-classifier",
      tool_choice: {
        type: "tool",
        name: CLOSURE_LOOP_CLASSIFIER_TOOL_NAME,
      },
    });
  });

  it("parses independent closure, content, and state-delta booleans", async () => {
    const supplied = [
      message({ role: "user", content: "Decision: rollback to v1.2.3. EOD.", ts: 1 }),
      message({ role: "user", content: "Thanks, goodnight.", ts: 2 }),
      message({ role: "user", content: "What's the timeline?", ts: 3 }),
    ];
    const llm = new FakeLLMClient({
      responses: [
        closureLoopResponse([
          classified(supplied[0]!, "signoff", {
            is_closure_shaped: true,
            has_substantive_content: true,
            has_substantive_state_delta: true,
          }),
          classified(supplied[1]!, "signoff", {
            is_closure_shaped: true,
            has_substantive_content: false,
            has_substantive_state_delta: false,
          }),
          classified(supplied[2]!, "substantive", {
            is_closure_shaped: false,
            has_substantive_content: true,
            has_substantive_state_delta: false,
          }),
        ]),
      ],
    });
    const classifier = new ClosureLoopClassifier({
      llmClient: llm,
      model: "test-recall",
    });

    const result = await classifier.classify({
      messages: supplied,
    });

    expect(result.messages).toEqual([
      expect.objectContaining({
        is_closure_shaped: true,
        has_substantive_content: true,
        has_substantive_state_delta: true,
      }),
      expect.objectContaining({
        is_closure_shaped: true,
        has_substantive_content: false,
        has_substantive_state_delta: false,
      }),
      expect.objectContaining({
        is_closure_shaped: false,
        has_substantive_content: true,
        has_substantive_state_delta: false,
      }),
    ]);
  });

  it("surfaces mixed closure plus state delta on the current user assessment", async () => {
    const supplied = [
      message({ role: "user", content: "Decision: rollback to v1.2.3. EOD.", ts: 1 }),
    ];
    const llm = new FakeLLMClient({
      responses: [
        closureLoopResponse([
          classified(supplied[0]!, "signoff", {
            is_closure_shaped: true,
            has_substantive_content: true,
            has_substantive_state_delta: true,
          }),
        ]),
      ],
    });
    const classifier = new ClosureLoopClassifier({
      llmClient: llm,
      model: "test-recall",
    });
    const result = await classifier.classify({
      messages: supplied,
    });

    const assessment = assessClosureLoopClassification({
      classification: result,
      suppliedMessages: supplied,
      currentUserRef: supplied[0]!.message_ref,
    });

    expect(assessment.currentUserClosureShaped).toBe(true);
    expect(assessment.currentUserSubstantive).toBe(true);
    expect(assessment.currentUserHasSubstantiveStateDelta).toBe(true);
  });

  it("fills omitted supplied message refs as substantive", async () => {
    const degraded: string[] = [];
    const supplied = [
      message({ role: "assistant", content: "Talk soon.", ts: 1 }),
      message({ role: "user", content: "phone down", ts: 2 }),
    ];
    const llm = new FakeLLMClient({
      responses: [closureLoopResponse([classified(supplied[0]!, "assistant_valediction")])],
    });
    const classifier = new ClosureLoopClassifier({
      llmClient: llm,
      model: "test-recall",
      onDegraded: (reason) => {
        degraded.push(reason);
      },
    });

    const result = await classifier.classify({
      messages: supplied,
    });

    expect(result.degraded).toBe(false);
    expect(result.messages).toEqual([
      classified(supplied[0]!, "assistant_valediction"),
      classified(supplied[1]!, "substantive", {
        has_substantive_state_delta: true,
      }),
    ]);
    expect(degraded).toEqual([]);
  });

  it("defaults omitted boolean axes to fail-open values and traces the normalized payload", async () => {
    const tracer = new TestTracer();
    const supplied = [message({ role: "user", content: "phone down", ts: 1 })];
    const llm = new FakeLLMClient({
      responses: [
        closureLoopResponse([
          {
            message_ref: supplied[0]!.message_ref,
            role: "user",
            act: "signoff",
          },
        ]),
      ],
    });
    const classifier = new ClosureLoopClassifier({
      llmClient: llm,
      model: "test-recall",
      tracer,
      turnId: "turn-default-closure-axes",
    });

    const result = await classifier.classify({
      messages: supplied,
    });
    const normalized = tracer.records.find(
      (record) => record.event === "closure_loop_classifier_payload_normalized",
    );

    expect(result.messages).toEqual([
      classified(supplied[0]!, "signoff", {
        is_closure_shaped: false,
        has_substantive_content: true,
        has_substantive_state_delta: true,
      }),
    ]);
    expect(normalized?.normalizations).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "is_closure_shaped", action: "defaulted", to: false }),
        expect.objectContaining({
          field: "has_substantive_content",
          action: "defaulted",
          to: true,
        }),
        expect.objectContaining({
          field: "has_substantive_state_delta",
          action: "defaulted",
          to: true,
        }),
      ]),
    );
    expect(normalized?.normalizedPayload).toMatchObject({
      messages: [
        {
          is_closure_shaped: false,
          has_substantive_content: true,
          has_substantive_state_delta: true,
        },
      ],
    });

    const assessment = assessClosureLoopClassification({
      classification: result,
      suppliedMessages: supplied,
      currentUserRef: supplied[0]!.message_ref,
    });

    expect(
      shouldSkipDecisionArtifactCompile({
        enabled: true,
        previousArtifact: null,
        perceptionMode: "problem_solving",
        frameAnomaly: null,
        closureLoopAssessment: assessment,
      }),
    ).toBeNull();
  });

  it("defaults invalid boolean axes to fail-open values", async () => {
    const supplied = [message({ role: "user", content: "phone down", ts: 1 })];
    const llm = new FakeLLMClient({
      responses: [
        closureLoopResponse([
          {
            message_ref: supplied[0]!.message_ref,
            role: "user",
            act: "signoff",
            is_closure_shaped: "yes",
            has_substantive_content: null,
            has_substantive_state_delta: "no",
          },
        ]),
      ],
    });
    const classifier = new ClosureLoopClassifier({
      llmClient: llm,
      model: "test-recall",
    });

    const result = await classifier.classify({
      messages: supplied,
    });

    expect(result.messages).toEqual([
      classified(supplied[0]!, "signoff", {
        is_closure_shaped: false,
        has_substantive_content: true,
        has_substantive_state_delta: true,
      }),
    ]);
  });

  it("takes the first duplicate message ref and traces the duplicate", async () => {
    const tracer = new TestTracer();
    const supplied = [
      message({ role: "assistant", content: "Talk soon.", ts: 1 }),
      message({ role: "user", content: "phone down", ts: 2 }),
    ];
    const llm = new FakeLLMClient({
      responses: [
        closureLoopResponse(
          [
            {
              ...classified(supplied[0]!, "assistant_valediction"),
              extra_field: "ignored",
            },
            classified(supplied[0]!, "substantive"),
            classified(supplied[1]!, "signoff"),
          ],
          { extra_payload_field: true },
        ),
      ],
    });
    const classifier = new ClosureLoopClassifier({
      llmClient: llm,
      model: "test-recall",
      tracer,
      turnId: "turn-closure-duplicate",
    });

    const result = await classifier.classify({
      messages: supplied,
    });
    const normalized = tracer.records.find(
      (record) => record.event === "closure_loop_classifier_payload_normalized",
    );

    expect(result.messages).toEqual([
      classified(supplied[0]!, "assistant_valediction"),
      classified(supplied[1]!, "signoff"),
    ]);
    expect(normalized?.normalizations).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          field: "message_ref",
          messageRef: supplied[0]!.message_ref,
          action: "duplicate_ref_ignored",
        }),
        expect.objectContaining({
          field: "*",
          messageRef: supplied[0]!.message_ref,
          action: "message_extra_fields_ignored",
        }),
        expect.objectContaining({
          field: "*",
          action: "extra_fields_ignored",
        }),
      ]),
    );
  });

  it("maps LLM dialogue-act aliases before closure-loop assessment", async () => {
    const tracer = new TestTracer();
    const supplied = [
      message({ role: "user", content: "going", ts: 1 }),
      message({ role: "assistant", content: "Go.", ts: 2 }),
      message({ role: "user", content: "really going", ts: 3 }),
      message({ role: "assistant", content: "Talk soon.", ts: 4 }),
      message({ role: "user", content: "phone down", ts: 5 }),
    ];
    const llm = new FakeLLMClient({
      responses: [
        closureLoopResponse([
          { ...classified(supplied[0]!, "signoff"), act: "user_signoff" },
          { ...classified(supplied[1]!, "assistant_imperative_closer"), act: "assistant_signoff" },
          { ...classified(supplied[2]!, "signoff"), act: "user_closure" },
          { ...classified(supplied[3]!, "assistant_valediction"), act: "assistant_goodnight" },
          { ...classified(supplied[4]!, "signoff"), act: "user_signoff" },
        ]),
      ],
    });
    const classifier = new ClosureLoopClassifier({
      llmClient: llm,
      model: "test-recall",
      tracer,
      turnId: "turn-closure-aliases",
    });

    const result = await classifier.classify({
      messages: supplied,
    });
    const assessment = assessClosureLoopClassification({
      classification: result,
      suppliedMessages: supplied,
      currentUserRef: supplied[4]!.message_ref,
    });
    const normalized = tracer.records.find(
      (record) => record.event === "closure_loop_classifier_payload_normalized",
    );

    expect(result.messages.map((item) => item.act)).toEqual([
      "signoff",
      "assistant_imperative_closer",
      "signoff",
      "assistant_valediction",
      "signoff",
    ]);
    expect(assessment.closureLoopDetected).toBe(true);
    expect(assessment.mutualClosureCycles).toBe(2);
    expect(normalized?.normalizations).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "act", action: "alias_mapped", to: "signoff" }),
        expect.objectContaining({
          field: "act",
          action: "alias_mapped",
          to: "assistant_imperative_closer",
        }),
        expect.objectContaining({
          field: "act",
          action: "alias_mapped",
          to: "assistant_valediction",
        }),
      ]),
    );
  });

  it("defaults genuinely unknown dialogue-act labels to substantive", async () => {
    const tracer = new TestTracer();
    const supplied = [message({ role: "user", content: "Can we inspect the scheduler?", ts: 1 })];
    const llm = new FakeLLMClient({
      responses: [
        closureLoopResponse([
          { ...classified(supplied[0]!, "substantive"), act: "not_a_known_closure_act" },
        ]),
      ],
    });
    const classifier = new ClosureLoopClassifier({
      llmClient: llm,
      model: "test-recall",
      tracer,
      turnId: "turn-closure-unknown-label",
    });

    const result = await classifier.classify({
      messages: supplied,
    });
    const normalized = tracer.records.find(
      (record) => record.event === "closure_loop_classifier_payload_normalized",
    );

    expect(result.messages).toEqual([classified(supplied[0]!, "substantive")]);
    expect(normalized?.normalizations).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          field: "act",
          action: "invalid_or_missing_defaulted",
          to: "substantive",
        }),
      ]),
    );
  });

  it("accepts explicit user closure requests as their own dialogue act", async () => {
    const supplied = [message({ role: "user", content: "Please say goodnight.", ts: 1 })];
    const llm = new FakeLLMClient({
      responses: [closureLoopResponse([classified(supplied[0]!, "user_requests_closure")])],
    });
    const classifier = new ClosureLoopClassifier({
      llmClient: llm,
      model: "test-recall",
    });

    const result = await classifier.classify({
      messages: supplied,
    });
    const assessment = assessClosureLoopClassification({
      classification: result,
      suppliedMessages: supplied,
      currentUserRef: supplied[0]!.message_ref,
    });

    expect(assessment.currentUserAct).toBe("user_requests_closure");
    expect(assessment.currentUserClosureShaped).toBe(false);
  });

  it("emits raw classifier output shape without full payloads on normalization", async () => {
    const tracer = new TestTracer(false);
    const supplied = [
      message({ role: "assistant", content: "Talk soon.", ts: 1 }),
      message({ role: "user", content: "phone down", ts: 2 }),
    ];
    const llm = new FakeLLMClient({
      responses: [
        closureLoopResponse(
          [
            {
              ...classified(supplied[0]!, "substantive"),
              act: "assistant_farewell",
              extra_message_field: "ignored",
            },
            classified(supplied[1]!, "signoff"),
          ],
          { extra_payload_field: true },
        ),
      ],
    });
    const classifier = new ClosureLoopClassifier({
      llmClient: llm,
      model: "test-recall",
      tracer,
      turnId: "turn-closure-shape",
    });

    await classifier.classify({
      messages: supplied,
    });
    const normalized = tracer.records.find(
      (record) => record.event === "closure_loop_classifier_payload_normalized",
    );

    expect(normalized?.rawToolInput).toBeUndefined();
    expect(normalized?.rawToolInputShape).toMatchObject({
      type: "object",
      fields: expect.arrayContaining([
        expect.objectContaining({ name: "messages", type: "array" }),
        expect.objectContaining({ name: "confidence", type: "number" }),
        expect.objectContaining({ name: "rationale", type: "string" }),
        expect.objectContaining({ name: "extra_payload_field", type: "boolean" }),
      ]),
    });
    expect(normalized?.normalizations).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ action: "extra_fields_ignored" }),
        expect.objectContaining({ action: "message_extra_fields_ignored" }),
        expect.objectContaining({ action: "alias_mapped", to: "assistant_valediction" }),
      ]),
    );
  });
});

describe("assessClosureLoopClassification", () => {
  it("detects a six-message mutual closure loop from LLM-emitted acts", () => {
    const supplied = [
      message({ role: "user", content: "going", ts: 1 }),
      message({ role: "assistant", content: "Go.", ts: 2 }),
      message({ role: "user", content: "really going", ts: 3 }),
      message({ role: "assistant", content: "Talk soon.", ts: 4 }),
      message({ role: "user", content: "phone down", ts: 5 }),
      message({ role: "assistant", content: "Same.", ts: 6 }),
    ];
    const classification = {
      messages: [
        classified(supplied[0]!, "signoff"),
        classified(supplied[1]!, "assistant_imperative_closer"),
        classified(supplied[2]!, "signoff"),
        classified(supplied[3]!, "assistant_valediction"),
        classified(supplied[4]!, "signoff"),
        classified(supplied[5]!, "minimal_acknowledgment"),
      ],
      confidence: 0.96,
      rationale: "Repeated goodbye beats.",
      degraded: false,
    };

    const assessment = assessClosureLoopClassification({
      classification,
      suppliedMessages: supplied,
      currentUserRef: supplied[4]!.message_ref,
    });

    expect(assessment.closureLoopDetected).toBe(true);
    expect(assessment.mutualClosureCycles).toBe(3);
    expect(assessment.currentUserClosureShaped).toBe(true);
    expect(assessment.sourceStreamEntryIds).toHaveLength(6);
  });

  it("does not detect a normal substantive exchange", () => {
    const supplied = [
      message({ role: "user", content: "Can we debug the scheduler?", ts: 1 }),
      message({ role: "assistant", content: "Yes, start with the queue.", ts: 2 }),
      message({ role: "user", content: "The retry count is wrong.", ts: 3 }),
    ];
    const classification = {
      messages: [
        classified(supplied[0]!, "substantive"),
        classified(supplied[1]!, "substantive"),
        classified(supplied[2]!, "substantive"),
      ],
      confidence: 0.97,
      rationale: "The exchange is substantive.",
      degraded: false,
    };

    const assessment = assessClosureLoopClassification({
      classification,
      suppliedMessages: supplied,
      currentUserRef: supplied[2]!.message_ref,
    });

    expect(assessment.closureLoopDetected).toBe(false);
    expect(assessment.currentUserSubstantive).toBe(true);
  });

  it("counts only the contiguous closure-shaped suffix after substantive content", () => {
    const supplied = [
      message({ role: "user", content: "going", ts: 1 }),
      message({ role: "assistant", content: "Go.", ts: 2 }),
      message({ role: "user", content: "really going", ts: 3 }),
      message({ role: "assistant", content: "Talk soon.", ts: 4 }),
      message({ role: "user", content: "Actually, the scheduler is broken.", ts: 5 }),
      message({ role: "assistant", content: "Let's inspect the queue.", ts: 6 }),
      message({ role: "user", content: "phone down", ts: 7 }),
    ];
    const classification = {
      messages: [
        classified(supplied[0]!, "signoff"),
        classified(supplied[1]!, "assistant_imperative_closer"),
        classified(supplied[2]!, "signoff"),
        classified(supplied[3]!, "assistant_valediction"),
        classified(supplied[4]!, "substantive"),
        classified(supplied[5]!, "substantive"),
        classified(supplied[6]!, "signoff"),
      ],
      confidence: 0.96,
      rationale: "Substantive content interrupts the prior closure beats.",
      degraded: false,
    };

    const assessment = assessClosureLoopClassification({
      classification,
      suppliedMessages: supplied,
      currentUserRef: supplied[6]!.message_ref,
    });

    expect(assessment.closureLoopDetected).toBe(false);
    expect(assessment.mutualClosureCycles).toBe(0);
    expect(assessment.currentUserClosureShaped).toBe(true);
  });

  it("marks the next signoff after naming as closure-shaped for no_output routing", () => {
    const currentUserEntryId = createStreamEntryId();
    const supplied: ClosureLoopMessageForClassification[] = [
      {
        message_ref: currentUserEntryId,
        role: "user",
        content: "phone actually down",
        stream_entry_id: currentUserEntryId as StreamEntryId,
        ts: 10,
      },
    ];
    const classification = {
      messages: [classified(supplied[0]!, "signoff")],
      confidence: 0.95,
      rationale: "The current turn is another closure beat.",
      degraded: false,
    };

    const assessment = assessClosureLoopClassification({
      classification,
      suppliedMessages: supplied,
      currentUserRef: currentUserEntryId,
    });

    expect(assessment.currentUserClosureShaped).toBe(true);
    expect(assessment.currentUserSubstantive).toBe(false);
    expect(assessment.currentUserHasSubstantiveStateDelta).toBe(false);
    expect(assessment.closureLoopDetected).toBe(false);
  });

  it("keeps degraded short-turn heuristic fail-open when suppression is ambiguous", () => {
    const currentUserEntryId = createStreamEntryId();
    const supplied: ClosureLoopMessageForClassification[] = [
      {
        message_ref: currentUserEntryId,
        role: "user",
        content: "phone down",
        stream_entry_id: currentUserEntryId as StreamEntryId,
        ts: 10,
      },
    ];

    const assessment = assessDegradedClosureLoopFallback({
      suppliedMessages: supplied,
      currentUserRef: currentUserEntryId,
      priorClosureLoopActive: true,
    });

    expect(assessment.closureLoopDetected).toBe(false);
    expect(assessment.currentUserClosureShaped).toBe(false);
    expect(assessment.currentUserSubstantive).toBe(false);
    expect(assessment.currentUserHasSubstantiveStateDelta).toBe(false);
    expect(assessment.reason).toContain("suppression failed open");
  });
});
