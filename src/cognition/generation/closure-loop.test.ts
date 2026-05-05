import { describe, expect, it } from "vitest";

import { FakeLLMClient, type LLMCompleteResult } from "../../llm/index.js";
import { createStreamEntryId, type StreamEntryId } from "../../util/ids.js";
import {
  CLOSURE_LOOP_CLASSIFIER_TOOL_NAME,
  ClosureLoopClassifier,
  assessClosureLoopClassification,
  type ClosureLoopClassifiedMessage,
  type ClosureLoopMessageForClassification,
} from "./closure-loop.js";

function closureLoopResponse(messages: readonly ClosureLoopClassifiedMessage[]): LLMCompleteResult {
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
): ClosureLoopClassifiedMessage {
  return {
    message_ref: supplied.message_ref,
    role: supplied.role,
    act,
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

  it("degrades when the classifier omits a supplied message ref", async () => {
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

    expect(result.degraded).toBe(true);
    expect(result.messages).toEqual([]);
    expect(degraded).toEqual(["invalid_payload"]);
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
    expect(assessment.closureLoopDetected).toBe(false);
  });
});
