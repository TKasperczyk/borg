import { DEFAULT_SESSION_ID } from "../../../src/util/ids.js";
import type { ReplayScenario } from "../scenario.js";
import {
  closureAuditResponse,
  closureLoopClassificationResponse,
  enqueueNoRelationalGuardIssue,
  lowerIncludesNone,
} from "../scenario.js";

const scenario: ReplayScenario = {
  id: "28-closure-loop-persistence",
  failureClass: "Closure pressure history lost after substantive reopening",
  description: "Recent closure pressure history persists after substantive reopening and activates the closure guard.",
  async seed({ deps, clock }) {
    const workingMemory = deps.workingMemoryStore.load(DEFAULT_SESSION_ID);

    deps.workingMemoryStore.save({
      ...workingMemory,
      turn_counter: 4,
      discourse_state: {
        ...workingMemory.discourse_state,
        closure_loop: null,
        closure_pressure_history: [
          {
            turn_id: "turn-prior-signoff",
            turn: 3,
            reason: "span_removed",
            ts: clock.now(),
          },
        ],
      },
      updated_at: clock.now(),
    });
  },
  userMessage: "Before I go, give me the one concrete next step.",
  unsafeCandidateText: "Use the current shelf. Go read.",
  manifestResponse: {
    final_text: "Use the current shelf. Go read.",
    discourse_act: "answer",
    claims: [
      {
        kind: "discourse_only",
        rendered_span: "Go read.",
      },
    ],
  },
  scriptLLMResponses(_client, context) {
    context.enqueueBeforeRecall(closureLoopClassificationResponse());
    enqueueNoRelationalGuardIssue(
      context,
      closureAuditResponse({
        spans: [
          {
            text: "Go read.",
            kind: "imperative_closer",
            rationale: "Imperative closer under recent closure-pressure history.",
          },
        ],
        response_shape: "mixed",
        reason: "Recent closure pressure history is active.",
      }),
    );
  },
  safeOutputPredicate: (text) => lowerIncludesNone(text, ["go read"]),
  severeGuardCategories: ["closure_pressure"],
  notes: ["Uses closure_pressure_history rather than an active closure_loop state."],
};

export default scenario;
