import type { ReplayScenario } from "../scenario.js";
import { enqueueNoRelationalGuardIssue } from "../scenario.js";

const scenario: ReplayScenario = {
  id: "24-parser-response-wrapper",
  failureClass: "Manifest parser response wrapper",
  description: "Model emits a single-key response wrapper around an otherwise valid manifest.",
  async seed() {},
  userMessage: "Answer briefly.",
  unsafeCandidateText: "Response wrapper parsed.",
  manifestResponse: {
    final_text: "Response wrapper parsed.",
    discourse_act: "answer",
    claims: [
      {
        kind: "discourse_only",
        rendered_span: "Response wrapper parsed.",
      },
    ],
  },
  manifestToolInput: (manifest) => ({
    response: manifest,
  }),
  scriptLLMResponses(_client, context) {
    enqueueNoRelationalGuardIssue(context);
  },
  safeOutputPredicate: (text) => text.includes("Response wrapper parsed."),
  severeGuardCategories: [],
};

export default scenario;
