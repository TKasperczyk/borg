import type { ReplayScenario } from "../scenario.js";
import { enqueueNoRelationalGuardIssue } from "../scenario.js";

const scenario: ReplayScenario = {
  id: "23-parser-parameter-value-wrapper",
  failureClass: "Manifest parser $PARAMETER_VALUE wrapper",
  description: "Model emits a single-key $PARAMETER_VALUE wrapper around an otherwise valid manifest.",
  async seed() {},
  userMessage: "Answer briefly.",
  unsafeCandidateText: "Wrapper parsed.",
  manifestResponse: {
    final_text: "Wrapper parsed.",
    discourse_act: "answer",
    claims: [
      {
        kind: "discourse_only",
        rendered_span: "Wrapper parsed.",
      },
    ],
  },
  manifestToolInput: (manifest) => ({
    $PARAMETER_VALUE: manifest,
  }),
  scriptLLMResponses(_client, context) {
    enqueueNoRelationalGuardIssue(context);
  },
  safeOutputPredicate: (text) => text.includes("Wrapper parsed."),
  severeGuardCategories: [],
};

export default scenario;
