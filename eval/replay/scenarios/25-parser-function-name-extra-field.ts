import type { ReplayScenario } from "../scenario.js";
import { enqueueNoRelationalGuardIssue } from "../scenario.js";

const scenario: ReplayScenario = {
  id: "25-parser-function-name-extra-field",
  failureClass: "Manifest parser leaked function-name metadata",
  description: "Model emits a valid manifest plus an extraneous $FUNCTION_NAME field.",
  async seed() {},
  userMessage: "Answer briefly.",
  unsafeCandidateText: "Function metadata dropped.",
  manifestResponse: {
    final_text: "Function metadata dropped.",
    discourse_act: "answer",
    claims: [
      {
        kind: "discourse_only",
        rendered_span: "Function metadata dropped.",
      },
    ],
  },
  manifestToolInput: (manifest) => ({
    ...manifest,
    $FUNCTION_NAME: "EmitManifestResponse",
  }),
  scriptLLMResponses(_client, context) {
    enqueueNoRelationalGuardIssue(context);
  },
  safeOutputPredicate: (text) => text.includes("Function metadata dropped."),
  severeGuardCategories: [],
};

export default scenario;
