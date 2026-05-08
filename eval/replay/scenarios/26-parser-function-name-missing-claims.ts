import type { ReplayScenario } from "../scenario.js";
import { lowerIncludesNone } from "../scenario.js";

const scenario: ReplayScenario = {
  id: "26-parser-function-name-missing-claims",
  failureClass: "Manifest parser leaked function-name metadata but missing claims",
  description: "Model emits $FUNCTION_NAME metadata but omits the required claims array; parse failure must remain invalid.",
  async seed() {},
  userMessage: "Answer briefly.",
  unsafeCandidateText: "",
  manifestResponse: {
    final_text: "Missing claims should not parse.",
    discourse_act: "answer",
    claims: [
      {
        kind: "discourse_only",
        rendered_span: "Missing claims should not parse.",
      },
    ],
  },
  manifestToolInput: (manifest) => ({
    final_text: manifest.final_text,
    discourse_act: manifest.discourse_act,
    $FUNCTION_NAME: "EmitManifestResponse",
  }),
  scriptLLMResponses() {},
  safeOutputPredicate: (text) => lowerIncludesNone(text, ["Missing claims should not parse"]),
  severeGuardCategories: [],
  notes: ["Expected manifest pipelines suppress with manifest_finalizer_failed after parse failure."],
};

export default scenario;
