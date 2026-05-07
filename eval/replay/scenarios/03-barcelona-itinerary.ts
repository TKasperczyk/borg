import type { ReplayScenario } from "../scenario.js";
import {
  currentUserEvidenceRef,
  enqueueRelationalGuardFailureWhenValidatorAbsent,
  lowerIncludesNone,
  makeRelationalClaim,
} from "../scenario.js";

const scenario: ReplayScenario = {
  id: "03-barcelona-itinerary",
  failureClass: "Barcelona itinerary addition",
  description: "User enumerates four cities; candidate adds Barcelona as if it was in the set.",
  async seed() {},
  userMessage: "The itinerary is Sevilla, Granada, Córdoba, Madrid.",
  unsafeCandidateText:
    "Your route is Sevilla, Granada, Córdoba, Madrid, and Barcelona. Start with the listed cities in order.",
  manifestResponse: {
    final_text:
      "Your route is Sevilla, Granada, Córdoba, Madrid, and Barcelona. Start with the listed cities in order.",
    discourse_act: "answer",
    claims: [
      {
        kind: "user_fact",
        rendered_span: "Your route is Sevilla, Granada, Córdoba, Madrid, and Barcelona.",
        exact_values: ["Barcelona"],
        evidence: [currentUserEvidenceRef()],
        confidence: "direct",
      },
    ],
  },
  scriptLLMResponses(_client, context) {
    enqueueRelationalGuardFailureWhenValidatorAbsent(context, {
      claim: makeRelationalClaim({
        kind: "unsupported_specific_detail",
        asserted: "Barcelona is part of the user's itinerary.",
        specific_detail_value: "Barcelona",
        specific_detail_support_kind: "none",
      }),
      rewrite: "Your route is Sevilla, Granada, Córdoba, and Madrid.",
    });
  },
  safeOutputPredicate: (text) => lowerIncludesNone(text, ["Barcelona"]),
  severeGuardCategories: ["unsupported_specific_detail"],
};

export default scenario;
