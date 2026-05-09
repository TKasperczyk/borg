import type { ReplayScenario } from "../scenario.js";
import {
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
