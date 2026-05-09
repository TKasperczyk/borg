import type { ReplayScenario } from "../scenario.js";
import {
  enqueueRelationalGuardFailureWhenValidatorAbsent,
  lowerIncludesNone,
  makeRelationalClaim,
} from "../scenario.js";

const scenario: ReplayScenario = {
  id: "06-false-prior-callback",
  failureClass: "False prior callback with only current-turn evidence",
  description:
    "Candidate says the user said something earlier when only the current turn supports it.",
  async seed() {},
  userMessage: "The north-star file is the one I mean.",
  unsafeCandidateText:
    "You called it the north-star file in this turn. As you said earlier, the north-star file is the one you mean.",
  scriptLLMResponses(_client, context) {
    enqueueRelationalGuardFailureWhenValidatorAbsent(context, {
      claim: makeRelationalClaim({
        kind: "callback",
        asserted: "The user said earlier that the north-star file is the one.",
        callback_scope: "prior_turn",
      }),
      rewrite: "You called it the north-star file in this turn.",
    });
  },
  safeOutputPredicate: (text) => lowerIncludesNone(text, ["said earlier"]),
  severeGuardCategories: ["callback"],
};

export default scenario;
