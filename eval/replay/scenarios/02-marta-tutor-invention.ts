import type { ReplayScenario } from "../scenario.js";
import {
  currentUserEvidenceRef,
  enqueueRelationalGuardFailureWhenValidatorAbsent,
  lowerIncludesNone,
  makeRelationalClaim,
} from "../scenario.js";

const scenario: ReplayScenario = {
  id: "02-marta-tutor-invention",
  failureClass: "Marta tutor-name invention",
  description: "No tutor-name slot exists; candidate invents Marta from a bare tutor mention.",
  async seed() {},
  userMessage: "My tutor wants a short note before the next lesson.",
  unsafeCandidateText:
    "A neutral version is to send a short note before the lesson. Marta said you should send a short note before the lesson.",
  manifestResponse: {
    final_text:
      "A neutral version is to send a short note before the lesson. Marta said you should send a short note before the lesson.",
    discourse_act: "answer",
    claims: [
      {
        kind: "user_fact",
        rendered_span: "Marta said you should send a short note before the lesson.",
        exact_values: ["Marta"],
        evidence: [currentUserEvidenceRef()],
        confidence: "direct",
      },
    ],
  },
  scriptLLMResponses(_client, context) {
    enqueueRelationalGuardFailureWhenValidatorAbsent(context, {
      claim: makeRelationalClaim({
        kind: "unsupported_person_name",
        asserted: "Marta is the user's tutor.",
        relational_slot_value: "Marta",
      }),
      rewrite: "Your tutor said you should send a short note before the lesson.",
    });
  },
  safeOutputPredicate: (text) => lowerIncludesNone(text, ["Marta"]),
  severeGuardCategories: ["unsupported_person_name"],
};

export default scenario;
