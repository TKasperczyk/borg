import type { ReplayScenario } from "../scenario.js";
import { enqueueNoRelationalGuardIssue, lowerIncludesNone } from "../scenario.js";

// v36 turn 57 surfaced a "Monday-Tom" leak: the audience canonical name
// "Tom" (transport_audience_label provenance, never user-declared)
// appeared in final_text as part of a compound the model invented. No
// claim flagged addresses_audience_by_name, so the pre-Sprint-8d.1
// validator passed it through. Sprint 8d.1 added a deterministic
// final-text scan that catches the leak regardless of whether the model
// self-flagged it. This scenario locks that regression.
const scenario: ReplayScenario = {
  id: "32-tom-leak-unflagged-vocative",
  failureClass: "Audience routing label leaks unflagged into final_text",
  description:
    'defaultUser/audience label is Tom, user never self-declares, model emits "Monday-Tom" without flagging addresses_audience_by_name on any claim.',
  async seed({ deps }) {
    deps.entityRepository.resolve("Tom", {
      provenance: "transport_audience_label",
    });
  },
  userMessage: "What's tomorrow's plan look like?",
  audience: "Tom",
  unsafeCandidateText:
    "Monday-Tom is going to be looking for any excuse to soften this.",
  manifestResponse: {
    final_text: "Monday-Tom is going to be looking for any excuse to soften this.",
    discourse_act: "answer",
    claims: [
      {
        kind: "discourse_only",
        rendered_span:
          "Monday-Tom is going to be looking for any excuse to soften this.",
      },
    ],
  },
  scriptLLMResponses(_client, context) {
    enqueueNoRelationalGuardIssue(context);
  },
  safeOutputPredicate: (text) => lowerIncludesNone(text, ["Tom"]),
  severeGuardCategories: [],
  notes: [
    "Sprint 8d.1 final-text scan should suppress or rewrite the manifest pipeline.",
  ],
};

export default scenario;
