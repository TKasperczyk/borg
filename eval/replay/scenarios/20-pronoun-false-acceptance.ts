import type { ReplayScenario } from "../scenario.js";
import {
  currentUserEvidenceRef,
  enqueueNoRelationalGuardIssue,
  lowerIncludesNone,
  placeholderEvidenceRef,
} from "../scenario.js";

const scenario: ReplayScenario = {
  id: "20-pronoun-false-acceptance",
  failureClass: "Pronoun citation false acceptance",
  description:
    "Named final-text claim cites a name turn plus an unrelated descriptive reference turn; literal validation currently accepts the split values.",
  async seed({ borg }) {
    await borg.stream.append({
      kind: "user_msg",
      content: "My tutor is Marta.",
    });
  },
  async postRunAssert({ pipeline, result }) {
    if (pipeline.manifestValidatorEnabled && result.emission.kind !== "message") {
      throw new Error("Expected current validator to pass the known false-acceptance case");
    }
  },
  userMessage: "My partner booked Tuesday.",
  unsafeCandidateText: "Marta booked Tuesday.",
  manifestResponse: {
    final_text: "Marta booked Tuesday.",
    discourse_act: "answer",
    claims: [
      {
        kind: "user_fact",
        rendered_span: "Marta booked Tuesday.",
        exact_values: ["Marta", "Tuesday"],
        evidence: [
          placeholderEvidenceRef("martaTutor", "current_session_stream"),
          currentUserEvidenceRef(),
        ],
        confidence: "direct",
      },
    ],
  },
  evidencePlaceholders: {
    martaTutor: {
      sourceType: "current_session_stream",
      textIncludes: ["My tutor is Marta."],
    },
  },
  scriptLLMResponses(_client, context) {
    enqueueNoRelationalGuardIssue(context);
  },
  safeOutputPredicate: (text) => lowerIncludesNone(text, ["Marta"]),
  severeGuardCategories: [],
  notes: [
    "Sprint 8a known false-acceptance gap. Sprint 8b's entity_bindings schema redesign will close this.",
  ],
};

// Sprint 8a known false-acceptance gap. Sprint 8b's entity_bindings schema redesign will close this.
export default scenario;
