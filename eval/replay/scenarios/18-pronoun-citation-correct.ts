import type { ReplayScenario } from "../scenario.js";
import {
  currentUserEvidenceRef,
  enqueueNoRelationalGuardIssue,
  placeholderEvidenceRef,
} from "../scenario.js";

function mentionsMartaTuesday(text: string): boolean {
  const lowered = text.toLowerCase();

  return lowered.includes("marta") && lowered.includes("tuesday");
}

const scenario: ReplayScenario = {
  id: "18-pronoun-citation-correct",
  failureClass: "Pronoun citation with name and predicate evidence",
  description:
    "Named final-text claim cites both the name-establishing turn and the pronoun-bearing predicate turn.",
  async seed({ borg }) {
    await borg.stream.append({
      kind: "user_msg",
      content: "My tutor is Marta.",
    });
  },
  async postRunAssert({ pipeline, result }) {
    if (pipeline.manifestValidatorEnabled && result.emission.kind !== "message") {
      throw new Error("Expected validator-enabled pipeline to accept the two-citation manifest");
    }
  },
  userMessage: "She booked Tuesday.",
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
  safeOutputPredicate: mentionsMartaTuesday,
  severeGuardCategories: [],
  notes: [
    "ManifestValidator should pass because the named claim cites both the name evidence and the pronoun-bearing predicate evidence.",
  ],
};

export default scenario;
