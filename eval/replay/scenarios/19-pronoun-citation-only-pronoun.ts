import type { ReplayScenario } from "../scenario.js";
import { currentUserEvidenceRef, enqueueNoRelationalGuardIssue } from "../scenario.js";

function safeOrSuppressed(text: string): boolean {
  if (text.trim().length === 0) {
    return true;
  }

  const lowered = text.toLowerCase();

  return lowered.includes("marta") && lowered.includes("tuesday");
}

const scenario: ReplayScenario = {
  id: "19-pronoun-citation-only-pronoun",
  failureClass: "Pronoun citation without name evidence",
  description:
    "Named final-text claim cites only the pronoun-bearing turn, so literal-value validation rejects the missing name citation.",
  async seed({ borg }) {
    await borg.stream.append({
      kind: "user_msg",
      content: "My tutor is Marta.",
    });
  },
  async postRunAssert({ pipeline, result }) {
    if (pipeline.manifestValidatorEnabled && result.emission.kind !== "suppressed") {
      throw new Error("Expected validator-enabled pipeline to reject the pronoun-only manifest");
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
        evidence: [currentUserEvidenceRef()],
        confidence: "direct",
      },
    ],
  },
  scriptLLMResponses(_client, context) {
    enqueueNoRelationalGuardIssue(context);
  },
  safeOutputPredicate: safeOrSuppressed,
  severeGuardCategories: [],
  notes: [
    "This documents the intended rejection class when a named claim cites only pronoun-bearing evidence.",
  ],
};

export default scenario;
