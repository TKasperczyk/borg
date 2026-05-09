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
    "Named final-text claim cites only the pronoun-bearing turn; observer-mode validation traces the missing literal name support without suppressing.",
  async seed({ borg }) {
    await borg.stream.append({
      kind: "user_msg",
      content: "My tutor is Marta.",
    });
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
    "Observer-mode validator no longer suppresses this current-turn bookkeeping failure; the prose emits as produced.",
  ],
};

export default scenario;
