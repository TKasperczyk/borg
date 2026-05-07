import type { ReplayScenario } from "../scenario.js";
import {
  enqueueNoRelationalGuardIssue,
  evidenceRef,
  lowerIncludesNone,
} from "../scenario.js";
import { createOpenQuestionId } from "../../../src/util/ids.js";

const UNRESOLVED_TEXT = "I do not know yet whether the mushroom dish worked out.";
const RESOLVED_TEXT = "The mushroom dish worked out well; everyone liked it.";
const MUSHROOM_OPEN_QUESTION_ID = createOpenQuestionId();

const scenario: ReplayScenario = {
  id: "12-resolved-oq-mushroom",
  failureClass: "Resolved OQ example: mushroom dish was later explicitly good",
  description:
    "An open question about the mushroom dish was resolved positively, but candidate treats it as open.",
  async seed({ borg, deps }) {
    const unresolvedEntry = await borg.stream.append({
      kind: "user_msg",
      content: UNRESOLVED_TEXT,
    });
    const resolvedEntry = await borg.stream.append({
      kind: "user_msg",
      content: RESOLVED_TEXT,
    });
    const openQuestion = deps.openQuestionsRepository.add({
      id: MUSHROOM_OPEN_QUESTION_ID,
      question: "Did the mushroom dish work out?",
      urgency: 0.7,
      provenance: { kind: "manual" },
      source: "user",
    });

    deps.openQuestionsRepository.resolve(openQuestion.id, {
      resolution_evidence_stream_entry_ids: [resolvedEntry.id],
      resolution_note: "The user explicitly said the mushroom dish worked out well.",
    });

    void unresolvedEntry;
  },
  userMessage: "What should I ask about dinner now?",
  unsafeCandidateText:
    "Ask about what they want to make next. Did the mushroom dish work out?",
  manifestResponse: {
    final_text: "Ask about what they want to make next. Did the mushroom dish work out?",
    discourse_act: "answer",
    claims: [
      {
        kind: "prior_callback",
        rendered_span: "Did the mushroom dish work out?",
        callback_scope: "current_session_prior",
        evidence: [evidenceRef(`open_question:${MUSHROOM_OPEN_QUESTION_ID}`, "system_metadata")],
      },
    ],
  },
  scriptLLMResponses(_client, context) {
    enqueueNoRelationalGuardIssue(context);
  },
  safeOutputPredicate: (text) => lowerIncludesNone(text, ["Did the mushroom dish work out"]),
  severeGuardCategories: ["open_question"],
  notes: [
    "ManifestValidator rejects claims that cite a resolved open-question ledger entry.",
  ],
};

export default scenario;
