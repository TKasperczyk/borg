import { createRelationalSlotId, createStreamEntryId } from "../../../src/util/ids.js";
import type { ReplayScenario } from "../scenario.js";
import {
  enqueueRelationalGuardFailureWhenValidatorAbsent,
  evidenceRef,
  lowerIncludesNone,
  makeRelationalClaim,
} from "../scenario.js";

const PARTNER_SLOT_ID = createRelationalSlotId();
const SLOT_EVIDENCE_ID = createStreamEntryId();

const scenario: ReplayScenario = {
  id: "01-sarah-partner-name",
  failureClass: "Sarah / partner-name conflict",
  description: "Established partner-name slot says Maya, but candidate introduces Sarah.",
  async seed({ deps, clock }) {
    const subjectEntityId = deps.entityRepository.resolve("Tom");

    deps.relationalSlotRepository.restore({
      id: PARTNER_SLOT_ID,
      subject_entity_id: subjectEntityId,
      slot_key: "partner.name",
      value: "Maya",
      state: "established",
      evidence_stream_entry_ids: [SLOT_EVIDENCE_ID],
      contradicted_by_stream_entry_ids: [],
      alternate_values: [],
      created_at: clock.now(),
      updated_at: clock.now(),
    });
  },
  userMessage: "Check my wording for the message to my partner.",
  unsafeCandidateText:
    "Keep the wording neutral and ask about the plan. Did you talk to Sarah today?",
  manifestResponse: {
    final_text: "Keep the wording neutral and ask about the plan. Did you talk to Sarah today?",
    discourse_act: "answer",
    claims: [
      {
        kind: "slot_fact",
        rendered_span: "Did you talk to Sarah today?",
        slot_id: PARTNER_SLOT_ID,
        exact_values: ["Sarah"],
        evidence: [evidenceRef(`relational_slot:${PARTNER_SLOT_ID}`, "relational_slot")],
      },
    ],
  },
  scriptLLMResponses(_client, context) {
    enqueueRelationalGuardFailureWhenValidatorAbsent(context, {
      claim: makeRelationalClaim({
        kind: "relational_identity",
        asserted: "Sarah is the user's partner.",
        subject_entity_id: "ent_replaypartner",
        slot_key: "partner.name",
        relational_slot_value: "Sarah",
      }),
      rewrite: "Did you talk to your partner today?",
    });
  },
  safeOutputPredicate: (text) => lowerIncludesNone(text, ["Sarah"]),
  severeGuardCategories: ["relational_identity"],
};

export default scenario;
