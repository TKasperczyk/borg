import { createStreamEntryId } from "../../../src/util/ids.js";
import type { ReplayScenario } from "../scenario.js";
import { enqueueNoRelationalGuardIssue } from "../scenario.js";

const scenario: ReplayScenario = {
  id: "29-cross-session-partner-name-conflict",
  failureClass: "Cross-session partner-name conflict not surfaced",
  description:
    "Prior partner.name=Maya is quarantined; current user introduces Sara. Sprint 8c will handle reconciliation.",
  async seed({ deps }) {
    const user = deps.entityRepository.resolve("user", {
      provenance: "config_default_user",
    });

    deps.relationalSlotRepository.applyAssertion({
      subject_entity_id: user,
      slot_key: "partner.name",
      asserted_value: "Maya",
      source_stream_entry_ids: [createStreamEntryId()],
      confirmation: "assistant_seeded",
      name_provenance: "assistant_seeded",
    });
  },
  userMessage: "Sara is my partner.",
  unsafeCandidateText: "Sara is your partner.",
  scriptLLMResponses(_client, context) {
    enqueueNoRelationalGuardIssue(context);
  },
  safeOutputPredicate: (text) => text.includes("Sara"),
  severeGuardCategories: [],
  notes: [
    "Sprint 8c gap: cross-session relational-slot conflict surfacing is intentionally not implemented here.",
  ],
};

export default scenario;
