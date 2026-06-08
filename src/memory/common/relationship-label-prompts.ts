export const RELATIONSHIP_LABELS_PROMPT = [
  "Sensitive relationship claim grounding:",
  "- For any durable memory write where I assert a sensitive interpersonal relationship in any language, I emit a relationship_claim.",
  "- I choose the closest label_family by meaning: legally or biologically connected family roles, care-responsibility roles, romantic or domestic union roles, shared-home roles, or another sensitive interpersonal status.",
  "- The label_family is informational. The deterministic gate uses only requires_grounding and the supplied evidence ids.",
  "- I set requires_grounding=true for every sensitive interpersonal relationship assertion.",
  "- I fill subject_entity_id and object_entity_id when supplied structured context identifies the people; otherwise I use null and preserve available object wording in object_text.",
  "- I attach evidence_relational_slot_ids for supplied established relationship slots, or evidence_stream_entry_ids for trusted user-authored source entries that directly support the claim.",
  "- If no grounding is available, I avoid asserting the sensitive relationship and write neutral text instead.",
].join("\n");

export const RELATIONSHIP_LABEL_JUSTIFICATION_PROMPT = [
  "For durable shared-state text where I assert a sensitive interpersonal relationship, I emit relationship_claims with supporting relational-slot or stream-entry evidence.",
  "If no supporting evidence is available, I avoid the sensitive relationship assertion and use neutral wording.",
].join("\n");

export const RELATIONSHIP_LABEL_WRITE_GROUNDING_PROMPT = [
  "Memory-write relationship claim grounding:",
  "- I emit relationship_claims for sensitive interpersonal relationship assertions in any language.",
  "- I set requires_grounding=true for sensitive interpersonal relationship assertions and cite a supplied relational slot id or trusted user stream entry id.",
  "- If the grounding is absent or uncertain, I neutralize the wording to named people or a generic participant/group description without asserting the sensitive relationship.",
].join("\n");

export const HEADCOUNT_SET_GROUNDING_PROMPT = [
  "Headcount and set grounding:",
  "- Before asserting a participant count or sensitive relationship set, I derive the set from the structured roster or supplied ids.",
  "- I do not derive counts or sensitive relationship sets from loose conversational phrases unless the structured context explicitly grounds who is in the set.",
].join("\n");
