/**
 * Strict tier protects kinship and caregiver labels because asserting these in
 * durable memory creates persistent identity claims that are hard to walk back.
 * Medical and professional role labels such as doctor, patient, therapist, and
 * lawyer are not in this tier because they commonly appear as logistics or
 * context nouns, where deterministic prose matching cannot distinguish role
 * assignment from context. Person-role medical assignments should rely on
 * prompt guidance and future structured claim extraction.
 */
export const STRICT_RELATIONSHIP_LABELS = [
  "sibling",
  "spouse",
  "parent",
  "child",
  "caregiver",
] as const;

export const CONTEXTUAL_RELATIONSHIP_LABELS = [
  "partner",
  "manager",
  "owner",
  "teammate",
  "stakeholder",
] as const;

export const PROTECTED_RELATIONSHIP_LABELS = [...STRICT_RELATIONSHIP_LABELS] as const;

const RELATIONSHIP_LABEL_MATCH_TERMS = [
  ...PROTECTED_RELATIONSHIP_LABELS,
  "siblings",
  "spouses",
  "parents",
  "children",
  "caregivers",
] as const;

function escapeRegExp(value: string): string {
  return value.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&");
}

export function protectedRelationshipLabelsInText(text: string): string[] {
  return RELATIONSHIP_LABEL_MATCH_TERMS.filter((label) =>
    new RegExp(`(^|[^\\p{L}\\p{N}_])${escapeRegExp(label)}([^\\p{L}\\p{N}_]|$)`, "iu").test(text),
  );
}

export const RELATIONSHIP_LABELS_PROMPT = [
  "Relationship label grounding:",
  `- Strict labels requiring direct support on memory writes: ${PROTECTED_RELATIONSHIP_LABELS.join(", ")}.`,
  `- Contextual role labels are prompt guidance only, not hard-gated: ${CONTEXTUAL_RELATIONSHIP_LABELS.join(", ")}.`,
  "- Contextual labels commonly appear in project, business, organizational, social, or game contexts. Use them naturally when describing grounded roles.",
  '- Context nouns describing logistics, appointments, scheduling, paperwork, portals, calls, visits, or other administrative or operational mentions of medical or professional services do not require relationship evidence. Examples: "doctor appointment", "doctor calls", "patient portal", "patient paperwork", "lawyer call", "therapist visit" are administrative context, not person-to-person role assignments.',
  'Person-to-person role assignments require grounding when using strict kinship or caregiver labels. Examples: "<person A>\'s sibling is <person B>", "<person A> is <person B>\'s caregiver", and "<person A>\'s spouse is <person B>" require relationship_evidence_relational_slot_ids or relationship_evidence_stream_entry_ids.',
  "- Use strict labels only when direct evidence supports them: a relational slot, an explicit user statement, a trusted participant profile, or a directly sourced prior memory.",
  '- If uncertain, use neutral wording such as "participant", "family member", "person in the thread", or name the people without assigning a role.',
  "- When a Thread roster is present, use it as compact structured context for participant membership, known relationship slots, and non-chat subjects; do not infer strict labels from vague group phrasing.",
].join("\n");

export const RELATIONSHIP_LABEL_JUSTIFICATION_PROMPT = [
  "For any durable shared-state text that uses a strict kinship or caregiver relationship label, cite the supporting relational slot or stream entry in source_stream_entry_ids.",
  "If no supporting slot or source entry is available, avoid the strict label and use neutral wording.",
].join("\n");

export const RELATIONSHIP_LABEL_WRITE_GROUNDING_PROMPT = [
  "Memory-write relationship label grounding:",
  "- When emitting durable memory text whose label, description, or entry text contains a strict kinship or caregiver relationship label, ground it in a supplied relational slot id, explicit user statement, or trusted profile.",
  '- Administrative or operational context nouns do not need relationship evidence: "doctor appointment", "doctor calls", "patient portal", "patient paperwork", "lawyer call", and "therapist visit" are context, not role assignments.',
  "- If the grounding is absent or uncertain, neutralize the wording to a domain-generic set term such as participants, group members, or named people without assigning a strict role.",
  "- Contextual role labels such as partner, manager, owner, teammate, and stakeholder may be used naturally for grounded project, organizational, relationship, or game roles.",
].join("\n");

export const HEADCOUNT_SET_GROUNDING_PROMPT = [
  "Headcount and set grounding:",
  '- Before asserting a participant count or set label such as "the N of you", "attendees", "members", or any strict relationship-set label, derive the set from the structured roster or supplied ids.',
  '- Do not derive counts or strict relationship sets from loose conversational phrases such as "us four" unless the structured context explicitly grounds who is in the set.',
].join("\n");
