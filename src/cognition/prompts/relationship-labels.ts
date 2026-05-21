export const RELATIONSHIP_LABELS = [
  "sibling",
  "partner",
  "spouse",
  "parent",
  "child",
  "caregiver",
  "manager",
  "owner",
  "patient",
  "doctor",
  "teammate",
  "stakeholder",
] as const;

const RELATIONSHIP_LABEL_MATCH_TERMS = [
  ...RELATIONSHIP_LABELS,
  "siblings",
  "spouses",
  "parents",
  "children",
  "caregivers",
  "managers",
  "owners",
  "patients",
  "doctors",
  "teammates",
  "stakeholders",
] as const;

function escapeRegExp(value: string): string {
  return value.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&");
}

export function protectedRelationshipLabelsInText(text: string): string[] {
  return RELATIONSHIP_LABEL_MATCH_TERMS.filter((label) =>
    new RegExp(`(^|[^\\p{L}\\p{N}_])${escapeRegExp(label)}([^\\p{L}\\p{N}_]|$)`, "iu").test(
      text,
    ),
  );
}

export const RELATIONSHIP_LABELS_PROMPT = [
  "Relationship label grounding:",
  `- Relationship labels needing direct support: ${RELATIONSHIP_LABELS.join(", ")}.`,
  "- Use one only when direct evidence supports it: a relational slot, an explicit user statement, a trusted participant profile, or a directly sourced prior memory.",
  '- If uncertain, use neutral wording such as "participant", "family member", "person in the thread", or name the people without assigning a role.',
  "- When a Thread roster is present, use it as compact structured context for participant membership, known relationship slots, and non-chat subjects; do not infer protected labels from vague group phrasing.",
].join("\n");

export const RELATIONSHIP_LABEL_JUSTIFICATION_PROMPT = [
  "For any durable shared-state text that uses a relationship label, cite the supporting relational slot or stream entry in source_stream_entry_ids.",
  "If no supporting slot or source entry is available, avoid the relationship label and use neutral wording.",
].join("\n");

export const RELATIONSHIP_LABEL_WRITE_GROUNDING_PROMPT = [
  "Memory-write relationship label grounding:",
  "- When emitting durable memory text whose label, description, or entry text contains a protected relationship label, ground it in a supplied relational slot id, explicit user statement, or trusted profile.",
  "- If the grounding is absent or uncertain, neutralize the wording to a domain-generic set term such as participants, group members, or named people without assigning a protected role.",
].join("\n");

export const HEADCOUNT_SET_GROUNDING_PROMPT = [
  "Headcount and set grounding:",
  '- Before asserting a participant count or set label such as "the N of you", "attendees", "members", or any protected relationship-set label, derive the set from the structured roster or supplied ids.',
  '- Do not derive counts or relationship sets from loose conversational phrases such as "us four" unless the structured context explicitly grounds who is in the set.',
].join("\n");
