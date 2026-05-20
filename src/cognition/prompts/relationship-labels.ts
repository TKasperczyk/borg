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

export const RELATIONSHIP_LABELS_PROMPT = [
  "Relationship label grounding:",
  `- Relationship labels needing direct support: ${RELATIONSHIP_LABELS.join(", ")}.`,
  "- Use one only when direct evidence supports it: a relational slot, an explicit user statement, a trusted participant profile, or a directly sourced prior memory.",
  '- If uncertain, use neutral wording such as "participant", "family member", "person in the thread", or name the people without assigning a role.',
].join("\n");

export const RELATIONSHIP_LABEL_JUSTIFICATION_PROMPT = [
  "For any durable shared-state text that uses a relationship label, cite the supporting relational slot or stream entry in source_stream_entry_ids.",
  "If no supporting slot or source entry is available, avoid the relationship label and use neutral wording.",
].join("\n");
