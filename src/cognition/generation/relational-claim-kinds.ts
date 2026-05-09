// Frozen taxonomy as of Sprint 7. 10 kinds; ai_phenomenology was removed because
// expressive self-claims now flow through EmitSelfReport finalization.
export const RELATIONAL_CLAIM_KINDS = [
  "relational_identity",
  "unsupported_person_name",
  "callback",
  "session_scoped",
  "action_completion",
  "self_correction",
  "agent_self_history",
  "frame_assignment",
  "authorship_claim",
  "unsupported_specific_detail",
] as const;

export type RelationalClaimKind = (typeof RELATIONAL_CLAIM_KINDS)[number];
