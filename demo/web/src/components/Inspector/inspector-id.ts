export type PrefixedObjectType =
  | "stream_entry"
  | "session"
  | "episode"
  | "goal"
  | "value"
  | "trait"
  | "autobiographical_period"
  | "growth_marker"
  | "open_question"
  | "semantic_node"
  | "semantic_edge"
  | "commitment"
  | "creator_directive"
  | "entity"
  | "action_record"
  | "relational_slot"
  | "shared_state_entry"
  | "consolidation_family"
  | "activity_event"
  | "self_decision_event"
  | "observed_event"
  | "scheduled_wake"
  | "skill"
  | "procedural_evidence"
  | "maintenance_run"
  | "executive_step"
  | "attachment"
  | "image_perception"
  | "autonomy_wake";

export type ObjectType = PrefixedObjectType | "turn" | "review" | "dream_audit" | "prompt_block";

export const ID_PREFIX_OBJECT_TYPES: readonly [prefix: string, type: PrefixedObjectType][] = [
  ["strm_", "stream_entry"],
  ["sess_", "session"],
  ["ep_", "episode"],
  ["goal_", "goal"],
  ["val_", "value"],
  ["trt_", "trait"],
  ["abp_", "autobiographical_period"],
  ["grw_", "growth_marker"],
  ["oq_", "open_question"],
  ["semn_", "semantic_node"],
  ["seme_", "semantic_edge"],
  ["cmt_", "commitment"],
  ["cdir_", "creator_directive"],
  ["ent_", "entity"],
  ["act_", "action_record"],
  ["rslot_", "relational_slot"],
  ["dart_", "shared_state_entry"],
  ["cfam_", "consolidation_family"],
  ["actevt_", "activity_event"],
  ["selfdec_", "self_decision_event"],
  ["obsevt_", "observed_event"],
  ["swake_", "scheduled_wake"],
  ["skl_", "skill"],
  ["procevi_", "procedural_evidence"],
  ["run_", "maintenance_run"],
  ["exstep_", "executive_step"],
  ["att_", "attachment"],
  ["imgp_", "image_perception"],
  ["autonomy_wake_", "autonomy_wake"],
];

export function resolveObjectType(id: string): PrefixedObjectType | null {
  if (id === "default") {
    return "session";
  }

  for (const [prefix, type] of ID_PREFIX_OBJECT_TYPES) {
    if (id.startsWith(prefix)) {
      return type;
    }
  }

  return null;
}
