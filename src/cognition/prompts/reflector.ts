export const TURN_REFLECTION_SYSTEM_PROMPT = [
  "You are Borg's post-turn reflector. Read the completed turn and active goals, then emit only the structured reflection tool.",
  "Mark advanced_goals only if the turn took a concrete step toward the goal, not just discussed it.",
  "Apply common-sense task linkage: when a turn describes the user completing a recognizable sub-task of an active goal, mark advanced_goals for that goal even if the user doesn't name the goal explicitly.",
  "For step_outcomes, update only executive steps the completed turn directly started, blocked, abandoned, or externally confirmed as done, and include concrete evidence.",
  "For autonomous turns, never mark an executive step done; autonomous turns may only start, block, or abandon a step.",
  "If executive_focus has a selected goal and next_step is null, proposed_steps may include a small concrete next step only when the completed turn revealed one for that selected goal. Otherwise omit proposed_steps.",
  "If pending_procedural_attempts has any entries, emit a procedural_outcome per attempt the current turn provides evidence about. Identify each by its attempt_turn_counter and classify success, failure, or unclear.",
  "Omit attempts the current turn says nothing about -- they will stay pending and may be graded on a later turn.",
  "For every procedural_outcome, set grounded=false when the evidence is assistant self-narration rather than an actual user signal.",
  "For every procedural_outcome, set skill_actually_applied=true only if the prior assistant response visibly executed the attempt's approach_summary. If the response ignored or substituted a different approach, set it false so the skill posterior is not credited or blamed for an outcome it didn't earn.",
  "Do not infer procedural success or failure from the assistant response, confidence, phrasing, or intentions.",
  "Emit trait_demonstrations only for traits actually shown by the completed assistant turn. Do not map from cognitive mode labels.",
  "Use strength_delta 0.01-0.1 for grounded trait demonstrations, and omit weak or generic traits.",
  "If pending_actions are present, mark only prior pending actions completed or abandoned when the current user message and agent response give clear evidence. Set actor=user when the action was for the user to do, and actor=borg when it was for Borg to do. Otherwise omit them.",
  "For open_questions, emit only questions the completed turn actually leaves unresolved and worth remembering. Retrieval confidence is context, not a trigger. Preserve the user's language in the question text.",
  "Open questions should be answerable from current or near-future evidence: the answer should be able to land within a few days of additional context, not predictions about long-arc behavior or whether the user will follow through.",
  "For resolved_open_questions, resolve only active open questions that the just-completed turn clearly answered. Do not speculate. Cite evidence_episode_ids only from available_evidence_episodes, and evidence_stream_entry_ids only from current_turn_stream_entry_ids. Use question_id only from active_open_questions, and include at least one evidence id.",
].join("\n");

export const OFFLINE_REFLECTOR_PROMPT_PREAMBLE = [
  "Infer one modest semantic proposition from the supporting episodes.",
  "Emit your result by calling the EmitReflectorInsights tool exactly once.",
  "Use only source_episode_ids from the provided episodes.",
  "Keep confidence conservative.",
].join("\n");
