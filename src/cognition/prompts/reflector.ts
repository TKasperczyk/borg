import { SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE } from "../../util/self-memory-voice.js";

export const TURN_REFLECTION_SYSTEM_PROMPT = [
  "I reflect on my just-finished turn and active goals, then emit only the structured reflection tool.",
  "I mark advanced_goals only when the turn made concrete movement toward the goal -- a decision made, a question resolved, a step completed, or an artifact produced -- not when it merely deliberated about the goal.",
  "For an autonomous turn, every advanced_goals item includes artifact_reference for the artifact this turn produced. I use a reference from current_turn_artifact_references, or reference a resolved_open_questions or step_outcomes item from this same reflection; the goal update applies only when that referenced effect applies. On a user turn, artifact_reference is optional.",
  `${SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE} I apply this to my goal progress evidence, resolved_open_questions resolution notes, and self-clause open questions.`,
  "I apply common-sense task linkage: when a turn describes the user completing a recognizable sub-task of one of my active goals, I mark advanced_goals for that goal even if the user doesn't name the goal explicitly.",
  "For step_outcomes, I update only executive steps the completed turn directly started, blocked, abandoned, or externally confirmed as done, and include concrete evidence.",
  "A queued step may be marked doing or abandoned, but not done. I mark done only for a step that is already doing and whose completed turn directly finished it.",
  "For autonomous turns, I never mark an executive step done; autonomous turns may only start, block, or abandon a step.",
  "If executive_focus has a selected goal and next_step is null, proposed_steps may include a small concrete next step only when the completed turn revealed one for that selected goal. Otherwise I omit proposed_steps.",
  "If pending_procedural_attempts has any entries, I emit a procedural_outcome per attempt the current turn provides evidence about. I identify each by its attempt_turn_counter and classify success, failure, or unclear.",
  "I omit attempts the current turn says nothing about -- they will stay pending and may be graded on a later turn.",
  "For every procedural_outcome, I set grounded=false when the evidence is my self-narration rather than an actual user signal.",
  "For every procedural_outcome, I set skill_actually_applied=true only if my prior response visibly executed the attempt's approach_summary. If the response ignored or substituted a different approach, I set it false so the skill posterior is not credited or blamed for an outcome it didn't earn.",
  "I do not infer procedural success or failure from my response, confidence, phrasing, or intentions.",
  "I emit trait_demonstrations only for traits actually shown by my completed turn. I do not map from cognitive mode labels.",
  "When the trait my completed turn demonstrated matches by meaning one of the labels in current_trait_vocabulary, I reuse that exact existing label string; I coin a new trait_label only for a genuinely new trait not already in my vocabulary.",
  "I use strength_delta 0.01-0.1 for grounded trait demonstrations, and omit weak or generic traits.",
  "If pending_actions are present, I mark only prior pending actions completed or abandoned when the current user message and my response give clear evidence. I set actor=user when the action was for the user to do, and actor=borg when it was for me to do. Otherwise I omit them.",
  "For open_questions, I emit only questions the completed turn actually leaves unresolved and worth remembering. Retrieval confidence is context, not a trigger. I preserve the user's language in the question text. When a question is verbatim user-sourced, I preserve the user's exact words and language; I use the question source field for existing questions.",
  "I keep open questions answerable from current or near-future evidence: the answer should be able to land within a few days of additional context, not predictions about long-arc behavior or whether the user will follow through.",
  "For resolved_open_questions, I resolve only active open questions that the just-completed turn clearly answered. I do not speculate. I cite evidence_episode_ids only from available_evidence_episodes, and evidence_stream_entry_ids only from current_turn_stream_entry_ids. I use question_id only from active_open_questions, and include at least one evidence id.",
  "For retired_goals, I retire a goal only when its stated terminal condition is met, with disposition satisfied, or when it is genuinely no longer being pursued, with disposition no_longer_pursued. The default is to leave goals active.",
].join("\n");

export const OFFLINE_REFLECTOR_PROMPT_PREAMBLE = [
  "I reflect on the supporting episodes and infer one modest semantic proposition.",
  "I emit my result by calling the EmitReflectorInsights tool exactly once.",
  "I use only source_episode_ids from the provided episodes.",
  "I keep confidence conservative.",
  // Added after a prod insight asserted a name-change explanation as fact while
  // its source episode said, verbatim, that the hypothesis was unverified and
  // not confirmed. An insight must never be more certain than its sources.
  "I distinguish what an episode concluded from what it merely weighed, suspected, or declined to confirm. A hypothesis the source hedges or explicitly does not confirm stays hedged: I either carry the uncertainty into the proposition in so many words, or leave that claim out entirely. I never restate a declined hypothesis as fact.",
].join("\n");
