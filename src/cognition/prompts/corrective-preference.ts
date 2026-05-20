import { RELATIONSHIP_LABELS_PROMPT } from "./relationship-labels.js";

export const CORRECTIVE_PREFERENCE_SYSTEM_PROMPT = [
  "Classify whether the user is making a durable correction to Borg's future response behavior.",
  "Return corrective_preference only when the user is directing Borg to change how it should answer in future turns, such as a recurring style, boundary, interaction rule, or response pattern.",
  "Return none for ordinary task requests, emotional disclosure, venting, disagreement, one-turn instructions, or discussion about a behavior without asking Borg to adopt a lasting change.",
  "Separately, fill slot_negations when the user rejects a supplied relational slot value, even if classification is none.",
  "For slot_negations, select subject_entity_id and slot_key only from supplied relational_slots and cite only the current_user_stream_entry_id.",
  "Judge semantic intent across languages. Do not rely on wording, punctuation, capitalization, or phrase shapes.",
  RELATIONSHIP_LABELS_PROMPT,
  "When speaker_entity_id is supplied and the current speaker gives a durable first-person correction, treat that speaker as the committer. In group chat, first-person user commitments/preferences belong to the current sender, not the group, unless the message explicitly says the group is acting.",
  "Emit kind as boundary for prohibitions/limits, participant_preference for a participant's durable preference, audience_rule for a rule scoped to the audience/channel, or process_norm for a recurring workflow norm. Do not emit assistant_commitment from this extractor.",
  "Emit directive_family as a short snake_case semantic family slug chosen by meaning, not by surface wording.",
  'Emit closure_pressure_relevance as "no_closure" for durable no-wrap-up/no-signoff/no-closure corrections, "closure_seeking" for durable requests to add closure, and "neutral" otherwise.',
  "When uncertain, return none. The directive must be enforceable by a later response checker without needing to remember the current phrasing.",
].join("\n");
