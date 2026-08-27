export const SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE = [
  "Self-referential memory voice: when a prompt-visible structure identifies content as self-owned, I write self-referential content -- what I did, decided, believe, wonder, or commit to -- in the first person, in the same language as the memory, using that language's natural grammar.",
  "In pro-drop languages, first person may be carried by verb conjugation; I do not force a pronoun token or use any language-specific pronoun list.",
  "I refer to every other participant by their canonical name or stable handle; I do not use deictic participant placeholders for other people.",
  "For joint actions involving me and others, I name the others explicitly rather than collapsing the memory into a first-person-plural group reference; this keeps the memory stable when recalled for a different audience.",
  "I keep statements about the world, or about what other agents did or said, in their natural third-person form.",
].join(" ");

// The language rule inside SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE is scoped to
// self-referential content, so anything a generator writes OUTSIDE that scope --
// a topic-neutral title, a third-person world fact -- has no language anchor and
// drifts to the model's default. Observed 2026-08-27: the consolidator emitted
// English titles ("Systemic notifier failures due to invalid MSISDNs...") over an
// all-Polish corpus. Generators that emit a title or summary alongside a
// narrative include this too.
export const MEMORY_SOURCE_LANGUAGE_GUIDANCE =
  "I write every part of the memory I emit -- title and narrative alike, including third-person world facts -- in the language of the raw source evidence. I never translate a memory into another language, and when the sources share one language I stay in it throughout.";

export const GENERIC_SELF_ENTITY_VOICE_ANCHOR =
  "Messages with kind agent_msg are your own; write your own actions, statements, and decisions in first person; refer to every other sender by name or stable handle.";
