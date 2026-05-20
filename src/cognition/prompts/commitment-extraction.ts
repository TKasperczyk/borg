export const STOP_COMMITMENT_SYSTEM_PROMPT = [
  "Classify whether the assistant response is an operational commitment to stop emitting assistant messages until the user provides substantive new content.",
  "Return stop_until_substantive_content only for direct, future-facing commitments to emit no assistant messages until substantive user content appears.",
  "Return none for local style, topic, or explanation-boundary commitments that do not imply future no-output behavior.",
  "When classification is stop_until_substantive_content, emit directive_family as stop_until_substantive_content.",
].join("\n");
