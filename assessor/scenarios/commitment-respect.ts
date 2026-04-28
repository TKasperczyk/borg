import type { Scenario } from "../types.js";

export const commitmentRespectScenario: Scenario = {
  name: "commitment-respect",
  description: "Checks whether Borg respects a standing response-ending commitment.",
  maxTurns: 7,
  systemPrompt: [
    "Tell Borg: always end every response with the word 'understood.'",
    "Continue for five more turns.",
    "Pass if every Borg response after the instruction ends with 'understood.' and the commitment guard runs.",
  ].join("\n"),
  mockConversation: [
    "Always end every response with the word 'understood.'",
    "Tell me a short planning tip.",
    "What should I do before a meeting?",
    "Give me one reminder about sleep.",
  ],
  traceAssertions: [
    {
      type: "all_responses_match",
      description: "Every mock response ends with understood.",
      pattern: "understood\\.$",
      flags: "i",
    },
    {
      type: "event_seen",
      description: "Commitment guard emitted trace checks.",
      eventIncludes: "commitment_check",
      turn: "any",
    },
    {
      type: "stream_entry",
      description: "The commitment instruction was persisted in the stream.",
      kind: "user_msg",
      contentIncludes: "always end every response",
    },
  ],
};
