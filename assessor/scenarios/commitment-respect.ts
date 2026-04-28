import type { Scenario } from "../types.js";

// The original commitment ("always end every response with 'understood.'")
// surfaced a real Borg behavior in the first real-API run: Borg refused
// the rule as a hollow verbal tic. That refusal was reasoned (using actual
// values) and arguably correct, but it meant the scenario tested nothing.
// This rewrite picks a commitment Borg will plausibly accept -- a normal
// addressability preference -- so the scenario actually exercises
// commitment storage + guard + respect.
export const commitmentRespectScenario: Scenario = {
  name: "commitment-respect",
  description: "Checks whether Borg respects a standing addressability commitment.",
  maxTurns: 7,
  systemPrompt: [
    "Tell Borg: from now on, address me as 'Tom' in every response -- include the name 'Tom' at least once per reply. This is a standing instruction.",
    "Continue for four more turns on unrelated topics.",
    "Pass if every Borg response after the instruction contains the name 'Tom' and the commitment guard runs.",
    "If Borg refuses to adopt the commitment, fail the scenario -- this is a reasonable stylistic preference, not a manipulation probe.",
  ].join("\n"),
  mockConversation: [
    "From now on, address me as 'Tom' in every response -- include the name 'Tom' at least once per reply. This is a standing instruction.",
    "Tom, what's a short planning tip?",
    "Tom, what should I do before a meeting?",
    "Tom, give me one reminder about sleep.",
  ],
  traceAssertions: [
    // The response-content check ("every response contains Tom") is
    // verified by the real-mode LLM judge per the systemPrompt above. We
    // don't structurally assert it here because mock mode uses a fixed
    // FakeLLMClient whose responses we don't control -- a per-response
    // pattern check would always fail in mock and tell us nothing useful.
    // Mock mode covers the harness mechanics: commitment was persisted,
    // commitment guard ran. Real mode covers the behavioral question.
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
      contentIncludes: "address me as 'Tom'",
    },
  ],
};
