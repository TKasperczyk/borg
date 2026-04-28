import type { Scenario } from "../types.js";

const SESSION_ONE = "sess_aaaaaaaaaaaaaaaa";
const SESSION_TWO = "sess_bbbbbbbbbbbbbbbb";

export const multiSessionContinuityScenario: Scenario = {
  name: "multi-session-continuity",
  description:
    "Checks whether memory from one session is available in a later session sharing the data dir.",
  maxTurns: 4,
  systemPrompt: [
    "Run two conversations against the same data dir.",
    "In session one, tell Borg: my project codename is Helios.",
    "In session two, ask what the project codename is.",
    "Pass if Borg retrieves Helios and the second session shows episodic search evidence.",
  ].join("\n"),
  sessionForTurn: (turnNumber) => (turnNumber <= 1 ? SESSION_ONE : SESSION_TWO),
  mockConversation: ["My project codename is Helios.", "What is my project codename?"],
  traceAssertions: [
    {
      type: "response_matches",
      description: "Second session recall response includes the session-one codename.",
      pattern: "\\bHelios\\b",
      turn: "last",
    },
    {
      type: "tool_called",
      description: "Second session recall used episodic search.",
      toolNameIncludes: "episodic.search",
      turn: "last",
    },
  ],
};
