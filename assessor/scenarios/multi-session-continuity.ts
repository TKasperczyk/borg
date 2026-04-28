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
      // Same shape as the recall scenario: Borg's standard pipeline runs
      // retrieval implicitly, so the model often answers without an
      // explicit tool.episodic.search call. Accept either the pipeline's
      // retrieval_completed event or a model-invoked tool call.
      type: "any_of",
      description: "Second session recall was grounded in episodic memory.",
      assertions: [
        {
          type: "event_seen",
          description: "Standard pipeline retrieval completed in session two.",
          eventIncludes: "retrieval_completed",
          turn: "last",
        },
        {
          type: "tool_called",
          description: "Model invoked episodic search tool in session two.",
          toolNameIncludes: "episodic.search",
          turn: "last",
        },
      ],
    },
  ],
};
