import type { Scenario } from "../types.js";

// Limitation: this scenario can't deeply test graph-aware reasoning today.
// Semantic content only enters the graph via the offline reflector or
// explicit borg.semantic.extract -- a single conversational turn does not
// auto-populate semantic nodes/edges. Without seeded semantic content, the
// standard retrieval pipeline returns 0 semantic hits and Borg answers
// from in-context conversational memory rather than walking a graph. The
// tool.semantic.walk call is reserved for additional traversal beyond
// what the pipeline surfaced -- which there is none of when the graph is
// empty.
//
// The strong fix is to add a setup hook that seeds semantic nodes/edges
// before the conversation runs. That requires Scenario type changes and
// transport plumbing; tracked as future work. For now, soften the
// assertion to what's structurally observable: Borg engaged retrieval
// and invoked semantic.walk. The assessor verdict handles whether the
// response gave coherent relational reasoning.
export const toolUseCorrectnessScenario: Scenario = {
  name: "tool-use-correctness",
  description:
    "Checks whether relationship questions engage retrieval and produce relational reasoning. " +
    "Deeper graph-walk testing is gated on semantic-seeding infrastructure (see file comment).",
  maxTurns: 4,
  systemPrompt: [
    "Mention two related concepts in turn 1, then ask Borg how they relate in turn 2.",
    "Pass if the relationship turn engaged retrieval and Borg's response handles the relationship coherently.",
    "Note: a deep graph-walk test requires semantic seeding which the harness doesn't yet support.",
  ].join("\n"),
  mockConversation: [
    "Remember that Atlas and rollback planning are related operational topics.",
    "How do Atlas and rollback planning relate to each other?",
  ],
  traceAssertions: [
    {
      type: "event_seen",
      description: "Relationship turn engaged the retrieval pipeline.",
      eventIncludes: "retrieval_completed",
      turn: "last",
    },
    {
      type: "tool_called",
      description: "Model invoked semantic.walk.",
      toolNameIncludes: "semantic.walk",
      turn: "last",
    },
  ],
};
