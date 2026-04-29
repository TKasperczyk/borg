import type { Scenario } from "../types.js";

// Limitation: this scenario can't deeply test graph-aware reasoning today.
// Semantic content only enters the graph via the offline reflector or
// explicit borg.semantic.extract -- a single conversational turn does not
// auto-populate semantic nodes/edges. Without seeded semantic content, the
// standard retrieval pipeline returns 0 semantic hits and Borg answers
// from in-context conversational memory rather than walking a graph.
//
// The strong fix is to add a setup hook that seeds semantic nodes/edges
// before the conversation runs. That requires Scenario type changes and
// transport plumbing; tracked as future work. Until then, the assertion
// surface is the retrieval pipeline running at all (a baseline structural
// fact); the assessor verdict handles whether the response gave coherent
// relational reasoning. Asserting on tool.semantic.walk specifically is
// not meaningful when the graph is empty -- the model has nothing to walk.
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
  ],
};
