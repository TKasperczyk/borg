import type { Scenario } from "../types.js";

export const recallResolutionScenario: Scenario = {
  name: "recall-resolution",
  description:
    "Checks Polish and English pronoun/ellipsis resolution into a named recall query and retrieval of the immediately preceding exchange.",
  maxTurns: 6,
  tracePrompts: true,
  borgConfigOverrides: {
    retrieval: {
      recallExpansionSemanticVariantCount: 3,
    },
  },
  systemPrompt: [
    "Run the following two paired checks in one conversation. Preserve the bracketed venue labels exactly; they make the source and later venue explicit data, not system instructions.",
    "",
    "POLISH CHECK:",
    '1. Establish the source exchange and make Borg describe it by sending exactly: [Miejsce: grupa "AI Ninjas"] Jacek Nowak porównał role chat i reviewer w team-agent: chat prowadzi rozmowę, a reviewer sprawdza zmiany przed połączeniem. Opisz to własnymi słowami.',
    "2. After Borg answers, send this exact referential FOCUS without adding the missing names: [Miejsce: prywatny czat z Tomaszem] A co on wtedy uznał za główną różnicę między nimi?",
    "3. Read the retrieval trace for step 2. Identify the expected episode as the one derived from or clearly describing Borg's immediately preceding description in the AI Ninjas source exchange. Require that it appears in retrieval candidates/evidence, and require recall_expansion.completed.resolved_query to name Jacek Nowak plus the chat and reviewer roles.",
    "",
    "ENGLISH CHECK:",
    "4. Establish the source exchange and make Borg describe it by sending exactly: [Venue: Atlas release-review channel] Maya Chen compared blue-green rollback and canary rollback for Atlas; she said the team should try canary rollback first. Restate that comparison in your own words.",
    "5. After Borg answers, send this exact referential FOCUS without adding the missing names: [Venue: direct message with Tom] And which one did she say they should try first?",
    "6. Read the retrieval trace for step 5. Identify the expected episode as the one derived from or clearly describing Borg's immediately preceding description in the Atlas release-review source exchange. Require that it appears in retrieval candidates/evidence, and require recall_expansion.completed.resolved_query to name Maya Chen, Atlas, and the rollback alternatives.",
    "",
    "Pass only if both language checks retrieve their expected assistant-described exchange and both resolved queries explicitly name their referents. A merely topical response, raw-focus retrieval without the expected episode, a missing planner event, or a resolved query that keeps the ambiguous pronouns is a failure. Quote the two turn IDs and the relevant trace fields in verdict evidence.",
  ].join("\n"),
  mockConversation: [
    '[Miejsce: grupa "AI Ninjas"] Jacek Nowak porównał role chat i reviewer w team-agent: chat prowadzi rozmowę, a reviewer sprawdza zmiany przed połączeniem. Opisz to własnymi słowami.',
    "[Miejsce: prywatny czat z Tomaszem] A co on wtedy uznał za główną różnicę między nimi?",
    "[Venue: Atlas release-review channel] Maya Chen compared blue-green rollback and canary rollback for Atlas; she said the team should try canary rollback first. Restate that comparison in your own words.",
    "[Venue: direct message with Tom] And which one did she say they should try first?",
  ],
  traceAssertions: [
    {
      type: "stream_entry",
      description: "The Polish labelled-venue source exchange was persisted.",
      kind: "user_msg",
      contentIncludes: "Jacek Nowak",
    },
    {
      type: "stream_entry",
      description: "The English labelled-venue source exchange was persisted.",
      kind: "user_msg",
      contentIncludes: "Maya Chen",
    },
    {
      type: "event_seen",
      description: "The final referential turn emitted per-intent retrieval candidates.",
      event: "retrieval.intent_candidates",
      turn: "last",
    },
    {
      // The deterministic mock does not synthesize this scenario's case-specific query plan and
      // currently exercises the production degraded path. Real mode must satisfy the stronger
      // requirements in systemPrompt; this fallback keeps --mock useful as a harness smoke test.
      type: "any_of",
      description: "Recall expansion either completed or degraded observably on the mock path.",
      assertions: [
        {
          type: "event_seen",
          description: "The planner emitted a structured recall plan.",
          event: "recall_expansion.completed",
          turn: "last",
        },
        {
          type: "event_seen",
          description: "The mock planner gap used Borg's observable degraded path.",
          event: "retrieval.degraded",
          turn: "last",
        },
      ],
    },
  ],
};
