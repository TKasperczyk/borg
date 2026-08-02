import { describe, expect, it } from "vitest";

import {
  collectProtectedEpisodeTokenLines,
  preserveProtectedEpisodeTokenLines,
} from "./protected-lines.js";

describe("protected episode token lines", () => {
  const outcomeLine = "OUTCOME fp=scheduled:triage:team-agent-ai role=triage tenant=team-agent-ai";
  const oldGrammarLine = "decision=teams_card:posted action=teams_card teams_card=yes card_count=1";
  const ticketActionLine =
    "ticket=AININJAS-1187 action=transition transition=Ready_for_dev verdict=approved";
  const overlappingNewGrammarLine = "ticket=AININJAS-1188 action=created summary=Prepare release";
  const bareTeamsCardLine = "action=teams_card";

  it("collects complete old- and new-grammar lines once in source order", () => {
    const first = [
      "Autonomous run completed salient outcomes.",
      outcomeLine,
      oldGrammarLine,
      ticketActionLine,
      overlappingNewGrammarLine,
      bareTeamsCardLine,
    ].join("\n");
    const replay = [ticketActionLine, oldGrammarLine, "No protocol tokens here."].join("\r\n");

    expect(collectProtectedEpisodeTokenLines([first, replay])).toEqual([
      outcomeLine,
      oldGrammarLine,
      ticketActionLine,
      overlappingNewGrammarLine,
      bareTeamsCardLine,
    ]);
  });

  it("preserves mixed-grammar source lines without appending any line twice", () => {
    const narrative = [
      "The triage run created and transitioned separate tickets, then posted a Teams card.",
      ticketActionLine,
    ].join("\n");
    const source = [
      outcomeLine,
      oldGrammarLine,
      ticketActionLine,
      overlappingNewGrammarLine,
      bareTeamsCardLine,
      overlappingNewGrammarLine,
    ].join("\n");

    const preserved = preserveProtectedEpisodeTokenLines(narrative, [source, source]);
    const lines = preserved.split(/\r\n|\n|\r/u);

    expect(lines).toEqual([
      "The triage run created and transitioned separate tickets, then posted a Teams card.",
      ticketActionLine,
      outcomeLine,
      oldGrammarLine,
      overlappingNewGrammarLine,
      bareTeamsCardLine,
    ]);
    for (const protectedLine of [
      outcomeLine,
      oldGrammarLine,
      ticketActionLine,
      overlappingNewGrammarLine,
      bareTeamsCardLine,
    ]) {
      expect(lines.filter((line) => line === protectedLine)).toHaveLength(1);
    }
  });

  it("accepts defined leading whitespace for protocol lines", () => {
    expect(
      collectProtectedEpisodeTokenLines([
        "  ticket=AININJAS-1189 action=mr mr=https://gitlab.example/project/-/merge_requests/12\n\taction=teams_card",
      ]),
    ).toEqual([
      "  ticket=AININJAS-1189 action=mr mr=https://gitlab.example/project/-/merge_requests/12",
      "\taction=teams_card",
    ]);
  });

  it("does not protect quoted tokens or ordinary-prose mentions of the new grammar", () => {
    expect(
      collectProtectedEpisodeTokenLines([
        [
          'The log quoted "ticket=AININJAS-1187 action=transition" for reference.',
          '"ticket=AININJAS-1187 action=transition"',
          "The note mentions action=teams_card in ordinary prose.",
          '"action=teams_card"',
          "action=teams_card was emitted earlier",
        ].join("\n"),
      ]),
    ).toEqual([]);
  });
});
