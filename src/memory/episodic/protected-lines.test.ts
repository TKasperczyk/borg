import { describe, expect, it } from "vitest";

import {
  buildConsolidationEpisodeEmbeddingText,
  buildEpisodeEmbeddingText,
  collectProtectedEpisodeTokenLines,
  EpisodeEmbeddingTextError,
  preserveProtectedEpisodeTokenLines,
} from "./protected-lines.js";

describe("protected episode token lines", () => {
  const outcomeLine = "OUTCOME fp=scheduled:triage:team-agent-ai role=triage tenant=team-agent-ai";
  const oldGrammarLine = "decision=teams_card:posted action=teams_card teams_card=yes card_count=1";
  const ticketActionLine =
    "ticket=AININJAS-1187 action=transition transition=Ready_for_dev verdict=approved";
  const overlappingNewGrammarLine = "ticket=AININJAS-1188 action=created summary=Prepare release";
  const bareTeamsCardLine = "action=teams_card";

  it("reconstructs the production consolidation recipe from its persisted narrative", () => {
    const prose = "The team completed the scheduled triage.";
    const source = `${outcomeLine}\n${oldGrammarLine}\n${ticketActionLine}`;
    const title = " Daily rollup ";
    const tags = ["triage"];
    const participants = ["team-agent-ai"];
    expect(
      buildEpisodeEmbeddingText({
        title,
        tags,
        participants,
        episode_kind: "consolidation_version",
        narrative: preserveProtectedEpisodeTokenLines(prose, [source]),
        legacyProtectedSourceTexts: [source],
      }),
    ).toBe(
      buildConsolidationEpisodeEmbeddingText({
        title,
        tags,
        participants,
        synthesizedNarrative: prose,
        protectedSourceTexts: [source],
      }),
    );
  });

  it("rejects ambiguous legacy inline protocol prose instead of adding it to the embedding", () => {
    const source = "I corrected the report. OUTCOME fp=legacy-correction decision=filter-by-author";
    const prose = "The report was corrected.";
    const persisted = preserveProtectedEpisodeTokenLines(prose, [source]);
    expect(persisted).toBe(preserveProtectedEpisodeTokenLines(`${prose}\n${source}`, [source]));
    expect(() =>
      buildEpisodeEmbeddingText({
        title: "Correction",
        narrative: persisted,
        tags: [],
        episode_kind: "consolidation_version",
        legacyProtectedSourceTexts: [source],
      }),
    ).toThrow("Legacy consolidation embedding input is ambiguous");
  });

  it("counts every distinct legacy candidate and exposes only a narrative-preserving longest prefix", () => {
    const sources = ["First prose OUTCOME fp=first", "Second prose decision=send"];
    const narrative = preserveProtectedEpisodeTokenLines("Synthesis", sources);
    for (const stored of [narrative, ` ${narrative} `]) {
      let caught: unknown;
      try {
        buildEpisodeEmbeddingText({
          title: "Title",
          narrative: stored,
          tags: [],
          episode_kind: "consolidation_version",
          legacyProtectedSourceTexts: sources,
        });
      } catch (error) {
        caught = error;
      }
      expect(caught).toBeInstanceOf(EpisodeEmbeddingTextError);
      expect(caught).toMatchObject({ candidateCount: 3, code: "EMBEDDING_TEXT_UNRECOVERABLE" });
      const error = caught as EpisodeEmbeddingTextError;
      if (stored === narrative) {
        expect(error.longestPrefix).toEqual({
          synthesized_narrative: narrative,
          protected_source_lines: sources,
        });
        expect(
          preserveProtectedEpisodeTokenLines(
            error.longestPrefix!.synthesized_narrative,
            error.longestPrefix!.protected_source_lines,
          ),
        ).toBe(narrative);
      } else expect(error.longestPrefix).toBeUndefined();
    }
  });

  it("uses recorded source order and refuses stale synthesized input", () => {
    const source = ["OUTCOME fp=second", "OUTCOME fp=first"];
    const prose = "A correction.\nOUTCOME fp=first";
    const input = {
      title: "Correction",
      narrative: preserveProtectedEpisodeTokenLines(prose, source),
      tags: [],
      episode_kind: "consolidation_version",
      consolidation_embedding_input: {
        synthesized_narrative: prose,
        protected_source_lines: source,
      },
    };
    expect(buildEpisodeEmbeddingText(input)).toBe(
      buildConsolidationEpisodeEmbeddingText({
        title: input.title,
        synthesizedNarrative: prose,
        protectedSourceTexts: source,
        tags: [],
        participants: [],
      }),
    );
    expect(() => buildEpisodeEmbeddingText({ ...input, narrative: "Changed" })).toThrow(
      "no longer matches",
    );
  });

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

  it("builds consolidation embeddings from prose and fp headers without mixed-grammar payloads", () => {
    const prose = "The triage run created and transitioned tickets.";
    const source = [
      `${prose} decision=created:AININJAS-1187 action=created ticket=AININJAS-1187 summary=Example`,
      `${outcomeLine} ticket=AININJAS-1188 action=created summary=Prepare release`,
      outcomeLine,
      oldGrammarLine,
      ticketActionLine,
      bareTeamsCardLine,
    ].join("\n");

    expect(
      buildConsolidationEpisodeEmbeddingText({
        title: "Daily triage rollup",
        synthesizedNarrative: source,
        protectedSourceTexts: [source],
        tags: ["triage", "daily"],
        participants: ["team"],
      }),
    ).toBe(["Daily triage rollup", prose, outcomeLine, "triage daily", "team"].join("\n"));
  });

  it("falls back to the fp token when a legacy outcome has no standalone header line", () => {
    const legacy = "I corrected the report. OUTCOME fp=legacy-correction decision=filter-by-author";

    expect(
      buildConsolidationEpisodeEmbeddingText({
        title: "Reporting correction",
        synthesizedNarrative: legacy,
        protectedSourceTexts: [legacy],
        tags: [],
        participants: [],
      }),
    ).toBe(
      [
        "Reporting correction",
        "I corrected the report.",
        "OUTCOME fp=legacy-correction",
        "",
        "",
      ].join("\n"),
    );
  });
});
