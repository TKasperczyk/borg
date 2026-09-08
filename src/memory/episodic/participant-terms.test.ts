import { describe, expect, it } from "vitest";

import { parseEntityId } from "../../util/ids.js";
import {
  episodeParticipantDisplayNames,
  episodeParticipantEntityId,
  episodeParticipantEntityIds,
  episodeParticipantEntityIdTerm,
  parseEpisodeParticipantEntityIdTerm,
} from "./participant-terms.js";

describe("episode participant terms", () => {
  const entityId = parseEntityId("ent_w4tgay3o9h06ogaa");
  const otherEntityId = parseEntityId("ent_g9a1j5imbhrgbo1q");

  it("mints and parses the prefixed term", () => {
    const term = episodeParticipantEntityIdTerm(entityId);

    expect(term).toBe(`entity_id:${entityId}`);
    expect(parseEpisodeParticipantEntityIdTerm(term)).toBe(entityId);
  });

  it("keeps the prefixed form distinct from the bare form", () => {
    expect(parseEpisodeParticipantEntityIdTerm(entityId)).toBeNull();
    expect(episodeParticipantEntityId(entityId)).toBe(entityId);
  });

  it("resolves both written forms of an id to the same entity", () => {
    expect(episodeParticipantEntityIds([episodeParticipantEntityIdTerm(entityId), entityId])).toEqual([
      entityId,
    ]);
  });

  it("never reports an id as a display name, whichever way it was written", () => {
    const participants = [
      "Claude Code",
      episodeParticipantEntityIdTerm(otherEntityId),
      entityId,
      "Tom",
    ];

    expect(episodeParticipantDisplayNames(participants)).toEqual(["Claude Code", "Tom"]);
    expect(episodeParticipantEntityIds(participants)).toEqual([otherEntityId, entityId]);
  });

  it("leaves names that merely resemble a term alone", () => {
    const participants = ["entity_id:not-an-id", "ent_short", "entity_id:"];

    expect(episodeParticipantDisplayNames(participants)).toEqual(participants);
    expect(episodeParticipantEntityIds(participants)).toEqual([]);
  });
});
