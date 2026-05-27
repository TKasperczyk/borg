import { describe, expect, it, vi } from "vitest";

import type { EntityRecord, EntityRepository } from "../../memory/commitments/index.js";
import type {
  RelationalSlot,
  RelationalSlotListOptions,
} from "../../memory/relational-slots/index.js";
import type { EntityId, RelationalSlotId, StreamEntryId } from "../../util/ids.js";
import {
  buildParticipantRoster,
  buildParticipantRosterFromRepositories,
  participantRosterRelationalSlotIds,
  renderParticipantRoster,
} from "./participant-roster.js";

const nora = "ent_noraaaaaaaaaaaa" as EntityId;
const julian = "ent_julianaaaaaaaaa" as EntityId;
const priya = "ent_priyaaaaaaaaaa" as EntityId;
const ruth = "ent_ruthaaaaaaaaaa" as EntityId;
const daniel = "ent_danielaaaaaaaa" as EntityId;
const maya = "ent_mayaaaaaaaaaaa" as EntityId;

function entity(id: EntityId, canonicalName: string, kind: EntityRecord["kind"] = "person") {
  return {
    id,
    canonical_name: canonicalName,
    aliases: [],
    kind,
    borg_role: null,
    name_provenance: "user_declared",
    created_at: 1,
  } satisfies EntityRecord;
}

function slot(input: {
  id: string;
  subject: EntityId;
  key: string;
  value: string;
  state?: RelationalSlot["state"];
}): RelationalSlot {
  return {
    id: input.id as RelationalSlotId,
    subject_entity_id: input.subject,
    slot_key: input.key,
    value: input.value,
    state: input.state ?? "established",
    evidence_stream_entry_ids: ["strm_rosteraaaaaaaa" as StreamEntryId],
    contradicted_by_stream_entry_ids: [],
    alternate_values: [],
    name_provenance: "user_declared",
    created_at: 1,
    updated_at: 1,
  };
}

function entityRepositoryFor(
  records: readonly EntityRecord[],
): Pick<EntityRepository, "get" | "findByName"> {
  const entities = new Map<EntityId, EntityRecord>(records.map((record) => [record.id, record]));

  return {
    get: (id) => entities.get(id) ?? null,
    findByName: (name) =>
      [...entities.values()].find((record) => record.canonical_name === name)?.id ?? null,
  };
}

describe("participant roster", () => {
  it("renders active participants, non-chat subjects, and uncertain slots from structured records", () => {
    const entityRepository = entityRepositoryFor([
      entity(nora, "Nora"),
      entity(julian, "Julian"),
      entity(priya, "Priya"),
      entity(ruth, "Ruth"),
    ]);

    const roster = buildParticipantRoster({
      activeParticipants: [
        { entityId: nora, displayName: "Nora", role: "speaker" },
        { entityId: julian, displayName: "Julian", role: "participant" },
        { entityId: priya, displayName: "Priya", role: "participant" },
      ],
      entityRepository,
      relationalSlots: [
        slot({ id: "rslot_parent_nora", subject: nora, key: "parent", value: "Ruth" }),
        slot({ id: "rslot_parent_julian", subject: julian, key: "parent", value: "Ruth" }),
        slot({ id: "rslot_spouse_priya", subject: priya, key: "spouse", value: "Nora" }),
        slot({
          id: "rslot_contested_manager",
          subject: priya,
          key: "manager",
          value: "Ruth",
          state: "contested",
        }),
      ],
    });
    const rendered = renderParticipantRoster(roster);

    expect(roster.participants).toEqual([
      expect.objectContaining({
        entity_id: nora,
        display_name: "Nora",
        audience_role: "speaker",
        known_relationships: ["parent:Ruth"],
        relationship_source: "relational_slot:rslot_parent_nora",
      }),
      expect.objectContaining({
        entity_id: julian,
        audience_role: "active_participant",
        known_relationships: ["parent:Ruth"],
      }),
      expect.objectContaining({
        entity_id: priya,
        known_relationships: ["spouse:Nora"],
      }),
    ]);
    expect(roster.non_chat_subjects).toEqual([
      expect.objectContaining({
        entity_id: ruth,
        display_name: "Ruth",
        relationship_source: "relational_slot:rslot_parent_nora",
      }),
    ]);
    expect(roster.unknown_or_uncertain).toEqual([
      expect.objectContaining({
        entity_id: priya,
        reason: "relational_slot_state:contested",
        known_relationships: ["manager:Ruth"],
      }),
    ]);
    expect(participantRosterRelationalSlotIds(roster)).toEqual(
      new Set(["rslot_parent_nora", "rslot_parent_julian", "rslot_spouse_priya"]),
    );
    expect(rendered).toContain("Thread roster:");
    expect(rendered).toContain("- Nora (id: ent_noraaaaaaaaaaaa; speaker");
    expect(rendered).toContain("Non-chat subjects:");
    expect(rendered).toContain("Unknown or uncertain:");
  });

  it("keeps quarantined slots visible as uncertain without making them grounding evidence", () => {
    const entityRepository = entityRepositoryFor([entity(nora, "Nora"), entity(ruth, "Ruth")]);

    const roster = buildParticipantRoster({
      activeParticipants: [{ entityId: nora, displayName: "Nora", role: "speaker" }],
      entityRepository,
      relationalSlots: [
        slot({
          id: "rslot_quarantined_parent",
          subject: nora,
          key: "parent",
          value: "Ruth",
          state: "quarantined",
        }),
      ],
    });

    expect(roster.participants).toEqual([
      expect.objectContaining({
        entity_id: nora,
        known_relationships: [],
        relationship_source: null,
      }),
    ]);
    expect(roster.non_chat_subjects).toEqual([]);
    expect(roster.unknown_or_uncertain).toEqual([
      expect.objectContaining({
        entity_id: nora,
        reason: "relational_slot_state:quarantined",
        relationship_source: "relational_slot:rslot_quarantined_parent",
      }),
    ]);
    expect(participantRosterRelationalSlotIds(roster)).toEqual(new Set());
  });

  it("returns an empty roster on repository cold start", () => {
    const entityRepository = entityRepositoryFor([]);
    const relationalSlotRepository = {
      list: vi.fn((_options?: RelationalSlotListOptions) => []),
    };

    const roster = buildParticipantRosterFromRepositories({
      activeParticipants: [],
      audienceEntityId: null,
      entityRepository,
      relationalSlotRepository,
    });

    expect(roster).toEqual({
      participants: [],
      non_chat_subjects: [],
      unknown_or_uncertain: [],
    });
    expect(relationalSlotRepository.list).not.toHaveBeenCalled();
  });

  it("includes structured stream evidence without treating it as relational-slot grounding", () => {
    const entityRepository = entityRepositoryFor([entity(nora, "Nora")]);
    const sourceStreamEntryId = "strm_streamroster01" as StreamEntryId;

    const roster = buildParticipantRoster({
      activeParticipants: [{ entityId: nora, displayName: "Nora", role: "speaker" }],
      entityRepository,
      relationalSlots: [],
      streamEvidence: [
        {
          entity_id: nora,
          display_name: "Nora",
          known_relationship: "project_role:incident lead",
          source_stream_entry_id: sourceStreamEntryId,
        },
      ],
    });

    expect(roster.participants[0]).toMatchObject({
      entity_id: nora,
      known_relationships: ["project_role:incident lead"],
      relationship_source: `stream_entry:${sourceStreamEntryId}`,
      relationship_sources: [`stream_entry:${sourceStreamEntryId}`],
    });
    expect(participantRosterRelationalSlotIds(roster)).toEqual(new Set());
  });

  it("does not pull unrelated global relational slots into non-chat subjects", () => {
    const entityRepository = entityRepositoryFor([
      entity(nora, "Nora"),
      entity(maya, "Maya"),
      entity(daniel, "Daniel"),
    ]);
    const unrelatedSlot = slot({
      id: "rslot_unrelated_manager",
      subject: maya,
      key: "manager",
      value: "Daniel",
    });
    const relationalSlotRepository = {
      list: vi.fn((options?: RelationalSlotListOptions) =>
        options?.subjectEntityId === nora ? [] : [unrelatedSlot],
      ),
    };

    const roster = buildParticipantRosterFromRepositories({
      activeParticipants: [{ entityId: nora, displayName: "Nora", role: "speaker" }],
      entityRepository,
      relationalSlotRepository,
    });

    expect(roster.participants).toEqual([
      expect.objectContaining({
        entity_id: nora,
        known_relationships: [],
      }),
    ]);
    expect(roster.non_chat_subjects).toEqual([]);
    expect(roster.unknown_or_uncertain).toEqual([]);
    expect(participantRosterRelationalSlotIds(roster)).toEqual(new Set());
  });
});
