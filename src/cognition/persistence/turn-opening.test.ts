import { describe, expect, it, vi } from "vitest";

import { createWorkingMemory } from "../../memory/working/index.js";
import type { StreamEntry, StreamEntryInput, StreamWriter } from "../../stream/index.js";
import { DEFAULT_SESSION_ID, type EntityId, type StreamEntryId } from "../../util/ids.js";
import { SuppressionSet } from "../attention/index.js";
import type { PerceptionResult } from "../types.js";
import { TurnOpeningPersistence } from "./turn-opening.js";

const userEntryId = "strm_abcdefghijklmnop" as StreamEntryId;
const perceptionEntryId = "strm_bcdefghijklmnopa" as StreamEntryId;
const entityId = "ent_abcdefghijklmnop" as EntityId;

function makePerception(): PerceptionResult {
  return {
    mode: "problem_solving",
    entities: ["Atlas"],
    temporalCue: null,
    affectiveSignal: {
      valence: -0.2,
      arousal: 0.5,
      dominant_emotion: "fear",
    },
  };
}

function makeStreamWriter(sequence: string[]) {
  const appended: StreamEntryInput[] = [];
  const streamWriter: Pick<StreamWriter, "append"> = {
    append: async (input) => {
      sequence.push(`append:${input.kind}`);
      appended.push(input);

      return {
        ...input,
        id: input.kind === "user_msg" ? userEntryId : perceptionEntryId,
        timestamp: input.kind === "user_msg" ? 1_000 : 1_001,
        session_id: DEFAULT_SESSION_ID,
        compressed: input.compressed ?? false,
      } satisfies StreamEntry;
    },
  };

  return { appended, streamWriter };
}

describe("TurnOpeningPersistence", () => {
  it("appends user message, saves working memory, then appends perception", async () => {
    const sequence: string[] = [];
    const { appended, streamWriter } = makeStreamWriter(sequence);
    const initialWorkingMemory = {
      ...createWorkingMemory(DEFAULT_SESSION_ID, 500),
      turn_counter: 4,
    };
    const save = vi.fn((memory) => {
      sequence.push("save:working_memory");
      return {
        ...memory,
        updated_at: 2_000,
      };
    });
    const now = vi.fn(() => {
      sequence.push("clock:now");
      return 2_000;
    });
    const pendingSocialAttribution = {
      entity_id: entityId,
      interaction_id: 1,
      agent_response_summary: "summary",
      turn_completed_ts: 900,
    };
    const pendingTraitAttribution = {
      trait_label: "careful",
      strength_delta: 0.05,
      source_stream_entry_ids: [userEntryId],
      source_episode_ids: [],
      turn_completed_ts: 900,
      audience_entity_id: entityId,
    };

    const result = await new TurnOpeningPersistence({
      workingMemoryStore: {
        save,
      },
    }).persist({
      streamWriter,
      userMessage: "Fix Atlas",
      audience: "alice",
      workingMemory: initialWorkingMemory,
      pendingSocialAttribution,
      pendingTraitAttribution,
      suppressionSet: SuppressionSet.fromEntries(
        [
          {
            id: "ep_1",
            reason: "duplicate",
            until_turn: 8,
          },
        ],
        4,
      ),
      perception: makePerception(),
      now,
    });

    expect(sequence).toEqual([
      "append:user_msg",
      "clock:now",
      "save:working_memory",
      "append:perception",
    ]);
    expect(appended).toEqual([
      {
        kind: "user_msg",
        content: "Fix Atlas",
        audience: "alice",
      },
      {
        kind: "perception",
        content: {
          mode: "problem_solving",
          entities: ["Atlas"],
          temporalCue: null,
          affectiveSignal: {
            valence: -0.2,
            arousal: 0.5,
            dominant_emotion: "fear",
          },
        },
        audience: "alice",
      },
    ]);
    expect(save).toHaveBeenCalledWith({
      ...initialWorkingMemory,
      pending_social_attribution: pendingSocialAttribution,
      pending_trait_attribution: pendingTraitAttribution,
      suppressed: [
        {
          id: "ep_1",
          reason: "duplicate",
          until_turn: 8,
        },
      ],
      updated_at: 2_000,
    });
    expect(result.persistedUserEntry.id).toBe(userEntryId);
    expect(result.workingMemory).toEqual(save.mock.results[0]?.value);
  });

  it("omits audience from opening stream entries when no audience was provided", async () => {
    const sequence: string[] = [];
    const { appended, streamWriter } = makeStreamWriter(sequence);

    await new TurnOpeningPersistence({
      workingMemoryStore: {
        save: vi.fn((memory) => memory),
      },
    }).persist({
      streamWriter,
      userMessage: "Hello",
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 500),
      pendingSocialAttribution: null,
      pendingTraitAttribution: null,
      suppressionSet: SuppressionSet.fromEntries([], 0),
      perception: makePerception(),
      now: () => 1_000,
    });

    expect(appended).toEqual([
      expect.not.objectContaining({
        audience: expect.anything(),
      }),
      expect.not.objectContaining({
        audience: expect.anything(),
      }),
    ]);
  });
});
