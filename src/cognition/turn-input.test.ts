import { describe, expect, it } from "vitest";

import type { Borg } from "../borg.js";
import { createEntityId, createSessionId, createStreamEntryId } from "../util/ids.js";
import { renderInboundBatch, type TurnInput, type TurnOrchestratorInput } from "./turn-input.js";

type PublicBorgTurnInput = Parameters<Borg["turn"]>[0];

const publicBorgTurnInput: PublicBorgTurnInput = {
  userMessage: "single public message",
};
const exportedPublicTurnInput: TurnInput = publicBorgTurnInput;

const publicBorgTurnRejectsInboundBatch: PublicBorgTurnInput = {
  userMessage: "single public message",
  // @ts-expect-error public Borg.turn does not accept internal catch-up batches
  inboundBatch: {
    kind: "stream_backlog",
    entryIds: [createStreamEntryId()],
  },
};

const publicBorgTurnRejectsLockMode: PublicBorgTurnInput = {
  userMessage: "single public message",
  // @ts-expect-error public Borg.turn does not accept lockMode
  lockMode: "try",
};

const internalOrchestratorAllowsCatchUp: TurnOrchestratorInput = {
  lockMode: "try",
  inboundBatch: {
    kind: "stream_backlog",
    entryIds: [createStreamEntryId()],
  },
};

void exportedPublicTurnInput;
void publicBorgTurnRejectsInboundBatch;
void publicBorgTurnRejectsLockMode;
void internalOrchestratorAllowsCatchUp;

describe("turn input batch rendering", () => {
  it("renders inbound batches oldest-first with structural sender metadata", () => {
    const sessionId = createSessionId();
    const sender = createEntityId();
    const older = createStreamEntryId();
    const newer = createStreamEntryId();

    const rendered = renderInboundBatch({
      entries: [
        {
          id: newer,
          session_id: sessionId,
          entry_index: 2,
          timestamp: 1_020,
          kind: "user_msg",
          content: "second <message>",
          sender_entity_id: sender,
        },
        {
          id: older,
          session_id: sessionId,
          entry_index: 1,
          timestamp: 1_010,
          kind: "user_msg",
          content: "first & message",
          sender_entity_id: sender,
        },
      ],
      senderDisplayNameById: (entityId) => (entityId === sender ? "Alice" : null),
    });

    expect(rendered).toBe(
      [
        '<inbound_batch kind="stream_backlog" count="2">',
        `<inbound_message index="1" stream_entry_id="${older}" timestamp_ms="1010" sender_entity_id="${sender}" sender_display_name="Alice">first &amp; message</inbound_message>`,
        `<inbound_message index="2" stream_entry_id="${newer}" timestamp_ms="1020" sender_entity_id="${sender}" sender_display_name="Alice">second &lt;message&gt;</inbound_message>`,
        "</inbound_batch>",
      ].join("\n"),
    );
  });
});
