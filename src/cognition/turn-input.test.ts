import { describe, expect, it } from "vitest";

import type { Borg } from "../borg.js";
import {
  createAttachmentId,
  createEntityId,
  createImagePerceptionId,
  createSessionId,
  createStreamEntryId,
} from "../util/ids.js";
import {
  renderInboundBatch,
  type HydratedInboundAttachment,
  type TurnInput,
  type TurnOrchestratorInput,
} from "./turn-input.js";

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

  it("renders one image perception as structural context with XML escaping", () => {
    const sessionId = createSessionId();
    const entryId = createStreamEntryId();
    const attachmentId = createAttachmentId();
    const perceptionId = createImagePerceptionId();

    const rendered = renderInboundBatch({
      entries: [
        {
          id: entryId,
          session_id: sessionId,
          entry_index: 1,
          timestamp: 2_010,
          kind: "user_msg",
          content: "look & compare",
          attachments: [
            {
              attachment_id: attachmentId,
              media_type: "image/png",
              width: 640,
              height: 480,
              perception: {
                perception_id: perceptionId,
                caption: 'screen <one> & "two"',
                image_kind: "screenshot",
                visible_text: ["alpha & beta", "<launch>"],
                search_terms: ['ops "panel"', "status > ready"],
              },
            },
          ],
        },
      ],
    });

    expect(rendered).toBe(
      [
        '<inbound_batch kind="stream_backlog" count="1">',
        `<inbound_message index="1" stream_entry_id="${entryId}" timestamp_ms="2010">`,
        "look &amp; compare",
        '<attachments count="1">',
        `<attachment index="1" kind="image" attachment_id="${attachmentId}" media_type="image/png" width="640" height="480">`,
        `<perception status="available" perception_id="${perceptionId}">`,
        '<caption>screen &lt;one&gt; &amp; "two"</caption>',
        "<image_kind>screenshot</image_kind>",
        '<visible_text count="2">',
        '<text index="1">alpha &amp; beta</text>',
        '<text index="2">&lt;launch&gt;</text>',
        "</visible_text>",
        '<search_terms count="2">',
        '<term index="1">ops "panel"</term>',
        '<term index="2">status &gt; ready</term>',
        "</search_terms>",
        "</perception>",
        "</attachment>",
        "</attachments>",
        "</inbound_message>",
        "</inbound_batch>",
      ].join("\n"),
    );
  });

  it("renders multiple image perceptions in attachment order", () => {
    const sessionId = createSessionId();
    const entryId = createStreamEntryId();
    const firstAttachmentId = createAttachmentId();
    const secondAttachmentId = createAttachmentId();
    const firstPerceptionId = createImagePerceptionId();
    const secondPerceptionId = createImagePerceptionId();
    const attachments: HydratedInboundAttachment[] = [
      {
        attachment_id: firstAttachmentId,
        media_type: "image/png",
        width: 100,
        height: 100,
        perception: {
          perception_id: firstPerceptionId,
          caption: "first image",
          image_kind: "photo",
          visible_text: [],
          search_terms: ["first"],
        },
      },
      {
        attachment_id: secondAttachmentId,
        media_type: "image/jpeg",
        width: 200,
        height: 150,
        perception: {
          perception_id: secondPerceptionId,
          caption: "second image",
          image_kind: "diagram",
          visible_text: ["second text"],
          search_terms: [],
        },
      },
    ];

    const rendered = renderInboundBatch({
      entries: [
        {
          id: entryId,
          session_id: sessionId,
          entry_index: 1,
          timestamp: 3_010,
          kind: "user_msg",
          content: "two images",
          attachments,
        },
      ],
    });

    expect(rendered.indexOf(`attachment_id="${firstAttachmentId}"`)).toBeLessThan(
      rendered.indexOf(`attachment_id="${secondAttachmentId}"`),
    );
    expect(rendered).toContain('<attachments count="2">');
    expect(rendered).toContain('<attachment index="1" kind="image"');
    expect(rendered).toContain('<attachment index="2" kind="image"');
    expect(rendered).toContain(
      `<perception status="available" perception_id="${firstPerceptionId}">`,
    );
    expect(rendered).toContain(
      `<perception status="available" perception_id="${secondPerceptionId}">`,
    );
  });

  it("renders image attachments without stored perception as unavailable", () => {
    const sessionId = createSessionId();
    const entryId = createStreamEntryId();
    const attachmentId = createAttachmentId();

    const rendered = renderInboundBatch({
      entries: [
        {
          id: entryId,
          session_id: sessionId,
          entry_index: 1,
          timestamp: 4_010,
          kind: "user_msg",
          content: "image without perception",
          attachments: [
            {
              attachment_id: attachmentId,
              media_type: "image/webp",
              width: 320,
              height: 240,
              perception: null,
            },
          ],
        },
      ],
    });

    expect(rendered).toContain(
      `<attachment index="1" kind="image" attachment_id="${attachmentId}" media_type="image/webp" width="320" height="240">`,
    );
    expect(rendered).toContain('<perception status="unavailable" />');
    expect(rendered).not.toContain("<caption>");
  });
});
