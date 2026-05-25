import { describe, expect, it } from "vitest";

import type { StreamEntry, StreamEntryIndexRecord, StreamReader } from "../stream/index.js";
import { DEFAULT_SESSION_ID, type StreamEntryId } from "../util/ids.js";
import {
  createLoadedUserStreamEntryRelationshipEvidenceTrustValidator,
  createUserStreamEntryRelationshipEvidenceTrustValidator,
} from "./source-trust.js";

const ATTACHMENT_ENTRY_ID = "strm_aaaaaaaaaaaaaaaa" as StreamEntryId;

function indexedEntry(overrides: Partial<StreamEntryIndexRecord> = {}): StreamEntryIndexRecord {
  return {
    entry_id: ATTACHMENT_ENTRY_ID,
    session_id: DEFAULT_SESSION_ID,
    byte_offset: 0,
    entry_index: 0,
    timestamp: 1,
    kind: "user_image_attachment",
    sender_entity_id: null,
    turn_id: "turn-image",
    turn_status: "active",
    active: true,
    ...overrides,
  };
}

function streamEntry(): StreamEntry {
  return {
    id: ATTACHMENT_ENTRY_ID,
    session_id: DEFAULT_SESSION_ID,
    timestamp: 1,
    kind: "user_image_attachment",
    content: { type: "image_ref" },
    turn_id: "turn-image",
    turn_status: "active",
    sender_entity_id: null,
    reply_target_entity_id: null,
    compressed: false,
  };
}

describe("source trust", () => {
  it("rejects indexed inactive image attachment entries", async () => {
    const validator = createUserStreamEntryRelationshipEvidenceTrustValidator({
      entryIndex: {
        lookup: () => indexedEntry({ active: false }),
      },
      createStreamReader: () =>
        ({
          async *iterate() {
            yield streamEntry();
          },
        }) as StreamReader,
    });

    await expect(validator(ATTACHMENT_ENTRY_ID)).resolves.toEqual({
      allowed: false,
      reason: "untrusted",
    });
  });

  it("rejects loaded image attachment entries when attachment state is inactive", () => {
    const validator = createLoadedUserStreamEntryRelationshipEvidenceTrustValidator({
      entries: [streamEntry()],
      isActiveAttachmentStreamEntry: () => false,
    });

    expect(validator(ATTACHMENT_ENTRY_ID)).toEqual({
      allowed: false,
      reason: "untrusted",
    });
  });
});
