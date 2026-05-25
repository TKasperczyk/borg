import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { composeMigrations, openDatabase, type SqliteDatabase } from "../storage/sqlite/index.js";
import {
  StreamEntryIndexRepository,
  StreamWriter,
  streamEntryIndexMigrations,
} from "../stream/index.js";
import { AttachmentError } from "../util/errors.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";
import { AttachmentBlobStore } from "./blob-store.js";
import { AttachmentRepository, attachmentMigrations } from "./repository.js";
import { AttachmentService } from "./service.js";

const PNG_1X1 = Uint8Array.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x04, 0x00, 0x00, 0x00, 0xb5, 0x1c, 0x0c,
  0x02, 0x00, 0x00, 0x00, 0x0b, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0xfc, 0xff, 0x1f, 0x00,
  0x03, 0x03, 0x02, 0x00, 0xef, 0xbf, 0x27, 0x8f, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44,
  0xae, 0x42, 0x60, 0x82,
]);
const MALFORMED_PNG_1X1 = PNG_1X1.subarray(0, 24);

describe("AttachmentService", () => {
  let tempDir: string | undefined;
  let db: SqliteDatabase | undefined;

  afterEach(() => {
    db?.close();
    db = undefined;

    if (tempDir !== undefined) {
      rmSync(tempDir, { recursive: true, force: true });
      tempDir = undefined;
    }
  });

  function setup() {
    tempDir = mkdtempSync(join(tmpdir(), "borg-attachments-"));
    db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(streamEntryIndexMigrations, attachmentMigrations),
    });
    const repository = new AttachmentRepository(db);
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const service = new AttachmentService({
      repository,
      blobStore: new AttachmentBlobStore(tempDir),
      config: {
        maxBytesPerImage: 1024,
        maxWidth: 64,
        maxHeight: 64,
        maxImagesPerTurn: 4,
      },
      entryIndex,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      entryIndex,
    });

    return { repository, service, writer, entryIndex };
  }

  it("stores image blobs by sha256 and writes only attachment refs to the stream", async () => {
    const { repository, service, writer } = setup();
    const parentEntry = await writer.append({
      kind: "user_msg",
      content: "What is in this image?",
      turn_id: "turn-image",
      audience: "Alice",
    });

    const [persisted] = await service.persistTurnAttachments({
      attachments: [{ mediaType: "image/png", bytes: PNG_1X1 }],
      streamWriter: writer,
      parentEntry,
      turnId: "turn-image",
      createdTurnGlobal: 7,
    });

    writer.close();

    expect(persisted?.contentBlock.type).toBe("image_ref");
    const record = repository.get(persisted!.attachmentId);
    expect(record).toMatchObject({
      media_type: "image/png",
      byte_size: PNG_1X1.byteLength,
      width: 1,
      height: 1,
      audience: "Alice",
      created_turn_global: 7,
      parent_entry_id: parentEntry.id,
      parent_turn_id: "turn-image",
      active: true,
    });
    expect(repository.isActiveForStreamEntry(persisted!.streamEntry.id)).toBe(true);
    expect(readFileSync(join(tempDir!, record!.storage_ref))).toEqual(Buffer.from(PNG_1X1));

    const streamJsonl = readFileSync(join(tempDir!, "stream", "default.jsonl"), "utf8");
    expect(streamJsonl).toContain('"kind":"user_image_attachment"');
    expect(streamJsonl).toContain('"attachment_id"');
    expect(streamJsonl).toContain(`"parent_entry_id":"${parentEntry.id}"`);
    expect(streamJsonl).not.toContain(Buffer.from(PNG_1X1).toString("base64"));
  });

  it("deduplicates repeated image bytes by sha256", async () => {
    const { repository, service, writer } = setup();
    const parentEntry = await writer.append({
      kind: "user_msg",
      content: "same image twice",
      turn_id: "turn-dedupe",
    });

    const first = await service.persistTurnAttachments({
      attachments: [{ mediaType: "image/png", bytes: PNG_1X1 }],
      streamWriter: writer,
      parentEntry,
      turnId: "turn-dedupe",
    });
    const second = await service.persistTurnAttachments({
      attachments: [{ mediaType: "image/png", bytes: PNG_1X1 }],
      streamWriter: writer,
      parentEntry,
      turnId: "turn-dedupe",
    });

    writer.close();

    expect(repository.get(first[0]!.attachmentId)?.sha256).toBe(
      repository.get(second[0]!.attachmentId)?.sha256,
    );
    expect(repository.get(first[0]!.attachmentId)?.storage_ref).toBe(
      repository.get(second[0]!.attachmentId)?.storage_ref,
    );
  });

  it("rejects unsupported media types visibly and logs a warning", () => {
    const { service } = setup();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    expect(() =>
      service.validateAttachments([
        {
          mediaType: "image/bmp" as "image/png",
          bytes: PNG_1X1,
        },
      ]),
    ).toThrow(AttachmentError);
    expect(warn).toHaveBeenCalled();

    warn.mockRestore();
  });

  it("rejects malformed images before storing them", () => {
    const { service } = setup();

    expect(() =>
      service.validateAttachments([
        {
          mediaType: "image/png",
          bytes: MALFORMED_PNG_1X1,
        },
      ]),
    ).toThrow(
      expect.objectContaining({
        code: "ATTACHMENT_IMAGE_MALFORMED",
      }),
    );
  });

  it("verifies blob integrity before returning image bytes", async () => {
    const { repository, service, writer } = setup();
    const parentEntry = await writer.append({
      kind: "user_msg",
      content: "hash check",
      turn_id: "turn-hash",
    });
    const [persisted] = await service.persistTurnAttachments({
      attachments: [{ mediaType: "image/png", bytes: PNG_1X1 }],
      streamWriter: writer,
      parentEntry,
      turnId: "turn-hash",
    });
    const record = repository.get(persisted!.attachmentId);
    writeFileSync(join(tempDir!, record!.storage_ref), Buffer.from("corrupt"));

    expect(() => service.fetchImageForLlm(persisted!.attachmentId)).toThrow(
      expect.objectContaining({
        code: "ATTACHMENT_BLOB_CORRUPTED",
      }),
    );
  });

  it("blocks fetches when the stream entry index marks an attachment inactive", async () => {
    const { service, writer } = setup();
    const parentEntry = await writer.append({
      kind: "user_msg",
      content: "abort me",
      turn_id: "turn-aborted-image",
    });
    const [persisted] = await service.persistTurnAttachments({
      attachments: [{ mediaType: "image/png", bytes: PNG_1X1 }],
      streamWriter: writer,
      parentEntry,
      turnId: "turn-aborted-image",
    });

    await writer.append({
      kind: "internal_event",
      content: {
        event: "aborted_turn",
        turn_id: "turn-aborted-image",
      },
    });

    expect(() => service.fetchImageForLlm(persisted!.attachmentId)).toThrow(
      expect.objectContaining({
        code: "ATTACHMENT_INACTIVE",
      }),
    );
  });

  it("reconciles stream entry links and inactive state from committed stream entries", async () => {
    const { repository, service, writer, entryIndex } = setup();
    const parentEntry = await writer.append({
      kind: "user_msg",
      content: "repair link",
      turn_id: "turn-repair",
    });
    const [persisted] = await service.persistTurnAttachments({
      attachments: [{ mediaType: "image/png", bytes: PNG_1X1 }],
      streamWriter: writer,
      parentEntry,
      turnId: "turn-repair",
    });
    repository.setActive(persisted!.attachmentId, false);
    repository.setStreamEntryId(persisted!.attachmentId, persisted!.streamEntry.id);
    await writer.append({
      kind: "internal_event",
      content: {
        event: "aborted_turn",
        turn_id: "turn-repair",
      },
    });

    await entryIndex.backfillSession(DEFAULT_SESSION_ID);
    expect(repository.reconcileFromStreamEntries([persisted!.streamEntry])).toBeGreaterThanOrEqual(
      0,
    );
    expect(repository.isActiveForStreamEntry(persisted!.streamEntry.id)).toBe(false);
  });
});
