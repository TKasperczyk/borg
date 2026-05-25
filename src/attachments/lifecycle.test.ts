import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { composeMigrations, openDatabase, type SqliteDatabase } from "../storage/sqlite/index.js";
import { createAttachmentId, createImagePerceptionId } from "../util/ids.js";
import { attachmentMigrations, AttachmentRepository } from "./repository.js";
import {
  createImagePerceptionTableSchema,
  imagePerceptionMigrations,
  ImagePerceptionRepository,
  type ImagePerceptionRecord,
} from "./perception.js";
import { LanceDbStore } from "../storage/lancedb/index.js";
import { ImageAttachmentLifecycleService } from "./lifecycle.js";

describe("ImageAttachmentLifecycleService", () => {
  let cleanup: (() => Promise<void> | void)[] = [];

  afterEach(async () => {
    for (const item of cleanup.splice(0).reverse()) {
      await item();
    }
  });

  it("quarantines attachments through the ordered cascade", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-image-lifecycle-"));
    const db: SqliteDatabase = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(attachmentMigrations, imagePerceptionMigrations),
    });
    const store = new LanceDbStore({ uri: join(tempDir, "lancedb") });
    const table = await store.openTable({
      name: "image_perception_embeddings",
      schema: createImagePerceptionTableSchema(4),
    });
    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const attachmentRepository = new AttachmentRepository(db);
    const imagePerceptionRepository = new ImagePerceptionRepository(db, table);
    const attachmentId = createAttachmentId();
    const perceptionId = createImagePerceptionId();
    const payloadId = createImagePerceptionId();

    attachmentRepository.insert({
      attachment_id: attachmentId,
      sha256: "sha",
      media_type: "image/png",
      byte_size: 12,
      width: 1,
      height: 1,
      storage_ref: "attachments/sha.png",
      thumbnail_ref: null,
      perception_id: perceptionId,
      text_embedding_ref: `image_perception_embeddings:${payloadId}`,
      visual_embedding_ref: null,
      active: true,
      audience: "Alice",
      created_turn_global: 7,
      parent_entry_id: "strm_aaaaaaaaaaaaaaaa" as never,
      stream_entry_id: "strm_bbbbbbbbbbbbbbbb" as never,
      parent_turn_id: "turn-image",
      created_at: 1,
    });

    const record: ImagePerceptionRecord = {
      perception_id: perceptionId,
      payload_id: payloadId,
      attachment_id: attachmentId,
      parent_entry_id: "strm_aaaaaaaaaaaaaaaa" as never,
      parent_turn_id: "turn-image",
      stream_entry_id: "strm_bbbbbbbbbbbbbbbb" as never,
      sha256: "sha",
      media_type: "image/png",
      perception_prompt_version: "test",
      model: "fake",
      caption: "caption",
      image_kind: "photo",
      visible_text: [],
      objects: [],
      people_or_roles: [],
      scene: "scene",
      colors_and_visual_attributes: [],
      spatial_relationships: [],
      possible_user_relevant_details: [],
      search_terms: ["caption"],
      uncertainties: [],
      audience: "Alice",
      active: true,
      created_turn_global: 7,
      created_at: 1,
      text_embedding_ref: `image_perception_embeddings:${payloadId}`,
      embedding_text: "caption",
      embedding_status: "pending",
    };
    imagePerceptionRepository.insertPayload(record);
    imagePerceptionRepository.upsertArtifact(record);

    const service = new ImageAttachmentLifecycleService({
      db,
      attachmentRepository,
    });

    const result = service.quarantineImageAttachment({
      attachmentId,
      reason: "test",
      turnId: "turn-test",
    });

    expect(result).toEqual({ attachmentChanged: true, perceptionArtifactsChanged: 1 });
    expect(attachmentRepository.get(attachmentId)?.active).toBe(false);
    expect(imagePerceptionRepository.get(perceptionId)?.active).toBe(false);
  });
});
