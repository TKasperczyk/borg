import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { FakeEmbeddingClient, type EmbeddingClient } from "../embeddings/index.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import type { LanceDbRow, LanceDbTable } from "../storage/lancedb/index.js";
import { composeMigrations, openDatabase, type SqliteDatabase } from "../storage/sqlite/index.js";
import { createAttachmentId, type StreamEntryId } from "../util/ids.js";
import { AttachmentRepository, attachmentMigrations } from "./repository.js";
import {
  IMAGE_PERCEPTION_TOOL_NAME,
  ImagePerceptionRepository,
  ImagePerceptionService,
  imagePerceptionMigrations,
} from "./perception.js";

describe("ImagePerceptionService", () => {
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
    tempDir = mkdtempSync(join(tmpdir(), "borg-image-perception-"));
    db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(attachmentMigrations, imagePerceptionMigrations),
    });
    const attachmentRepository = new AttachmentRepository(db);
    const tableRows: LanceDbRow[] = [];
    const table = {
      async upsert(rows: readonly LanceDbRow[]) {
        tableRows.push(...rows);
      },
      async search() {
        return tableRows.map((row) => ({
          ...row,
          _distance: 0.05,
        }));
      },
    } as unknown as LanceDbTable;
    const repository = new ImagePerceptionRepository(db, table);
    const attachmentId = createAttachmentId();
    attachmentRepository.insert({
      attachment_id: attachmentId,
      sha256: "abc123",
      media_type: "image/png",
      byte_size: 12,
      width: 1,
      height: 1,
      storage_ref: "attachments/abc123.png",
      thumbnail_ref: null,
      perception_id: null,
      text_embedding_ref: null,
      visual_embedding_ref: null,
      active: true,
      audience: "Alice",
      created_turn_global: 42,
      parent_entry_id: "strm_aaaaaaaaaaaaaaaa" as StreamEntryId,
      stream_entry_id: "strm_bbbbbbbbbbbbbbbb" as StreamEntryId,
      parent_turn_id: "turn-image",
      created_at: 1_000,
    });

    return {
      attachmentId,
      attachmentRepository,
      repository,
      tableRows,
    };
  }

  class FailingOnceEmbeddingClient implements EmbeddingClient {
    attempts = 0;

    async embed(): Promise<Float32Array> {
      this.attempts += 1;
      if (this.attempts === 1) {
        throw new Error("embedding unavailable");
      }

      return Float32Array.from([1, 0, 0, 0]);
    }

    async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
      return texts.map(() => Float32Array.from([1, 0, 0, 0]));
    }
  }

  it("stores structured perception, embeds recall text, and populates attachment refs", async () => {
    const { attachmentId, attachmentRepository, repository, tableRows } = setup();
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_image",
            name: IMAGE_PERCEPTION_TOOL_NAME,
            input: {
              caption: "A screenshot of a release checklist.",
              image_kind: "screenshot",
              visible_text: ["Release v88"],
              objects: ["checklist"],
              people_or_roles: [],
              scene: "A software planning interface.",
              colors_and_visual_attributes: ["dark header"],
              spatial_relationships: ["checklist centered"],
              possible_user_relevant_details: ["release v88 planning"],
              search_terms: ["release checklist", "v88 screenshot", "planning UI"],
              uncertainties: [],
            },
          },
        ],
      ],
    });
    const service = new ImagePerceptionService({
      repository,
      attachmentRepository,
      llmClient: llm,
      embeddingClient: new FakeEmbeddingClient(4),
      model: "haiku-test",
      promptVersion: "test-v1",
    });

    const record = await service.perceiveAttachment({
      attachmentId,
      turnId: "turn-image",
    });

    expect(record?.created_turn_global).toBe(42);
    expect(record?.audience).toBe("Alice");
    expect(record?.search_terms).toContain("v88 screenshot");
    expect(record?.embedding_text).toContain("search_terms: release checklist");
    expect(tableRows).toHaveLength(1);
    expect(attachmentRepository.get(attachmentId)?.perception_id).toBe(record?.perception_id);
    expect(attachmentRepository.get(attachmentId)?.text_embedding_ref).toBe(
      record?.text_embedding_ref,
    );
  });

  it("uses the sha/media/prompt/model cache instead of rerunning perception", async () => {
    const { attachmentId, attachmentRepository, repository } = setup();
    const firstLlm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_image",
            name: IMAGE_PERCEPTION_TOOL_NAME,
            input: {
              caption: "A diagram.",
              image_kind: "diagram",
              visible_text: [],
              objects: ["box"],
              people_or_roles: [],
              scene: "A simple diagram.",
              colors_and_visual_attributes: [],
              spatial_relationships: [],
              possible_user_relevant_details: ["box diagram"],
              search_terms: ["box diagram"],
              uncertainties: [],
            },
          },
        ],
      ],
    });
    const firstService = new ImagePerceptionService({
      repository,
      attachmentRepository,
      llmClient: firstLlm,
      embeddingClient: new FakeEmbeddingClient(4),
      model: "haiku-test",
      promptVersion: "test-v1",
    });
    const first = await firstService.perceiveAttachment({ attachmentId, turnId: "turn-image" });
    const secondLlm = new FakeLLMClient();
    const secondService = new ImagePerceptionService({
      repository,
      attachmentRepository,
      llmClient: secondLlm,
      embeddingClient: new FakeEmbeddingClient(4),
      model: "haiku-test",
      promptVersion: "test-v1",
    });

    const second = await secondService.perceiveAttachment({ attachmentId, turnId: "turn-image" });

    expect(second?.perception_id).toBe(first?.perception_id);
    expect(secondLlm.converseRequests).toHaveLength(0);
  });

  it("reuses the payload cache while creating source-scoped artifacts per audience", async () => {
    const { attachmentId, attachmentRepository, repository, tableRows } = setup();
    const bobAttachmentId = createAttachmentId();
    attachmentRepository.insert({
      attachment_id: bobAttachmentId,
      sha256: "abc123",
      media_type: "image/png",
      byte_size: 12,
      width: 1,
      height: 1,
      storage_ref: "attachments/abc123-bob.png",
      thumbnail_ref: null,
      perception_id: null,
      text_embedding_ref: null,
      visual_embedding_ref: null,
      active: true,
      audience: "Bob",
      created_turn_global: 43,
      parent_entry_id: "strm_cccccccccccccccc" as StreamEntryId,
      stream_entry_id: "strm_dddddddddddddddd" as StreamEntryId,
      parent_turn_id: "turn-image-bob",
      created_at: 2_000,
    });
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_image",
            name: IMAGE_PERCEPTION_TOOL_NAME,
            input: {
              caption: "A shared diagram.",
              image_kind: "diagram",
              visible_text: [],
              objects: ["box"],
              people_or_roles: [],
              scene: "A simple diagram.",
              colors_and_visual_attributes: [],
              spatial_relationships: [],
              possible_user_relevant_details: ["box diagram"],
              search_terms: ["box diagram"],
              uncertainties: [],
            },
          },
        ],
      ],
    });
    const service = new ImagePerceptionService({
      repository,
      attachmentRepository,
      llmClient: llm,
      embeddingClient: new FakeEmbeddingClient(4),
      model: "haiku-test",
      promptVersion: "test-v1",
    });

    const alice = await service.perceiveAttachment({ attachmentId, turnId: "turn-image" });
    const bob = await service.perceiveAttachment({
      attachmentId: bobAttachmentId,
      turnId: "turn-image-bob",
    });

    expect(llm.converseRequests).toHaveLength(1);
    expect(tableRows).toHaveLength(1);
    expect(bob?.payload_id).toBe(alice?.payload_id);
    expect(bob?.perception_id).not.toBe(alice?.perception_id);
    expect(bob?.attachment_id).toBe(bobAttachmentId);
    expect(bob?.audience).toBe("Bob");
    expect(bob?.parent_entry_id).toBe("strm_cccccccccccccccc");
  });

  it("keeps a payload after embedding failure and retries embedding on cache hit", async () => {
    const { attachmentId, attachmentRepository, repository, tableRows } = setup();
    const embedding = new FailingOnceEmbeddingClient();
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_image",
            name: IMAGE_PERCEPTION_TOOL_NAME,
            input: {
              caption: "A screenshot.",
              image_kind: "screenshot",
              visible_text: ["Retry me"],
              objects: [],
              people_or_roles: [],
              scene: "A retry case.",
              colors_and_visual_attributes: [],
              spatial_relationships: [],
              possible_user_relevant_details: ["retry screenshot"],
              search_terms: ["retry screenshot"],
              uncertainties: [],
            },
          },
        ],
      ],
    });
    const service = new ImagePerceptionService({
      repository,
      attachmentRepository,
      llmClient: llm,
      embeddingClient: embedding,
      model: "haiku-test",
      promptVersion: "test-v1",
    });

    const first = await service.perceiveAttachment({ attachmentId, turnId: "turn-image" });
    const second = await service.perceiveAttachment({ attachmentId, turnId: "turn-image" });

    expect(first?.embedding_status).toBe("failed");
    expect(second?.embedding_status).toBe("complete");
    expect(embedding.attempts).toBe(2);
    expect(llm.converseRequests).toHaveLength(1);
    expect(tableRows).toHaveLength(1);
  });

  it("marks perception artifacts inactive when the source attachment becomes inactive", async () => {
    const { attachmentId, attachmentRepository, repository } = setup();
    const service = new ImagePerceptionService({
      repository,
      attachmentRepository,
      llmClient: new FakeLLMClient({
        responses: [
          [
            {
              type: "tool_use",
              id: "toolu_image",
              name: IMAGE_PERCEPTION_TOOL_NAME,
              input: {
                caption: "A photo.",
                image_kind: "photo",
                visible_text: [],
                objects: ["table"],
                people_or_roles: [],
                scene: "Indoor table.",
                colors_and_visual_attributes: [],
                spatial_relationships: [],
                possible_user_relevant_details: ["table photo"],
                search_terms: ["table photo"],
                uncertainties: [],
              },
            },
          ],
        ],
      }),
      embeddingClient: new FakeEmbeddingClient(4),
      model: "haiku-test",
      promptVersion: "test-v1",
    });
    const record = await service.perceiveAttachment({ attachmentId, turnId: "turn-image" });

    attachmentRepository.setActive(attachmentId, false);

    expect(repository.get(record!.perception_id)?.active).toBe(false);
  });
});
