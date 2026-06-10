import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { FakeEmbeddingClient, type EmbeddingClient } from "../embeddings/index.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import type { LanceDbRow, LanceDbTable } from "../storage/lancedb/index.js";
import { composeMigrations, openDatabase, type SqliteDatabase } from "../storage/sqlite/index.js";
import { createAttachmentId, type StreamEntryId } from "../util/ids.js";
import { imagePerceptionMemoryDisclosureLabel } from "../memory/common/disclosure-serializers.js";
import { AttachmentRepository, attachmentMigrations } from "./repository.js";
import {
  IMAGE_PERCEPTION_TOOL_NAME,
  ImagePerceptionRepository,
  ImagePerceptionService,
  imagePerceptionMigrations,
} from "./perception.js";

const ALICE_ENTITY_ID = "ent_aaaaaaaaaaaaaaaa" as never;
const BOB_ENTITY_ID = "ent_bbbbbbbbbbbbbbbb" as never;

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
      audience_entity_id: ALICE_ENTITY_ID,
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

  class CountingEmbeddingClient implements EmbeddingClient {
    attempts = 0;

    async embed(): Promise<Float32Array> {
      this.attempts += 1;
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

  it("can use deterministic sha256-indexed perception artifacts for tests and sims", async () => {
    const { attachmentId, attachmentRepository, repository } = setup();
    const service = new ImagePerceptionService({
      repository,
      attachmentRepository,
      llmClient: new FakeLLMClient({ responses: [] }),
      embeddingClient: new FakeEmbeddingClient(4),
      model: "haiku-test",
      promptVersion: "test-v1",
      artifactBySha256: {
        abc123: {
          caption: "Fixture perception by hash.",
          image_kind: "diagram",
          visible_text: ["fixture"],
          objects: ["box"],
          people_or_roles: [],
          scene: "A deterministic simulator fixture.",
          colors_and_visual_attributes: [],
          spatial_relationships: [],
          possible_user_relevant_details: ["hash fixture"],
          search_terms: ["hash fixture"],
          uncertainties: [],
        },
      },
    });

    const record = await service.perceiveAttachment({
      attachmentId,
      turnId: "turn-image",
    });

    expect(record?.caption).toBe("Fixture perception by hash.");
    expect(record?.search_terms).toEqual(["hash fixture"]);
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
      audience_entity_id: BOB_ENTITY_ID,
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

  it("filters image disclosure by audience entity id, not alias, and fails closed for null origins", async () => {
    const { attachmentId, attachmentRepository, repository } = setup();
    const bobAttachmentId = createAttachmentId();
    const unknownAttachmentId = createAttachmentId();
    db!.prepare("UPDATE stream_attachments SET audience = ? WHERE attachment_id = ?").run(
      "Alex",
      attachmentId,
    );

    for (const record of [
      {
        attachment_id: bobAttachmentId,
        audience_entity_id: BOB_ENTITY_ID,
        parent_entry_id: "strm_cccccccccccccccc" as StreamEntryId,
        stream_entry_id: "strm_dddddddddddddddd" as StreamEntryId,
        parent_turn_id: "turn-image-bob",
        created_turn_global: 43,
      },
      {
        attachment_id: unknownAttachmentId,
        audience_entity_id: null,
        parent_entry_id: "strm_eeeeeeeeeeeeeeee" as StreamEntryId,
        stream_entry_id: "strm_ffffffffffffffff" as StreamEntryId,
        parent_turn_id: "turn-image-unknown",
        created_turn_global: 44,
      },
    ]) {
      attachmentRepository.insert({
        attachment_id: record.attachment_id,
        sha256: "abc123",
        media_type: "image/png",
        byte_size: 12,
        width: 1,
        height: 1,
        storage_ref: `attachments/${record.attachment_id}.png`,
        thumbnail_ref: null,
        perception_id: null,
        text_embedding_ref: null,
        visual_embedding_ref: null,
        active: true,
        audience: "Alex",
        audience_entity_id: record.audience_entity_id,
        created_turn_global: record.created_turn_global,
        parent_entry_id: record.parent_entry_id,
        stream_entry_id: record.stream_entry_id,
        parent_turn_id: record.parent_turn_id,
        created_at: record.created_turn_global * 1_000,
      });
    }

    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_image",
            name: IMAGE_PERCEPTION_TOOL_NAME,
            input: {
              caption: "A shared alias diagram.",
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
    const unknown = await service.perceiveAttachment({
      attachmentId: unknownAttachmentId,
      turnId: "turn-image-unknown",
    });

    const aliceHits = await repository.searchForDisclosure({
      vector: Float32Array.from([1, 0, 0, 0]),
      limit: 5,
      audienceEntityId: ALICE_ENTITY_ID,
    });
    const bobHits = await repository.searchForDisclosure({
      vector: Float32Array.from([1, 0, 0, 0]),
      limit: 5,
      audienceEntityId: BOB_ENTITY_ID,
    });
    const nullAudienceHits = await repository.searchForDisclosure({
      vector: Float32Array.from([1, 0, 0, 0]),
      limit: 5,
      audienceEntityId: null,
    });
    const adminHits = await repository.searchForDisclosure({
      vector: Float32Array.from([1, 0, 0, 0]),
      limit: 5,
      audienceEntityId: ALICE_ENTITY_ID,
      crossAudience: true,
    });

    expect(aliceHits.map((hit) => hit.record.attachment_id)).toEqual([attachmentId]);
    expect(bobHits.map((hit) => hit.record.attachment_id)).toEqual([bobAttachmentId]);
    expect(nullAudienceHits).toEqual([]);
    expect(adminHits.map((hit) => hit.record.attachment_id)).toEqual(
      expect.arrayContaining([attachmentId, bobAttachmentId, unknownAttachmentId]),
    );
    expect(imagePerceptionMemoryDisclosureLabel(alice!)).toMatchObject({
      disclosureClass: "relationship_private",
      privateToEntityIds: [ALICE_ENTITY_ID],
    });
    expect(imagePerceptionMemoryDisclosureLabel(bob!)).toMatchObject({
      disclosureClass: "relationship_private",
      privateToEntityIds: [BOB_ENTITY_ID],
    });
    expect(imagePerceptionMemoryDisclosureLabel(unknown!)).toMatchObject({
      disclosureClass: "unknown",
      privateToEntityIds: [],
    });
  });

  it("canonicalizes concurrent same-byte payload writes before linking artifacts", async () => {
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
      audience_entity_id: BOB_ENTITY_ID,
      created_turn_global: 43,
      parent_entry_id: "strm_cccccccccccccccc" as StreamEntryId,
      stream_entry_id: "strm_dddddddddddddddd" as StreamEntryId,
      parent_turn_id: "turn-image-bob",
      created_at: 2_000,
    });
    const imageResponse = {
      type: "tool_use" as const,
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
    };
    const llm = new FakeLLMClient({
      responses: [[imageResponse], [imageResponse]],
    });
    const embeddingClient = new CountingEmbeddingClient();
    const service = new ImagePerceptionService({
      repository,
      attachmentRepository,
      llmClient: llm,
      embeddingClient,
      model: "haiku-test",
      promptVersion: "test-v1",
    });

    const [alice, bob] = await Promise.all([
      service.perceiveAttachment({ attachmentId, turnId: "turn-image" }),
      service.perceiveAttachment({ attachmentId: bobAttachmentId, turnId: "turn-image-bob" }),
    ]);
    const payloadRows = db!
      .prepare("SELECT payload_id FROM image_perception_payloads")
      .all() as Array<{ payload_id: string }>;
    const artifactRows = db!
      .prepare(
        "SELECT attachment_id, payload_id FROM image_perception_artifacts ORDER BY attachment_id",
      )
      .all() as Array<{ attachment_id: string; payload_id: string }>;

    expect(payloadRows).toHaveLength(1);
    expect(alice?.payload_id).toBe(payloadRows[0]?.payload_id);
    expect(bob?.payload_id).toBe(payloadRows[0]?.payload_id);
    expect(new Set(artifactRows.map((row) => row.payload_id))).toEqual(
      new Set([payloadRows[0]?.payload_id]),
    );
    expect(artifactRows.map((row) => row.attachment_id).sort()).toEqual(
      [attachmentId, bobAttachmentId].sort(),
    );
    expect(tableRows).toHaveLength(1);
    expect(embeddingClient.attempts).toBe(1);
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
