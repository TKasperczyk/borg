import type { Migration, SqliteDatabase } from "../storage/sqlite/index.js";
import { AttachmentError } from "../util/errors.js";
import {
  type AttachmentId,
  type StreamEntryId,
  attachmentIdHelpers,
  streamEntryIdHelpers,
} from "../util/ids.js";
import type { ImageMediaType, StoredAttachmentRecord } from "./types.js";

export const attachmentMigrations: Migration[] = [
  {
    id: 1,
    name: "create-stream-attachments",
    up: `
      CREATE TABLE IF NOT EXISTS stream_attachments (
        attachment_id TEXT PRIMARY KEY,
        sha256 TEXT NOT NULL,
        media_type TEXT NOT NULL,
        byte_size INTEGER NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        storage_ref TEXT NOT NULL,
        thumbnail_ref TEXT NULL,
        perception_id TEXT NULL,
        text_embedding_ref TEXT NULL,
        visual_embedding_ref TEXT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        audience TEXT NULL,
        created_turn_global INTEGER NULL,
        parent_entry_id TEXT NOT NULL,
        stream_entry_id TEXT NULL UNIQUE,
        parent_turn_id TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_stream_attachments_sha256
      ON stream_attachments(sha256);
      CREATE INDEX IF NOT EXISTS idx_stream_attachments_parent_entry
      ON stream_attachments(parent_entry_id);
      CREATE INDEX IF NOT EXISTS idx_stream_attachments_stream_entry
      ON stream_attachments(stream_entry_id);
      CREATE INDEX IF NOT EXISTS idx_stream_attachments_parent_turn
      ON stream_attachments(parent_turn_id);
      CREATE INDEX IF NOT EXISTS idx_stream_attachments_active
      ON stream_attachments(active);
      CREATE INDEX IF NOT EXISTS idx_stream_attachments_audience
      ON stream_attachments(audience);
    `,
  },
];

type AttachmentRow = {
  attachment_id: string;
  sha256: string;
  media_type: string;
  byte_size: number;
  width: number;
  height: number;
  storage_ref: string;
  thumbnail_ref: string | null;
  perception_id: string | null;
  text_embedding_ref: string | null;
  visual_embedding_ref: string | null;
  active: number;
  audience: string | null;
  created_turn_global: number | null;
  parent_entry_id: string;
  stream_entry_id: string | null;
  parent_turn_id: string;
  created_at: number;
};

function rowToRecord(row: AttachmentRow): StoredAttachmentRecord {
  return {
    attachment_id: attachmentIdHelpers.parse(row.attachment_id) as AttachmentId,
    sha256: row.sha256,
    media_type: row.media_type as ImageMediaType,
    byte_size: row.byte_size,
    width: row.width,
    height: row.height,
    storage_ref: row.storage_ref,
    thumbnail_ref: row.thumbnail_ref,
    perception_id: row.perception_id,
    text_embedding_ref: row.text_embedding_ref,
    visual_embedding_ref: row.visual_embedding_ref,
    active: row.active !== 0,
    audience: row.audience,
    created_turn_global: row.created_turn_global,
    parent_entry_id: row.parent_entry_id as StreamEntryId,
    stream_entry_id: row.stream_entry_id as StreamEntryId | null,
    parent_turn_id: row.parent_turn_id,
    created_at: row.created_at,
  };
}

export class AttachmentRepository {
  constructor(private readonly db: SqliteDatabase) {}

  private hasImagePerceptionArtifactsTable(): boolean {
    return (
      this.db
        .prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?")
        .get("image_perception_artifacts") !== undefined
    );
  }

  private cascadePerceptionActiveByAttachment(attachmentId: AttachmentId, active: boolean): number {
    if (!this.hasImagePerceptionArtifactsTable()) {
      return 0;
    }

    return this.db
      .prepare(
        `UPDATE image_perception_artifacts
         SET active = ?
         WHERE attachment_id = ?`,
      )
      .run(active ? 1 : 0, attachmentId).changes;
  }

  private cascadeInactivePerceptionsFromInactiveAttachments(): void {
    if (!this.hasImagePerceptionArtifactsTable()) {
      return;
    }

    this.db
      .prepare(
        `UPDATE image_perception_artifacts
         SET active = 0
         WHERE active != 0
           AND EXISTS (
             SELECT 1
             FROM stream_attachments
             WHERE stream_attachments.attachment_id = image_perception_artifacts.attachment_id
               AND stream_attachments.active = 0
           )`,
      )
      .run();
  }

  insert(record: StoredAttachmentRecord): void {
    this.db
      .prepare(
        `INSERT INTO stream_attachments (
           attachment_id, sha256, media_type, byte_size, width, height, storage_ref,
           thumbnail_ref, perception_id, text_embedding_ref, visual_embedding_ref,
           active, audience, created_turn_global, parent_entry_id, stream_entry_id,
           parent_turn_id, created_at
         )
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        record.attachment_id,
        record.sha256,
        record.media_type,
        record.byte_size,
        record.width,
        record.height,
        record.storage_ref,
        record.thumbnail_ref,
        record.perception_id,
        record.text_embedding_ref,
        record.visual_embedding_ref,
        record.active ? 1 : 0,
        record.audience,
        record.created_turn_global,
        record.parent_entry_id,
        record.stream_entry_id,
        record.parent_turn_id,
        record.created_at,
      );
  }

  get(attachmentId: AttachmentId): StoredAttachmentRecord | null {
    const row = this.db
      .prepare(
        `SELECT attachment_id, sha256, media_type, byte_size, width, height, storage_ref,
                thumbnail_ref, perception_id, text_embedding_ref, visual_embedding_ref,
                active, audience, created_turn_global, parent_entry_id, stream_entry_id,
                parent_turn_id, created_at
         FROM stream_attachments
         WHERE attachment_id = ?`,
      )
      .get(attachmentId) as AttachmentRow | undefined;

    return row === undefined ? null : rowToRecord(row);
  }

  getActive(attachmentId: AttachmentId): StoredAttachmentRecord {
    const record = this.get(attachmentId);

    if (record === null) {
      throw new AttachmentError(`Attachment ${attachmentId} was not found`, {
        code: "ATTACHMENT_NOT_FOUND",
      });
    }

    if (!record.active) {
      throw new AttachmentError(`Attachment ${attachmentId} is inactive`, {
        code: "ATTACHMENT_INACTIVE",
      });
    }

    return record;
  }

  listByParentEntry(parentEntryId: StreamEntryId): StoredAttachmentRecord[] {
    const rows = this.db
      .prepare(
        `SELECT attachment_id, sha256, media_type, byte_size, width, height, storage_ref,
                thumbnail_ref, perception_id, text_embedding_ref, visual_embedding_ref,
                active, audience, created_turn_global, parent_entry_id, stream_entry_id,
                parent_turn_id, created_at
         FROM stream_attachments
         WHERE parent_entry_id = ?
         ORDER BY created_at ASC, attachment_id ASC`,
      )
      .all(parentEntryId) as AttachmentRow[];

    return rows.map(rowToRecord);
  }

  setActive(attachmentId: AttachmentId, active: boolean): number {
    return this.setActiveWithCascade(attachmentId, active).attachmentChanges;
  }

  setActiveWithCascade(
    attachmentId: AttachmentId,
    active: boolean,
  ): { attachmentChanges: number; perceptionArtifactChanges: number } {
    const result = this.db
      .prepare(
        `UPDATE stream_attachments
         SET active = ?
         WHERE attachment_id = ?`,
      )
      .run(active ? 1 : 0, attachmentId);
    let perceptionArtifactChanges = 0;
    if (result.changes > 0) {
      perceptionArtifactChanges = this.cascadePerceptionActiveByAttachment(attachmentId, active);
    }
    return {
      attachmentChanges: result.changes,
      perceptionArtifactChanges,
    };
  }

  setPerceptionRefs(
    attachmentId: AttachmentId,
    refs: {
      perceptionId: string | null;
      textEmbeddingRef: string | null;
    },
  ): void {
    this.db
      .prepare(
        `UPDATE stream_attachments
         SET perception_id = ?,
             text_embedding_ref = ?
         WHERE attachment_id = ?`,
      )
      .run(refs.perceptionId, refs.textEmbeddingRef, attachmentId);
  }

  setStreamEntryId(attachmentId: AttachmentId, streamEntryId: StreamEntryId): void {
    this.db
      .prepare(
        `UPDATE stream_attachments
         SET stream_entry_id = ?,
             active = 1
         WHERE attachment_id = ?`,
      )
      .run(streamEntryId, attachmentId);
  }

  reconcileStreamEntryLink(input: {
    attachmentId: AttachmentId;
    streamEntryId: StreamEntryId;
    parentEntryId: StreamEntryId;
  }): void {
    this.db
      .prepare(
        `UPDATE stream_attachments
         SET stream_entry_id = ?,
             active = CASE WHEN active = 0 AND stream_entry_id IS NULL THEN 1 ELSE active END
         WHERE attachment_id = ?
           AND parent_entry_id = ?
           AND (stream_entry_id IS NULL OR stream_entry_id = ?)`,
      )
      .run(input.streamEntryId, input.attachmentId, input.parentEntryId, input.streamEntryId);
  }

  reconcileActiveStateFromStreamIndex(): number {
    const result = this.db
      .prepare(
        `UPDATE stream_attachments
         SET active = 0
         WHERE stream_entry_id IS NOT NULL
           AND active != 0
           AND (
             EXISTS (
               SELECT 1
               FROM stream_entry_index
               WHERE stream_entry_index.entry_id = stream_attachments.stream_entry_id
                 AND stream_entry_index.active = 0
             )
             OR EXISTS (
               SELECT 1
               FROM stream_entry_index
               WHERE stream_entry_index.entry_id = stream_attachments.parent_entry_id
                 AND stream_entry_index.active = 0
             )
           )`,
      )
      .run();

    this.cascadeInactivePerceptionsFromInactiveAttachments();
    return result.changes;
  }

  isActiveForStreamEntry(streamEntryId: StreamEntryId): boolean | null {
    const row = this.db
      .prepare(
        `SELECT stream_attachments.active AS attachment_active,
                stream_entry_index.active AS stream_active,
                parent_entry_index.active AS parent_active
         FROM stream_attachments
         LEFT JOIN stream_entry_index
           ON stream_entry_index.entry_id = stream_attachments.stream_entry_id
         LEFT JOIN stream_entry_index AS parent_entry_index
           ON parent_entry_index.entry_id = stream_attachments.parent_entry_id
         WHERE stream_entry_id = ?`,
      )
      .get(streamEntryId) as
      | { attachment_active: number; stream_active: number | null; parent_active: number | null }
      | undefined;

    if (row === undefined) {
      return null;
    }

    return row.attachment_active !== 0 && row.stream_active === 1 && row.parent_active === 1;
  }

  reconcileFromStreamEntries(
    entries: readonly {
      id: StreamEntryId;
      kind: string;
      content: unknown;
    }[],
  ): number {
    let reconciled = 0;

    const update = this.db.prepare(
      `UPDATE stream_attachments
       SET stream_entry_id = ?,
           active = CASE WHEN active = 0 AND stream_entry_id IS NULL THEN 1 ELSE active END
       WHERE attachment_id = ?
         AND parent_entry_id = ?
         AND (stream_entry_id IS NULL OR stream_entry_id = ?)`,
    );

    const apply = this.db.transaction(() => {
      for (const entry of entries) {
        if (entry.kind !== "user_image_attachment") {
          continue;
        }

        if (
          entry.content === null ||
          typeof entry.content !== "object" ||
          Array.isArray(entry.content)
        ) {
          continue;
        }

        const content = entry.content as Record<string, unknown>;
        const attachmentId = content.attachment_id;
        const parentEntryId = content.parent_entry_id;

        if (
          typeof attachmentId !== "string" ||
          !attachmentIdHelpers.is(attachmentId) ||
          typeof parentEntryId !== "string" ||
          !streamEntryIdHelpers.is(parentEntryId)
        ) {
          continue;
        }

        reconciled += update.run(entry.id, attachmentId, parentEntryId, entry.id).changes;
      }
    });

    apply();
    reconciled += this.reconcileActiveStateFromStreamIndex();
    return reconciled;
  }
}
