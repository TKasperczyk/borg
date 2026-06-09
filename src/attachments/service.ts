import { createHash } from "node:crypto";

import { AttachmentError } from "../util/errors.js";
import {
  attachmentIdHelpers,
  createAttachmentId,
  streamEntryIdHelpers,
  type AttachmentId,
  type EntityId,
  type StreamEntryId,
} from "../util/ids.js";
import type {
  StreamEntry,
  StreamEntryIndexRepository,
  StreamEntryInput,
  StreamReader,
  StreamWriter,
} from "../stream/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { AttachmentBlobStore } from "./blob-store.js";
import { readImageDimensions } from "./image-info.js";
import { validateImageForFinalizerRender } from "./render-validation.js";
import type { ImageAttachmentLifecycleService } from "./lifecycle.js";
import { AttachmentRepository } from "./repository.js";
import {
  SUPPORTED_IMAGE_MEDIA_TYPES,
  imageMediaTypeSchema,
  type BorgUserContentBlock,
  type StoredAttachmentRecord,
  type TurnInputAttachment,
} from "./types.js";

type LoggerLike = Pick<Console, "warn">;

export type AttachmentValidationConfig = {
  maxBytesPerImage: number;
  maxWidth: number;
  maxHeight: number;
  maxImagesPerTurn: number;
};

export type AttachmentServiceOptions = {
  repository: AttachmentRepository;
  blobStore: AttachmentBlobStore;
  config: AttachmentValidationConfig;
  entryIndex?: Pick<StreamEntryIndexRepository, "lookup">;
  createStreamReader?: (sessionId: StreamEntry["session_id"]) => Pick<StreamReader, "iterate">;
  lifecycle?: Pick<
    ImageAttachmentLifecycleService,
    "quarantineImageAttachment" | "unquarantineImageAttachment"
  >;
  logger?: LoggerLike;
  tracer?: TurnTracer;
};

export type PersistTurnAttachmentsInput = {
  attachments: readonly TurnInputAttachment[];
  streamWriter: Pick<StreamWriter, "appendMany">;
  parentEntry: StreamEntry;
  turnId: string;
  audienceEntityId?: EntityId | null;
  createdTurnGlobal?: number;
};

export type PersistedTurnAttachment = {
  attachmentId: AttachmentId;
  streamEntry: StreamEntry;
  contentBlock: BorgUserContentBlock;
};

export type PersistParentEntryAttachmentsInput = {
  attachments: readonly TurnInputAttachment[];
  streamWriter: Pick<StreamWriter, "appendMany">;
  parentEntry: StreamEntry;
  audienceEntityId?: EntityId | null;
  createdTurnGlobal?: number;
};

export type PersistedParentEntryAttachment = {
  attachmentId: AttachmentId;
  streamEntry: StreamEntry | null;
  record: StoredAttachmentRecord;
  contentBlock: BorgUserContentBlock;
};

export class AttachmentService {
  private readonly repository: AttachmentRepository;
  private readonly blobStore: AttachmentBlobStore;
  private readonly config: AttachmentValidationConfig;
  private readonly entryIndex?: Pick<StreamEntryIndexRepository, "lookup">;
  private readonly createStreamReader?: (
    sessionId: StreamEntry["session_id"],
  ) => Pick<StreamReader, "iterate">;
  private readonly lifecycle?: Pick<
    ImageAttachmentLifecycleService,
    "quarantineImageAttachment" | "unquarantineImageAttachment"
  >;
  private readonly logger: LoggerLike;
  private readonly tracer?: TurnTracer;

  constructor(options: AttachmentServiceOptions) {
    this.repository = options.repository;
    this.blobStore = options.blobStore;
    this.config = options.config;
    this.entryIndex = options.entryIndex;
    this.createStreamReader = options.createStreamReader;
    this.lifecycle = options.lifecycle;
    this.logger = options.logger ?? console;
    this.tracer = options.tracer;
  }

  private emitRejected(reason: AttachmentRejectedReason): void {
    if (this.tracer?.enabled === true) {
      this.tracer.emit("attachment.rejected", {
        turnId: "attachment_validation",
        reason,
      });
    }
  }

  validateAttachments(attachments: readonly TurnInputAttachment[]): void {
    if (attachments.length > this.config.maxImagesPerTurn) {
      this.emitRejected("too_many_images");
      this.logger.warn(
        `Rejected image attachments: ${attachments.length} exceeds maxImagesPerTurn=${this.config.maxImagesPerTurn}`,
      );
      throw new AttachmentError("Too many image attachments for one turn", {
        code: "ATTACHMENT_TOO_MANY_IMAGES",
      });
    }

    for (const attachment of attachments) {
      const mediaType = imageMediaTypeSchema.safeParse(attachment.mediaType);

      if (!mediaType.success) {
        this.emitRejected("unsupported_media_type");
        this.logger.warn(
          `Rejected unsupported image media type: ${String(attachment.mediaType)}; supported=${SUPPORTED_IMAGE_MEDIA_TYPES.join(",")}`,
        );
        throw new AttachmentError(`Unsupported image media type: ${String(attachment.mediaType)}`, {
          code: "ATTACHMENT_UNSUPPORTED_MEDIA_TYPE",
        });
      }

      if (attachment.bytes.byteLength > this.config.maxBytesPerImage) {
        this.emitRejected("image_too_large");
        this.logger.warn(
          `Rejected oversized image: ${attachment.bytes.byteLength} bytes exceeds maxBytesPerImage=${this.config.maxBytesPerImage}`,
        );
        throw new AttachmentError("Image attachment exceeds maximum byte size", {
          code: "ATTACHMENT_IMAGE_TOO_LARGE",
        });
      }

      let dimensions: ReturnType<typeof readImageDimensions>;
      try {
        dimensions = readImageDimensions(attachment.bytes, mediaType.data);
      } catch (error) {
        if (error instanceof AttachmentError) {
          this.emitRejected(
            error.code === "ATTACHMENT_IMAGE_MALFORMED"
              ? "image_malformed"
              : "dimensions_unreadable",
          );
        }
        throw error;
      }

      if (dimensions.width > this.config.maxWidth || dimensions.height > this.config.maxHeight) {
        this.emitRejected("dimensions_too_large");
        this.logger.warn(
          `Rejected image dimensions: ${dimensions.width}x${dimensions.height} exceeds max=${this.config.maxWidth}x${this.config.maxHeight}`,
        );
        throw new AttachmentError("Image attachment exceeds maximum dimensions", {
          code: "ATTACHMENT_DIMENSIONS_TOO_LARGE",
        });
      }
    }
  }

  async persistTurnAttachments(
    input: PersistTurnAttachmentsInput,
  ): Promise<PersistedTurnAttachment[]> {
    this.validateAttachments(input.attachments);

    const prepared = input.attachments.map((attachment, ordinal) =>
      this.prepareNewAttachment({
        attachment,
        parentEntry: input.parentEntry,
        parentTurnId: input.turnId,
        audienceEntityId: input.audienceEntityId ?? null,
        createdTurnGlobal: input.createdTurnGlobal ?? null,
        ordinal,
        traceTurnId: input.turnId,
      }),
    );

    const streamEntries = await input.streamWriter.appendMany(
      prepared.map((item) => item.streamInput),
    );

    const persisted: PersistedTurnAttachment[] = [];

    for (let index = 0; index < prepared.length; index += 1) {
      const item = prepared[index];
      const streamEntry = streamEntries[index];

      if (item === undefined || streamEntry === undefined) {
        continue;
      }

      this.repository.setStreamEntryId(item.attachmentId, streamEntry.id);

      persisted.push({
        attachmentId: item.attachmentId,
        streamEntry,
        contentBlock: {
          type: "image_ref",
          attachment_id: item.attachmentId,
        },
      });
    }

    return persisted;
  }

  async persistParentEntryAttachments(
    input: PersistParentEntryAttachmentsInput,
  ): Promise<PersistedParentEntryAttachment[]> {
    this.validateAttachments(input.attachments);

    if (input.attachments.length === 0) {
      return [];
    }

    const existing = this.repository
      .listByParentEntry(input.parentEntry.id)
      .slice(0, input.attachments.length);
    const existingResults = await this.linkExistingParentEntryAttachments({
      records: existing,
      streamWriter: input.streamWriter,
      parentEntry: input.parentEntry,
    });
    const missingAttachments = input.attachments.slice(existing.length);
    const prepared = missingAttachments.map((attachment, missingIndex) =>
      this.prepareNewAttachment({
        attachment,
        parentEntry: input.parentEntry,
        parentTurnId: null,
        audienceEntityId: input.audienceEntityId ?? null,
        createdTurnGlobal: input.createdTurnGlobal ?? null,
        ordinal: existing.length + missingIndex,
        traceTurnId: input.parentEntry.id,
      }),
    );
    const streamEntries = await input.streamWriter.appendMany(
      prepared.map((item) => item.streamInput),
    );
    const persisted: PersistedParentEntryAttachment[] = [...existingResults];

    for (let index = 0; index < prepared.length; index += 1) {
      const item = prepared[index];
      const streamEntry = streamEntries[index];

      if (item === undefined || streamEntry === undefined) {
        continue;
      }

      this.repository.setStreamEntryId(item.attachmentId, streamEntry.id);
      const record = this.repository.get(item.attachmentId);

      if (record === null) {
        continue;
      }

      persisted.push({
        attachmentId: item.attachmentId,
        streamEntry,
        record,
        contentBlock: {
          type: "image_ref",
          attachment_id: item.attachmentId,
        },
      });
    }

    return persisted;
  }

  private async linkExistingParentEntryAttachments(input: {
    records: readonly StoredAttachmentRecord[];
    streamWriter: Pick<StreamWriter, "appendMany">;
    parentEntry: StreamEntry;
  }): Promise<PersistedParentEntryAttachment[]> {
    if (input.records.length === 0) {
      return [];
    }

    const unlinked = input.records.filter((record) => record.stream_entry_id === null);
    const linkedStreamEntries = new Map<AttachmentId, StreamEntry>();

    if (unlinked.length > 0) {
      const recoveredStreamEntries = await this.reconcileCommittedParentEntryAttachmentStreams({
        records: unlinked,
        parentEntry: input.parentEntry,
      });
      for (const [attachmentId, streamEntry] of recoveredStreamEntries) {
        linkedStreamEntries.set(attachmentId, streamEntry);
      }

      const stillUnlinked = unlinked.filter(
        (record) => !recoveredStreamEntries.has(record.attachment_id),
      );
      const streamEntries = await input.streamWriter.appendMany(
        stillUnlinked.map((record) =>
          this.buildAttachmentStreamInput({
            attachmentId: record.attachment_id,
            mediaType: record.media_type,
            parentEntry: input.parentEntry,
            turnId: null,
          }),
        ),
      );

      for (let index = 0; index < stillUnlinked.length; index += 1) {
        const record = stillUnlinked[index];
        const streamEntry = streamEntries[index];

        if (record === undefined || streamEntry === undefined) {
          continue;
        }

        this.repository.setStreamEntryId(record.attachment_id, streamEntry.id);
        linkedStreamEntries.set(record.attachment_id, streamEntry);
      }
    }

    return input.records.flatMap((record) => {
      const refreshed = this.repository.get(record.attachment_id);

      if (refreshed === null) {
        return [];
      }

      return [
        {
          attachmentId: refreshed.attachment_id,
          streamEntry: linkedStreamEntries.get(refreshed.attachment_id) ?? null,
          record: refreshed,
          contentBlock: {
            type: "image_ref" as const,
            attachment_id: refreshed.attachment_id,
          },
        },
      ];
    });
  }

  private async reconcileCommittedParentEntryAttachmentStreams(input: {
    records: readonly StoredAttachmentRecord[];
    parentEntry: StreamEntry;
  }): Promise<Map<AttachmentId, StreamEntry>> {
    if (this.createStreamReader === undefined || input.records.length === 0) {
      return new Map();
    }

    const wanted = new Set(input.records.map((record) => record.attachment_id));
    const recovered = new Map<AttachmentId, StreamEntry>();

    for await (const entry of this.createStreamReader(input.parentEntry.session_id).iterate({
      kinds: ["user_image_attachment"],
    })) {
      if (wanted.size === 0) {
        break;
      }

      const ref = this.attachmentStreamRef(entry);

      if (
        ref === null ||
        ref.parentEntryId !== input.parentEntry.id ||
        !wanted.has(ref.attachmentId)
      ) {
        continue;
      }

      this.repository.reconcileStreamEntryLink({
        attachmentId: ref.attachmentId,
        streamEntryId: entry.id,
        parentEntryId: ref.parentEntryId,
      });
      recovered.set(ref.attachmentId, entry);
      wanted.delete(ref.attachmentId);
    }

    return recovered;
  }

  private attachmentStreamRef(
    entry: Pick<StreamEntry, "content">,
  ): { attachmentId: AttachmentId; parentEntryId: StreamEntryId } | null {
    if (
      entry.content === null ||
      typeof entry.content !== "object" ||
      Array.isArray(entry.content)
    ) {
      return null;
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
      return null;
    }

    return {
      attachmentId,
      parentEntryId,
    };
  }

  private prepareNewAttachment(input: {
    attachment: TurnInputAttachment;
    parentEntry: StreamEntry;
    parentTurnId: string | null;
    audienceEntityId: EntityId | null;
    createdTurnGlobal: number | null;
    ordinal: number;
    traceTurnId: string;
  }): {
    attachmentId: AttachmentId;
    streamInput: StreamEntryInput;
  } {
    const dimensions = readImageDimensions(input.attachment.bytes, input.attachment.mediaType);
    const blob = this.blobStore.write(input.attachment.bytes, input.attachment.mediaType);
    const attachmentId = createAttachmentId();

    this.repository.insert({
      attachment_id: attachmentId,
      sha256: blob.sha256,
      media_type: input.attachment.mediaType,
      byte_size: input.attachment.bytes.byteLength,
      width: dimensions.width,
      height: dimensions.height,
      storage_ref: blob.storageRef,
      thumbnail_ref: null,
      perception_id: null,
      text_embedding_ref: null,
      visual_embedding_ref: null,
      active: false,
      audience: input.parentEntry.audience ?? null,
      audience_entity_id: input.audienceEntityId,
      created_turn_global: input.createdTurnGlobal,
      ordinal: input.ordinal,
      parent_entry_id: input.parentEntry.id,
      stream_entry_id: null,
      parent_turn_id: input.parentTurnId,
      created_at: input.parentEntry.timestamp,
    });

    if (this.tracer?.enabled === true) {
      this.tracer.emit("attachment.write", {
        turnId: input.traceTurnId,
        attachment_id: attachmentId,
        sha256: blob.sha256,
        media_type: input.attachment.mediaType,
        byte_size: input.attachment.bytes.byteLength,
        width: dimensions.width,
        height: dimensions.height,
        deduplicated: blob.deduplicated,
      });
    }

    return {
      attachmentId,
      streamInput: this.buildAttachmentStreamInput({
        attachmentId,
        mediaType: input.attachment.mediaType,
        parentEntry: input.parentEntry,
        turnId: input.parentTurnId,
      }),
    };
  }

  private buildAttachmentStreamInput(input: {
    attachmentId: AttachmentId;
    mediaType: TurnInputAttachment["mediaType"];
    parentEntry: StreamEntry;
    turnId: string | null;
  }): StreamEntryInput {
    return {
      kind: "user_image_attachment",
      content: {
        type: "image_ref",
        attachment_id: input.attachmentId,
        media_type: input.mediaType,
        parent_entry_id: input.parentEntry.id,
      },
      ...(input.turnId === null ? {} : { turn_id: input.turnId }),
      turn_status: "active",
      ...(input.parentEntry.audience === undefined ? {} : { audience: input.parentEntry.audience }),
      ...(input.parentEntry.sender_entity_id === null
        ? {}
        : { sender_entity_id: input.parentEntry.sender_entity_id }),
    };
  }

  fetchImageForLlm(attachmentId: AttachmentId): {
    mediaType: string;
    bytes: Buffer;
  } {
    const record = this.repository.getActive(attachmentId);

    if (record.stream_entry_id === null) {
      throw new AttachmentError(`Attachment ${attachmentId} is not linked to a stream entry`, {
        code: "ATTACHMENT_STREAM_ENTRY_UNLINKED",
      });
    }

    if (this.repository.isActiveForStreamEntry(record.stream_entry_id) !== true) {
      throw new AttachmentError(`Attachment ${attachmentId} stream entry is inactive`, {
        code: "ATTACHMENT_INACTIVE",
      });
    }

    const indexed = this.entryIndex?.lookup(record.stream_entry_id);
    if (indexed !== undefined && (indexed === null || indexed.active === false)) {
      throw new AttachmentError(`Attachment ${attachmentId} stream entry is inactive`, {
        code: "ATTACHMENT_INACTIVE",
      });
    }

    validateImageForFinalizerRender(record, {
      maxBytes: this.config.maxBytesPerImage,
      maxDimension: Math.max(this.config.maxWidth, this.config.maxHeight),
    });

    const bytes = this.blobStore.read(record.storage_ref);
    const sha256 = createHash("sha256").update(bytes).digest("hex");

    if (sha256 !== record.sha256) {
      if (this.tracer?.enabled === true) {
        this.tracer.emit("attachment.blob_corrupted", {
          turnId: record.parent_turn_id ?? "attachment",
          attachment_id: attachmentId,
          expected_sha256: record.sha256,
          actual_sha256: sha256,
        });
      }

      throw new AttachmentError(`Attachment ${attachmentId} blob hash mismatch`, {
        code: "ATTACHMENT_BLOB_CORRUPTED",
      });
    }

    if (this.tracer?.enabled === true) {
      this.tracer.emit("attachment.fetch_for_ledger", {
        turnId: record.parent_turn_id ?? "attachment",
        attachment_id: attachmentId,
        media_type: record.media_type,
        byte_size: record.byte_size,
      });
    }

    return {
      mediaType: record.media_type,
      bytes,
    };
  }

  setAttachmentActive(attachmentId: AttachmentId, active: boolean, turnId = "attachment"): void {
    if (this.lifecycle !== undefined) {
      if (active) {
        this.lifecycle.unquarantineImageAttachment({
          attachmentId,
          reason: "set_attachment_active",
          turnId,
        });
      } else {
        this.lifecycle.quarantineImageAttachment({
          attachmentId,
          reason: "set_attachment_active",
          turnId,
        });
      }
      return;
    }

    this.repository.setActive(attachmentId, active);

    if (this.tracer?.enabled === true) {
      if (!active) {
        this.tracer.emit("attachment.quarantined", {
          turnId,
          attachment_id: attachmentId,
        });
        this.tracer.emit("image_perception.deactivated", {
          turnId,
          attachment_id: attachmentId,
          active: false,
          reason: "set_attachment_active",
          changed_count: 0,
        });
        return;
      }

      this.tracer.emit("image_perception.reactivated", {
        turnId,
        attachment_id: attachmentId,
        active: true,
        reason: "set_attachment_active",
        changed_count: 0,
      });
    }
  }
}

type AttachmentRejectedReason =
  | "too_many_images"
  | "unsupported_media_type"
  | "image_too_large"
  | "dimensions_too_large"
  | "dimensions_unreadable"
  | "image_malformed";
