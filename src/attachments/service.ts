import { createHash } from "node:crypto";

import { AttachmentError } from "../util/errors.js";
import { createAttachmentId, type AttachmentId, type StreamEntryId } from "../util/ids.js";
import type {
  StreamEntry,
  StreamEntryIndexRepository,
  StreamEntryInput,
  StreamWriter,
} from "../stream/index.js";
import type { TurnTracer } from "../cognition/tracing/tracer.js";
import { AttachmentBlobStore } from "./blob-store.js";
import { readImageDimensions } from "./image-info.js";
import { AttachmentRepository } from "./repository.js";
import {
  SUPPORTED_IMAGE_MEDIA_TYPES,
  imageMediaTypeSchema,
  type BorgUserContentBlock,
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
  logger?: LoggerLike;
  tracer?: TurnTracer;
};

export type PersistTurnAttachmentsInput = {
  attachments: readonly TurnInputAttachment[];
  streamWriter: Pick<StreamWriter, "appendMany">;
  parentEntry: StreamEntry;
  turnId: string;
  createdTurnGlobal?: number;
};

export type PersistedTurnAttachment = {
  attachmentId: AttachmentId;
  streamEntry: StreamEntry;
  contentBlock: BorgUserContentBlock;
};

export class AttachmentService {
  private readonly repository: AttachmentRepository;
  private readonly blobStore: AttachmentBlobStore;
  private readonly config: AttachmentValidationConfig;
  private readonly entryIndex?: Pick<StreamEntryIndexRepository, "lookup">;
  private readonly logger: LoggerLike;
  private readonly tracer?: TurnTracer;

  constructor(options: AttachmentServiceOptions) {
    this.repository = options.repository;
    this.blobStore = options.blobStore;
    this.config = options.config;
    this.entryIndex = options.entryIndex;
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

    const prepared = input.attachments.map((attachment) => {
      const dimensions = readImageDimensions(attachment.bytes, attachment.mediaType);
      const blob = this.blobStore.write(attachment.bytes, attachment.mediaType);
      const attachmentId = createAttachmentId();

      this.repository.insert({
        attachment_id: attachmentId,
        sha256: blob.sha256,
        media_type: attachment.mediaType,
        byte_size: attachment.bytes.byteLength,
        width: dimensions.width,
        height: dimensions.height,
        storage_ref: blob.storageRef,
        thumbnail_ref: null,
        perception_id: null,
        text_embedding_ref: null,
        visual_embedding_ref: null,
        active: false,
        audience: input.parentEntry.audience ?? null,
        created_turn_global: input.createdTurnGlobal ?? null,
        parent_entry_id: input.parentEntry.id,
        stream_entry_id: null,
        parent_turn_id: input.turnId,
        created_at: input.parentEntry.timestamp,
      });

      if (this.tracer?.enabled === true) {
        this.tracer.emit("attachment.write", {
          turnId: input.turnId,
          attachment_id: attachmentId,
          sha256: blob.sha256,
          media_type: attachment.mediaType,
          byte_size: attachment.bytes.byteLength,
          width: dimensions.width,
          height: dimensions.height,
          deduplicated: blob.deduplicated,
        });
      }

      const streamInput: StreamEntryInput = {
        kind: "user_image_attachment",
        content: {
          type: "image_ref",
          attachment_id: attachmentId,
          media_type: attachment.mediaType,
          parent_entry_id: input.parentEntry.id,
        },
        turn_id: input.turnId,
        turn_status: "active",
        ...(input.parentEntry.audience === undefined
          ? {}
          : { audience: input.parentEntry.audience }),
        ...(input.parentEntry.sender_entity_id === null
          ? {}
          : { sender_entity_id: input.parentEntry.sender_entity_id }),
      };

      return {
        attachment,
        attachmentId,
        streamInput,
      };
    });

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

    const bytes = this.blobStore.read(record.storage_ref);
    const sha256 = createHash("sha256").update(bytes).digest("hex");

    if (sha256 !== record.sha256) {
      if (this.tracer?.enabled === true) {
        this.tracer.emit("attachment.blob_corrupted", {
          turnId: record.parent_turn_id,
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
        turnId: record.parent_turn_id,
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
    this.repository.setActive(attachmentId, active);

    if (!active && this.tracer?.enabled === true) {
      this.tracer.emit("attachment.quarantine", {
        turnId,
        attachment_id: attachmentId,
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
