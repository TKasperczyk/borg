import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { AttachmentError } from "../util/errors.js";
import type { AttachmentId } from "../util/ids.js";
import type { TurnTracer } from "../cognition/tracing/tracer.js";
import type { AttachmentRepository } from "./repository.js";

export type ImageAttachmentLifecycleServiceOptions = {
  db: SqliteDatabase;
  attachmentRepository: AttachmentRepository;
  tracer?: TurnTracer;
};

export type ImageAttachmentLifecycleResult = {
  attachmentChanged: boolean;
  perceptionArtifactsChanged: number;
};

export class ImageAttachmentLifecycleService {
  constructor(private readonly options: ImageAttachmentLifecycleServiceOptions) {}

  quarantineImageAttachment(input: {
    attachmentId: AttachmentId;
    reason: string;
    turnId?: string;
  }): ImageAttachmentLifecycleResult {
    return this.setImageAttachmentActive({
      attachmentId: input.attachmentId,
      active: false,
      reason: input.reason,
      turnId: input.turnId,
    });
  }

  unquarantineImageAttachment(input: {
    attachmentId: AttachmentId;
    reason: string;
    turnId?: string;
  }): ImageAttachmentLifecycleResult {
    return this.setImageAttachmentActive({
      attachmentId: input.attachmentId,
      active: true,
      reason: input.reason,
      turnId: input.turnId,
    });
  }

  private setImageAttachmentActive(input: {
    attachmentId: AttachmentId;
    active: boolean;
    reason: string;
    turnId?: string;
  }): ImageAttachmentLifecycleResult {
    const existing = this.options.attachmentRepository.get(input.attachmentId);
    if (existing === null) {
      throw new AttachmentError(`Attachment ${input.attachmentId} was not found`, {
        code: "ATTACHMENT_NOT_FOUND",
      });
    }

    const apply = this.options.db.transaction(() => {
      const changes = this.options.attachmentRepository.setActiveWithCascade(
        input.attachmentId,
        input.active,
      );

      return {
        attachmentChanged: changes.attachmentChanges > 0,
        perceptionArtifactsChanged: changes.perceptionArtifactChanges,
      };
    });
    const result = apply();

    if (this.options.tracer?.enabled === true) {
      const turnId = input.turnId ?? existing.parent_turn_id ?? "attachment";
      if (input.active) {
        this.options.tracer.emit("image_perception.reactivated", {
          turnId,
          attachment_id: input.attachmentId,
          active: true,
          reason: input.reason,
          changed_count: result.perceptionArtifactsChanged,
        });
      } else {
        this.options.tracer.emit("attachment.quarantined", {
          turnId,
          attachment_id: input.attachmentId,
          reason: input.reason,
        });
        this.options.tracer.emit("image_perception.deactivated", {
          turnId,
          attachment_id: input.attachmentId,
          active: false,
          reason: input.reason,
          changed_count: result.perceptionArtifactsChanged,
        });
        this.options.tracer.emit("shared_state.attachment_rejected", {
          turnId,
          attachment_id: input.attachmentId,
          reason: input.reason,
        });
      }
    }

    return result;
  }
}
