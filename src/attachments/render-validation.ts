import { AttachmentError } from "../util/errors.js";

export type ImageRenderValidationRecord = {
  attachment_id: string;
  byte_size: number;
  width: number;
  height: number;
};

export function validateImageForFinalizerRender(
  record: ImageRenderValidationRecord,
  options: {
    maxBytes?: number;
    maxDimension: number;
  },
): void {
  if (options.maxBytes !== undefined && record.byte_size > options.maxBytes) {
    throw new AttachmentError(
      `Attachment ${record.attachment_id} exceeds finalizer render byte budget`,
      { code: "ATTACHMENT_LEDGER_RENDER_BYTES_TOO_LARGE" },
    );
  }

  if (record.width > options.maxDimension || record.height > options.maxDimension) {
    throw new AttachmentError(
      `Attachment ${record.attachment_id} exceeds finalizer render max dimension`,
      { code: "ATTACHMENT_LEDGER_RENDER_DIMENSIONS_TOO_LARGE" },
    );
  }
}
