import { z } from "zod";

import { attachmentIdHelpers, type AttachmentId, type StreamEntryId } from "../util/ids.js";

export const SUPPORTED_IMAGE_MEDIA_TYPES = [
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
] as const;

export const imageMediaTypeSchema = z.enum(SUPPORTED_IMAGE_MEDIA_TYPES);

export const attachmentIdSchema = z
  .string()
  .refine((value) => attachmentIdHelpers.is(value), {
    message: "Invalid attachment id",
  })
  .transform((value) => value as AttachmentId);

export type ImageMediaType = z.infer<typeof imageMediaTypeSchema>;

export type BorgUserContentBlock =
  | {
      type: "text";
      text: string;
    }
  | {
      type: "image_ref";
      attachment_id: AttachmentId;
    };

export type TurnInputAttachment = {
  mediaType: ImageMediaType;
  bytes: Uint8Array;
};

export type StoredAttachmentRecord = {
  attachment_id: AttachmentId;
  sha256: string;
  media_type: ImageMediaType;
  byte_size: number;
  width: number;
  height: number;
  storage_ref: string;
  thumbnail_ref: string | null;
  perception_id: string | null;
  text_embedding_ref: string | null;
  visual_embedding_ref: string | null;
  active: boolean;
  audience: string | null;
  created_turn_global: number | null;
  ordinal?: number;
  parent_entry_id: StreamEntryId;
  stream_entry_id: StreamEntryId | null;
  parent_turn_id: string | null;
  created_at: number;
};
