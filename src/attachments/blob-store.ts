import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { writeFileAtomic } from "../util/atomic-write.js";
import { AttachmentError } from "../util/errors.js";
import type { ImageMediaType } from "./types.js";

const EXTENSION_BY_MEDIA_TYPE: Record<ImageMediaType, string> = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/gif": "gif",
  "image/webp": "webp",
};

export type BlobWriteResult = {
  sha256: string;
  storageRef: string;
  deduplicated: boolean;
};

export class AttachmentBlobStore {
  constructor(private readonly dataDir: string) {}

  storageRefFor(input: { sha256: string; mediaType: ImageMediaType }): string {
    return join(
      "blobs",
      input.sha256.slice(0, 2),
      `${input.sha256}.${EXTENSION_BY_MEDIA_TYPE[input.mediaType]}`,
    );
  }

  absolutePath(storageRef: string): string {
    return join(this.dataDir, storageRef);
  }

  write(bytes: Uint8Array, mediaType: ImageMediaType): BlobWriteResult {
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    const storageRef = this.storageRefFor({ sha256, mediaType });
    const path = this.absolutePath(storageRef);

    if (existsSync(path)) {
      return {
        sha256,
        storageRef,
        deduplicated: true,
      };
    }

    writeFileAtomic(path, bytes);

    return {
      sha256,
      storageRef,
      deduplicated: false,
    };
  }

  read(storageRef: string): Buffer {
    try {
      return readFileSync(this.absolutePath(storageRef));
    } catch (error) {
      throw new AttachmentError(`Failed to read attachment blob ${storageRef}`, {
        code: "ATTACHMENT_BLOB_READ_FAILED",
        cause: error,
      });
    }
  }
}
