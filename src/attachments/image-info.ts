import { AttachmentError } from "../util/errors.js";
import type { ImageMediaType } from "./types.js";

export type ImageDimensions = {
  width: number;
  height: number;
};

function pngDimensions(bytes: Uint8Array): ImageDimensions | null {
  if (
    bytes.length < 33 ||
    bytes[0] !== 0x89 ||
    bytes[1] !== 0x50 ||
    bytes[2] !== 0x4e ||
    bytes[3] !== 0x47 ||
    bytes[4] !== 0x0d ||
    bytes[5] !== 0x0a ||
    bytes[6] !== 0x1a ||
    bytes[7] !== 0x0a
  ) {
    throw new AttachmentError("Malformed PNG image", {
      code: "ATTACHMENT_IMAGE_MALFORMED",
    });
  }

  const ihdrLength = (bytes[8]! << 24) | (bytes[9]! << 16) | (bytes[10]! << 8) | bytes[11]!;
  const ihdrType = Buffer.from(bytes.subarray(12, 16)).toString("ascii");

  if (ihdrLength !== 13 || ihdrType !== "IHDR") {
    throw new AttachmentError("Malformed PNG image", {
      code: "ATTACHMENT_IMAGE_MALFORMED",
    });
  }

  if (bytes.length < 16 + ihdrLength + 4) {
    return null;
  }

  return {
    width: (bytes[16]! << 24) | (bytes[17]! << 16) | (bytes[18]! << 8) | bytes[19]!,
    height: (bytes[20]! << 24) | (bytes[21]! << 16) | (bytes[22]! << 8) | bytes[23]!,
  };
}

function gifDimensions(bytes: Uint8Array): ImageDimensions | null {
  if (bytes.length < 10) {
    throw new AttachmentError("Malformed GIF image", {
      code: "ATTACHMENT_IMAGE_MALFORMED",
    });
  }

  const header = Buffer.from(bytes.subarray(0, 6)).toString("ascii");

  if (header !== "GIF87a" && header !== "GIF89a") {
    throw new AttachmentError("Malformed GIF image", {
      code: "ATTACHMENT_IMAGE_MALFORMED",
    });
  }

  return {
    width: bytes[6]! | (bytes[7]! << 8),
    height: bytes[8]! | (bytes[9]! << 8),
  };
}

function webpDimensions(bytes: Uint8Array): ImageDimensions | null {
  if (
    bytes.length < 20 ||
    Buffer.from(bytes.subarray(0, 4)).toString("ascii") !== "RIFF" ||
    Buffer.from(bytes.subarray(8, 12)).toString("ascii") !== "WEBP"
  ) {
    throw new AttachmentError("Malformed WebP image", {
      code: "ATTACHMENT_IMAGE_MALFORMED",
    });
  }

  const riffPayloadLength = bytes[4]! | (bytes[5]! << 8) | (bytes[6]! << 16) | (bytes[7]! << 24);

  if (riffPayloadLength + 8 > bytes.length) {
    throw new AttachmentError("Malformed WebP image", {
      code: "ATTACHMENT_IMAGE_MALFORMED",
    });
  }

  if (bytes.length < 30) {
    return null;
  }

  const chunk = Buffer.from(bytes.subarray(12, 16)).toString("ascii");

  if (chunk === "VP8 " && bytes.length >= 30) {
    return {
      width: bytes[26]! | ((bytes[27]! & 0x3f) << 8),
      height: bytes[28]! | ((bytes[29]! & 0x3f) << 8),
    };
  }

  if (chunk === "VP8L" && bytes.length >= 25) {
    const b0 = bytes[21]!;
    const b1 = bytes[22]!;
    const b2 = bytes[23]!;
    const b3 = bytes[24]!;
    return {
      width: 1 + (((b1 & 0x3f) << 8) | b0),
      height: 1 + (((b3 & 0x0f) << 10) | (b2 << 2) | ((b1 & 0xc0) >> 6)),
    };
  }

  if (chunk === "VP8X" && bytes.length >= 30) {
    return {
      width: 1 + (bytes[24]! | (bytes[25]! << 8) | (bytes[26]! << 16)),
      height: 1 + (bytes[27]! | (bytes[28]! << 8) | (bytes[29]! << 16)),
    };
  }

  return null;
}

function jpegDimensions(bytes: Uint8Array): ImageDimensions | null {
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) {
    throw new AttachmentError("Malformed JPEG image", {
      code: "ATTACHMENT_IMAGE_MALFORMED",
    });
  }

  let offset = 2;

  while (offset + 9 < bytes.length) {
    if (bytes[offset] !== 0xff) {
      offset += 1;
      continue;
    }

    const marker = bytes[offset + 1]!;
    offset += 2;

    if (marker === 0xd8 || marker === 0xd9) {
      continue;
    }

    if (offset + 2 > bytes.length) {
      return null;
    }

    const length = (bytes[offset]! << 8) | bytes[offset + 1]!;

    if (length < 2 || offset + length > bytes.length) {
      return null;
    }

    if (
      (marker >= 0xc0 && marker <= 0xc3) ||
      (marker >= 0xc5 && marker <= 0xc7) ||
      (marker >= 0xc9 && marker <= 0xcb) ||
      (marker >= 0xcd && marker <= 0xcf)
    ) {
      return {
        height: (bytes[offset + 3]! << 8) | bytes[offset + 4]!,
        width: (bytes[offset + 5]! << 8) | bytes[offset + 6]!,
      };
    }

    offset += length;
  }

  return null;
}

export function readImageDimensions(bytes: Uint8Array, mediaType: ImageMediaType): ImageDimensions {
  const dimensions =
    mediaType === "image/png"
      ? pngDimensions(bytes)
      : mediaType === "image/gif"
        ? gifDimensions(bytes)
        : mediaType === "image/webp"
          ? webpDimensions(bytes)
          : jpegDimensions(bytes);

  if (dimensions === null || dimensions.width <= 0 || dimensions.height <= 0) {
    throw new AttachmentError(`Unable to read ${mediaType} dimensions`, {
      code: "ATTACHMENT_DIMENSIONS_UNREADABLE",
    });
  }

  return dimensions;
}
