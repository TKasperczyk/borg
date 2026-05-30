import type { StreamEntryId } from "../util/ids.js";

import type { StreamCursor } from "./types.js";
import type { StreamWatermark } from "./watermark.js";

export function streamCursorsEqual(left: StreamCursor | null, right: StreamCursor | null): boolean {
  if (left === null || right === null) {
    return left === right;
  }

  return left.ts === right.ts && left.entryId === right.entryId;
}

export function streamCursorFromWatermark(watermark: StreamWatermark | null): StreamCursor | null {
  if (watermark === null) {
    return null;
  }

  return {
    ts: watermark.lastTs,
    entryId: watermark.lastEntryId as StreamEntryId,
  };
}
