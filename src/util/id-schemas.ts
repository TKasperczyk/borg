import { z } from "zod";

import {
  isSessionId,
  parseSessionId,
  streamEntryIdHelpers,
  type StreamEntryId,
} from "./ids.js";

export const streamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const sessionIdSchema = z
  .string()
  .refine((value) => isSessionId(value), {
    message: "Invalid session id",
  })
  .transform((value) => parseSessionId(value));
