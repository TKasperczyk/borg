import { z } from "zod";

import {
  PROMPT_SURFACE_CHANGES_MAX_LIMIT,
  promptSurfaceChangeRecordSchema,
  promptSurfaceCurrentSchema,
  type PromptSurfaceChangeRecord,
  type PromptSurfaceCurrent,
} from "../../cognition/prompts/prompt-surface-history.js";
import type { ToolDefinition } from "../dispatcher.js";

const promptSurfaceChangesInputSchema = z
  .object({
    limit: z.number().int().positive().max(PROMPT_SURFACE_CHANGES_MAX_LIMIT).optional(),
    since_version: z.string().length(64).optional(),
  })
  .strict();

const promptSurfaceChangesOutputSchema = z
  .object({
    current: promptSurfaceCurrentSchema,
    changes: z.array(promptSurfaceChangeRecordSchema),
  })
  .strict();

export type PromptSurfaceChangesToolOptions = {
  current: () => PromptSurfaceCurrent;
  listChanges: (options: {
    limit?: number;
    sinceVersion?: string;
  }) => PromptSurfaceChangeRecord[];
};

export function createPromptSurfaceChangesTool(
  options: PromptSurfaceChangesToolOptions,
): ToolDefinition<
  z.infer<typeof promptSurfaceChangesInputSchema>,
  z.infer<typeof promptSurfaceChangesOutputSchema>
> {
  return {
    name: "tool.promptSurface.changes",
    description:
      "Read structural prompt-surface version changes: hashes, observation times, block ids, surfaces, and placement orders only. since_version is an exclusive to_hash cursor; unknown cursors return no changes.",
    menuSummary: "Review structural prompt-surface changes.",
    allowedOrigins: ["autonomous"],
    writeScope: "read",
    inputSchema: promptSurfaceChangesInputSchema,
    outputSchema: promptSurfaceChangesOutputSchema,
    async invoke(input) {
      return {
        current: options.current(),
        changes: options.listChanges({
          limit: input.limit,
          sinceVersion: input.since_version,
        }),
      };
    },
  };
}
