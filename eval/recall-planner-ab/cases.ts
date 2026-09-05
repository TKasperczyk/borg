import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { z } from "zod";

import { episodeIdSchema } from "../../src/memory/episodic/index.js";
import { streamConversationSchema } from "../../src/stream/index.js";
import { parseIsoInstant } from "../../src/util/iso-instant.js";

import type { RecallPlannerCase } from "./types.js";

const contextTurnSchema = z
  .object({
    role: z.enum(["user", "assistant"]),
    content: z.string().min(1),
  })
  .strict();

const identitySchema = z
  .object({
    memory_owner_name: z.string().min(1),
    current_sender_name: z.string().min(1).optional(),
    current_audience_name: z.string().min(1).optional(),
    current_venue: streamConversationSchema.strict().optional(),
    entity_terms: z.array(z.string().min(1)).optional(),
  })
  .strict();

const recentActivitySchema = z
  .object({
    excerpt: z.string().min(1),
    occurred_at: z.number().int().nonnegative(),
    venue: streamConversationSchema.strict(),
    counterparty_name: z.string().min(1).optional(),
  })
  .strict();

export const recallPlannerCaseSchema: z.ZodType<RecallPlannerCase> = z
  .object({
    id: z.string().min(1),
    focus: z.string().min(1),
    context_turns: z.array(contextTurnSchema),
    identity: identitySchema,
    owner_recent_activity: z.array(recentActivitySchema),
    expected_episode_ids: z.array(episodeIdSchema).min(1),
    now: z
      .string()
      .refine((value) => parseIsoInstant(value, { requireOffset: true }) !== undefined, {
        message: "now must be an ISO-8601 instant with an explicit offset (Z or ±hh:mm)",
      })
      .optional(),
    notes: z.string().min(1).optional(),
  })
  .strict();

export const recallPlannerCasesSchema = z.array(recallPlannerCaseSchema).min(1);

export function parseRecallPlannerCases(value: unknown): RecallPlannerCase[] {
  const cases = recallPlannerCasesSchema.parse(value);
  const seen = new Set<string>();

  for (const item of cases) {
    if (seen.has(item.id)) {
      throw new Error(`Duplicate recall-planner case id: ${item.id}`);
    }
    seen.add(item.id);
  }

  return cases;
}

export function loadRecallPlannerCases(pathLike: string): {
  path: string;
  cases: RecallPlannerCase[];
} {
  const path = resolve(pathLike);
  let parsed: unknown;

  try {
    parsed = JSON.parse(readFileSync(path, "utf8")) as unknown;
  } catch (error) {
    throw new Error(
      `Unable to read recall-planner cases from ${path}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  return {
    path,
    cases: parseRecallPlannerCases(parsed),
  };
}
