import { z } from "zod";

import { toToolInputSchema, type LLMClient, type LLMToolDefinition } from "../../llm/index.js";
import type { TemporalCue } from "../types.js";

const temporalCueJudgeSchema = z.object({
  has_cue: z.boolean(),
  since_ts: z.number().nullable().optional(),
  until_ts: z.number().nullable().optional(),
  label: z.string().min(1).nullable().optional(),
});
const TEMPORAL_CUE_TOOL_NAME = "EmitTemporalCue";
const TEMPORAL_CUE_TOOL = {
  name: TEMPORAL_CUE_TOOL_NAME,
  description:
    "Extract a temporal reference from the user's message. Emit has_cue=true only if the message refers to a specific past/future time window. Fill since_ts and until_ts as Unix milliseconds relative to the supplied 'now' timestamp.",
  inputSchema: toToolInputSchema(temporalCueJudgeSchema),
} satisfies LLMToolDefinition;

export type TemporalCueDetectorOptions = {
  llmClient?: LLMClient;
  model?: string;
  onDegraded?: (reason: "llm_unavailable" | "llm_failed", error?: unknown) => Promise<void> | void;
};

/**
 * Detect a temporal reference in the user's message. Returns `null` if the
 * message doesn't refer to a specific time window, or if no LLM client is
 * configured for extraction.
 *
 * Previously this module hardcoded six English phrases (yesterday, last
 * week, this morning, this week, today, tonight) and silently returned
 * `null` for everything else -- including very common phrasings like
 * "last Tuesday", "earlier today", "a few days ago", "this past weekend".
 * That patch-work has been replaced with an LLM classifier that interprets
 * the message directly against the current clock.
 */
export async function detectTemporalCue(
  text: string,
  nowMs: number,
  options: TemporalCueDetectorOptions = {},
): Promise<TemporalCue | null> {
  if (options.llmClient === undefined || options.model === undefined) {
    await options.onDegraded?.("llm_unavailable");
    return null;
  }

  try {
    const response = await options.llmClient.complete({
      model: options.model,
      system:
        "Identify whether the user's message contains a temporal reference -- a specific past or future time window. Examples: 'yesterday', 'last Tuesday', 'earlier today', 'this morning', 'a week ago', 'tonight', 'next month'. If there is no concrete time window being referenced, return has_cue=false. When a cue is present, compute since_ts and until_ts as Unix milliseconds relative to the supplied 'now' timestamp (also in ms). Prefer narrower ranges when the phrase is specific (e.g. 'yesterday' is a 24h window, not a week). Label should be a short human-readable form of the phrase.",
      messages: [
        {
          role: "user",
          content: JSON.stringify({
            text,
            now_ms: nowMs,
          }),
        },
      ],
      tools: [TEMPORAL_CUE_TOOL],
      tool_choice: { type: "tool", name: TEMPORAL_CUE_TOOL_NAME },
      max_tokens: 400,
      budget: "perception-temporal-cue",
    });

    const call = response.tool_calls.find((toolCall) => toolCall.name === TEMPORAL_CUE_TOOL_NAME);
    if (call === undefined) {
      return null;
    }

    const parsed = temporalCueJudgeSchema.safeParse(call.input);
    if (!parsed.success || !parsed.data.has_cue) {
      return null;
    }

    const sinceTs = parsed.data.since_ts ?? undefined;
    const untilTs = parsed.data.until_ts ?? undefined;
    const label = parsed.data.label ?? undefined;

    // If the judge returns no actionable window, treat as no cue.
    if (sinceTs === undefined && untilTs === undefined) {
      return null;
    }

    const cue: TemporalCue = {};
    if (sinceTs !== undefined) {
      cue.sinceTs = sinceTs;
    }
    if (untilTs !== undefined) {
      cue.untilTs = untilTs;
    }
    if (label !== undefined) {
      cue.label = label;
    }
    return cue;
  } catch (error) {
    // Any failure on this cheap enrichment path degrades gracefully to
    // "no temporal filter" rather than breaking the turn.
    await options.onDegraded?.("llm_failed", error);
    return null;
  }
}
