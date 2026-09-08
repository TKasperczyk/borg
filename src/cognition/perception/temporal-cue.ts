import { z } from "zod";

import {
  callStructuredTool,
  isStructuredToolCallError,
  toToolInputSchema,
  type LLMClient,
  type LLMToolDefinition,
} from "../../llm/index.js";
import { EXTRACTOR_MAX_TOKENS_DEFAULT } from "../prompts/constants.js";
import type { TemporalCue } from "../types.js";
import { parseIsoInstant } from "../../util/iso-instant.js";

const temporalCueJudgeSchema = z.object({
  has_cue: z.boolean(),
  since: z.string().min(1).nullable().optional(),
  until: z.string().min(1).nullable().optional(),
  label: z.string().min(1).nullable().optional(),
});
const TEMPORAL_CUE_TOOL_NAME = "EmitTemporalCue";
const TEMPORAL_CUE_TOOL = {
  name: TEMPORAL_CUE_TOOL_NAME,
  description:
    "Extract the time window the user's message is asking to be looked at. Emit has_cue=true only if the message scopes the request to a specific past/future window; a window the message merely mentions, quotes or reports on is not a cue. Express since and until as ISO 8601 UTC instants (e.g. 2026-08-14T00:00:00Z), never as epoch numbers.",
  inputSchema: toToolInputSchema(temporalCueJudgeSchema),
} satisfies LLMToolDefinition;

/**
 * Convert an ISO 8601 instant to epoch milliseconds, or `undefined` when the
 * string does not parse. This is format parsing, not a judgment about whether
 * the window the model chose is a good one.
 */

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
 *
 * The classifier used to be asked for `since_ts`/`until_ts` as raw Unix
 * milliseconds computed against a `now_ms` it was also given as a bare
 * number -- i.e. 13-digit arithmetic done in-head, with no way for anything
 * downstream to notice a slip. It slipped often: on the demo data dir over
 * 2026-08-15/16, 9 of 30 emitted cues carried epoch values landing in 2024
 * (or further out) while their own `label` named 2026. A cue window that
 * misses by two years costs the time-relevance boost in retrieval scoring
 * and empties the autobiographical recall window outright, which removes
 * ledger section 14 with no header, no count and no trace of its own.
 * Asking for ISO 8601 instead moves the arithmetic to the harness, where it
 * is exact.
 *
 * The judge was then asked whether the message *contains* a temporal
 * reference, which a message discussing an earlier window satisfies just by
 * naming it. Because `resolveAutobiographicalRecallWindow` treats a cue as a
 * replacement for the recent default rather than an addition to it, a window
 * quoted in passing closes autobiographical recall on everything since:
 * observed 2026-09-03, where a message quoting the sentence "a
 * perception_temporal_cue window can close well before now" re-minted the
 * 24h window that sentence was about, and the reader's section 14 came back
 * scoped to 2026-08-30/31 with three days of its own record outside the
 * bounds. Withdrawing an attribution re-minted the cue the withdrawal was
 * about, so the act of discussing the loop reproduced it. Asking for the
 * window the message is asking to be looked *in*, rather than any window it
 * looks *at*, is a distinction the judge can make from the text it already
 * has; the failure is worth naming to it because the two directions cost
 * differently -- a missed cue falls back to the recent default, a spurious
 * one deletes the present.
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
    const result = await callStructuredTool({
      llmClient: options.llmClient,
      request: {
        model: options.model,
        system:
          "Identify the time window the user's message is asking to be looked at -- a specific past or future window that scopes the request. Examples: 'yesterday', 'last Tuesday', 'earlier today', 'this morning', 'a week ago', 'tonight', 'next month'. A window the message only talks about rather than asks you to look inside is not a cue: quoted, cited, restated or reported windows return has_cue=false, however precisely they are stated, including when the message is discussing a window an earlier turn established. The asymmetry is worth knowing: this cue replaces the default recent window outright rather than adding to it, so a window minted from a passing mention closes the reader's window on everything since and costs it its own recent record for the turn, while no cue simply leaves the recent default in place. If no concrete time window is being scoped, return has_cue=false. When a cue is present, express since and until as ISO 8601 UTC instants (e.g. 2026-08-14T00:00:00Z) resolved against the supplied 'now', which is also ISO 8601 UTC. Prefer narrower ranges when the phrase is specific (e.g. 'yesterday' is a 24h window, not a week). Label should be a short human-readable form of the phrase.",
        messages: [
          {
            role: "user",
            content: JSON.stringify({
              text,
              now: new Date(nowMs).toISOString(),
            }),
          },
        ],
        tools: [TEMPORAL_CUE_TOOL],
        tool_choice: { type: "tool", name: TEMPORAL_CUE_TOOL_NAME },
        max_tokens: EXTRACTOR_MAX_TOKENS_DEFAULT,
        budget: "perception-temporal-cue",
      },
      toolName: TEMPORAL_CUE_TOOL_NAME,
      parse: (input) => temporalCueJudgeSchema.parse(input),
    });
    const parsed = result.parsed;
    if (!parsed.has_cue) {
      return null;
    }

    const sinceTs = parseIsoInstant(parsed.since);
    const untilTs = parseIsoInstant(parsed.until);
    const label = parsed.label ?? undefined;

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
    if (
      isStructuredToolCallError(error, "missing_tool_call") ||
      isStructuredToolCallError(error, "invalid_payload")
    ) {
      return null;
    }

    // Any failure on this cheap enrichment path degrades gracefully to
    // "no temporal filter" rather than breaking the turn.
    await options.onDegraded?.(
      "llm_failed",
      isStructuredToolCallError(error, "llm_failed") ? (error.cause ?? error) : error,
    );
    return null;
  }
}
