import { z } from "zod";

export const narrativeCoverageSchema = z
  .object({
    shown_episodes: z.number().int().nonnegative(),
    omitted_episodes: z.number().int().nonnegative(),
    window_rule: z.literal("newest_contiguous_suffix"),
    limiting_bound: z
      .enum(["maxEpisodesPerRun", "maxInputTokensPerRun", "maxPromptTokens"])
      .nullable(),
  })
  .strict();

export type NarrativeCoverage = z.infer<typeof narrativeCoverageSchema>;

// This describes the last pass's direct episode input, not the accumulated story.
// Keep it beside the narrative so a prose excerpt cannot silently cut it away.
export function renderNarrativeCoverage(coverage: NarrativeCoverage | null | undefined): string {
  if (coverage == null) return "Latest narration pass input: coverage unknown.";
  return `Latest narration pass input: newest ${coverage.shown_episodes} episodes shown (contiguous suffix, chronological order); ${coverage.omitted_episodes} earlier episodes omitted; limiting bound: ${coverage.limiting_bound ?? "none"}.`;
}
