import { z } from "zod";

export type RetrievalDegradation = { subsystem: string; reason: string };

export const retrievalStatusSchema = z.object({
  degraded: z.literal(true),
  subsystems: z.array(z.string()),
});

/** Execution metadata only: never turn a partial search into a confidence judgment. */
export function retrievalStatus(degradations: readonly RetrievalDegradation[] = []) {
  if (degradations.length === 0) return undefined;
  return {
    degraded: true as const,
    subsystems: [...new Set(degradations.map((entry) => entry.subsystem))],
  };
}
