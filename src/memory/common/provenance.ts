import { z } from "zod";

import { episodeIdHelpers, type EpisodeId } from "../../util/ids.js";

const episodeIdSchema = z
  .string()
  .refine((value) => episodeIdHelpers.is(value), {
    message: "Invalid episode id",
  })
  .transform((value) => value as EpisodeId);

export const provenanceKindSchema = z.enum(["episodes", "manual", "system", "offline"]);

export const provenanceSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("episodes"),
    episode_ids: z.array(episodeIdSchema).min(1),
  }),
  z.object({
    kind: z.literal("manual"),
  }),
  z.object({
    kind: z.literal("system"),
  }),
  z.object({
    kind: z.literal("offline"),
    process: z.string().min(1),
  }),
]);

export type Provenance = z.infer<typeof provenanceSchema>;
export type ProvenanceKind = z.infer<typeof provenanceKindSchema>;

export type StoredProvenance = {
  // SQL columns stay nullable for migration compatibility; repositories
  // enforce provenance on all production writes.
  provenance_kind: ProvenanceKind;
  provenance_episode_ids: string;
  provenance_process: string | null;
};

const DEFAULT_MANUAL_PROVENANCE = {
  kind: "manual" as const,
};

export function isEpisodeProvenance(
  provenance: Provenance,
): provenance is Extract<Provenance, { kind: "episodes" }> {
  return provenance.kind === "episodes";
}

export function getEpisodeProvenanceIds(provenance: Provenance): EpisodeId[] {
  return isEpisodeProvenance(provenance) ? provenance.episode_ids : [];
}

export function toStoredProvenance(provenance: Provenance): StoredProvenance {
  return {
    provenance_kind: provenance.kind,
    provenance_episode_ids: JSON.stringify(
      provenance.kind === "episodes" ? provenance.episode_ids : [],
    ),
    provenance_process: provenance.kind === "offline" ? provenance.process : null,
  };
}

export function summarizeProvenanceForPrompt(provenance: Provenance, limit = 2): string {
  switch (provenance.kind) {
    case "episodes": {
      const shown = provenance.episode_ids.slice(0, limit);
      const suffix = provenance.episode_ids.length > limit ? ", ..." : "";
      return `(from ${shown.join(", ")}${suffix})`;
    }
    case "manual":
      return "(manual)";
    case "system":
      return "(system)";
    case "offline":
      return `(offline: ${provenance.process})`;
  }
}

export function parseStoredProvenance(input: {
  provenance_kind: unknown;
  provenance_episode_ids?: unknown;
  provenance_process?: unknown;
}): Provenance {
  return provenanceSchema.parse({
    kind: input.provenance_kind,
    ...(input.provenance_kind === "episodes"
      ? {
          episode_ids: parseStoredProvenanceEpisodeIds(input.provenance_episode_ids),
        }
      : {}),
    ...(input.provenance_kind === "offline"
      ? {
          process: input.provenance_process,
        }
      : {}),
  });
}

export function parseReviewProvenance(refs: Record<string, unknown>): Provenance {
  return provenanceSchema.parse(
    refs.proposed_provenance ?? refs.provenance ?? DEFAULT_MANUAL_PROVENANCE,
  );
}

export function parseStoredProvenanceEpisodeIds(value: unknown): EpisodeId[] {
  if (typeof value !== "string") {
    return [];
  }

  const parsed = JSON.parse(value) as unknown;
  return z.array(episodeIdSchema).parse(parsed);
}
