import { z } from "zod";

import {
  episodeIdHelpers,
  streamEntryIdHelpers,
  type EpisodeId,
  type StreamEntryId,
} from "../../util/ids.js";

const episodeIdSchema = z
  .string()
  .refine((value) => episodeIdHelpers.is(value), {
    message: "Invalid episode id",
  })
  .transform((value) => value as EpisodeId);

const streamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const provenanceKindSchema = z.enum([
  "episodes",
  "manual",
  "system",
  "offline",
  "online",
  "online_reflector",
]);

const baseProvenanceSchema = z.discriminatedUnion("kind", [
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
  z.object({
    kind: z.literal("online"),
    process: z.string().min(1),
  }),
  z.object({
    kind: z.literal("online_reflector"),
    evidence_episode_ids: z.array(episodeIdSchema),
    evidence_stream_entry_ids: z.array(streamEntryIdSchema),
  }),
]);

export const provenanceSchema = baseProvenanceSchema.superRefine((value, ctx) => {
  if (
    value.kind === "online_reflector" &&
    value.evidence_episode_ids.length === 0 &&
    value.evidence_stream_entry_ids.length === 0
  ) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "online_reflector provenance requires episode or stream evidence",
      path: ["evidence_episode_ids"],
    });
  }
});

export type Provenance = z.infer<typeof provenanceSchema>;
export type ProvenanceKind = z.infer<typeof provenanceKindSchema>;

export type StoredProvenance = {
  provenance_kind: ProvenanceKind;
  provenance_episode_ids: string;
  provenance_process: string | null;
  provenance_stream_entry_ids?: string;
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
      provenance.kind === "episodes"
        ? provenance.episode_ids
        : provenance.kind === "online_reflector"
          ? provenance.evidence_episode_ids
          : [],
    ),
    provenance_process:
      provenance.kind === "offline" || provenance.kind === "online"
        ? provenance.process
        : provenance.kind === "online_reflector"
          ? "reflector"
          : null,
    ...(provenance.kind === "online_reflector"
      ? {
          provenance_stream_entry_ids: JSON.stringify(provenance.evidence_stream_entry_ids),
        }
      : {}),
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
    case "online":
      return `(online: ${provenance.process})`;
    case "online_reflector":
      return "(online: reflector)";
  }
}

export function parseStoredProvenance(input: {
  provenance_kind: unknown;
  provenance_episode_ids?: unknown;
  provenance_stream_entry_ids?: unknown;
  provenance_process?: unknown;
}): Provenance {
  return provenanceSchema.parse({
    kind: input.provenance_kind,
    ...(input.provenance_kind === "episodes"
      ? {
          episode_ids: parseStoredProvenanceEpisodeIds(input.provenance_episode_ids),
        }
      : {}),
    ...(input.provenance_kind === "offline" || input.provenance_kind === "online"
      ? {
          process: input.provenance_process,
        }
      : {}),
    ...(input.provenance_kind === "online_reflector"
      ? {
          evidence_episode_ids: parseStoredProvenanceEpisodeIds(input.provenance_episode_ids),
          evidence_stream_entry_ids: parseStoredProvenanceStreamEntryIds(
            input.provenance_stream_entry_ids,
          ),
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

export function parseStoredProvenanceStreamEntryIds(value: unknown): StreamEntryId[] {
  if (typeof value !== "string") {
    return [];
  }

  const parsed = JSON.parse(value) as unknown;
  return z.array(streamEntryIdSchema).parse(parsed);
}
