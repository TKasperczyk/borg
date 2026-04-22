import { z } from "zod";

import { emotionalArcSchema, type EmotionalArc } from "../affective/types.js";
import type { StreamEntryId } from "../../util/ids.js";
import {
  entityIdHelpers,
  episodeIdHelpers,
  streamEntryIdHelpers,
  type EpisodeId,
  type EntityId,
} from "../../util/ids.js";

export const EPISODE_TIERS = ["T1", "T2", "T3", "T4"] as const;

export const episodeIdSchema = z
  .string()
  .refine((value) => episodeIdHelpers.is(value), {
    message: "Invalid episode id",
  })
  .transform((value) => value as EpisodeId);

export const streamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const episodeAudienceEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid episode audience entity id",
  })
  .transform((value) => value as EntityId);

export const episodeTierSchema = z.enum(EPISODE_TIERS);

const float32ArraySchema = z.custom<Float32Array>((value) => value instanceof Float32Array, {
  message: "Expected Float32Array embedding",
});

export const episodeLineageSchema = z.object({
  derived_from: z.array(episodeIdSchema),
  supersedes: z.array(episodeIdSchema),
});

const episodeShape = z.object({
  id: episodeIdSchema,
  title: z.string().min(1),
  narrative: z.string().min(1),
  participants: z.array(z.string().min(1)),
  location: z.string().min(1).nullable(),
  start_time: z.number().finite(),
  end_time: z.number().finite(),
  source_stream_ids: z.array(streamEntryIdSchema).min(1),
  significance: z.number().min(0).max(1),
  tags: z.array(z.string().min(1)),
  confidence: z.number().min(0).max(1),
  lineage: episodeLineageSchema,
  emotional_arc: emotionalArcSchema.nullable().default(null),
  audience_entity_id: episodeAudienceEntityIdSchema.nullable().optional(),
  shared: z.boolean().optional(),
  embedding: float32ArraySchema,
  created_at: z.number().finite(),
  updated_at: z.number().finite(),
});

export const episodeSchema = episodeShape
  .refine((value) => value.end_time >= value.start_time, {
    message: "Episode end_time must be greater than or equal to start_time",
    path: ["end_time"],
  })
  .refine((value) => value.updated_at >= value.created_at, {
    message: "Episode updated_at must be greater than or equal to created_at",
    path: ["updated_at"],
  });

export const episodeInsertSchema = episodeSchema;

export const episodePatchSchema = episodeShape
  .omit({
    id: true,
    created_at: true,
  })
  .partial();

export const episodeStatsSchema = z.object({
  episode_id: episodeIdSchema,
  retrieval_count: z.number().int().nonnegative(),
  use_count: z.number().int().nonnegative(),
  last_retrieved: z.number().finite().nullable(),
  win_rate: z.number().min(0).max(1),
  tier: episodeTierSchema,
  promoted_at: z.number().finite(),
  promoted_from: z.string().min(1).nullable(),
  gist: z.string().min(1).nullable(),
  gist_generated_at: z.number().finite().nullable(),
  last_decayed_at: z.number().finite().nullable(),
  valence_mean: z.number().min(-1).max(1).default(0),
  archived: z.boolean().default(false),
});

export const episodeStatsPatchSchema = episodeStatsSchema
  .omit({
    episode_id: true,
  })
  .partial();

export type Episode = z.infer<typeof episodeSchema>;
export type EpisodePatch = z.infer<typeof episodePatchSchema>;
export type EpisodeTier = z.infer<typeof episodeTierSchema>;
export type EpisodeStats = z.infer<typeof episodeStatsSchema>;
export type EpisodeStatsPatch = z.infer<typeof episodeStatsPatchSchema>;
export type { EmotionalArc };

export type EpisodeListOptions = {
  limit?: number;
  cursor?: string;
};

export type EpisodeListResult = {
  items: Episode[];
  nextCursor?: string;
};

export type EpisodeSearchOptions = {
  limit?: number;
  minSimilarity?: number;
  tagFilter?: readonly string[];
  tierFilter?: readonly EpisodeTier[];
  audienceEntityId?: EntityId | null;
  crossAudience?: boolean;
  timeRange?: {
    start: number;
    end: number;
  };
};

export type EpisodeSearchCandidate = {
  episode: Episode;
  stats: EpisodeStats;
  similarity: number;
};
