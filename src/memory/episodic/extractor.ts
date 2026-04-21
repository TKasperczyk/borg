import type { EmbeddingClient } from "../../embeddings/index.js";
import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { analyzeAffectiveSignalHeuristically, type EmotionalArc } from "../affective/index.js";
import { StreamReader, type StreamEntry } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { EmbeddingError, LLMError, StorageError } from "../../util/errors.js";
import { createEpisodeId, DEFAULT_SESSION_ID, type SessionId } from "../../util/ids.js";
import { EpisodicRepository } from "./repository.js";
import { type Episode } from "./types.js";

import { z } from "zod";

const extractorCandidateSchema = z.object({
  title: z.string().min(1),
  narrative: z.string().min(1),
  source_stream_ids: z.array(z.string().min(1)).min(1),
  participants: z.array(z.string().min(1)),
  location: z.string().min(1).nullable(),
  tags: z.array(z.string().min(1)),
  confidence: z.number().min(0).max(1),
  significance: z.number().min(0).max(1),
});

const extractorResponseSchema = z.object({
  episodes: z.array(extractorCandidateSchema),
});

type ExtractorCandidate = z.infer<typeof extractorCandidateSchema>;
const DEDUP_THRESHOLD = 0.85;
const EXTRACT_EPISODES_TOOL_NAME = "EmitEpisodeCandidates";
export const EXTRACT_EPISODES_TOOL = {
  name: EXTRACT_EPISODES_TOOL_NAME,
  description: "Emit grounded episodic memory candidates for the provided stream chunk.",
  inputSchema: toToolInputSchema(extractorResponseSchema),
} satisfies LLMToolDefinition;

export type EpisodicExtractorOptions = {
  dataDir: string;
  episodicRepository: EpisodicRepository;
  embeddingClient: EmbeddingClient;
  llmClient: LLMClient;
  model: string;
  clock?: Clock;
  chunkTokenLimit?: number;
  maxTokens?: number;
};

export type ExtractFromStreamOptions = {
  session?: SessionId;
  sinceTs?: number;
  untilTs?: number;
};

export type ExtractFromStreamResult = {
  inserted: number;
  updated: number;
  skipped: number;
};

function estimateTokens(entry: StreamEntry): number {
  if (entry.token_estimate !== undefined) {
    return entry.token_estimate;
  }

  const content =
    typeof entry.content === "string" ? entry.content : JSON.stringify(entry.content ?? null);
  return Math.max(1, Math.ceil(content.length / 4));
}

function chunkEntries(entries: readonly StreamEntry[], maxTokens: number): StreamEntry[][] {
  if (entries.length === 0) {
    return [];
  }

  const chunks: StreamEntry[][] = [];
  let currentChunk: StreamEntry[] = [];
  let currentTokens = 0;

  for (const entry of entries) {
    const entryTokens = estimateTokens(entry);

    if (currentChunk.length > 0 && currentTokens + entryTokens > maxTokens) {
      chunks.push(currentChunk);
      currentChunk = [];
      currentTokens = 0;
    }

    currentChunk.push(entry);
    currentTokens += entryTokens;
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk);
  }

  return chunks;
}

function uniqueStrings(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function uniqueStreamEntryIds(entries: readonly StreamEntry[]): Episode["source_stream_ids"] {
  return [...new Set(entries.map((entry) => entry.id))];
}

function entryContentText(entry: StreamEntry): string {
  return typeof entry.content === "string" ? entry.content : JSON.stringify(entry.content ?? null);
}

function buildEmotionalArc(sourceEntries: readonly StreamEntry[]): EmotionalArc | null {
  if (sourceEntries.length === 0) {
    return null;
  }

  const signals = sourceEntries.map((entry) =>
    analyzeAffectiveSignalHeuristically(entryContentText(entry)),
  );
  const byPeak = [...signals].sort(
    (left, right) =>
      Math.abs(right.valence) + right.arousal - (Math.abs(left.valence) + left.arousal),
  );

  return {
    start: {
      valence: signals[0]?.valence ?? 0,
      arousal: signals[0]?.arousal ?? 0,
    },
    peak: {
      valence: byPeak[0]?.valence ?? 0,
      arousal: byPeak[0]?.arousal ?? 0,
    },
    end: {
      valence: signals.at(-1)?.valence ?? 0,
      arousal: signals.at(-1)?.arousal ?? 0,
    },
    dominant_emotion:
      signals.find((signal) => signal.dominant_emotion !== "neutral")?.dominant_emotion ?? null,
  };
}

function buildExtractorPrompt(chunk: readonly StreamEntry[]): string {
  const lines = chunk.map((entry) =>
    JSON.stringify({
      id: entry.id,
      timestamp: entry.timestamp,
      kind: entry.kind,
      content: entry.content,
      audience: entry.audience,
    }),
  );

  return [
    "You extract episodic memories from a stream chunk.",
    `Emit your result by calling the ${EXTRACT_EPISODES_TOOL_NAME} tool exactly once.`,
    "source_stream_ids MUST only reference ids present in the chunk.",
    "Narrative should be 2-5 concise sentences.",
    "Chunk:",
    ...lines,
  ].join("\n");
}

function parseLlmResponse(result: LLMCompleteResult): ExtractorCandidate[] {
  const call = result.tool_calls.find((toolCall) => toolCall.name === EXTRACT_EPISODES_TOOL_NAME);

  if (call === undefined) {
    throw new LLMError(`Extractor did not emit tool ${EXTRACT_EPISODES_TOOL_NAME}`, {
      code: "EXTRACTOR_OUTPUT_INVALID",
    });
  }

  const parsed = extractorResponseSchema.safeParse(call.input);

  if (!parsed.success) {
    throw new LLMError("Extractor returned invalid episode payload", {
      cause: parsed.error,
      code: "EXTRACTOR_OUTPUT_INVALID",
    });
  }

  return parsed.data.episodes;
}

function sourceEntriesFromCandidate(
  candidate: ExtractorCandidate,
  chunkEntriesById: Map<string, StreamEntry>,
): StreamEntry[] {
  const entries = candidate.source_stream_ids
    .map((sourceId) => chunkEntriesById.get(sourceId))
    .filter((entry): entry is StreamEntry => entry !== undefined);

  if (entries.length !== candidate.source_stream_ids.length) {
    throw new LLMError("Extractor referenced stream ids outside the chunk", {
      code: "EXTRACTOR_SOURCE_ID_INVALID",
    });
  }

  return entries;
}

function buildEpisodeFromCandidate(
  candidate: ExtractorCandidate,
  sourceEntries: readonly StreamEntry[],
  embedding: Float32Array,
  nowMs: number,
): Episode {
  const timestamps = sourceEntries.map((entry) => entry.timestamp);

  return {
    id: createEpisodeId(),
    title: candidate.title.trim(),
    narrative: candidate.narrative.trim(),
    participants: uniqueStrings(candidate.participants),
    location: candidate.location ?? null,
    start_time: Math.min(...timestamps),
    end_time: Math.max(...timestamps),
    source_stream_ids: uniqueStreamEntryIds(sourceEntries),
    significance: candidate.significance,
    tags: uniqueStrings(candidate.tags),
    confidence: candidate.confidence,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    emotional_arc: buildEmotionalArc(sourceEntries),
    embedding,
    created_at: nowMs,
    updated_at: nowMs,
  };
}

function episodeToPatch(episode: Episode) {
  return {
    title: episode.title,
    narrative: episode.narrative,
    participants: episode.participants,
    location: episode.location,
    start_time: episode.start_time,
    end_time: episode.end_time,
    source_stream_ids: episode.source_stream_ids,
    significance: episode.significance,
    tags: episode.tags,
    confidence: episode.confidence,
    lineage: episode.lineage,
    embedding: episode.embedding,
    updated_at: episode.updated_at,
  };
}

type CandidateOutcome = "inserted" | "updated" | "skipped";

export class EpisodicExtractor {
  private readonly clock: Clock;
  private readonly chunkTokenLimit: number;
  private readonly maxTokens: number;

  constructor(private readonly options: EpisodicExtractorOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.chunkTokenLimit = options.chunkTokenLimit ?? 16_000;
    this.maxTokens = options.maxTokens ?? 16_000;
  }

  private async processCandidate(
    candidate: ExtractorCandidate,
    chunkById: Map<string, StreamEntry>,
  ): Promise<CandidateOutcome> {
    const sourceEntries = sourceEntriesFromCandidate(candidate, chunkById);

    try {
      const embedding = await this.options.embeddingClient.embed(
        `${candidate.title}\n${candidate.narrative}\n${candidate.tags.join(" ")}`,
      );
      const nowMs = this.clock.now();
      const nextEpisode = buildEpisodeFromCandidate(candidate, sourceEntries, embedding, nowMs);
      const matches = await this.options.episodicRepository.searchByVector(embedding, {
        limit: 1,
        minSimilarity: DEDUP_THRESHOLD,
      });
      const existing = matches[0];

      if (existing === undefined || existing.similarity < DEDUP_THRESHOLD) {
        await this.options.episodicRepository.insert(nextEpisode);
        return "inserted";
      }

      const merged = this.options.episodicRepository.mergeEpisodeFields(existing.episode, {
        title:
          nextEpisode.confidence >= existing.episode.confidence
            ? nextEpisode.title
            : existing.episode.title,
        narrative:
          nextEpisode.confidence >= existing.episode.confidence
            ? nextEpisode.narrative
            : existing.episode.narrative,
        participants: nextEpisode.participants,
        location: nextEpisode.location ?? existing.episode.location,
        start_time: Math.min(existing.episode.start_time, nextEpisode.start_time),
        end_time: Math.max(existing.episode.end_time, nextEpisode.end_time),
        source_stream_ids: nextEpisode.source_stream_ids,
        significance: Math.max(existing.episode.significance, nextEpisode.significance),
        tags: nextEpisode.tags,
        confidence: Math.max(existing.episode.confidence, nextEpisode.confidence),
        embedding,
        lineage: existing.episode.lineage,
      });
      const updatedEpisode = await this.options.episodicRepository.update(
        existing.episode.id,
        episodeToPatch(merged),
      );

      return updatedEpisode === null ? "skipped" : "updated";
    } catch (error) {
      if (error instanceof EmbeddingError || error instanceof StorageError) {
        return "skipped";
      }

      throw error;
    }
  }

  async extractFromStream(
    extractOptions: ExtractFromStreamOptions = {},
  ): Promise<ExtractFromStreamResult> {
    const session = extractOptions.session ?? DEFAULT_SESSION_ID;
    const reader = new StreamReader({
      dataDir: this.options.dataDir,
      sessionId: session,
    });
    const streamEntries: StreamEntry[] = [];

    for await (const entry of reader.iterate({
      sinceTs: extractOptions.sinceTs,
      untilTs: extractOptions.untilTs,
    })) {
      streamEntries.push(entry);
    }

    if (streamEntries.length === 0) {
      return {
        inserted: 0,
        updated: 0,
        skipped: 0,
      };
    }

    let inserted = 0;
    let updated = 0;
    let skipped = 0;
    const chunks = chunkEntries(streamEntries, this.chunkTokenLimit);

    for (const chunk of chunks) {
      const chunkById = new Map(chunk.map((entry) => [entry.id, entry]));
      const result = await this.options.llmClient.complete({
        model: this.options.model,
        system: "Extract episodic memories grounded only in the provided stream chunk.",
        messages: [
          {
            role: "user",
            content: buildExtractorPrompt(chunk),
          },
        ],
        tools: [EXTRACT_EPISODES_TOOL],
        tool_choice: { type: "tool", name: EXTRACT_EPISODES_TOOL_NAME },
        max_tokens: this.maxTokens,
        budget: "episodic-extraction",
      });
      const candidates = parseLlmResponse(result);

      for (const candidate of candidates) {
        const outcome = await this.processCandidate(candidate, chunkById);

        if (outcome === "inserted") {
          inserted += 1;
          continue;
        }

        if (outcome === "skipped") {
          skipped += 1;
          continue;
        }

        updated += 1;
      }
    }

    return {
      inserted,
      updated,
      skipped,
    };
  }
}
