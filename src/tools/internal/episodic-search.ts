import { z } from "zod";
import {
  readRecordEvidenceContexts,
  recordEvidenceContext,
  recordEvidenceShape,
  renderRecordEvidenceObject,
  type RecordEvidenceReader,
} from "../../memory/common/record-evidence.js";
import { EVIDENCE_GUIDANCE, evidencePromptFields } from "../../memory/common/evidence-prompt.js";
import {
  frameAnomalyAnnotationFields,
  frameAnomalyAnnotationSchema,
} from "../../stream/frame-anomaly.js";

import { memoryDisclosurePayloadFields } from "../../memory/common/disclosure-serializers.js";
import {
  MEMORY_DISCLOSURE_CLASSES,
  memoryDisclosureLabelFromEpisodeAccess,
  memoryDisclosureLabelMetadata,
  renderMemoryDisclosureLabelForModel,
  type RetrievedEpisode,
} from "../../retrieval/index.js";
import {
  MAX_RECALLED_EVIDENCE_TEXT_CHARS,
  MAX_RECALLED_SOURCE_MESSAGES_PER_EPISODE,
} from "../../retrieval/evidence-bounds.js";
import type { ToolDefinition, ToolInvocationContext } from "../dispatcher.js";
import {
  retrievalStatus,
  retrievalStatusSchema,
  type RetrievalDegradation,
} from "../../retrieval/degradation.js";

const DEFAULT_EPISODIC_SEARCH_LIMIT = 5;
const MAX_EPISODIC_SEARCH_LIMIT = 5;
const MAX_NARRATIVE_CHARS = 400;

const episodicSearchInputSchema = z.object({
  query: z.string().min(1),
  limit: z.number().int().positive().max(MAX_EPISODIC_SEARCH_LIMIT).optional(),
});

const episodicSearchOutputSchema = z.object({
  evidence_guidance: z.string().optional(),
  retrieval_status: retrievalStatusSchema.optional(),
  episodes: z.array(
    z.object({
      ...recordEvidenceShape,
      id: z.string().min(1),
      title: z.string().min(1),
      narrative: z.string(),
      participants: z.array(z.string()),
      tags: z.array(z.string()),
      start_time: z.number().finite(),
      end_time: z.number().finite(),
      source_stream_ids: z.array(z.string().min(1)),
      disclosure: z.string().min(1),
      disclosure_label: z.object({
        disclosure_class: z.enum(MEMORY_DISCLOSURE_CLASSES),
        origin_audience_entity_ids: z.array(z.string().min(1)),
        private_to_entity_ids: z.array(z.string().min(1)),
        public_to_entity_ids: z.array(z.string().min(1)),
      }),
      score: z.number().finite(),
      score_breakdown: z.object({
        similarity: z.number().finite(),
        decayed_salience: z.number().finite(),
        time_relevance: z.number().finite(),
      }),
      citation_chain: z.array(
        z.object({
          id: z.string().min(1),
          kind: z.string().min(1),
          timestamp: z.number().finite(),
          content: z.string(),
          frame_anomalies: z.array(frameAnomalyAnnotationSchema).optional(),
          disclosure: z.string().min(1),
          disclosure_label: z.object({
            disclosure_class: z.enum(MEMORY_DISCLOSURE_CLASSES),
            origin_audience_entity_ids: z.array(z.string().min(1)),
            private_to_entity_ids: z.array(z.string().min(1)),
            public_to_entity_ids: z.array(z.string().min(1)),
          }),
        }),
      ),
    }),
  ),
});

export type EpisodicSearchToolOptions = {
  recordEvidenceReader?: RecordEvidenceReader;
  searchEpisodes: (
    query: string,
    limit: number | undefined,
    context: ToolInvocationContext,
  ) => Promise<
    RetrievedEpisode[] | { episodes: RetrievedEpisode[]; degradations: RetrievalDegradation[] }
  >;
};

function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function truncateText(text: string, maxChars: number): string {
  const normalized = normalizeWhitespace(text);

  if (normalized.length <= maxChars) {
    return normalized;
  }

  return `${normalized.slice(0, maxChars - 3).trimEnd()}...`;
}

function summarizeCitationContent(content: unknown): string {
  const serialized = JSON.stringify(content ?? null);
  const text = typeof content === "string" ? content : (serialized ?? String(content));

  return truncateText(text, MAX_RECALLED_EVIDENCE_TEXT_CHARS);
}

export function createEpisodicSearchTool(
  options: EpisodicSearchToolOptions,
): ToolDefinition<
  z.infer<typeof episodicSearchInputSchema>,
  z.infer<typeof episodicSearchOutputSchema>
> {
  return {
    name: "tool.episodic.search",
    description: "Search episodic memory for relevant episodes.",
    menuSummary: "Search episodic memory by relevance.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "read",
    // Vector search over the Lance index; between compactions fragment counts
    // climb into the hundreds and honest queries exceed the 5s dispatcher
    // default (observed live: seven consecutive ~5s timeouts across two days).
    timeoutClass: "search",
    inputSchema: episodicSearchInputSchema,
    outputSchema: episodicSearchOutputSchema,
    async invoke(input, context) {
      const found = await options.searchEpisodes(
        input.query,
        Math.min(input.limit ?? DEFAULT_EPISODIC_SEARCH_LIMIT, MAX_EPISODIC_SEARCH_LIMIT),
        context,
      );
      const results = Array.isArray(found) ? found : found.episodes;
      const status = Array.isArray(found) ? undefined : retrievalStatus(found.degradations);
      const contexts = await readRecordEvidenceContexts(
        options.recordEvidenceReader,
        results.map(({ episode }) => ({
          record_type: "episode",
          record_id: episode.id,
        })),
        {
          readLoadedText: ({ record_id }) =>
            results.find(({ episode }) => episode.id === record_id)?.episode.narrative,
        },
      );

      return renderRecordEvidenceObject({
        evidence_guidance: EVIDENCE_GUIDANCE,
        ...(status === undefined ? {} : { retrieval_status: status }),
        episodes: results.map((result) => {
          const disclosureLabel =
            result.disclosureLabel ?? memoryDisclosureLabelFromEpisodeAccess(result.episode);

          return {
            ...evidencePromptFields(
              recordEvidenceContext(contexts, {
                record_type: "episode",
                record_id: result.episode.id,
              }),
            ),
            id: result.episode.id,
            title: result.episode.title,
            narrative: truncateText(result.episode.narrative, MAX_NARRATIVE_CHARS),
            participants: result.episode.participants,
            tags: result.episode.tags,
            start_time: result.episode.start_time,
            end_time: result.episode.end_time,
            source_stream_ids: result.episode.source_stream_ids,
            disclosure: renderMemoryDisclosureLabelForModel(disclosureLabel),
            disclosure_label: memoryDisclosureLabelMetadata(disclosureLabel),
            score: result.score,
            score_breakdown: {
              similarity: result.scoreBreakdown.similarity,
              decayed_salience: result.scoreBreakdown.decayedSalience,
              time_relevance: result.scoreBreakdown.timeRelevance,
            },
            citation_chain: result.citationChain
              .slice(0, MAX_RECALLED_SOURCE_MESSAGES_PER_EPISODE)
              .map((entry) => ({
                id: entry.id,
                kind: entry.kind,
                timestamp: entry.timestamp,
                content: summarizeCitationContent(entry.content),
                ...frameAnomalyAnnotationFields(entry.frame_anomalies),
                ...memoryDisclosurePayloadFields(disclosureLabel),
              })),
          };
        }),
      });
    },
  };
}
