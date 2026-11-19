import { z } from "zod";
import {
  readRecordEvidenceContexts,
  recordEvidenceContext,
  recordEvidenceShape,
  renderRecordEvidenceObject,
  type RecordEvidenceReader,
} from "../../memory/common/record-evidence.js";
import { EVIDENCE_GUIDANCE, evidencePromptFields } from "../../memory/common/evidence-prompt.js";

import { memoryDisclosurePayloadFields } from "../../memory/common/disclosure-serializers.js";
import { memoryDisclosureLabelMetadataSchema } from "../../memory/common/disclosure-label.js";
import type { EpisodeSearchCandidate } from "../../memory/episodic/index.js";
import {
  memoryDisclosureLabelFromEpisodeAccess,
  MEMORY_DISCLOSURE_CLASSES,
} from "../../retrieval/index.js";
import type { ToolDefinition, ToolInvocationContext } from "../dispatcher.js";

const DEFAULT_EPISODIC_RECENT_LIMIT = 5;
const MAX_EPISODIC_RECENT_LIMIT = 10;
const MAX_NARRATIVE_CHARS = 400;

const episodicRecentInputSchema = z
  .object({
    limit: z.number().int().positive().max(MAX_EPISODIC_RECENT_LIMIT).optional(),
  })
  .strict();

const episodicRecentOutputSchema = z.object({
  evidence_guidance: z.string().optional(),
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
      created_at: z.number().finite(),
      updated_at: z.number().finite(),
      disclosure: z.string().min(1),
      disclosure_label: memoryDisclosureLabelMetadataSchema.safeExtend({
        disclosure_class: z.enum(MEMORY_DISCLOSURE_CLASSES),
      }),
    }),
  ),
});

export type EpisodicRecentToolOptions = {
  recordEvidenceReader?: RecordEvidenceReader;
  listRecentEpisodes: (
    limit: number,
    context: ToolInvocationContext,
  ) => Promise<EpisodeSearchCandidate[]>;
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

export function createEpisodicRecentTool(
  options: EpisodicRecentToolOptions,
): ToolDefinition<
  z.infer<typeof episodicRecentInputSchema>,
  z.infer<typeof episodicRecentOutputSchema>
> {
  return {
    name: "tool.episodic.recent",
    description:
      "List my most recent episodic memories in time order for autonomous reflection when I need recent context rather than similarity search.",
    menuSummary: "Read the most recent episodic memories.",
    allowedOrigins: ["autonomous"],
    writeScope: "read",
    inputSchema: episodicRecentInputSchema,
    outputSchema: episodicRecentOutputSchema,
    async invoke(input, context) {
      const results = await options.listRecentEpisodes(
        Math.min(input.limit ?? DEFAULT_EPISODIC_RECENT_LIMIT, MAX_EPISODIC_RECENT_LIMIT),
        context,
      );

      const contexts = readRecordEvidenceContexts(
        options.recordEvidenceReader,
        results.map(({ episode }) => ({ record_type: "episode", record_id: episode.id })),
      );
      return renderRecordEvidenceObject({
        evidence_guidance: EVIDENCE_GUIDANCE,
        episodes: results.map((result) => ({
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
          created_at: result.episode.created_at,
          updated_at: result.episode.updated_at,
          ...memoryDisclosurePayloadFields(memoryDisclosureLabelFromEpisodeAccess(result.episode)),
        })),
      });
    },
  };
}
