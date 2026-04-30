import { StreamWriter } from "../../stream/index.js";
import type { ReviewQueueItem } from "../semantic/review-queue.js";
import {
  entityIdHelpers,
  episodeIdHelpers,
  semanticNodeIdHelpers,
  type EpisodeId,
  type SemanticNodeId,
} from "../../util/ids.js";
import type { IdentityService } from "../identity/index.js";

import type { OpenQuestionsRepository } from "./open-questions.js";
import type {
  OpenQuestionProposal,
  ReviewOpenQuestionContext,
  ReviewOpenQuestionExtractor,
} from "./review-open-question-extractor.js";

type OpenQuestionCreateInput = Parameters<OpenQuestionsRepository["add"]>[0];
type OpenQuestionWriter = OpenQuestionsRepository | Pick<IdentityService, "addOpenQuestion">;
export type ReviewOpenQuestionExtractorLike = Pick<ReviewOpenQuestionExtractor, "extract">;

export type ReviewOpenQuestionHookOptions = {
  extractor?: ReviewOpenQuestionExtractorLike | null;
};

function addOpenQuestion(writer: OpenQuestionWriter, input: OpenQuestionCreateInput): void {
  if ("addOpenQuestion" in writer) {
    writer.addOpenQuestion(input);
    return;
  }

  writer.add(input);
}

function reviewItemAudienceEntityId(item: ReviewQueueItem) {
  const audienceEntityId = item.refs.audience_entity_id;

  return typeof audienceEntityId === "string" && entityIdHelpers.is(audienceEntityId)
    ? audienceEntityId
    : null;
}

function isReviewKindWithOpenQuestion(item: ReviewQueueItem): boolean {
  return (
    item.kind === "contradiction" ||
    item.kind === "misattribution" ||
    item.kind === "identity_inconsistency"
  );
}

function collectAllowedReviewReferenceIds(
  value: unknown,
  episodeIds: Set<EpisodeId>,
  semanticNodeIds: Set<SemanticNodeId>,
): void {
  if (typeof value === "string") {
    if (episodeIdHelpers.is(value)) {
      episodeIds.add(value);
    }

    if (semanticNodeIdHelpers.is(value)) {
      semanticNodeIds.add(value);
    }

    return;
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      collectAllowedReviewReferenceIds(item, episodeIds, semanticNodeIds);
    }

    return;
  }

  if (value !== null && typeof value === "object") {
    for (const item of Object.values(value)) {
      collectAllowedReviewReferenceIds(item, episodeIds, semanticNodeIds);
    }
  }
}

function buildReviewOpenQuestionContext(item: ReviewQueueItem): ReviewOpenQuestionContext {
  const episodeIds = new Set<EpisodeId>();
  const semanticNodeIds = new Set<SemanticNodeId>();

  collectAllowedReviewReferenceIds(item.refs, episodeIds, semanticNodeIds);

  return {
    audience_entity_id: reviewItemAudienceEntityId(item),
    allowed_episode_ids: [...episodeIds],
    allowed_semantic_node_ids: [...semanticNodeIds],
  };
}

function filterProposalIds(
  proposal: OpenQuestionProposal,
  context: ReviewOpenQuestionContext,
): Pick<OpenQuestionCreateInput, "related_episode_ids" | "related_semantic_node_ids"> {
  const allowedEpisodeIds = new Set(context.allowed_episode_ids);
  const allowedSemanticNodeIds = new Set(context.allowed_semantic_node_ids);

  return {
    related_episode_ids: proposal.related_episode_ids.filter((id) => allowedEpisodeIds.has(id)),
    related_semantic_node_ids: proposal.related_semantic_node_ids.filter((id) =>
      allowedSemanticNodeIds.has(id),
    ),
  };
}

function provenanceForFilteredProposal(
  relatedIds: Pick<OpenQuestionCreateInput, "related_episode_ids" | "related_semantic_node_ids">,
): OpenQuestionCreateInput["provenance"] {
  if (
    (relatedIds.related_episode_ids?.length ?? 0) > 0 ||
    (relatedIds.related_semantic_node_ids?.length ?? 0) > 0
  ) {
    return null;
  }

  return {
    kind: "offline",
    process: "overseer",
  };
}

function sourceForReviewItem(item: ReviewQueueItem): OpenQuestionCreateInput["source"] {
  return item.kind === "contradiction" ? "contradiction" : "overseer";
}

export function formatHookError(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }

  return String(error);
}

export async function enqueueOpenQuestionForReview(
  writer: OpenQuestionWriter,
  item: ReviewQueueItem,
  options: ReviewOpenQuestionHookOptions = {},
): Promise<void> {
  if (!isReviewKindWithOpenQuestion(item)) {
    return;
  }

  if (options.extractor === undefined || options.extractor === null) {
    return;
  }

  const context = buildReviewOpenQuestionContext(item);
  const proposal = await options.extractor.extract(item, context);

  if (proposal === null) {
    return;
  }

  const relatedIds = filterProposalIds(proposal, context);

  addOpenQuestion(writer, {
    question: proposal.question,
    urgency: proposal.urgency,
    audience_entity_id: context.audience_entity_id,
    ...relatedIds,
    provenance: provenanceForFilteredProposal(relatedIds),
    source: sourceForReviewItem(item),
  });
}

export async function appendInternalFailureEvent(
  streamWriter: StreamWriter,
  hook: string,
  error: unknown,
  details?: Record<string, unknown>,
): Promise<void> {
  try {
    await streamWriter.append({
      kind: "internal_event",
      content: {
        ...details,
        hook,
        error: formatHookError(error),
      },
    });
  } catch {
    // Best-effort logging only.
  }
}

export async function appendOpenQuestionHookFailureEvent(
  streamWriter: StreamWriter,
  hook: string,
  error: unknown,
): Promise<void> {
  await appendInternalFailureEvent(streamWriter, hook, error);
}
