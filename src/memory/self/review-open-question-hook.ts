import { StreamWriter } from "../../stream/index.js";
import type { ReviewQueueItem } from "../review-queue/review-queue.js";
import {
  entityIdHelpers,
  episodeIdHelpers,
  semanticNodeIdHelpers,
  type EpisodeId,
  type SemanticNodeId,
} from "../../util/ids.js";
import type { IdentityService } from "../identity/index.js";
import { parseIdentityEventValueDisclosureSources } from "../identity/index.js";
import type { Provenance } from "../common/provenance.js";
import { expectedRecordVersion } from "../common/cas.js";
import {
  combineMemoryDisclosureLabels,
  relationshipPrivateMemoryDisclosureLabel,
  selfPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
} from "../common/disclosure-label.js";

import { OpenQuestionsRepository } from "./open-questions.js";
import type { OpenQuestion } from "./open-questions.js";
import {
  buildOpenQuestionDuplicatePresentation,
  buildOpenQuestionReinforcementPatch,
  findOpenQuestionDuplicateBackstop,
  type OpenQuestionDuplicatePresentation,
} from "./open-question-duplicates.js";
import type {
  OpenQuestionProposal,
  ReviewOpenQuestionContext,
  ReviewOpenQuestionExtractor,
} from "./review-open-question-extractor.js";

type OpenQuestionCreateInput = Parameters<OpenQuestionsRepository["add"]>[0];
type OpenQuestionWriter =
  | OpenQuestionsRepository
  | Pick<IdentityService, "addOpenQuestion" | "findSimilarOpenQuestion" | "updateOpenQuestion">;
export type ReviewOpenQuestionExtractorLike = Pick<ReviewOpenQuestionExtractor, "extract">;

export type ReviewOpenQuestionHookOptions = {
  extractor?: ReviewOpenQuestionExtractorLike | null;
  openQuestionsRepository?: OpenQuestionsRepository;
};

function addOpenQuestion(writer: OpenQuestionWriter, input: OpenQuestionCreateInput): OpenQuestion {
  if ("addOpenQuestion" in writer) {
    return writer.addOpenQuestion(input);
  }

  return writer.add(input);
}

function reinforceOpenQuestion(
  writer: OpenQuestionWriter,
  question: OpenQuestion,
  incoming: OpenQuestionCreateInput,
  provenance: Provenance,
): void {
  const patch = buildOpenQuestionReinforcementPatch({
    existing: question,
    incomingRelatedEpisodeIds: incoming.related_episode_ids ?? [],
    incomingRelatedSemanticNodeIds: incoming.related_semantic_node_ids ?? [],
    incomingDisclosureLabel: incoming.disclosure_label ?? unknownMemoryDisclosureLabel(),
    urgencyDelta: 0.02,
  });

  if ("updateOpenQuestion" in writer) {
    writer.updateOpenQuestion(question.id, patch, provenance, {
      throughReview: true,
      reason: "Similar review-derived open question already exists.",
      preserveRecordProvenance: true,
    });
    return;
  }

  writer.update(question.id, patch, {
    expectedVersion: expectedRecordVersion(question),
  });
}

function repositoryForDuplicateHandling(
  writer: OpenQuestionWriter,
  options: ReviewOpenQuestionHookOptions,
): OpenQuestionsRepository | null {
  if (options.openQuestionsRepository !== undefined) {
    return options.openQuestionsRepository;
  }

  return writer instanceof OpenQuestionsRepository ? writer : null;
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
    item.kind === "identity_inconsistency" ||
    item.kind === "commitment_reconciliation"
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
    open_question_duplicate_candidates: {
      complete: false,
      total_open_questions: 0,
      presented_count: 0,
      omitted_count: 0,
      rows: [],
    },
  };
}

function reviewOpenQuestionDisclosureLabel(
  item: ReviewQueueItem,
  context: ReviewOpenQuestionContext,
) {
  const sources = parseIdentityEventValueDisclosureSources(item.refs, "open_question");
  const baseLabel =
    context.audience_entity_id === null
      ? selfPrivateMemoryDisclosureLabel()
      : relationshipPrivateMemoryDisclosureLabel([context.audience_entity_id]);
  const sourceLabels =
    sources.disclosureLabels.length === 0 || sources.malformed
      ? [unknownMemoryDisclosureLabel()]
      : sources.disclosureLabels;

  return combineMemoryDisclosureLabels([baseLabel, ...sourceLabels]);
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
  const duplicateRepository = repositoryForDuplicateHandling(writer, options);

  if (duplicateRepository !== null) {
    context.open_question_duplicate_candidates = await buildOpenQuestionDuplicatePresentation({
      repository: duplicateRepository,
      sourceTextProxy: JSON.stringify({
        kind: item.kind,
        reason: item.reason,
        refs: item.refs,
      }),
      onSearchFailure: (error) => {
        console.warn("Review open-question candidate search failed open", {
          review_item_id: item.id,
          error,
        });
      },
    });
  }

  const proposal = await options.extractor.extract(item, context);

  if (proposal === null) {
    return;
  }

  const relatedIds = filterProposalIds(proposal, context);
  const provenance = provenanceForFilteredProposal(relatedIds);
  const createInput: OpenQuestionCreateInput = {
    question: proposal.question,
    urgency: proposal.urgency,
    audience_entity_id: context.audience_entity_id,
    disclosure_label: reviewOpenQuestionDisclosureLabel(item, context),
    ...relatedIds,
    provenance,
    source: sourceForReviewItem(item),
  };
  const presentedIds = new Set(
    (context.open_question_duplicate_candidates?.rows ?? []).map((candidate) => candidate.id),
  );
  const advisoryDuplicate =
    duplicateRepository !== null &&
    proposal.duplicate_of_open_question_id != null &&
    presentedIds.has(proposal.duplicate_of_open_question_id)
      ? duplicateRepository.get(proposal.duplicate_of_open_question_id)
      : null;
  const backstop =
    advisoryDuplicate?.status === "open"
      ? null
      : duplicateRepository === null
        ? await writer.findSimilarOpenQuestion({ question: createInput.question })
        : await findOpenQuestionDuplicateBackstop({
            repository: duplicateRepository,
            question: createInput.question,
            onSearchFailure: (error) => {
              console.warn("Review open-question duplicate backstop failed open", {
                review_item_id: item.id,
                error,
              });
            },
          });
  const existing = advisoryDuplicate?.status === "open" ? advisoryDuplicate : backstop?.question;

  if (existing !== null && existing !== undefined) {
    reinforceOpenQuestion(
      writer,
      existing,
      createInput,
      provenance ?? {
        kind: "offline",
        process: sourceForReviewItem(item),
      },
    );
    return;
  }

  addOpenQuestion(writer, createInput);
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
