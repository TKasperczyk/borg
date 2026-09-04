import { openQuestionMemoryDisclosureLabel } from "../common/disclosure-serializers.js";
import {
  combineMemoryDisclosureLabels,
  type MemoryDisclosureLabel,
} from "../common/disclosure-label.js";
import { utf16SafePrefixEnd } from "../../util/utf16-boundary.js";

import type { OpenQuestionsRepository } from "./open-questions.js";
import type { OpenQuestion, OpenQuestionPatch, OpenQuestionSearchCandidate } from "./types.js";

export const OPEN_QUESTION_DUPLICATE_PRESENTATION_LIMIT = 150;
export const OPEN_QUESTION_DUPLICATE_BACKSTOP_SIMILARITY_THRESHOLD = 0.9;
const OPEN_QUESTION_DUPLICATE_TEXT_EXCERPT_CHARS = 240;

export type OpenQuestionDuplicatePresentationRow = {
  id: OpenQuestion["id"];
  text_excerpt: string;
  urgency: number;
  source: OpenQuestion["source"];
  disclosure_label: MemoryDisclosureLabel;
};

export type OpenQuestionDuplicatePresentation = {
  complete: boolean;
  total_open_questions: number;
  presented_count: number;
  omitted_count: number;
  rows: OpenQuestionDuplicatePresentationRow[];
};

function compactOpenQuestionText(text: string): string {
  const trimmed = text.trim();

  if (trimmed.length <= OPEN_QUESTION_DUPLICATE_TEXT_EXCERPT_CHARS) {
    return trimmed;
  }

  const prefixEnd = utf16SafePrefixEnd(trimmed, OPEN_QUESTION_DUPLICATE_TEXT_EXCERPT_CHARS - 3);

  return `${trimmed.slice(0, prefixEnd).trimEnd()}...`;
}

export function openQuestionDuplicatePresentationRow(
  question: OpenQuestion,
): OpenQuestionDuplicatePresentationRow {
  return {
    id: question.id,
    text_excerpt: compactOpenQuestionText(question.question),
    urgency: question.urgency,
    source: question.source,
    disclosure_label: openQuestionMemoryDisclosureLabel(question),
  };
}

export async function buildOpenQuestionDuplicatePresentation(input: {
  repository: OpenQuestionsRepository;
  sourceTextProxy: string;
  onSearchFailure?: (error: unknown) => void;
}): Promise<OpenQuestionDuplicatePresentation> {
  const allOpenQuestions = input.repository.listAllOpen();
  let presentedQuestions = allOpenQuestions;
  const complete = allOpenQuestions.length <= OPEN_QUESTION_DUPLICATE_PRESENTATION_LIMIT;

  if (!complete) {
    try {
      presentedQuestions = (
        await input.repository.searchByText(input.sourceTextProxy, {
          status: "open",
          limit: OPEN_QUESTION_DUPLICATE_PRESENTATION_LIMIT,
        })
      ).map((candidate) => candidate.question);
    } catch (error) {
      input.onSearchFailure?.(error);
      presentedQuestions = [];
    }
  }

  return {
    complete,
    total_open_questions: allOpenQuestions.length,
    presented_count: presentedQuestions.length,
    omitted_count: allOpenQuestions.length - presentedQuestions.length,
    rows: presentedQuestions.map(openQuestionDuplicatePresentationRow),
  };
}

export async function findOpenQuestionDuplicateBackstop(input: {
  repository: OpenQuestionsRepository;
  question: string;
  onSearchFailure?: (error: unknown) => void;
}): Promise<OpenQuestionSearchCandidate | null> {
  const exact = await input.repository.findSimilarOpenQuestion({
    question: input.question,
  });

  if (exact !== null) {
    return exact;
  }

  try {
    return (
      (
        await input.repository.searchByText(input.question, {
          status: "open",
          limit: 1,
          minSimilarity: OPEN_QUESTION_DUPLICATE_BACKSTOP_SIMILARITY_THRESHOLD,
        })
      )[0] ?? null
    );
  } catch (error) {
    input.onSearchFailure?.(error);
    return null;
  }
}

export function buildOpenQuestionReinforcementPatch(input: {
  existing: OpenQuestion;
  incomingRelatedEpisodeIds: readonly OpenQuestion["related_episode_ids"][number][];
  incomingRelatedSemanticNodeIds: readonly OpenQuestion["related_semantic_node_ids"][number][];
  incomingDisclosureLabel: MemoryDisclosureLabel;
  urgencyDelta?: number;
}): Pick<
  OpenQuestionPatch,
  "urgency" | "related_episode_ids" | "related_semantic_node_ids" | "disclosure_label"
> {
  return {
    urgency: Math.min(1, input.existing.urgency + (input.urgencyDelta ?? 0)),
    related_episode_ids: [
      ...new Set([...input.existing.related_episode_ids, ...input.incomingRelatedEpisodeIds]),
    ],
    related_semantic_node_ids: [
      ...new Set([
        ...input.existing.related_semantic_node_ids,
        ...input.incomingRelatedSemanticNodeIds,
      ]),
    ],
    disclosure_label: combineMemoryDisclosureLabels([
      openQuestionMemoryDisclosureLabel(input.existing),
      input.incomingDisclosureLabel,
    ]),
  };
}
