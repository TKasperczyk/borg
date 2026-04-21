import { StreamWriter } from "../../stream/index.js";
import type { ReviewQueueItem } from "../semantic/review-queue.js";
import { parseEpisodeId, parseSemanticNodeId } from "../../util/ids.js";

import type { OpenQuestionsRepository } from "./open-questions.js";

export function formatHookError(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }

  return String(error);
}

export function enqueueOpenQuestionForReview(
  repository: OpenQuestionsRepository,
  item: ReviewQueueItem,
): void {
  if (item.kind === "contradiction") {
    const nodeIds = Array.isArray(item.refs.node_ids)
      ? item.refs.node_ids
          .filter((value): value is string => typeof value === "string")
          .map((value) => parseSemanticNodeId(value))
      : [];
    const nodeLabels = Array.isArray(item.refs.node_labels)
      ? item.refs.node_labels.filter((value): value is string => typeof value === "string")
      : [];
    const summary =
      nodeLabels.length >= 2
        ? `${nodeLabels[0]} vs ${nodeLabels[1]}`
        : nodeIds.slice(0, 2).join(" vs ") || "these claims";

    repository.add({
      question: `Which of these claims is right: ${summary}?`,
      urgency: 0.7,
      related_semantic_node_ids: nodeIds.slice(0, 2),
      source: "contradiction",
    });
    return;
  }

  if (item.kind === "misattribution" || item.kind === "identity_inconsistency") {
    const relatedEpisodeIds =
      item.refs.target_type === "episode" && typeof item.refs.target_id === "string"
        ? [parseEpisodeId(item.refs.target_id)]
        : [];
    const relatedSemanticNodeIds =
      item.refs.target_type === "semantic_node" && typeof item.refs.target_id === "string"
        ? [parseSemanticNodeId(item.refs.target_id)]
        : [];

    repository.add({
      question:
        item.kind === "misattribution"
          ? "What needs to be corrected about this memory's attribution?"
          : "How should I reconcile this memory with my active values, goals, or traits?",
      urgency: 0.55,
      related_episode_ids: relatedEpisodeIds,
      related_semantic_node_ids: relatedSemanticNodeIds,
      source: "overseer",
    });
  }
}

export async function appendInternalFailureEvent(
  streamWriter: StreamWriter,
  hook: string,
  error: unknown,
): Promise<void> {
  try {
    await streamWriter.append({
      kind: "internal_event",
      content: {
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
