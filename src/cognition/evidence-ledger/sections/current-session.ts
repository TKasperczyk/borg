import { currentUserMessageStateMetadata, optionalStateMetadata } from "../entry-metadata.js";
import { CURRENT_USER_TRUST_RANK, addEntry } from "../section-buckets.js";
import type { BuilderSectionContext } from "../builder-context.js";

export function addCurrentUserMessageSection(context: BuilderSectionContext): void {
  const stateMetadata = currentUserMessageStateMetadata(context.input, context.repos.entities);

  addEntry(context.buckets, "current_user_message", {
    id: `current_user_message:${context.input.currentUserEntry?.id ?? context.input.turnId ?? "unpersisted"}`,
    source_type: "current_user_message",
    session_scope: "current_session",
    actor: "user",
    trust_rank: CURRENT_USER_TRUST_RANK,
    text: context.input.currentUserMessage,
    ...optionalStateMetadata(stateMetadata),
    stream_index:
      context.input.currentUserEntry === undefined
        ? undefined
        : context.resolver.streamOrderById.get(context.input.currentUserEntry.id),
    taint:
      context.input.frameAnomaly?.status === "ok" && context.input.frameAnomaly.kind !== "normal"
        ? "quarantined"
        : "none",
  });
}

export function addCurrentSessionTranscriptSection(context: BuilderSectionContext): void {
  for (const entry of context.transcript.entries) {
    addEntry(context.buckets, "current_session_transcript", entry);
  }
}
