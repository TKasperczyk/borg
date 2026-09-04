import type { EntityRecord } from "../memory/commitments/index.js";
import type { StreamEntry } from "../stream/index.js";
import type { EntityId, StreamEntryId } from "../util/ids.js";
import type { RecencyMessage } from "./recency/index.js";
import type { CurrentTurnUserInputSenderAttribution } from "./turn-input.js";

export type ExtractorSelfIdentity = Pick<EntityRecord, "id" | "canonical_name" | "aliases">;

export type ExtractorConversationContextInput = {
  selfIdentity: ExtractorSelfIdentity | null;
  recentHistory: readonly RecencyMessage[];
  currentMessageEntries?: readonly StreamEntry[];
  currentMessageStreamEntryIds?: readonly StreamEntryId[];
  currentMessageSenderAttribution?: readonly CurrentTurnUserInputSenderAttribution[];
  audienceEntityId: EntityId | null;
  speakerEntityId?: EntityId | null;
  speakerDisplayName?: string | null;
  senderDisplayNameById?: (entityId: EntityId) => string | null | undefined;
};

function entryKindIsSelfAuthored(kind: StreamEntry["kind"] | null): boolean {
  return kind === "agent_msg" || kind === "agent_suppressed";
}

function attributedSender(input: {
  kind: StreamEntry["kind"] | null;
  senderEntityId: EntityId | null;
  attributedDisplayName?: string | null;
  selfIdentity: ExtractorSelfIdentity | null;
  senderDisplayNameById?: (entityId: EntityId) => string | null | undefined;
}): {
  senderEntityId: EntityId | null;
  senderDisplayName: string | null;
  senderIsSelf: boolean;
} {
  const senderIsSelfByKind = entryKindIsSelfAuthored(input.kind);
  const senderEntityId =
    input.senderEntityId ??
    (senderIsSelfByKind && input.selfIdentity !== null ? input.selfIdentity.id : null);
  const senderIsSelf =
    senderIsSelfByKind || (input.selfIdentity !== null && senderEntityId === input.selfIdentity.id);
  const senderDisplayName =
    input.attributedDisplayName ??
    (senderEntityId === null
      ? null
      : senderIsSelf && input.selfIdentity !== null
        ? input.selfIdentity.canonical_name
        : (input.senderDisplayNameById?.(senderEntityId) ?? null));

  return {
    senderEntityId,
    senderDisplayName,
    senderIsSelf,
  };
}

export function buildExtractorConversationContext(input: ExtractorConversationContextInput) {
  const presentedEntityIds = new Set<EntityId>();
  const addPresentedEntityId = (entityId: EntityId | null | undefined): void => {
    if (entityId !== null && entityId !== undefined) {
      presentedEntityIds.add(entityId);
    }
  };

  addPresentedEntityId(input.selfIdentity?.id);
  addPresentedEntityId(input.audienceEntityId);
  addPresentedEntityId(input.speakerEntityId);

  const recentHistory = input.recentHistory.slice(-8).map((message) => {
    const sender = attributedSender({
      kind: message.kind ?? null,
      senderEntityId: message.sender_entity_id ?? null,
      selfIdentity: input.selfIdentity,
      senderDisplayNameById: input.senderDisplayNameById,
    });
    addPresentedEntityId(sender.senderEntityId);

    return {
      stream_entry_id: message.stream_entry_id,
      timestamp_ms: message.ts,
      role: message.role,
      kind: message.kind ?? null,
      sender_entity_id: sender.senderEntityId,
      sender_display_name: sender.senderDisplayName,
      sender_is_self: sender.senderIsSelf,
      content: message.content,
    };
  });

  const currentEntriesById = new Map(
    (input.currentMessageEntries ?? []).map((entry) => [entry.id, entry]),
  );
  const currentAttributionById = new Map(
    (input.currentMessageSenderAttribution ?? []).map((attribution) => [
      attribution.entryId,
      attribution,
    ]),
  );
  const orderedCurrentEntryIds: StreamEntryId[] = [];
  const seenCurrentEntryIds = new Set<StreamEntryId>();

  for (const entryId of [
    ...(input.currentMessageEntries ?? []).map((entry) => entry.id),
    ...(input.currentMessageStreamEntryIds ?? []),
    ...(input.currentMessageSenderAttribution ?? []).map((attribution) => attribution.entryId),
  ]) {
    if (!seenCurrentEntryIds.has(entryId)) {
      seenCurrentEntryIds.add(entryId);
      orderedCurrentEntryIds.push(entryId);
    }
  }

  const currentMessageEntries = orderedCurrentEntryIds.map((entryId) => {
    const entry = currentEntriesById.get(entryId);
    const attribution = currentAttributionById.get(entryId);
    const sender = attributedSender({
      kind: entry?.kind ?? null,
      senderEntityId: entry?.sender_entity_id ?? attribution?.senderEntityId ?? null,
      attributedDisplayName: attribution?.senderDisplayName ?? null,
      selfIdentity: input.selfIdentity,
      senderDisplayNameById: input.senderDisplayNameById,
    });
    const replyTargetEntityId = entry?.reply_target_entity_id ?? null;

    addPresentedEntityId(sender.senderEntityId);
    addPresentedEntityId(replyTargetEntityId);

    return {
      stream_entry_id: entryId,
      timestamp_ms: entry?.timestamp ?? null,
      kind: entry?.kind ?? null,
      sender_entity_id: sender.senderEntityId,
      sender_display_name: sender.senderDisplayName,
      sender_is_self: sender.senderIsSelf,
      audience_routing_label: entry?.audience ?? null,
      audience_entity_id: input.audienceEntityId,
      reply_target_entity_id: replyTargetEntityId,
    };
  });

  return {
    self_identity:
      input.selfIdentity === null
        ? null
        : {
            entity_id: input.selfIdentity.id,
            canonical_name: input.selfIdentity.canonical_name,
            handles: [...input.selfIdentity.aliases],
          },
    recent_history: recentHistory,
    current_message_entries: currentMessageEntries,
    audience_entity_id: input.audienceEntityId,
    speaker_entity_id: input.speakerEntityId ?? null,
    speaker_display_name: input.speakerDisplayName ?? null,
    presented_entity_ids: [...presentedEntityIds],
  };
}
