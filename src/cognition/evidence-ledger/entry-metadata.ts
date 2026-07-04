import type { RelationalSlot } from "../../memory/relational-slots/index.js";
import {
  memoryDisclosureInternalUseNote,
  memoryDisclosureLabelMetadata,
  renderMemoryDisclosureLabelForModel,
  type MemoryDisclosureLabel,
  type MemoryDisclosureLabelRenderContext,
} from "../../retrieval/index.js";
import type { StreamEntry, TranscriptStreamEntry } from "../../stream/index.js";
import type { EntityId } from "../../util/ids.js";
import { undeliveredDraftFromContent } from "../generation/types.js";
import type { EvidenceLedgerBuildInput } from "./builder-types.js";
import { resolveSpeakerDisplayName, type SpeakerEntityRepository } from "../speaker-tags.js";
import type { ScopeResolver } from "./scope-resolver.js";
import type { EvidenceLedgerActor, EvidenceLedgerEntry, EvidenceLedgerTaint } from "./types.js";

export function actorForStreamEntry(entry: Pick<StreamEntry, "kind">): EvidenceLedgerActor {
  if (entry.kind === "user_msg") {
    return "user";
  }

  if (
    entry.kind === "agent_msg" ||
    entry.kind === "agent_suppressed" ||
    entry.kind === "agent_observed"
  ) {
    return "assistant";
  }

  return "system";
}

export function transcriptState(entry: TranscriptStreamEntry): string | undefined {
  if (entry.kind === "agent_suppressed") {
    return undeliveredDraftFromContent(entry.content) === undefined
      ? "suppressed"
      : "undelivered_draft";
  }

  if (entry.kind === "agent_observed") {
    return "observed";
  }

  return undefined;
}

export function streamPersistenceClass(entry: Pick<StreamEntry, "persistence_class">) {
  return entry.persistence_class === undefined
    ? {}
    : { persistence_class: entry.persistence_class };
}

export function speakerStateMetadata(
  entityRepository: SpeakerEntityRepository | undefined,
  senderEntityId: EntityId | null | undefined,
): Record<string, unknown> | undefined {
  if (senderEntityId === null || senderEntityId === undefined) {
    return undefined;
  }

  const displayName = resolveSpeakerDisplayName(entityRepository, senderEntityId);

  return {
    sender_entity_id: senderEntityId,
    ...(displayName === null ? {} : { sender_display_name: displayName }),
  };
}

export function replyTargetStateMetadata(
  entry: TranscriptStreamEntry,
  entityRepository: SpeakerEntityRepository | undefined,
): Record<string, unknown> | undefined {
  if (entry.kind !== "agent_msg") {
    return speakerStateMetadata(entityRepository, entry.sender_entity_id);
  }

  const replyTargetEntityId = entry.reply_target_entity_id ?? null;

  if (replyTargetEntityId === null) {
    return undefined;
  }

  const displayName = resolveSpeakerDisplayName(entityRepository, replyTargetEntityId);

  return {
    reply_target_kind: "entity",
    reply_target_entity_id: replyTargetEntityId,
    ...(displayName === null ? {} : { reply_target_display_name: displayName }),
  };
}

export function optionalStateMetadata(
  stateMetadata: Record<string, unknown> | undefined,
): Pick<EvidenceLedgerEntry, "state_metadata"> {
  return stateMetadata === undefined ? {} : { state_metadata: stateMetadata };
}

export function appendMemoryDisclosureState(input: {
  state: string | undefined;
  disclosureLabel: MemoryDisclosureLabel | undefined;
  renderContext?: MemoryDisclosureLabelRenderContext;
}): string | undefined {
  if (input.disclosureLabel === undefined) {
    return input.state;
  }

  if (input.disclosureLabel.disclosureClass === "public") {
    return input.state;
  }

  const rendered = renderMemoryDisclosureLabelForModel(input.disclosureLabel, {
    context: input.renderContext,
  });

  return input.state === undefined || input.state.length === 0
    ? rendered
    : `${input.state} ${rendered}`;
}

export function memoryDisclosureStateMetadata(input: {
  disclosureLabel: MemoryDisclosureLabel | undefined;
  renderContext?: MemoryDisclosureLabelRenderContext;
  currentAudienceEntityId?: EntityId | null;
}): Record<string, unknown> {
  if (input.disclosureLabel === undefined || input.disclosureLabel.disclosureClass === "public") {
    return {};
  }

  return {
    disclosure_label: memoryDisclosureLabelMetadata(input.disclosureLabel),
    disclosure_note: memoryDisclosureInternalUseNote(input.renderContext),
    ...(input.currentAudienceEntityId === undefined
      ? {}
      : { current_audience_entity_id: input.currentAudienceEntityId }),
  };
}

export function appendMemoryDisclosureStateMetadata(input: {
  stateMetadata: Record<string, unknown> | undefined;
  disclosureLabel: MemoryDisclosureLabel | undefined;
  renderContext?: MemoryDisclosureLabelRenderContext;
  currentAudienceEntityId?: EntityId | null;
}): Record<string, unknown> | undefined {
  const metadata = {
    ...(input.stateMetadata ?? {}),
    ...memoryDisclosureStateMetadata({
      disclosureLabel: input.disclosureLabel,
      renderContext: input.renderContext,
      currentAudienceEntityId: input.currentAudienceEntityId,
    }),
  };

  return Object.keys(metadata).length === 0 ? undefined : metadata;
}

export function rawStreamActor(
  streamEntryIds: readonly string[] | undefined,
  resolver: ScopeResolver,
): EvidenceLedgerActor {
  const actors = new Set<EvidenceLedgerActor>();

  for (const streamEntryId of streamEntryIds ?? []) {
    const entry = resolver.streamEntriesById.get(streamEntryId);

    if (entry !== undefined) {
      actors.add(actorForStreamEntry(entry));
    }
  }

  return actors.size === 1 ? ([...actors][0] ?? "memory") : "memory";
}

export function slotTaint(slot: RelationalSlot): EvidenceLedgerTaint {
  if (slot.state === "quarantined") {
    return "quarantined";
  }

  if (slot.state === "contested") {
    return "contested";
  }

  return "none";
}

export function semanticTaint(input: {
  underReview?: unknown;
  status?: string;
  validTo?: number | null;
  invalidatedAt?: number | null;
}): EvidenceLedgerTaint {
  if (
    input.underReview !== undefined ||
    (input.status !== undefined && input.status !== "active") ||
    (input.validTo !== undefined && input.validTo !== null) ||
    (input.invalidatedAt !== undefined && input.invalidatedAt !== null)
  ) {
    return "contested";
  }

  return "none";
}

export function semanticNodeStateMetadata(node: {
  id?: string;
  source_episode_ids?: readonly string[];
  status?: string;
  corrected_by?: string | null;
  superseded_at?: number | null;
  partial_source_visibility?: boolean;
  source_visibility_fraction?: number;
}): Record<string, unknown> | undefined {
  const nonActiveStatus = node.status !== undefined && node.status !== "active";
  const metadata: Record<string, unknown> = {
    ...(node.id === undefined ? {} : { node_id: node.id }),
    ...(!nonActiveStatus ? {} : { status: node.status }),
    ...(!nonActiveStatus || node.superseded_at === undefined || node.superseded_at === null
      ? {}
      : { superseded_at: node.superseded_at }),
    ...(node.source_episode_ids === undefined || node.source_episode_ids.length === 0
      ? {}
      : { source_episode_ids: [...node.source_episode_ids] }),
  };

  if (node.partial_source_visibility === true) {
    metadata.partial_source_visibility = true;
    if (node.source_visibility_fraction !== undefined) {
      metadata.source_visibility_fraction = node.source_visibility_fraction;
    }
  }

  return Object.keys(metadata).length === 0 ? undefined : metadata;
}

export function currentUserMessageStateMetadata(
  input: EvidenceLedgerBuildInput,
  entityRepository: SpeakerEntityRepository | undefined,
): Record<string, unknown> | undefined {
  const senderIds = [
    ...new Set(
      (input.currentUserEntries ?? []).flatMap((entry) =>
        entry.sender_entity_id === null ? [] : [entry.sender_entity_id],
      ),
    ),
  ];

  if (senderIds.length === 1) {
    return speakerStateMetadata(entityRepository, senderIds[0]);
  }

  if ((input.currentUserEntries ?? []).length > 0) {
    return undefined;
  }

  return speakerStateMetadata(entityRepository, input.currentUserEntry?.sender_entity_id);
}
