import {
  type SharedStateArtifact,
  type SharedStateCanonicalizes,
  type SharedStateEntry,
  type SharedStateEntryKind,
  type SharedStateOperation,
  type SharedStateSourceTrustValidator,
} from "../../memory/decision-artifacts/index.js";
import {
  actionIdHelpers,
  commitmentIdHelpers,
  entityIdHelpers,
  goalIdHelpers,
  openQuestionIdHelpers,
  sharedStateEntryIdHelpers,
  type ActionId,
  type CommitmentId,
  type EntityId,
  type GoalId,
  type OpenQuestionId,
  type SharedStateEntryId,
  type StreamEntryId,
} from "../../util/ids.js";
import { checkRelationshipLabelGrounding } from "../memory-write-relationship-gate.js";
import type { ParticipantRoster } from "../perception/index.js";
import type {
  AllowedCanonicalizationIds,
  CanonicalizationDuplicateDrop,
  CanonicalizeIdChannel,
  DroppedCanonicalizeId,
  EmptyUpdateDrop,
  EmitSharedStatePatch,
  NonLockedCanonicalizesDrop,
  ParsedCanonicalizes,
  ParsedPatchOperation,
  PatchRejection,
  SharedStateArtifactParticipantContext,
  SharedStateCanonicalizationCandidates,
} from "./schema.js";
import { parseSourceStreamEntryIds } from "./source-trust.js";
import { sharedStateKeyTokens, stateKeysAreNearDuplicate } from "./state-key.js";
import type { SyncRelationshipEvidenceStreamEntryTrustValidator } from "../../memory/source-trust.js";
import { buildExistingStateKeyRegistry } from "./summary.js";

const DEFAULT_MAX_LIVE_ENTRIES_PER_KEY = 2;

function parseEntryId(value: string): SharedStateEntryId | null {
  try {
    return sharedStateEntryIdHelpers.parse(value);
  } catch {
    return null;
  }
}

function normalizeOwnerEntityId(
  value: string | null | undefined,
  allowedOwnerEntityIds: ReadonlySet<EntityId>,
): EntityId | null | "invalid" {
  if (value === undefined || value === null) {
    return null;
  }

  if (!entityIdHelpers.is(value)) {
    return "invalid";
  }

  return allowedOwnerEntityIds.has(value) ? value : "invalid";
}

function normalizeUpdateOwnerEntityId(
  value: string | null | undefined,
  entry: SharedStateEntry,
  allowedOwnerEntityIds: ReadonlySet<EntityId>,
): EntityId | null | "invalid" {
  if (value === undefined || value === null) {
    return null;
  }

  if (!entityIdHelpers.is(value)) {
    return "invalid";
  }

  if (value === entry.owner_entity_id) {
    return value;
  }

  return allowedOwnerEntityIds.has(value) ? value : "invalid";
}

export function emptyCanonicalizes(): SharedStateCanonicalizes {
  return {
    goal_ids: [],
    commitment_ids: [],
    action_ids: [],
    open_question_ids: [],
  };
}

export function hasCanonicalizes(value: SharedStateCanonicalizes | undefined): boolean {
  return (
    value !== undefined &&
    (value.goal_ids.length > 0 ||
      value.commitment_ids.length > 0 ||
      value.action_ids.length > 0 ||
      value.open_question_ids.length > 0)
  );
}

function parsedCanonicalizesTraceIds(
  value: ParsedCanonicalizes | undefined,
): NonLockedCanonicalizesDrop["dropped_ids"] | null {
  if (value === undefined) {
    return null;
  }

  const droppedIds = {
    goal_ids: [...(value.goal_ids ?? [])],
    commitment_ids: [...(value.commitment_ids ?? [])],
    action_ids: [...(value.action_ids ?? [])],
    open_question_ids: [...(value.open_question_ids ?? [])],
  };

  return droppedIds.goal_ids.length > 0 ||
    droppedIds.commitment_ids.length > 0 ||
    droppedIds.action_ids.length > 0 ||
    droppedIds.open_question_ids.length > 0
    ? droppedIds
    : null;
}

export function allowedCanonicalizationIds(
  candidates: SharedStateCanonicalizationCandidates | undefined,
): AllowedCanonicalizationIds {
  return {
    goalIds: new Set(
      (candidates?.goals ?? [])
        .map((candidate) => candidate.id)
        .filter((id): id is GoalId => goalIdHelpers.is(id)),
    ),
    commitmentIds: new Set(
      (candidates?.commitments ?? [])
        .map((candidate) => candidate.id)
        .filter((id): id is CommitmentId => commitmentIdHelpers.is(id)),
    ),
    actionIds: new Set(
      (candidates?.actions ?? [])
        .map((candidate) => candidate.id)
        .filter((id): id is ActionId => actionIdHelpers.is(id)),
    ),
    openQuestionIds: new Set(
      (candidates?.openQuestions ?? [])
        .map((candidate) => candidate.id)
        .filter((id): id is OpenQuestionId => openQuestionIdHelpers.is(id)),
    ),
  };
}

function normalizeCanonicalizeIds<TId extends string>(input: {
  values: readonly string[] | undefined;
  channel: CanonicalizeIdChannel;
  isId: (value: string) => value is TId;
  allowedIds: ReadonlySet<TId>;
  operation: ParsedPatchOperation;
  operationIndex: number;
  dropped: DroppedCanonicalizeId[];
}): TId[] {
  const ids: TId[] = [];

  for (const value of input.values ?? []) {
    if (!input.isId(value)) {
      input.dropped.push({
        channel: input.channel,
        id: value,
        reason: "invalid_id",
        operationType: input.operation.type,
        operationIndex: input.operationIndex,
      });
      continue;
    }

    if (!input.allowedIds.has(value)) {
      input.dropped.push({
        channel: input.channel,
        id: value,
        reason: "unknown_id",
        operationType: input.operation.type,
        operationIndex: input.operationIndex,
      });
      continue;
    }

    if (!ids.some((id) => id === value)) {
      ids.push(value);
    }
  }

  return ids;
}

function normalizeCanonicalizes(input: {
  value: ParsedCanonicalizes | undefined;
  kind: SharedStateEntryKind;
  allowedIds: AllowedCanonicalizationIds;
  operation: ParsedPatchOperation;
  operationIndex: number;
  dropped: DroppedCanonicalizeId[];
  nonLockedDrops: NonLockedCanonicalizesDrop[];
}): SharedStateCanonicalizes | undefined {
  if (input.value === undefined) {
    return undefined;
  }

  const nonLockedDroppedIds = parsedCanonicalizesTraceIds(input.value);

  if (input.kind !== "locked" && nonLockedDroppedIds !== null) {
    input.nonLockedDrops.push({
      operation_index: input.operationIndex,
      kind: input.kind,
      dropped_ids: nonLockedDroppedIds,
    });

    return undefined;
  }

  const canonicalizes: SharedStateCanonicalizes = {
    goal_ids: normalizeCanonicalizeIds({
      values: input.value.goal_ids,
      channel: "goal",
      isId: goalIdHelpers.is,
      allowedIds: input.allowedIds.goalIds,
      operation: input.operation,
      operationIndex: input.operationIndex,
      dropped: input.dropped,
    }),
    commitment_ids: normalizeCanonicalizeIds({
      values: input.value.commitment_ids,
      channel: "commitment",
      isId: commitmentIdHelpers.is,
      allowedIds: input.allowedIds.commitmentIds,
      operation: input.operation,
      operationIndex: input.operationIndex,
      dropped: input.dropped,
    }),
    action_ids: normalizeCanonicalizeIds({
      values: input.value.action_ids,
      channel: "action",
      isId: actionIdHelpers.is,
      allowedIds: input.allowedIds.actionIds,
      operation: input.operation,
      operationIndex: input.operationIndex,
      dropped: input.dropped,
    }),
    open_question_ids: normalizeCanonicalizeIds({
      values: input.value.open_question_ids,
      channel: "open_question",
      isId: openQuestionIdHelpers.is,
      allowedIds: input.allowedIds.openQuestionIds,
      operation: input.operation,
      operationIndex: input.operationIndex,
      dropped: input.dropped,
    }),
  };

  return hasCanonicalizes(canonicalizes) ? canonicalizes : emptyCanonicalizes();
}

function previousEntryById(
  previousEntries: ReadonlyMap<SharedStateEntryId, SharedStateEntry>,
  id: string,
): { id: SharedStateEntryId | null; entry: SharedStateEntry | null } {
  const parsedId = parseEntryId(id);

  if (parsedId === null) {
    return { id: null, entry: null };
  }

  return {
    id: parsedId,
    entry: previousEntries.get(parsedId) ?? null,
  };
}

function canonicalizesAddNewIds(
  existing: SharedStateCanonicalizes,
  proposed: SharedStateCanonicalizes | undefined,
): boolean {
  if (proposed === undefined) {
    return false;
  }

  return (
    proposed.goal_ids.some((id) => !existing.goal_ids.some((existingId) => existingId === id)) ||
    proposed.commitment_ids.some(
      (id) => !existing.commitment_ids.some((existingId) => existingId === id),
    ) ||
    proposed.action_ids.some(
      (id) => !existing.action_ids.some((existingId) => existingId === id),
    ) ||
    proposed.open_question_ids.some(
      (id) => !existing.open_question_ids.some((existingId) => existingId === id),
    )
  );
}

function isMaterialNoopUpdate(input: {
  operation: Extract<ParsedPatchOperation, { type: "update" }>;
  entry: SharedStateEntry;
  nextKind: SharedStateEntryKind;
  ownerEntityId: EntityId | null;
  canonicalizes: SharedStateCanonicalizes | undefined;
}): boolean {
  return (
    input.operation.state_key === input.entry.state_key &&
    input.nextKind === input.entry.kind &&
    (input.operation.text === undefined || input.operation.text === input.entry.text) &&
    (input.operation.owner_entity_id === undefined ||
      input.ownerEntityId === input.entry.owner_entity_id) &&
    !canonicalizesAddNewIds(input.entry.canonicalizes, input.canonicalizes)
  );
}

function emptyUpdateDrop(input: {
  operation: Extract<ParsedPatchOperation, { type: "update" }>;
  operationIndex: number;
  id: SharedStateEntryId;
  entry: SharedStateEntry;
}): EmptyUpdateDrop {
  return {
    operationIndex: input.operationIndex,
    operationId: input.id,
    stateKey: input.entry.state_key,
    fieldPresence: {
      kind: input.operation.kind !== undefined,
      text: input.operation.text !== undefined,
      owner_entity_id: input.operation.owner_entity_id !== undefined,
      canonicalizes: input.operation.canonicalizes !== undefined,
    },
  };
}

type StateKeyTrackedEntry = Pick<
  SharedStateEntry,
  "kind" | "state_key" | "created_at" | "last_updated_at" | "rank"
> & {
  id: string;
  fromPreviousArtifact: boolean;
};

function normalizeMaxLiveEntriesPerKey(value: number | undefined): number {
  return Number.isFinite(value) && value !== undefined && value > 0
    ? Math.floor(value)
    : DEFAULT_MAX_LIVE_ENTRIES_PER_KEY;
}

function activePreviousEntriesByStateKey(
  artifact: SharedStateArtifact | null,
): Map<string, StateKeyTrackedEntry> {
  const entries = new Map<string, StateKeyTrackedEntry>();

  for (const entry of artifact?.entries ?? []) {
    if (entry.superseded_by_id !== null) {
      continue;
    }

    entries.set(entry.id, {
      id: entry.id,
      kind: entry.kind,
      state_key: entry.state_key,
      created_at: entry.created_at,
      last_updated_at: entry.last_updated_at,
      rank: entry.rank,
      fromPreviousArtifact: true,
    });
  }

  return entries;
}

function compareTrackedEntryRecency(
  left: StateKeyTrackedEntry,
  right: StateKeyTrackedEntry,
): number {
  return (
    right.last_updated_at - left.last_updated_at ||
    right.created_at - left.created_at ||
    left.rank - right.rank ||
    left.id.localeCompare(right.id)
  );
}

function liveEntriesForStateKey(
  entries: ReadonlyMap<string, StateKeyTrackedEntry>,
  stateKey: string,
): StateKeyTrackedEntry[] {
  return [...entries.values()].filter(
    (entry) => entry.kind === "live" && entry.state_key === stateKey,
  );
}

function lockedEntriesForStateKey(
  entries: ReadonlyMap<string, StateKeyTrackedEntry>,
  stateKey: string,
): StateKeyTrackedEntry[] {
  return [...entries.values()].filter(
    (entry) => entry.kind === "locked" && entry.state_key === stateKey,
  );
}

function mostRecentPreviousEntry(
  entries: readonly StateKeyTrackedEntry[],
): StateKeyTrackedEntry | null {
  return (
    entries.filter((entry) => entry.fromPreviousArtifact).sort(compareTrackedEntryRecency)[0] ??
    null
  );
}

function rejection(
  operation: ParsedPatchOperation,
  operationIndex: number,
  reason: PatchRejection["reason"],
  details: Pick<
    PatchRejection,
    | "sourceStreamEntryId"
    | "sourceTrustReason"
    | "stateKey"
    | "currentCount"
    | "proposedCount"
    | "maxLiveEntriesPerKey"
    | "targetEntryId"
    | "lockedEntryIds"
    | "similarStateKeys"
    | "sharedStateKeyTokens"
    | "protectedRelationshipLabels"
    | "relationshipEvidenceRelationalSlotIds"
    | "relationshipEvidenceStreamEntryIds"
    | "rejectedRelationshipEvidenceRelationalSlotIds"
    | "rejectedRelationshipEvidenceStreamEntryIds"
  > = {},
): PatchRejection {
  return {
    reason,
    operationType: operation.type,
    operationIndex,
    ...details,
  };
}

export function normalizePatch(input: {
  patch: EmitSharedStatePatch;
  previousArtifact: SharedStateArtifact | null;
  audienceEntityId: EntityId;
  selfEntityId: EntityId;
  speakerEntityId: EntityId | null;
  participants: readonly SharedStateArtifactParticipantContext[];
  allowedSourceStreamEntryIds: ReadonlySet<StreamEntryId> | null;
  sourceTrustValidator?: SharedStateSourceTrustValidator;
  participantRoster?: ParticipantRoster | null;
  relationshipEvidenceStreamEntryTrust?: SyncRelationshipEvidenceStreamEntryTrustValidator;
  allowedCanonicalizationIds: AllowedCanonicalizationIds;
  maxLiveEntriesPerKey?: number;
}): {
  operations: SharedStateOperation[];
  rejected: PatchRejection[];
  droppedCanonicalizeIds: DroppedCanonicalizeId[];
  nonLockedCanonicalizesDrops: NonLockedCanonicalizesDrop[];
  emptyUpdateDrops: EmptyUpdateDrop[];
  emptyUpdateAttemptedCount: number;
} {
  const allowedOwnerEntityIds = new Set<EntityId>([
    input.audienceEntityId,
    input.selfEntityId,
    ...input.participants.map((participant) => participant.entityId),
  ]);

  if (input.speakerEntityId !== null) {
    allowedOwnerEntityIds.add(input.speakerEntityId);
  }

  const previousEntries = new Map<SharedStateEntryId, SharedStateEntry>(
    (input.previousArtifact?.entries ?? []).map((entry) => [entry.id, entry]),
  );
  const activeEntriesByStateKey = activePreviousEntriesByStateKey(input.previousArtifact);
  const operations: SharedStateOperation[] = [];
  const rejected: PatchRejection[] = [];
  const droppedCanonicalizeIds: DroppedCanonicalizeId[] = [];
  const nonLockedCanonicalizesDrops: NonLockedCanonicalizesDrop[] = [];
  const emptyUpdateDrops: EmptyUpdateDrop[] = [];
  let emptyUpdateAttemptedCount = 0;
  const baseRank = input.previousArtifact?.entries.length ?? 0;
  const maxLiveEntriesPerKey = normalizeMaxLiveEntriesPerKey(input.maxLiveEntriesPerKey);
  const initialActiveStateKeyCount = buildExistingStateKeyRegistry(input.previousArtifact).length;

  const relationshipLabelRejection = (
    operation: Extract<ParsedPatchOperation, { type: "add" | "update" | "supersede" }>,
    operationIndex: number,
    text: string | undefined,
    evidence: {
      relationship_evidence_relational_slot_ids?: readonly string[];
      relationship_evidence_stream_entry_ids?: readonly string[];
    },
  ): PatchRejection | null => {
    if (text === undefined) {
      return null;
    }

    const check = checkRelationshipLabelGrounding({
      text,
      participantRoster: input.participantRoster,
      relationshipEvidenceRelationalSlotIds:
        evidence.relationship_evidence_relational_slot_ids ?? [],
      relationshipEvidenceStreamEntryIds: evidence.relationship_evidence_stream_entry_ids ?? [],
      allowedRelationshipEvidenceStreamEntryIds: input.allowedSourceStreamEntryIds,
      relationshipEvidenceStreamEntryTrust: input.relationshipEvidenceStreamEntryTrust,
    });

    if (check.grounded) {
      return null;
    }

    return rejection(operation, operationIndex, "relationship_label_ungrounded", {
      ...(operation.type === "add" ? {} : { targetEntryId: operation.id }),
      protectedRelationshipLabels: check.protectedLabels,
      relationshipEvidenceRelationalSlotIds: [
        ...(evidence.relationship_evidence_relational_slot_ids ?? []),
      ],
      relationshipEvidenceStreamEntryIds: [
        ...(evidence.relationship_evidence_stream_entry_ids ?? []),
      ],
      rejectedRelationshipEvidenceRelationalSlotIds: check.rejectedRelationalSlotIds,
      rejectedRelationshipEvidenceStreamEntryIds: check.rejectedStreamEntryIds,
    });
  };

  const addTrackedEntry = (
    operationIndex: number,
    entry: Pick<SharedStateEntry, "kind" | "state_key"> & {
      id?: SharedStateEntryId;
      created_at?: number;
      last_updated_at?: number;
      rank?: number;
    },
  ): void => {
    const id = entry.id ?? `operation:${operationIndex}`;
    const createdAt = entry.created_at ?? 0;

    activeEntriesByStateKey.set(id, {
      id,
      kind: entry.kind,
      state_key: entry.state_key,
      created_at: createdAt,
      last_updated_at: entry.last_updated_at ?? createdAt,
      rank: entry.rank ?? baseRank + operations.length,
      fromPreviousArtifact: false,
    });
  };

  const validateAddStateKey = (
    operation: Extract<ParsedPatchOperation, { type: "add" }>,
    operationIndex: number,
  ): PatchRejection | null => {
    const activeStateKeys = [
      ...new Set(
        [...activeEntriesByStateKey.values()]
          .map((entry) => entry.state_key)
          .filter((stateKey): stateKey is string => stateKey !== null),
      ),
    ].sort((left, right) => left.localeCompare(right));
    const exactStateKeyExists = activeStateKeys.some(
      (stateKey) => stateKey === operation.state_key,
    );
    const lockedEntries = lockedEntriesForStateKey(activeEntriesByStateKey, operation.state_key);

    if ((operation.kind === "locked" || operation.kind === "live") && lockedEntries.length > 0) {
      const target = mostRecentPreviousEntry(lockedEntries);

      return rejection(operation, operationIndex, "locked_state_key_collision", {
        stateKey: operation.state_key,
        currentCount: lockedEntries.length,
        targetEntryId: target?.id,
        lockedEntryIds: lockedEntries
          .filter((entry) => entry.fromPreviousArtifact)
          .map((entry) => entry.id),
      });
    }

    if (!exactStateKeyExists) {
      const similarStateKeys = activeStateKeys.filter((stateKey) =>
        stateKeysAreNearDuplicate(operation.state_key, stateKey),
      );

      if (similarStateKeys.length > 0) {
        return rejection(operation, operationIndex, "near_duplicate_state_key", {
          stateKey: operation.state_key,
          similarStateKeys,
          sharedStateKeyTokens: sharedStateKeyTokens(
            operation.state_key,
            similarStateKeys[0] ?? "",
          ),
        });
      }

      if (initialActiveStateKeyCount > 0 && operation.new_key_reason === undefined) {
        return rejection(operation, operationIndex, "missing_new_key_reason", {
          stateKey: operation.state_key,
        });
      }
    }

    if (operation.kind !== "live") {
      return null;
    }

    const currentLiveEntries = liveEntriesForStateKey(activeEntriesByStateKey, operation.state_key);
    const proposedCount = currentLiveEntries.length + 1;

    if (proposedCount <= maxLiveEntriesPerKey) {
      return null;
    }

    const target = mostRecentPreviousEntry(currentLiveEntries);

    return rejection(operation, operationIndex, "live_entry_cap_exceeded_for_key", {
      stateKey: operation.state_key,
      currentCount: currentLiveEntries.length,
      proposedCount,
      maxLiveEntriesPerKey,
      targetEntryId: target?.id,
    });
  };

  input.patch.operations.forEach((operation, operationIndex) => {
    switch (operation.type) {
      case "add": {
        const stateKeyRejection = validateAddStateKey(operation, operationIndex);

        if (stateKeyRejection !== null) {
          rejected.push(stateKeyRejection);
          return;
        }

        const ownerEntityId = normalizeOwnerEntityId(
          operation.owner_entity_id,
          allowedOwnerEntityIds,
        );

        if (ownerEntityId === "invalid") {
          rejected.push(rejection(operation, operationIndex, "invalid_owner_entity_id"));
          return;
        }

        const citations = parseSourceStreamEntryIds(
          operation.source_stream_entry_ids,
          input.allowedSourceStreamEntryIds,
          input.sourceTrustValidator,
        );

        if (citations.reason !== null) {
          rejected.push(
            rejection(operation, operationIndex, citations.reason, {
              sourceStreamEntryId: citations.rejectedStreamEntryId,
              sourceTrustReason: citations.sourceTrustReason,
            }),
          );
          return;
        }

        const labelRejection = relationshipLabelRejection(
          operation,
          operationIndex,
          operation.text,
          operation,
        );

        if (labelRejection !== null) {
          rejected.push(labelRejection);
          return;
        }

        const canonicalizes = normalizeCanonicalizes({
          value: operation.canonicalizes,
          kind: operation.kind,
          allowedIds: input.allowedCanonicalizationIds,
          operation,
          operationIndex,
          dropped: droppedCanonicalizeIds,
          nonLockedDrops: nonLockedCanonicalizesDrops,
        });
        const rank = baseRank + operations.length;

        operations.push({
          type: "add",
          state_key: operation.state_key,
          kind: operation.kind,
          text: operation.text,
          owner_entity_id: ownerEntityId,
          provenance_stream_entry_ids: citations.streamEntryIds,
          last_updated_stream_entry_ids: citations.streamEntryIds,
          rank,
          ...(canonicalizes === undefined ? {} : { canonicalizes }),
        });
        addTrackedEntry(operationIndex, {
          state_key: operation.state_key,
          kind: operation.kind,
          rank,
        });
        return;
      }

      case "update": {
        const { id, entry } = previousEntryById(previousEntries, operation.id);

        if (id === null) {
          rejected.push(rejection(operation, operationIndex, "invalid_entry_id"));
          return;
        }

        if (entry === null) {
          rejected.push(rejection(operation, operationIndex, "unknown_entry_id"));
          return;
        }

        const nextKind = operation.kind ?? entry.kind;

        const ownerEntityId = normalizeUpdateOwnerEntityId(
          operation.owner_entity_id,
          entry,
          allowedOwnerEntityIds,
        );

        if (ownerEntityId === "invalid") {
          rejected.push(rejection(operation, operationIndex, "invalid_owner_entity_id"));
          return;
        }

        const operationDroppedCanonicalizeIds: DroppedCanonicalizeId[] = [];
        const operationNonLockedCanonicalizesDrops: NonLockedCanonicalizesDrop[] = [];
        const canonicalizes = normalizeCanonicalizes({
          value: operation.canonicalizes,
          kind: nextKind,
          allowedIds: input.allowedCanonicalizationIds,
          operation,
          operationIndex,
          dropped: operationDroppedCanonicalizeIds,
          nonLockedDrops: operationNonLockedCanonicalizesDrops,
        });
        emptyUpdateAttemptedCount += 1;

        if (
          isMaterialNoopUpdate({
            operation,
            entry,
            nextKind,
            ownerEntityId,
            canonicalizes,
          })
        ) {
          emptyUpdateDrops.push(emptyUpdateDrop({ operation, operationIndex, id, entry }));
          return;
        }

        const citations = parseSourceStreamEntryIds(
          operation.source_stream_entry_ids,
          input.allowedSourceStreamEntryIds,
          input.sourceTrustValidator,
        );

        if (citations.reason !== null) {
          rejected.push(
            rejection(operation, operationIndex, citations.reason, {
              sourceStreamEntryId: citations.rejectedStreamEntryId,
              sourceTrustReason: citations.sourceTrustReason,
            }),
          );
          return;
        }

        const labelRejection = relationshipLabelRejection(
          operation,
          operationIndex,
          operation.text,
          operation,
        );

        if (labelRejection !== null) {
          rejected.push(labelRejection);
          return;
        }

        droppedCanonicalizeIds.push(...operationDroppedCanonicalizeIds);
        nonLockedCanonicalizesDrops.push(...operationNonLockedCanonicalizesDrops);

        operations.push({
          type: "update",
          id,
          state_key: operation.state_key,
          kind: nextKind,
          text: operation.text,
          owner_entity_id: operation.owner_entity_id === undefined ? undefined : ownerEntityId,
          add_provenance_stream_entry_ids: citations.streamEntryIds,
          last_updated_stream_entry_ids: citations.streamEntryIds,
          ...(canonicalizes === undefined ? {} : { canonicalizes }),
        });
        const tracked = activeEntriesByStateKey.get(id);
        if (tracked !== undefined) {
          activeEntriesByStateKey.set(id, {
            ...tracked,
            kind: nextKind,
            state_key: operation.state_key,
          });
        }
        return;
      }

      case "supersede": {
        const { id, entry } = previousEntryById(previousEntries, operation.id);

        if (id === null) {
          rejected.push(rejection(operation, operationIndex, "invalid_entry_id"));
          return;
        }

        if (entry === null) {
          rejected.push(rejection(operation, operationIndex, "unknown_entry_id"));
          return;
        }

        const ownerEntityId = normalizeOwnerEntityId(
          operation.replacement.owner_entity_id,
          allowedOwnerEntityIds,
        );

        if (ownerEntityId === "invalid") {
          rejected.push(rejection(operation, operationIndex, "invalid_owner_entity_id"));
          return;
        }

        const replacementCitations = parseSourceStreamEntryIds(
          operation.replacement.source_stream_entry_ids,
          input.allowedSourceStreamEntryIds,
          input.sourceTrustValidator,
        );

        if (replacementCitations.reason !== null) {
          rejected.push(
            rejection(operation, operationIndex, replacementCitations.reason, {
              sourceStreamEntryId: replacementCitations.rejectedStreamEntryId,
              sourceTrustReason: replacementCitations.sourceTrustReason,
            }),
          );
          return;
        }

        const updateCitationValues =
          operation.source_stream_entry_ids ?? operation.replacement.source_stream_entry_ids;
        const updateCitations = parseSourceStreamEntryIds(
          updateCitationValues,
          input.allowedSourceStreamEntryIds,
          input.sourceTrustValidator,
        );

        if (updateCitations.reason !== null) {
          rejected.push(
            rejection(operation, operationIndex, updateCitations.reason, {
              sourceStreamEntryId: updateCitations.rejectedStreamEntryId,
              sourceTrustReason: updateCitations.sourceTrustReason,
            }),
          );
          return;
        }

        const labelRejection = relationshipLabelRejection(
          operation,
          operationIndex,
          operation.replacement.text,
          {
            relationship_evidence_relational_slot_ids: [
              ...(operation.relationship_evidence_relational_slot_ids ?? []),
              ...(operation.replacement.relationship_evidence_relational_slot_ids ?? []),
            ],
            relationship_evidence_stream_entry_ids: [
              ...(operation.relationship_evidence_stream_entry_ids ?? []),
              ...(operation.replacement.relationship_evidence_stream_entry_ids ?? []),
            ],
          },
        );

        if (labelRejection !== null) {
          rejected.push(labelRejection);
          return;
        }

        const canonicalizes = normalizeCanonicalizes({
          value: operation.canonicalizes,
          kind: operation.replacement.kind,
          allowedIds: input.allowedCanonicalizationIds,
          operation,
          operationIndex,
          dropped: droppedCanonicalizeIds,
          nonLockedDrops: nonLockedCanonicalizesDrops,
        });
        const rank = baseRank + operations.length;

        operations.push({
          type: "supersede",
          id,
          replacement: {
            state_key: operation.replacement.state_key,
            kind: operation.replacement.kind,
            text: operation.replacement.text,
            owner_entity_id: ownerEntityId,
            provenance_stream_entry_ids: replacementCitations.streamEntryIds,
            last_updated_stream_entry_ids: replacementCitations.streamEntryIds,
            rank,
            ...(canonicalizes === undefined ? {} : { canonicalizes }),
          },
          last_updated_stream_entry_ids: updateCitations.streamEntryIds,
        });
        activeEntriesByStateKey.delete(id);
        addTrackedEntry(operationIndex, {
          state_key: operation.replacement.state_key,
          kind: operation.replacement.kind,
          rank,
        });
        return;
      }

      case "prune": {
        const { id, entry } = previousEntryById(previousEntries, operation.id);

        if (id === null) {
          rejected.push(rejection(operation, operationIndex, "invalid_entry_id"));
          return;
        }

        if (entry === null) {
          rejected.push(rejection(operation, operationIndex, "unknown_entry_id"));
          return;
        }

        operations.push({
          type: "prune",
          id,
        });
        activeEntriesByStateKey.delete(id);
      }
    }
  });

  return {
    operations,
    rejected,
    droppedCanonicalizeIds,
    nonLockedCanonicalizesDrops,
    emptyUpdateDrops,
    emptyUpdateAttemptedCount,
  };
}

function dedupeCanonicalizeIds<TId extends string>(
  ids: readonly TId[],
  seen: Set<TId>,
): { kept: TId[]; dropped: TId[] } {
  const kept: TId[] = [];
  const dropped: TId[] = [];

  for (const id of ids) {
    if (seen.has(id)) {
      dropped.push(id);
      continue;
    }

    seen.add(id);
    kept.push(id);
  }

  return { kept, dropped };
}

export function dedupeCanonicalizesAcrossOperations(operations: readonly SharedStateOperation[]): {
  operations: SharedStateOperation[];
  duplicateDrops: CanonicalizationDuplicateDrop[];
} {
  const prunedEntryIds = new Set(
    operations
      .filter((operation): operation is Extract<SharedStateOperation, { type: "prune" }> => {
        return operation.type === "prune";
      })
      .map((operation) => operation.id),
  );
  const seenGoalIds = new Set<GoalId>();
  const seenCommitmentIds = new Set<CommitmentId>();
  const seenActionIds = new Set<ActionId>();
  const seenOpenQuestionIds = new Set<OpenQuestionId>();
  const duplicateDrops: CanonicalizationDuplicateDrop[] = [];

  const dedupe = (input: {
    id: SharedStateEntryId | undefined;
    kind: SharedStateEntryKind;
    canonicalizes: SharedStateCanonicalizes | undefined;
  }): SharedStateCanonicalizes | undefined => {
    if (
      input.canonicalizes === undefined ||
      input.id === undefined ||
      prunedEntryIds.has(input.id)
    ) {
      return input.canonicalizes;
    }

    const goals = dedupeCanonicalizeIds(input.canonicalizes.goal_ids, seenGoalIds);
    const commitments = dedupeCanonicalizeIds(
      input.canonicalizes.commitment_ids,
      seenCommitmentIds,
    );
    const actions = dedupeCanonicalizeIds(input.canonicalizes.action_ids, seenActionIds);
    const openQuestions = dedupeCanonicalizeIds(
      input.canonicalizes.open_question_ids,
      seenOpenQuestionIds,
    );
    const dropped: SharedStateCanonicalizes = {
      goal_ids: goals.dropped,
      commitment_ids: commitments.dropped,
      action_ids: actions.dropped,
      open_question_ids: openQuestions.dropped,
    };

    if (hasCanonicalizes(dropped)) {
      duplicateDrops.push({
        artifact_entry_id: input.id,
        kind: input.kind,
        dropped_ids: dropped,
      });
    }

    return {
      goal_ids: goals.kept,
      commitment_ids: commitments.kept,
      action_ids: actions.kept,
      open_question_ids: openQuestions.kept,
    };
  };

  return {
    duplicateDrops,
    operations: operations.map((operation) => {
      switch (operation.type) {
        case "add":
          return {
            ...operation,
            canonicalizes: dedupe({
              id: operation.id,
              kind: operation.kind,
              canonicalizes: operation.canonicalizes,
            }),
          };
        case "update":
          return {
            ...operation,
            canonicalizes: dedupe({
              id: operation.id,
              kind: operation.kind ?? "locked",
              canonicalizes: operation.canonicalizes,
            }),
          };
        case "supersede":
          return {
            ...operation,
            replacement: {
              ...operation.replacement,
              canonicalizes: dedupe({
                id: operation.replacement.id,
                kind: operation.replacement.kind,
                canonicalizes: operation.replacement.canonicalizes,
              }),
            },
          };
        case "prune":
        case "transition_kind":
          return operation;
      }
    }),
  };
}
