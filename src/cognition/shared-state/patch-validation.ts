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
import type {
  AllowedCanonicalizationIds,
  CanonicalizationDuplicateDrop,
  CanonicalizeIdChannel,
  DroppedCanonicalizeId,
  EmitSharedStatePatch,
  NonLockedCanonicalizesDrop,
  ParsedCanonicalizes,
  ParsedPatchOperation,
  PatchRejection,
  SharedStateArtifactParticipantContext,
  SharedStateCanonicalizationCandidates,
} from "./schema.js";
import { parseSourceStreamEntryIds } from "./source-trust.js";

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

function rejection(
  operation: ParsedPatchOperation,
  operationIndex: number,
  reason: PatchRejection["reason"],
  details: Pick<PatchRejection, "sourceStreamEntryId" | "sourceTrustReason"> = {},
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
  allowedCanonicalizationIds: AllowedCanonicalizationIds;
}): {
  operations: SharedStateOperation[];
  rejected: PatchRejection[];
  droppedCanonicalizeIds: DroppedCanonicalizeId[];
  nonLockedCanonicalizesDrops: NonLockedCanonicalizesDrop[];
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
  const operations: SharedStateOperation[] = [];
  const rejected: PatchRejection[] = [];
  const droppedCanonicalizeIds: DroppedCanonicalizeId[] = [];
  const nonLockedCanonicalizesDrops: NonLockedCanonicalizesDrop[] = [];
  const baseRank = input.previousArtifact?.entries.length ?? 0;

  input.patch.operations.forEach((operation, operationIndex) => {
    switch (operation.type) {
      case "add": {
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

        const canonicalizes = normalizeCanonicalizes({
          value: operation.canonicalizes,
          kind: operation.kind,
          allowedIds: input.allowedCanonicalizationIds,
          operation,
          operationIndex,
          dropped: droppedCanonicalizeIds,
          nonLockedDrops: nonLockedCanonicalizesDrops,
        });

        operations.push({
          type: "add",
          kind: operation.kind,
          text: operation.text,
          owner_entity_id: ownerEntityId,
          provenance_stream_entry_ids: citations.streamEntryIds,
          last_updated_stream_entry_ids: citations.streamEntryIds,
          rank: baseRank + operations.length,
          ...(canonicalizes === undefined ? {} : { canonicalizes }),
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

        if (
          operation.kind === undefined &&
          operation.text === undefined &&
          operation.owner_entity_id === undefined &&
          operation.canonicalizes === undefined
        ) {
          rejected.push(rejection(operation, operationIndex, "empty_update"));
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

        const canonicalizes = normalizeCanonicalizes({
          value: operation.canonicalizes,
          kind: nextKind,
          allowedIds: input.allowedCanonicalizationIds,
          operation,
          operationIndex,
          dropped: droppedCanonicalizeIds,
          nonLockedDrops: nonLockedCanonicalizesDrops,
        });

        operations.push({
          type: "update",
          id,
          kind: nextKind,
          text: operation.text,
          owner_entity_id: operation.owner_entity_id === undefined ? undefined : ownerEntityId,
          add_provenance_stream_entry_ids: citations.streamEntryIds,
          last_updated_stream_entry_ids: citations.streamEntryIds,
          ...(canonicalizes === undefined ? {} : { canonicalizes }),
        });
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

        const canonicalizes = normalizeCanonicalizes({
          value: operation.canonicalizes,
          kind: operation.replacement.kind,
          allowedIds: input.allowedCanonicalizationIds,
          operation,
          operationIndex,
          dropped: droppedCanonicalizeIds,
          nonLockedDrops: nonLockedCanonicalizesDrops,
        });

        operations.push({
          type: "supersede",
          id,
          replacement: {
            kind: operation.replacement.kind,
            text: operation.replacement.text,
            owner_entity_id: ownerEntityId,
            provenance_stream_entry_ids: replacementCitations.streamEntryIds,
            last_updated_stream_entry_ids: replacementCitations.streamEntryIds,
            rank: baseRank + operations.length,
            ...(canonicalizes === undefined ? {} : { canonicalizes }),
          },
          last_updated_stream_entry_ids: updateCitations.streamEntryIds,
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
      }
    }
  });

  return { operations, rejected, droppedCanonicalizeIds, nonLockedCanonicalizesDrops };
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
          return operation;
      }
    }),
  };
}
