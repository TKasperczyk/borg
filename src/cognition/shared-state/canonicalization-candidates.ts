import {
  type SharedStateArtifact,
  type SharedStateCanonicalizes,
  type SharedStateEntry,
  type SharedStateEntryKind,
  type SharedStateOperation,
} from "../../memory/decision-artifacts/index.js";
import type { SharedStateEntryId } from "../../util/ids.js";
import {
  findUnsettledSharedStateReconciliation,
  type SharedStateReconciliationRepositories,
  type SharedStateUnsettledReconciliationSummary,
} from "./reconciliation.js";
import { emptyCanonicalizes, hasCanonicalizes } from "./patch-validation.js";
import type { CanonicalizeIdChannel } from "./schema.js";

function canonicalizedEntryIdsFromOperations(
  operations: readonly SharedStateOperation[],
): Set<SharedStateEntryId> {
  const ids = new Set<SharedStateEntryId>();

  for (const operation of operations) {
    switch (operation.type) {
      case "add":
        if (operation.id !== undefined && hasCanonicalizes(operation.canonicalizes)) {
          ids.add(operation.id);
        }
        break;
      case "update":
        if (hasCanonicalizes(operation.canonicalizes)) {
          ids.add(operation.id);
        }
        break;
      case "supersede":
        if (
          operation.replacement.id !== undefined &&
          hasCanonicalizes(operation.replacement.canonicalizes)
        ) {
          ids.add(operation.replacement.id);
        }
        break;
      case "prune":
        break;
    }
  }

  return ids;
}

function canonicalizedEntriesFromOperations(input: {
  artifact: SharedStateArtifact;
  operations: readonly SharedStateOperation[];
}): SharedStateEntry[] {
  const ids = canonicalizedEntryIdsFromOperations(input.operations);

  if (ids.size === 0) {
    return [];
  }

  return input.artifact.entries.filter((entry) => ids.has(entry.id));
}

export type SharedStateReconciliationWorkSet = {
  entries: SharedStateEntry[];
  currentOperationCanonicalizationCount: number;
  retriedStrandedCanonicalizationCount: number;
  retrySummary: SharedStateUnsettledReconciliationSummary | null;
};

function canonicalizationKey(input: {
  entryId: SharedStateEntryId;
  channel: CanonicalizeIdChannel;
  id: string;
}): string {
  return `${input.entryId}:${input.channel}:${input.id}`;
}

function emptyReconciliationEntry(entry: SharedStateEntry): SharedStateEntry {
  return {
    ...entry,
    canonicalizes: emptyCanonicalizes(),
  };
}

function appendReconciliationEntry(input: {
  entry: SharedStateEntry;
  entriesById: Map<SharedStateEntryId, SharedStateEntry>;
  seen: Set<string>;
}): number {
  const mergedEntry =
    input.entriesById.get(input.entry.id) ?? emptyReconciliationEntry(input.entry);
  let count = 0;

  if (!input.entriesById.has(input.entry.id)) {
    input.entriesById.set(input.entry.id, mergedEntry);
  }

  for (const goalId of input.entry.canonicalizes.goal_ids) {
    const key = canonicalizationKey({
      entryId: input.entry.id,
      channel: "goal",
      id: goalId,
    });

    if (!input.seen.has(key)) {
      input.seen.add(key);
      mergedEntry.canonicalizes.goal_ids.push(goalId);
      count += 1;
    }
  }

  for (const commitmentId of input.entry.canonicalizes.commitment_ids) {
    const key = canonicalizationKey({
      entryId: input.entry.id,
      channel: "commitment",
      id: commitmentId,
    });

    if (!input.seen.has(key)) {
      input.seen.add(key);
      mergedEntry.canonicalizes.commitment_ids.push(commitmentId);
      count += 1;
    }
  }

  for (const actionId of input.entry.canonicalizes.action_ids) {
    const key = canonicalizationKey({
      entryId: input.entry.id,
      channel: "action",
      id: actionId,
    });

    if (!input.seen.has(key)) {
      input.seen.add(key);
      mergedEntry.canonicalizes.action_ids.push(actionId);
      count += 1;
    }
  }

  for (const openQuestionId of input.entry.canonicalizes.open_question_ids) {
    const key = canonicalizationKey({
      entryId: input.entry.id,
      channel: "open_question",
      id: openQuestionId,
    });

    if (!input.seen.has(key)) {
      input.seen.add(key);
      mergedEntry.canonicalizes.open_question_ids.push(openQuestionId);
      count += 1;
    }
  }

  return count;
}

export function buildSharedStateReconciliationWorkSet(input: {
  artifact: SharedStateArtifact | null;
  operations: readonly SharedStateOperation[];
  repositories?: SharedStateReconciliationRepositories;
  nowMs: number;
}): SharedStateReconciliationWorkSet {
  if (input.artifact === null) {
    return {
      entries: [],
      currentOperationCanonicalizationCount: 0,
      retriedStrandedCanonicalizationCount: 0,
      retrySummary: null,
    };
  }

  const entriesById = new Map<SharedStateEntryId, SharedStateEntry>();
  const seen = new Set<string>();
  let currentOperationCanonicalizationCount = 0;
  let retriedStrandedCanonicalizationCount = 0;

  for (const entry of canonicalizedEntriesFromOperations({
    artifact: input.artifact,
    operations: input.operations,
  })) {
    currentOperationCanonicalizationCount += appendReconciliationEntry({
      entry,
      entriesById,
      seen,
    });
  }

  const retry = findUnsettledSharedStateReconciliation({
    previousArtifact: input.artifact,
    repositories: input.repositories,
    nowMs: input.nowMs,
  });

  for (const entry of retry?.entries ?? []) {
    retriedStrandedCanonicalizationCount += appendReconciliationEntry({
      entry,
      entriesById,
      seen,
    });
  }

  return {
    entries: [...entriesById.values()].filter((entry) => hasCanonicalizes(entry.canonicalizes)),
    currentOperationCanonicalizationCount,
    retriedStrandedCanonicalizationCount,
    retrySummary: retry?.summary ?? null,
  };
}
