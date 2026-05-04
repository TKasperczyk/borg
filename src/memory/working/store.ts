import { existsSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";

import { readJsonFile, writeJsonFileAtomic } from "../../util/atomic-write.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { WorkingMemoryError } from "../../util/errors.js";
import type { SessionId } from "../../util/ids.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import { cosineSimilarity } from "../../retrieval/embedding-similarity.js";

import {
  createWorkingMemory,
  type PendingActionRecord,
  type WorkingMemory,
  workingMemorySchema,
} from "./types.js";

export type WorkingMemoryStoreOptions = {
  dataDir?: string;
  clock?: Clock;
};

export type RelationalSlotPendingActionSanitization = {
  sessionId: SessionId;
  values: readonly string[];
  neutralPhrase: string;
};

const PENDING_ACTIONS_LIMIT = 16;
const HOT_ENTITIES_LIMIT = 32;
export const PENDING_ACTION_SEMANTIC_MERGE_THRESHOLD = 0.85;

function cloneWorkingMemory(state: WorkingMemory): WorkingMemory {
  return structuredClone(state) as WorkingMemory;
}

function normalizePendingActions(
  actions: WorkingMemory["pending_actions"],
): WorkingMemory["pending_actions"] {
  const deduped: WorkingMemory["pending_actions"] = [];
  const seenNextActions = new Set<string>();

  for (let index = actions.length - 1; index >= 0; index -= 1) {
    const action = actions[index];

    if (action === undefined) {
      continue;
    }

    const nextAction = action.next_action?.trim().toLowerCase();

    if (nextAction !== undefined && nextAction.length > 0) {
      if (seenNextActions.has(nextAction)) {
        continue;
      }

      seenNextActions.add(nextAction);
    }

    deduped.push(action);

    if (deduped.length >= PENDING_ACTIONS_LIMIT) {
      break;
    }
  }

  return deduped.reverse();
}

function pendingActionEmbeddingText(action: PendingActionRecord): string {
  return [action.description, action.next_action ?? ""].join("\n").trim();
}

function withPendingActionTimestamp(
  action: PendingActionRecord,
  nowMs: number,
): PendingActionRecord {
  return action.created_at === undefined
    ? {
        ...action,
        created_at: nowMs,
      }
    : action;
}

export async function mergePendingActionsBySimilarity(input: {
  existing: WorkingMemory["pending_actions"];
  incoming: readonly PendingActionRecord[];
  embeddingClient?: EmbeddingClient;
  nowMs: number;
  threshold?: number;
}): Promise<WorkingMemory["pending_actions"]> {
  const threshold = input.threshold ?? PENDING_ACTION_SEMANTIC_MERGE_THRESHOLD;
  let merged = normalizePendingActions([...input.existing]);

  for (const action of input.incoming) {
    const incoming =
      input.embeddingClient === undefined
        ? action
        : withPendingActionTimestamp(action, input.nowMs);

    if (input.embeddingClient === undefined || merged.length === 0) {
      merged = normalizePendingActions([...merged, incoming]);
      continue;
    }

    const vectors = await input.embeddingClient.embedBatch([
      pendingActionEmbeddingText(incoming),
      ...merged.map((candidate) => pendingActionEmbeddingText(candidate)),
    ]);
    const incomingVector = vectors[0];

    if (incomingVector === undefined) {
      merged = normalizePendingActions([...merged, incoming]);
      continue;
    }

    let bestMatchIndex = -1;
    let bestSimilarity = Number.NEGATIVE_INFINITY;

    for (let index = 0; index < merged.length; index += 1) {
      const candidateVector = vectors[index + 1];

      if (candidateVector === undefined) {
        continue;
      }

      const similarity = cosineSimilarity(incomingVector, candidateVector);

      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestMatchIndex = index;
      }
    }

    if (bestMatchIndex >= 0 && bestSimilarity >= threshold) {
      merged = merged.map((candidate, index) =>
        index === bestMatchIndex
          ? {
              ...candidate,
              created_at: Math.max(candidate.created_at ?? 0, incoming.created_at ?? input.nowMs),
            }
          : candidate,
      );
      continue;
    }

    merged = normalizePendingActions([...merged, incoming]);
  }

  return normalizePendingActions(merged);
}

function normalizeHotEntities(hotEntities: readonly string[]): string[] {
  const normalized: string[] = [];
  const seen = new Set<string>();

  for (const entity of hotEntities) {
    const trimmed = entity.trim();

    if (trimmed.length === 0) {
      continue;
    }

    const key = trimmed.toLowerCase();

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    normalized.push(trimmed);

    if (normalized.length >= HOT_ENTITIES_LIMIT) {
      break;
    }
  }

  return normalized;
}

function normalizeWorkingMemory(state: WorkingMemory): WorkingMemory {
  return {
    ...state,
    hot_entities: normalizeHotEntities(state.hot_entities),
    pending_actions: normalizePendingActions(state.pending_actions),
  };
}

function uniqueReplacementValues(values: readonly string[], neutralPhrase: string): string[] {
  const unique: string[] = [];
  const replacement = neutralPhrase.trim();

  for (const value of values) {
    const trimmed = value.trim();

    if (trimmed.length === 0 || trimmed === replacement) {
      continue;
    }

    if (unique.some((existing) => existing === trimmed)) {
      continue;
    }

    unique.push(trimmed);
  }

  return unique.sort((left, right) => right.length - left.length || left.localeCompare(right));
}

function replaceKnownValues(text: string, values: readonly string[], replacement: string): string {
  let next = text;

  for (const value of values) {
    next = next.split(value).join(replacement);
  }

  return next;
}

export class WorkingMemoryStore {
  private readonly clock: Clock;

  constructor(private readonly options: WorkingMemoryStoreOptions = {}) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get workingDirectory(): string | undefined {
    return this.options.dataDir === undefined ? undefined : join(this.options.dataDir, "working");
  }

  private getPath(sessionId: SessionId): string {
    const workingDirectory = this.workingDirectory;

    if (workingDirectory === undefined) {
      throw new WorkingMemoryError("Working memory persistence is not configured", {
        code: "WORKING_MEMORY_PERSISTENCE_DISABLED",
      });
    }

    return join(workingDirectory, `${sessionId}.json`);
  }

  load(sessionId: SessionId): WorkingMemory {
    const persisted = this.readPersisted(sessionId);
    const state = persisted ?? createWorkingMemory(sessionId, this.clock.now());
    return cloneWorkingMemory(state);
  }

  save(state: WorkingMemory): WorkingMemory {
    const parsed = workingMemorySchema.safeParse(normalizeWorkingMemory(state));

    if (!parsed.success) {
      throw new WorkingMemoryError("Invalid working memory state", {
        cause: parsed.error,
        code: "WORKING_MEMORY_INVALID",
      });
    }

    const next = cloneWorkingMemory(parsed.data);
    this.writePersisted(next);
    return cloneWorkingMemory(next);
  }

  sanitizePendingActionsForRelationalSlot(
    input: RelationalSlotPendingActionSanitization,
  ): WorkingMemory {
    const replacement = input.neutralPhrase.trim();
    const values = uniqueReplacementValues(input.values, replacement);
    const current = this.load(input.sessionId);

    if (replacement.length === 0 || values.length === 0 || current.pending_actions.length === 0) {
      return current;
    }

    let changed = false;
    const pendingActions = current.pending_actions.map((action) => {
      const description = replaceKnownValues(action.description, values, replacement);
      const nextAction =
        action.next_action === null
          ? null
          : replaceKnownValues(action.next_action, values, replacement);

      if (description !== action.description || nextAction !== action.next_action) {
        changed = true;
      }

      return {
        ...action,
        description,
        next_action: nextAction,
      };
    });

    if (!changed) {
      return current;
    }

    return this.save({
      ...current,
      pending_actions: pendingActions,
      updated_at: this.clock.now(),
    });
  }

  async addPendingAction(input: {
    sessionId: SessionId;
    action: PendingActionRecord;
    embeddingClient?: EmbeddingClient;
    similarityThreshold?: number;
  }): Promise<WorkingMemory> {
    const nowMs = this.clock.now();
    const current = this.load(input.sessionId);
    const pendingActions = await mergePendingActionsBySimilarity({
      existing: current.pending_actions,
      incoming: [input.action],
      embeddingClient: input.embeddingClient,
      threshold: input.similarityThreshold,
      nowMs,
    });

    return this.save({
      ...current,
      pending_actions: pendingActions,
      updated_at: nowMs,
    });
  }

  clear(sessionId: SessionId): void {
    const workingDirectory = this.workingDirectory;

    if (workingDirectory === undefined) {
      return;
    }

    const filePath = join(workingDirectory, `${sessionId}.json`);

    if (!existsSync(filePath)) {
      return;
    }

    try {
      rmSync(filePath, { force: true });
    } catch (error) {
      throw new WorkingMemoryError(`Failed to clear working memory for ${sessionId}`, {
        cause: error,
        code: "WORKING_MEMORY_CLEAR_FAILED",
      });
    }
  }

  private readPersisted(sessionId: SessionId): WorkingMemory | undefined {
    if (this.workingDirectory === undefined) {
      return undefined;
    }

    try {
      const raw = readJsonFile<unknown>(this.getPath(sessionId));

      if (raw === undefined) {
        return undefined;
      }

      const parsed = workingMemorySchema.safeParse(raw);

      if (!parsed.success) {
        throw new WorkingMemoryError(`Invalid working memory file for ${sessionId}`, {
          cause: parsed.error,
          code: "WORKING_MEMORY_INVALID",
        });
      }

      return parsed.data;
    } catch (error) {
      if (error instanceof WorkingMemoryError) {
        throw error;
      }

      throw new WorkingMemoryError(`Failed to load working memory for ${sessionId}`, {
        cause: error,
        code: "WORKING_MEMORY_LOAD_FAILED",
      });
    }
  }

  private writePersisted(state: WorkingMemory): void {
    if (this.workingDirectory === undefined) {
      return;
    }

    try {
      mkdirSync(this.workingDirectory, { recursive: true });
      writeJsonFileAtomic(this.getPath(state.session_id), state);
    } catch (error) {
      throw new WorkingMemoryError(`Failed to save working memory for ${state.session_id}`, {
        cause: error,
        code: "WORKING_MEMORY_SAVE_FAILED",
      });
    }
  }
}
