import type { EntityRepository } from "../commitments/repository.js";
import { similarityThresholds, type SimilarityConfigSource } from "../../config/similarity.js";
import { existsSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";

import { readJsonFile, writeJsonFileAtomic } from "../../util/atomic-write.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { WorkingMemoryError } from "../../util/errors.js";
import {
  isSessionId,
  actionIdHelpers,
  pendingActionIdHelpers,
  type ActionId,
  type EntityId,
  type SessionId,
} from "../../util/ids.js";
import { memoryDisclosurePayloadFields } from "../common/disclosure-serializers.js";
import { selfPrivateMemoryDisclosureLabel } from "../common/disclosure-label.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import { cosineSimilarity } from "../../retrieval/embedding-similarity.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import type { ActionRepository } from "../actions/repository.js";

import {
  createWorkingMemory,
  type PendingActionRecord,
  type WorkingMemory,
  workingMemorySchema,
} from "./types.js";

export type WorkingMemoryStoreOptions = {
  entityRepository?: Pick<EntityRepository, "resolveReferences">;
  similarityConfig?: SimilarityConfigSource;
  dataDir?: string;
  clock?: Clock;
  audienceForSession?: (sessionId: SessionId) => EntityId | null;
  tracer?: Pick<TurnTracer, "emit">;
  actionRepository?: Pick<ActionRepository, "get"> & Partial<Pick<ActionRepository, "delete">>;
};

export type PendingActionSessionLock = {
  tryAcquire(sessionId: SessionId): Promise<{ release(): Promise<void> } | null>;
};

export type PendingActionProjection = PendingActionRecord & {
  pending_action_id: string;
  session_id: SessionId;
  audience_entity_id: EntityId | null;
  completion_status?: "completion_pending";
} & ReturnType<typeof memoryDisclosurePayloadFields>;

export type PendingActionProjectionResult = {
  actions: readonly PendingActionProjection[];
  degraded: readonly SessionId[];
  completed_pending_cleanup: readonly (Pick<
    PendingActionProjection,
    "session_id" | "pending_action_id" | "audience_entity_id" | "disclosure" | "disclosure_label"
  > & { action_id: ActionId })[];
};

export function pendingActionResolutionId(update: {
  session_id: SessionId;
  pending_action_id: string;
}): ActionId {
  // Retries must find the same durable action. Scope the identity to both
  // handles, never description text.
  return actionIdHelpers.parse(
    `act_${createHash("sha256")
      .update(JSON.stringify([update.session_id, update.pending_action_id]))
      .digest("hex")
      .slice(0, 16)}`,
  );
}

export type RelationalSlotPendingActionSanitization = {
  sessionId: SessionId;
  values: readonly string[];
  neutralPhrase: string;
};

const PENDING_ACTIONS_LIMIT = 16;
const HOT_ENTITIES_LIMIT = 32;

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

    deduped.push({
      ...action,
      pending_action_id: action.pending_action_id ?? pendingActionIdHelpers.create(),
    });

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
  onSemanticMerge?: () => void;
}): Promise<WorkingMemory["pending_actions"]> {
  const threshold =
    input.threshold ??
    similarityThresholds({ embedding: input.embeddingClient?.profile }).pendingActionMerge;
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
      input.onSemanticMerge?.();
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
  private pendingActionMergeCount = 0;

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
    return cloneWorkingMemory(this.options.entityRepository?.resolveReferences(state) ?? state);
  }

  async migratePendingActionIds(
    sessionId: SessionId,
    sessionLock: PendingActionSessionLock,
    input: { lockedSessionId?: SessionId; turnId?: string } = {},
  ): Promise<boolean> {
    const own = input.lockedSessionId === sessionId;
    const lease = own ? null : await sessionLock.tryAcquire(sessionId);
    if (!own && lease === null) return false;
    try {
      // Re-read only after acquiring the origin lock; ordinary reads never write.
      // Holding that lock proves an earlier marker's turn is no longer in flight.
      // A caller already holding it supplies its current turn id; never recover
      // that turn's provisional actions while its rollback is still running.
      let current = this.readPersisted(sessionId);
      const orphanedTurns = new Set(
        (current?.pending_actions ?? []).flatMap((action) =>
          action.completing_action?.phase === "provisional" &&
          action.completing_action.turn_id !== input.turnId
            ? [action.completing_action.turn_id]
            : [],
        ),
      );
      for (const turnId of orphanedTurns) {
        current = this.save(current!, { pendingActionTurn: { turnId, outcome: "rollback" } });
      }
      for (const action of current?.pending_actions ?? []) {
        const restored = action.restored_from_action;
        if (restored === undefined || restored.turn_id === input.turnId) continue;
        const repository = this.options.actionRepository;
        if (repository?.get(restored.action_id) == null) continue;
        if (repository.delete === undefined) {
          throw new WorkingMemoryError("Provisional action recovery requires action deletion", {
            code: "WORKING_MEMORY_ROLLBACK_RECOVERY_FAILED",
          });
        }
        await repository.delete(restored.action_id);
        this.options.tracer?.emit("working_memory.pending_action_resolved", {
          turnId: input.turnId ?? "working_memory",
          session_id: sessionId,
          pending_action_id: action.pending_action_id,
          action_id: restored.action_id,
          rollback_turn_id: restored.turn_id,
          step: "rollback_action",
          outcome: "deleted",
        });
      }
      this.readPersisted(sessionId, true);
      return true;
    } catch (error) {
      this.options.tracer?.emit("working_memory.degraded", {
        turnId: input.turnId ?? "working_memory",
        session_id: sessionId,
        operation: "pending_action_rollback_recovery",
        reason:
          error instanceof WorkingMemoryError
            ? error.code
            : "WORKING_MEMORY_ROLLBACK_RECOVERY_FAILED",
      });
      throw error;
    } finally {
      await lease?.release();
    }
  }

  async projectPendingActions(input: {
    current: WorkingMemory;
    turnId?: string;
    sessionLock: PendingActionSessionLock;
    currentAudienceEntityId?: EntityId | null;
    includeOtherSessions: boolean;
  }): Promise<PendingActionProjectionResult> {
    const sessionIds = new Set<SessionId>([input.current.session_id]);
    if (
      input.includeOtherSessions &&
      this.workingDirectory !== undefined &&
      existsSync(this.workingDirectory)
    ) {
      for (const name of readdirSync(this.workingDirectory)) {
        if (!name.endsWith(".json")) continue;
        const id = name.slice(0, -5);
        if (isSessionId(id)) sessionIds.add(id);
      }
    }
    const projection: PendingActionProjection[] = [];
    const degraded: SessionId[] = [];
    const completedPendingCleanup: Array<
      PendingActionProjectionResult["completed_pending_cleanup"][number]
    > = [];
    for (const sessionId of [...sessionIds].sort()) {
      const own = sessionId === input.current.session_id;
      try {
        // Atomic file reads remain global even while a session is busy. Projection
        // never migrates on disk or changes another session's mutable state.
        const state = own ? input.current : this.load(sessionId);
        const audience = own
          ? (input.currentAudienceEntityId ?? this.options.audienceForSession?.(sessionId) ?? null)
          : (this.options.audienceForSession?.(sessionId) ?? null);
        for (const action of state.pending_actions) {
          if (action.pending_action_id === undefined) continue;
          const actionId = pendingActionResolutionId({
            session_id: sessionId,
            pending_action_id: action.pending_action_id,
          });
          if (
            action.restored_from_action === undefined &&
            action.completing_action?.phase !== "provisional" &&
            this.options.actionRepository?.get(actionId) != null
          ) {
            completedPendingCleanup.push({
              session_id: sessionId,
              pending_action_id: action.pending_action_id,
              action_id: actionId,
              audience_entity_id: audience,
              ...memoryDisclosurePayloadFields(
                selfPrivateMemoryDisclosureLabel(audience === null ? [] : [audience]),
              ),
            });
            continue;
          }
          projection.push({
            ...action,
            ...(action.completing_action?.phase === "provisional"
              ? { completion_status: "completion_pending" as const }
              : {}),
            created_at: action.created_at ?? state.updated_at,
            pending_action_id: action.pending_action_id,
            session_id: sessionId,
            audience_entity_id: audience,
            ...memoryDisclosurePayloadFields(
              selfPrivateMemoryDisclosureLabel(audience === null ? [] : [audience]),
            ),
          });
        }
      } catch (error) {
        degraded.push(sessionId);
        this.options.tracer?.emit("working_memory.degraded", {
          turnId: input.turnId ?? "working_memory",
          session_id: sessionId,
          operation: "pending_action_projection",
          reason: error instanceof WorkingMemoryError ? error.code : "WORKING_MEMORY_LOAD_FAILED",
        });
      }
    }
    return { actions: projection, degraded, completed_pending_cleanup: completedPendingCleanup };
  }

  async resolvePendingAction(input: {
    sessionId: SessionId;
    turnId?: string;
    pendingActionId: string;
    lockedSessionId: SessionId;
    sessionLock: PendingActionSessionLock;
    apply: (action: PendingActionRecord) => Promise<boolean>;
    persist?: boolean;
  }): Promise<"applied" | "applied_cleanup_pending" | "missing" | "busy" | "failed"> {
    const own = input.sessionId === input.lockedSessionId;
    const lease = own ? null : await input.sessionLock.tryAcquire(input.sessionId);
    if (!own && lease === null) return "busy";
    try {
      await this.migratePendingActionIds(input.sessionId, input.sessionLock, {
        lockedSessionId: input.sessionId,
        ...(input.turnId === undefined ? {} : { turnId: input.turnId }),
      });
      const current = this.load(input.sessionId);
      const action = current.pending_actions.find(
        (candidate) => candidate.pending_action_id === input.pendingActionId,
      );
      const actionId = pendingActionResolutionId({
        session_id: input.sessionId,
        pending_action_id: input.pendingActionId,
      });
      // A marker belonging to this still-running turn is not a completion.
      if (action?.restored_from_action !== undefined) return "busy";
      const alreadyCommitted = this.options.actionRepository?.get(actionId) != null;
      if (!alreadyCommitted && action === undefined) return "missing";
      if (own && input.persist === false && !alreadyCommitted && action !== undefined) {
        if (input.turnId === undefined) {
          throw new WorkingMemoryError("A provisional pending completion requires a turn id", {
            code: "WORKING_MEMORY_INVALID",
          });
        }
        // Marker first: no provisional action can exist without its durable
        // intention and recovery handle. A marker-only crash is safe to retry.
        this.save({
          ...current,
          pending_actions: current.pending_actions.map((candidate) =>
            candidate.pending_action_id === input.pendingActionId
              ? {
                  ...candidate,
                  completing_action: {
                    action_id: actionId,
                    turn_id: input.turnId!,
                    phase: "provisional",
                  },
                }
              : candidate,
          ),
        });
      }
      let committed = alreadyCommitted;
      if (!committed && action !== undefined) {
        try {
          committed = await input.apply(action);
        } catch (error) {
          this.options.tracer?.emit("working_memory.degraded", {
            turnId: input.turnId ?? "working_memory",
            session_id: input.sessionId,
            pending_action_id: input.pendingActionId,
            operation: "pending_action_resolution_apply",
            reason: "ACTION_WRITE_FAILED",
          });
        }
        // A repository callback can fail after its SQL insert committed.
        committed ||= this.options.actionRepository?.get(actionId) != null;
      }
      if (!committed) return "failed";
      const provisional =
        own &&
        input.persist === false &&
        (!alreadyCommitted || action?.completing_action?.phase === "provisional");
      this.options.tracer?.emit("working_memory.pending_action_resolved", {
        turnId: input.turnId ?? "working_memory",
        session_id: input.sessionId,
        pending_action_id: input.pendingActionId,
        action_id: actionId,
        step: "action",
        outcome: provisional
          ? alreadyCommitted
            ? "already_provisional"
            : "provisional"
          : alreadyCommitted
            ? "already_committed"
            : "committed",
      });
      if (input.persist === false) return "applied";
      try {
        // Action first: a crash leaves a durable completion and a pending row
        // that the read-only projection excludes and the next locked write cleans.
        if (action !== undefined) {
          this.save({
            ...current,
            pending_actions: current.pending_actions.filter(
              (candidate) => candidate.pending_action_id !== input.pendingActionId,
            ),
            updated_at: this.clock.now(),
          });
        }
      } catch (error) {
        this.options.tracer?.emit("working_memory.degraded", {
          turnId: input.turnId ?? "working_memory",
          session_id: input.sessionId,
          pending_action_id: input.pendingActionId,
          operation: "pending_action_resolution_save",
          reason: error instanceof WorkingMemoryError ? error.code : "WORKING_MEMORY_SAVE_FAILED",
        });
        return "applied_cleanup_pending";
      }
      this.options.tracer?.emit("working_memory.pending_action_resolved", {
        turnId: input.turnId ?? "working_memory",
        session_id: input.sessionId,
        pending_action_id: input.pendingActionId,
        action_id: actionId,
        step: "cleanup",
        outcome: "saved",
      });
      return "applied";
    } finally {
      await lease?.release();
    }
  }

  save(
    state: WorkingMemory,
    input: { pendingActionTurn?: { turnId: string; outcome: "commit" | "rollback" } } = {},
  ): WorkingMemory {
    const normalized = normalizeWorkingMemory(state);
    // A stale turn snapshot or lane cap must not remove recovery information.
    // Re-read under the caller's origin lock; persisted markers take precedence.
    let persisted: WorkingMemory | undefined;
    try {
      persisted = this.readPersisted(state.session_id);
    } catch (error) {
      if (
        !(error instanceof WorkingMemoryError) ||
        (error.code !== "WORKING_MEMORY_INVALID" &&
          error.code !== "WORKING_MEMORY_SESSION_MISMATCH" &&
          !(error.code === "WORKING_MEMORY_LOAD_FAILED" && error.cause instanceof SyntaxError))
      ) {
        throw error;
      }
      // Only an own-session save may overwrite malformed persisted content.
      // Preserve incoming markers below: orphan recovery returns their provisional
      // completions to pending once the owning turn is no longer in flight. Invalid
      // disk content supplies no trusted markers; foreign/recovery reads stay strict.
      this.options.tracer?.emit("working_memory.degraded", {
        turnId: input.pendingActionTurn?.turnId ?? "working_memory",
        session_id: state.session_id,
        operation: "save_persisted_read",
        reason: error.code,
      });
    }
    for (const action of [...state.pending_actions, ...(persisted?.pending_actions ?? [])]) {
      if (action.completing_action === undefined && action.restored_from_action === undefined)
        continue;
      const index = normalized.pending_actions.findIndex(
        (candidate) => candidate.pending_action_id === action.pending_action_id,
      );
      const protectedAction = { ...(normalized.pending_actions[index] ?? action) };
      delete protectedAction.completing_action;
      delete protectedAction.restored_from_action;
      if (action.completing_action !== undefined)
        protectedAction.completing_action = { ...action.completing_action };
      if (action.restored_from_action !== undefined)
        protectedAction.restored_from_action = { ...action.restored_from_action };
      if (index < 0) normalized.pending_actions.push(protectedAction);
      else normalized.pending_actions[index] = protectedAction;
    }
    for (const action of normalized.pending_actions) {
      const marker = action.completing_action;
      if (
        marker === undefined ||
        marker.phase === "committed" ||
        marker.turn_id !== input.pendingActionTurn?.turnId
      )
        continue;
      if (input.pendingActionTurn.outcome === "commit") {
        marker.phase = "committed";
      } else {
        action.restored_from_action = { action_id: marker.action_id, turn_id: marker.turn_id };
        delete action.completing_action;
      }
    }
    const parsed = workingMemorySchema.safeParse(normalized);

    if (!parsed.success) {
      throw new WorkingMemoryError("Invalid working memory state", {
        cause: parsed.error,
        code: "WORKING_MEMORY_INVALID",
      });
    }

    const next = cloneWorkingMemory(parsed.data);
    next.pending_actions = next.pending_actions.map((action, index) => ({
      ...normalized.pending_actions[index],
      ...action,
    }));
    this.writePersisted(
      next,
      next.session_id,
      input.pendingActionTurn?.outcome === "commit" ? input.pendingActionTurn.turnId : undefined,
    );
    return cloneWorkingMemory(next);
  }

  recordPendingActionMerges(count: number): void {
    if (!Number.isFinite(count) || count <= 0) {
      return;
    }

    this.pendingActionMergeCount += Math.floor(count);
  }

  getPendingActionMergeCount(): number {
    return this.pendingActionMergeCount;
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
    let semanticMergeCount = 0;
    const pendingActions = await mergePendingActionsBySimilarity({
      existing: current.pending_actions,
      incoming: [input.action],
      embeddingClient: input.embeddingClient,
      threshold:
        input.similarityThreshold ??
        similarityThresholds(
          this.options.similarityConfig ?? { embedding: input.embeddingClient?.profile },
        ).pendingActionMerge,
      nowMs,
      onSemanticMerge: () => {
        semanticMergeCount += 1;
      },
    });
    this.recordPendingActionMerges(semanticMergeCount);

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

  private readPersisted(sessionId: SessionId, migrate = false): WorkingMemory | undefined {
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

      if (parsed.data.session_id !== sessionId) {
        throw new WorkingMemoryError(
          `Working memory session identity does not match ${sessionId}`,
          {
            code: "WORKING_MEMORY_SESSION_MISMATCH",
          },
        );
      }

      const original = raw as WorkingMemory;
      const needsMigration = parsed.data.pending_actions.some(
        (action) => action.pending_action_id === undefined,
      );
      if (needsMigration) {
        for (const [index, action] of original.pending_actions.entries()) {
          // Hashing exact serialized legacy records assigns handles; it does not
          // compare or interpret their language. Preserve every field and row.
          action.pending_action_id ??= `pact_${createHash("sha256")
            .update(JSON.stringify([sessionId, index, action]))
            .digest("hex")
            .slice(0, 16)}`;
        }
      }
      parsed.data.pending_actions = parsed.data.pending_actions.map((action, index) => ({
        ...original.pending_actions[index],
        ...action,
        pending_action_id: original.pending_actions[index]!.pending_action_id,
      }));
      for (const action of parsed.data.pending_actions) {
        const marker = action.restored_from_action ?? action.completing_action;
        if (
          (action.restored_from_action !== undefined && action.completing_action !== undefined) ||
          (marker !== undefined &&
            marker.action_id !==
              pendingActionResolutionId({
                session_id: sessionId,
                pending_action_id: action.pending_action_id!,
              }))
        ) {
          throw new WorkingMemoryError("Invalid pending action turn marker", {
            code: "WORKING_MEMORY_INVALID",
          });
        }
      }
      if (migrate) {
        // This entrypoint holds the origin lock. Also clean previously committed
        // completions even when the file already has pending-action ids.
        this.writePersisted(original, sessionId);
        const remainingIds = new Set(
          original.pending_actions.map((action) => action.pending_action_id),
        );
        parsed.data.pending_actions = parsed.data.pending_actions.filter((action) =>
          remainingIds.has(action.pending_action_id),
        );
      }
      return parsed.data;
    } catch (error) {
      this.options.tracer?.emit("working_memory.degraded", {
        turnId: "working_memory",
        session_id: sessionId,
        operation: migrate ? "pending_action_migration" : "load",
        reason: error instanceof WorkingMemoryError ? error.code : "WORKING_MEMORY_LOAD_FAILED",
      });
      if (error instanceof WorkingMemoryError) {
        throw error;
      }

      throw new WorkingMemoryError(`Failed to load working memory for ${sessionId}`, {
        cause: error,
        code: "WORKING_MEMORY_LOAD_FAILED",
      });
    }
  }

  private writePersisted(
    state: WorkingMemory,
    sessionId: SessionId = state.session_id,
    committingTurnId?: string,
  ): void {
    if (this.workingDirectory === undefined) {
      return;
    }

    try {
      // Like every working-memory write, callers must own the origin lock.
      // Cleanup covers turn saves, origin updates, and locked migration; never reads.
      const completed: Array<{ pendingActionId: string; actionId: ActionId }> = [];
      const completing: Array<{
        pendingActionId: string;
        marker: NonNullable<PendingActionRecord["completing_action"]>;
      }> = [];
      const restored: Array<{
        pendingActionId: string;
        actionId: ActionId;
        turnId: string;
        cleared: boolean;
      }> = [];
      const pendingActions = state.pending_actions.filter((action) => {
        if (action.pending_action_id === undefined) return true;
        const actionId = pendingActionResolutionId({
          session_id: sessionId,
          pending_action_id: action.pending_action_id,
        });
        if (action.completing_action !== undefined) {
          const marker = action.completing_action;
          if (marker.action_id !== actionId || action.restored_from_action !== undefined) {
            throw new WorkingMemoryError("Invalid pending action turn marker", {
              code: "WORKING_MEMORY_INVALID",
            });
          }
          if (marker.phase === "provisional" || marker.turn_id === committingTurnId) {
            completing.push({ pendingActionId: action.pending_action_id, marker });
            return true;
          }
          // A durable commit decision makes the normal completion-id cleanup safe.
          delete action.completing_action;
        }
        if (action.restored_from_action !== undefined) {
          if (action.restored_from_action.action_id !== actionId) {
            throw new WorkingMemoryError("Rollback marker does not match its pending action", {
              code: "WORKING_MEMORY_INVALID",
            });
          }
          const cleared =
            this.options.actionRepository !== undefined &&
            this.options.actionRepository.get(actionId) == null;
          restored.push({
            pendingActionId: action.pending_action_id,
            actionId,
            turnId: action.restored_from_action.turn_id,
            cleared,
          });
          if (cleared) {
            delete action.restored_from_action;
          }
          // Even while the action exists, the durable rollback marker makes this
          // a pending intention, not an action-first completion awaiting cleanup.
          return true;
        }
        if (this.options.actionRepository?.get(actionId) == null) return true;
        completed.push({ pendingActionId: action.pending_action_id, actionId });
        return false;
      });
      mkdirSync(this.workingDirectory, { recursive: true });
      writeJsonFileAtomic(this.getPath(sessionId), { ...state, pending_actions: pendingActions });
      state.pending_actions = pendingActions;
      for (const { pendingActionId, marker } of completing) {
        this.options.tracer?.emit("working_memory.pending_action_resolved", {
          turnId: marker.turn_id,
          session_id: sessionId,
          pending_action_id: pendingActionId,
          action_id: marker.action_id,
          step: marker.phase === "provisional" ? "completion_pending" : "completion_commit",
          outcome: "saved",
        });
      }
      for (const { pendingActionId, actionId, turnId, cleared } of restored) {
        this.options.tracer?.emit("working_memory.pending_action_resolved", {
          turnId: "working_memory",
          session_id: sessionId,
          pending_action_id: pendingActionId,
          action_id: actionId,
          rollback_turn_id: turnId,
          step: cleared ? "rollback_cleanup" : "rollback_restore",
          outcome: "saved",
        });
      }
      for (const { pendingActionId, actionId } of completed) {
        this.options.tracer?.emit("working_memory.pending_action_resolved", {
          turnId: "working_memory",
          session_id: sessionId,
          pending_action_id: pendingActionId,
          action_id: actionId,
          step: "cleanup",
          outcome: "saved",
        });
      }
    } catch (error) {
      throw new WorkingMemoryError(`Failed to save working memory for ${state.session_id}`, {
        cause: error,
        code: "WORKING_MEMORY_SAVE_FAILED",
      });
    }
  }
}
