import { z } from "zod";

import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { StorageError } from "../util/errors.js";
import {
  commitmentIdHelpers,
  episodeIdHelpers,
  openQuestionIdHelpers,
  parseCommitmentId,
  parseEpisodeId,
  parseOpenQuestionId,
  parseSemanticEdgeId,
  parseSemanticNodeId,
  semanticEdgeIdHelpers,
  semanticNodeIdHelpers,
  streamEntryIdHelpers,
} from "../util/ids.js";
import { serializeJsonValue } from "../util/json-value.js";

import type { EvidenceItem, RecallEvidenceHandle } from "./recall-types.js";

export const DEFAULT_RECALL_STATE_TTL_TURNS = 6;
export const DEFAULT_RECALL_STATE_WARM_SUPPRESSION_TURNS = 2;
export const DEFAULT_RECALL_STATE_MAX_ACTIVE_HANDLES = 24;
export const DEFAULT_RECALL_STATE_MAX_NEW_HANDLES_PER_TURN = 6;
export const DEFAULT_RECALL_STATE_MAX_WARM_EVIDENCE_RENDERED = 4;

const streamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid stream entry id",
  })
  .transform((value) => streamEntryIdHelpers.parse(value));

const episodeIdSchema = z
  .string()
  .refine((value) => episodeIdHelpers.is(value), {
    message: "Invalid episode id",
  })
  .transform((value) => parseEpisodeId(value));

const semanticNodeIdSchema = z
  .string()
  .refine((value) => semanticNodeIdHelpers.is(value), {
    message: "Invalid semantic node id",
  })
  .transform((value) => parseSemanticNodeId(value));

const semanticEdgeIdSchema = z
  .string()
  .refine((value) => semanticEdgeIdHelpers.is(value), {
    message: "Invalid semantic edge id",
  })
  .transform((value) => parseSemanticEdgeId(value));

const commitmentIdSchema = z
  .string()
  .refine((value) => commitmentIdHelpers.is(value), {
    message: "Invalid commitment id",
  })
  .transform((value) => parseCommitmentId(value));

const openQuestionIdSchema = z
  .string()
  .refine((value) => openQuestionIdHelpers.is(value), {
    message: "Invalid open question id",
  })
  .transform((value) => parseOpenQuestionId(value));

export const recallEvidenceHandleSchema = z.discriminatedUnion("source", [
  z
    .object({
      source: z.literal("episode"),
      episodeId: episodeIdSchema,
    })
    .strict(),
  z
    .object({
      source: z.literal("raw_stream"),
      streamIds: z.array(streamEntryIdSchema).min(1),
      parentEpisodeId: episodeIdSchema.optional(),
    })
    .strict(),
  z
    .object({
      source: z.literal("semantic_node"),
      nodeId: semanticNodeIdSchema,
    })
    .strict(),
  z
    .object({
      source: z.literal("semantic_edge"),
      edgeId: semanticEdgeIdSchema,
      nodeId: semanticNodeIdSchema.optional(),
    })
    .strict(),
  z
    .object({
      source: z.literal("commitment"),
      commitmentId: commitmentIdSchema,
    })
    .strict(),
  z
    .object({
      source: z.literal("open_question"),
      openQuestionId: openQuestionIdSchema,
    })
    .strict(),
]);

export type RecallStateHandle = {
  handle: RecallEvidenceHandle;
  firstSeenTurn: number;
  lastSeenTurn: number;
  lastRenderedTurn: number | null;
  expiresAfterTurn: number;
  reinforcementCount: number;
};

export type RecallState = {
  scopeKey: string;
  activeHandles: RecallStateHandle[];
  suppressedHandles: Record<string, number>;
  lastRefreshTurn: number;
  updatedAt: number;
  ttlTurns: number;
};

export const recallStateHandleSchema = z
  .object({
    handle: recallEvidenceHandleSchema,
    firstSeenTurn: z.number().int().nonnegative(),
    lastSeenTurn: z.number().int().nonnegative(),
    lastRenderedTurn: z.number().int().nonnegative().nullable(),
    expiresAfterTurn: z.number().int().nonnegative(),
    reinforcementCount: z.number().int().nonnegative(),
  })
  .strict();

export const recallStateSchema = z
  .object({
    scopeKey: z.string().min(1),
    activeHandles: z.array(recallStateHandleSchema),
    suppressedHandles: z.record(z.string(), z.number().int().nonnegative()),
    lastRefreshTurn: z.number().int().nonnegative(),
    updatedAt: z.number().finite(),
    ttlTurns: z.number().int().positive(),
  })
  .strict();

type RecallStateRow = {
  scope_key: string;
  state_json: string;
  updated_at: number;
};

export type RecallStateRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

export class RecallStateRepository {
  private readonly clock: Clock;

  constructor(private readonly options: RecallStateRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  load(scopeKey: string): RecallState | null {
    const row = this.db
      .prepare(
        `
          SELECT scope_key, state_json, updated_at
          FROM recall_state
          WHERE scope_key = ?
        `,
      )
      .get(scopeKey) as RecallStateRow | undefined;

    if (row === undefined) {
      return null;
    }

    return parseRecallStateRow(row);
  }

  save(state: RecallState): RecallState {
    const parsed = recallStateSchema.safeParse(state);

    if (!parsed.success) {
      throw new StorageError("Invalid recall state", {
        cause: parsed.error,
        code: "RECALL_STATE_INVALID",
      });
    }

    const next: RecallState = {
      ...parsed.data,
      updatedAt: parsed.data.updatedAt || this.clock.now(),
    };

    this.db
      .prepare(
        `
          INSERT INTO recall_state (scope_key, state_json, updated_at)
          VALUES (?, ?, ?)
          ON CONFLICT (scope_key) DO UPDATE SET
            state_json = excluded.state_json,
            updated_at = excluded.updated_at
        `,
      )
      .run(next.scopeKey, serializeJsonValue(next), next.updatedAt);

    return next;
  }
}

export function createEmptyRecallState(input: {
  scopeKey: string;
  nowMs: number;
  ttlTurns?: number;
}): RecallState {
  return {
    scopeKey: input.scopeKey,
    activeHandles: [],
    suppressedHandles: {},
    lastRefreshTurn: 0,
    updatedAt: input.nowMs,
    ttlTurns: input.ttlTurns ?? DEFAULT_RECALL_STATE_TTL_TURNS,
  };
}

export function deriveRecallEvidenceHandle(item: EvidenceItem): RecallEvidenceHandle | null {
  if (item.source === "recent_raw_stream" || item.source === "working_state") {
    return null;
  }

  const provenance = item.provenance;

  if (provenance === undefined) {
    return null;
  }

  if (provenance.episodeId !== undefined) {
    return {
      source: "episode",
      episodeId: provenance.episodeId,
    };
  }

  if (provenance.streamIds !== undefined && provenance.streamIds.length > 0) {
    return normalizeRecallEvidenceHandle({
      source: "raw_stream",
      streamIds: provenance.streamIds,
      ...(provenance.parentEpisodeId === undefined
        ? {}
        : { parentEpisodeId: provenance.parentEpisodeId }),
    });
  }

  if (provenance.edgeId !== undefined) {
    return {
      source: "semantic_edge",
      edgeId: provenance.edgeId,
      ...(provenance.nodeId === undefined ? {} : { nodeId: provenance.nodeId }),
    };
  }

  if (provenance.nodeId !== undefined) {
    return {
      source: "semantic_node",
      nodeId: provenance.nodeId,
    };
  }

  if (provenance.commitmentId !== undefined) {
    return {
      source: "commitment",
      commitmentId: provenance.commitmentId,
    };
  }

  if (provenance.openQuestionId !== undefined) {
    return {
      source: "open_question",
      openQuestionId: provenance.openQuestionId,
    };
  }

  return null;
}

export function normalizeRecallEvidenceHandle(handle: RecallEvidenceHandle): RecallEvidenceHandle {
  if (handle.source !== "raw_stream") {
    return handle;
  }

  const streamIds = [...new Set(handle.streamIds)].sort();

  return {
    source: "raw_stream",
    streamIds,
    ...(handle.parentEpisodeId === undefined ? {} : { parentEpisodeId: handle.parentEpisodeId }),
  };
}

export function recallEvidenceHandleKey(handle: RecallEvidenceHandle): string {
  if (handle.source === "episode") {
    return `episode:${handle.episodeId}`;
  }

  if (handle.source === "raw_stream") {
    return `raw_stream:${[...new Set(handle.streamIds)].sort().join("|")}`;
  }

  if (handle.source === "semantic_node") {
    return `semantic_node:${handle.nodeId}`;
  }

  if (handle.source === "semantic_edge") {
    return `semantic_edge:${handle.edgeId}`;
  }

  if (handle.source === "commitment") {
    return `commitment:${handle.commitmentId}`;
  }

  return `open_question:${handle.openQuestionId}`;
}

function parseRecallStateRow(row: RecallStateRow): RecallState {
  try {
    const raw = JSON.parse(row.state_json) as unknown;
    const parsed = recallStateSchema.safeParse(raw);

    if (!parsed.success) {
      throw parsed.error;
    }

    if (parsed.data.scopeKey !== row.scope_key) {
      throw new TypeError("Recall state scope key mismatch");
    }

    return parsed.data;
  } catch (error) {
    throw new StorageError(`Invalid recall state row for ${row.scope_key}`, {
      cause: error,
      code: "RECALL_STATE_ROW_INVALID",
    });
  }
}
