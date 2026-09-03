import { createServer, request as httpRequest, type Server } from "node:http";
import { AddressInfo } from "node:net";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { createHash } from "node:crypto";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  createMemoryHandler,
  type MemoryHandlerOptions,
  type MemoryPool,
} from "./memory-handler.js";
import { MemoryTraceRegistry } from "./memory-trace.js";
import type { Borg } from "../borg.js";
import { BorgPool } from "../borg/pool.js";
import type { BorgDependencies } from "../borg/types.js";
import { FakeEmbeddingClient } from "../embeddings/index.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import {
  commitmentSchema,
  type CommitmentRecord,
  type EntityRecord,
} from "../memory/commitments/index.js";
import {
  creatorDirectiveQueueInputSchema,
  creatorDirectiveSchema,
  type CreatorDirective,
  type CreatorDirectiveApplicable,
  type CreatorDirectiveQueueInput,
} from "../memory/creator-directives/index.js";
import type {
  ActivityEventRecordInput,
  ActivityVisibleSessionEvent,
} from "../memory/activity/index.js";
import { episodeParticipantEntityIdTerm, type Episode } from "../memory/episodic/index.js";
import { EmbeddingError } from "../util/errors.js";
import {
  createCommitmentId,
  createCreatorDirectiveId,
  createEntityId,
  createEpisodeId,
  createMaintenanceRunId,
  parseSessionId,
  parseStreamEntryId,
  type EntityId,
  type SessionId,
} from "../util/ids.js";

const TOKEN = "secret-token";
// Default discovery for stub pools: the tenant these tests exercise. Only the
// fan-out tests override it.
const STUB_TENANTS = ["acme"];

const servers: Server[] = [];

afterEach(() => {
  while (servers.length > 0) {
    servers.pop()?.close();
  }
});

type Recorder = {
  tenants: string[];
  exclusives: Array<boolean | undefined>;
  lastRecallLimit?: number;
  lastRecallTraceTurnId?: string;
  lastListOptions?: {
    limit?: number;
    cursor?: string;
  };
  inspectIds: string[];
  appendMany?: {
    inputs: unknown[];
    session?: string;
  };
  appendManyCalls: Array<{
    inputs: unknown[];
    session?: string;
  }>;
  appendCalls: Array<{ input: unknown; session?: string }>;
  resolvedExternalSenders: unknown[];
  resolvedExternalEntities: unknown[];
  lookedUpExternalSenders: Array<{ source: string; externalId: string }>;
  externalSenderIds: Map<string, EntityId>;
  externalEntityIds: Map<string, EntityId>;
  entities: EntityRecord[];
  entityListCalls: number;
  entityGetIds: EntityId[];
  selfEntityGetCalls: number;
  episodeOverrides: Partial<Episode>;
  ingestSessions: string[];
  extractOptions: unknown[];
  commitments: CommitmentRecord[];
  commitmentAdds: unknown[];
  sessionEnsures: unknown[];
  sessionTouches: unknown[];
  activityRecords: ActivityEventRecordInput[];
  activityProjectionInputs: Array<Parameters<Borg["activity"]["projectCompletedTurn"]>[0]>;
  activityObservationProjectionInputs: Array<
    Parameters<Borg["activity"]["projectObservedTurn"]>[0]
  >;
  activityReplyProjectionInputs: Array<Parameters<Borg["activity"]["projectRepliedTurn"]>[0]>;
  activityProjectionError?: Error;
  observedGroupAudienceIds: EntityId[];
  visibleActivityEvents: ActivityVisibleSessionEvent[];
  lastVisibleActivityInput?: unknown;
  lastRecallOptions?: Record<string, unknown>;
  recallOptionsCalls: Array<Record<string, unknown>>;
  retrievalRecords: Array<{ episodeId: Episode["id"]; score: number }>;
  recallEpisodes?: Episode[];
  venueEpisodes: Episode[];
  lastVenueOptions?: unknown;
  recallError?: Error;
  recallPromise?: Promise<never>;
  creatorDirectives: CreatorDirective[];
  directiveQueueInputs: unknown[];
  directiveQueueError?: Error;
  directiveRevokeError?: Error;
  directiveApplicable: CreatorDirectiveApplicable[];
  directiveApplicableOptions: unknown[];
};

function testEpisode(
  id: Episode["id"] = "ep_aaaaaaaaaaaaaaaa" as Episode["id"],
  overrides: Partial<Episode> = {},
): Episode {
  return {
    id,
    title: "Title",
    narrative: "Narrative",
    participants: ["Ada"],
    location: null,
    start_time: 10,
    end_time: 20,
    source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as Episode["source_stream_ids"][number]],
    significance: 0.72,
    tags: ["planning", "admin"],
    confidence: 0.9,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    emotional_arc: null,
    audience_entity_id: null,
    origin_audience_entity_ids: [],
    shared: false,
    episode_kind: "raw",
    consolidation_family_id: null,
    consolidation_coverage_hash: null,
    embedding: Float32Array.from([1, 0, 0, 0]),
    created_at: 1,
    updated_at: 2,
    ...overrides,
  };
}

function testCommitment(overrides: Partial<CommitmentRecord> = {}): CommitmentRecord {
  return commitmentSchema.parse({
    id: overrides.id ?? createCommitmentId(),
    record_version: overrides.record_version ?? 1,
    type: overrides.type ?? "preference",
    kind: overrides.kind ?? "participant_preference",
    enforcement_class: overrides.enforcement_class ?? "advisory",
    critical_domain: overrides.critical_domain ?? null,
    directive_family: overrides.directive_family ?? "concise_replies",
    closure_pressure_relevance: overrides.closure_pressure_relevance ?? "neutral",
    directive: overrides.directive ?? "Keep replies concise.",
    priority: overrides.priority ?? 5,
    made_to_entity: overrides.made_to_entity ?? null,
    restricted_audience: overrides.restricted_audience ?? null,
    about_entity: overrides.about_entity ?? null,
    committed_by_entity_id: overrides.committed_by_entity_id ?? null,
    provenance: overrides.provenance ?? { kind: "manual" },
    source_stream_entry_ids: overrides.source_stream_entry_ids,
    created_at: overrides.created_at ?? 100,
    expires_at: overrides.expires_at ?? null,
    expired_at: overrides.expired_at ?? null,
    revoked_at: overrides.revoked_at ?? null,
    revoked_reason: overrides.revoked_reason ?? null,
    revoke_provenance: overrides.revoke_provenance ?? null,
    superseded_by: overrides.superseded_by ?? null,
    canonicalized_by_artifact_entry_id: overrides.canonicalized_by_artifact_entry_id ?? null,
    last_reinforced_at: overrides.last_reinforced_at ?? 100,
  });
}

function testDirective(overrides: Partial<CreatorDirective> = {}): CreatorDirective {
  return creatorDirectiveSchema.parse({
    id: overrides.id ?? createCreatorDirectiveId(),
    record_version: overrides.record_version ?? 1,
    status: overrides.status ?? "active",
    kind: overrides.kind ?? "response_policy",
    created_by_entity_id: overrides.created_by_entity_id ?? createEntityId(),
    source_session_id: overrides.source_session_id ?? "sess_aaaaaaaaaaaaaaaa",
    authorization_stream_entry_ids: overrides.authorization_stream_entry_ids ?? [
      "strm_aaaaaaaaaaaaaaaa",
    ],
    content_source_stream_entry_ids: overrides.content_source_stream_entry_ids ?? [
      "strm_aaaaaaaaaaaaaaaa",
    ],
    subject_kind: overrides.subject_kind ?? "system",
    subject_entity_id: overrides.subject_entity_id ?? null,
    semantic_slot: overrides.semantic_slot ?? null,
    canonical_fact: overrides.canonical_fact ?? null,
    operational_directive: overrides.operational_directive ?? "Use concise replies.",
    disclosure_policy: overrides.disclosure_policy ?? {
      content_scope: "public",
      allowed_entity_ids: [],
      excluded_entity_ids: [],
      subject_may_know: null,
      mention_policy: "answer_if_asked",
      denied_audience_behavior: "omit",
      boundary_prompt: "A private operator rule applies.",
      topic_tags: [],
    },
    activation_policy: overrides.activation_policy ?? {
      scope: "same_as_disclosure",
      allowed_entity_ids: [],
      excluded_entity_ids: [],
    },
    priority: overrides.priority ?? 0,
    superseded_by: overrides.superseded_by ?? null,
    revoked_reason: overrides.revoked_reason ?? null,
    created_at: overrides.created_at ?? 100,
    updated_at: overrides.updated_at ?? 100,
  });
}

function stubBorg(rec: Recorder): Borg {
  return {
    stream: {
      append: async (
        input: { kind?: string; content: unknown },
        options?: { session?: string },
      ) => {
        rec.appendCalls.push({
          input,
          session: options?.session,
        });
        return {
          ...input,
          id: "strm_bbbbbbbbbbbbbbbb",
          timestamp: 1000,
          kind: input.kind ?? "user_msg",
          content: input.content,
          sender_entity_id: null,
          reply_target_entity_id: null,
          session_id: options?.session ?? "sess_0000000000000000",
          compressed: false,
        };
      },
      appendMany: async (inputs: unknown[], options?: { session?: string }) => {
        rec.appendMany = { inputs, session: options?.session };
        rec.appendManyCalls.push(rec.appendMany);
        return inputs.map((input, index) => ({
          ...(input as Record<string, unknown>),
          id: `strm_${String(index).padStart(16, "a")}`,
          timestamp: 1000 + index,
          kind: (input as { kind?: string }).kind,
          sender_entity_id: (input as { sender_entity_id?: EntityId }).sender_entity_id ?? null,
          reply_target_entity_id: null,
          session_id: options?.session ?? "sess_0000000000000000",
          compressed: false,
        }));
      },
    },
    entities: {
      resolveExternal: (input: {
        source: string;
        externalId: string;
        canonicalName: string;
        kind: "person" | "group" | "abstract";
        provenance?: EntityRecord["name_provenance"];
      }) => {
        rec.resolvedExternalEntities.push(input);
        if (input.source === "team-agent.sender") {
          rec.resolvedExternalSenders.push(input);
        }
        const key = `${input.source}\u0000${input.externalId}`;
        const existing = rec.externalEntityIds.get(key);

        if (existing !== undefined) {
          return existing;
        }

        const entityId = createEntityId();
        rec.externalEntityIds.set(key, entityId);
        if (input.source === "team-agent.sender") {
          rec.externalSenderIds.set(input.externalId, entityId);
        }
        rec.entities.push({
          id: entityId,
          canonical_name: input.canonicalName,
          aliases: [],
          kind: input.kind,
          borg_role: null,
          name_provenance: input.provenance ?? "transport_sender",
          created_at: 1,
        });
        return entityId;
      },
      findByExternalId: (source: string, externalId: string) => {
        rec.lookedUpExternalSenders.push({ source, externalId });
        return (
          rec.externalEntityIds.get(`${source}\u0000${externalId}`) ??
          (source === "team-agent.sender" ? rec.externalSenderIds.get(externalId) : undefined) ??
          null
        );
      },
      get: (id: EntityId) => {
        rec.entityGetIds.push(id);
        return (
          rec.entities.find((entity) => entity.id === id) ??
          ([...rec.externalSenderIds.values()].some((entityId) => entityId === id)
            ? {
                id,
                canonical_name: "Known sender",
                aliases: [],
                kind: "person",
                borg_role: null,
                created_at: 1,
              }
            : null)
        );
      },
      getSelf: () => {
        rec.selfEntityGetCalls += 1;
        return rec.entities.find((entity) => entity.kind === "self") ?? null;
      },
      list: () => {
        rec.entityListCalls += 1;
        return [...rec.entities];
      },
    },
    identity: {
      addCommitment: (input: {
        type: CommitmentRecord["type"];
        kind: CommitmentRecord["kind"];
        enforcementClass: CommitmentRecord["enforcement_class"];
        criticalDomain: CommitmentRecord["critical_domain"];
        directiveFamily: string;
        directive: string;
        priority: number;
        restrictedAudience: EntityId | null;
      }) => {
        rec.commitmentAdds.push(input);
        const commitment = testCommitment({
          type: input.type,
          kind: input.kind,
          enforcement_class: input.enforcementClass,
          critical_domain: input.criticalDomain,
          directive_family: input.directiveFamily,
          directive: input.directive,
          priority: input.priority,
          restricted_audience: input.restrictedAudience,
          created_at: 100 + rec.commitments.length,
          last_reinforced_at: 100 + rec.commitments.length,
        });
        rec.commitments.push(commitment);
        return commitment;
      },
    },
    commitments: {
      get: (id: CommitmentRecord["id"]) =>
        rec.commitments.find((commitment) => commitment.id === id) ?? null,
      list: (options?: { activeOnly?: boolean; audienceEntityId?: EntityId | null }) =>
        rec.commitments.filter((commitment) => {
          if (
            options?.activeOnly === true &&
            (commitment.revoked_at !== null ||
              commitment.expired_at !== null ||
              commitment.superseded_by !== null)
          ) {
            return false;
          }

          if (options?.audienceEntityId === undefined) {
            return true;
          }

          const scope = commitment.restricted_audience ?? commitment.made_to_entity;
          return scope === null || scope === options.audienceEntityId;
        }),
      revoke: (id: CommitmentRecord["id"], reason: string) => {
        const index = rec.commitments.findIndex((commitment) => commitment.id === id);
        const current = rec.commitments[index];
        if (current === undefined) {
          return null;
        }
        const revoked = testCommitment({
          ...current,
          record_version: (current.record_version ?? 1) + 1,
          revoked_at: 500,
          revoked_reason: reason,
          revoke_provenance: { kind: "manual" },
        });
        rec.commitments[index] = revoked;
        return revoked;
      },
    },
    sessions: {
      get: () => null,
      ensure: (input: unknown) => {
        rec.sessionEnsures.push(input);
        return input;
      },
      touch: (sessionId: string, update: unknown) => {
        rec.sessionTouches.push({ sessionId, update });
        return null;
      },
    },
    activity: {
      record: (input: ActivityEventRecordInput) => {
        rec.activityRecords.push(input);
        return input;
      },
      projectObservedTurn: (input: Parameters<Borg["activity"]["projectObservedTurn"]>[0]) => {
        rec.activityObservationProjectionInputs.push(input);
        if (rec.activityProjectionError !== undefined) {
          throw rec.activityProjectionError;
        }

        rec.sessionEnsures.push(input.session);
        rec.activityRecords.push(input.userContact);
        rec.sessionTouches.push({ sessionId: input.session.session_id, update: input.touch });
        return {
          userContact: input.userContact,
          session: input.session,
        };
      },
      projectRepliedTurn: (input: Parameters<Borg["activity"]["projectRepliedTurn"]>[0]) => {
        rec.activityReplyProjectionInputs.push(input);
        if (rec.activityProjectionError !== undefined) {
          throw rec.activityProjectionError;
        }

        rec.sessionEnsures.push(input.session);
        rec.activityRecords.push(input.borgReplied);
        rec.sessionTouches.push({ sessionId: input.session.session_id, update: input.touch });
        return {
          borgReplied: input.borgReplied,
          session: input.session,
        };
      },
      projectCompletedTurn: (input: Parameters<Borg["activity"]["projectCompletedTurn"]>[0]) => {
        rec.activityProjectionInputs.push(input);
        if (rec.activityProjectionError !== undefined) {
          throw rec.activityProjectionError;
        }

        rec.sessionEnsures.push(input.session);
        rec.activityRecords.push(input.userContact, input.borgReplied);
        rec.sessionTouches.push({ sessionId: input.session.session_id, update: input.touch });
        return {
          userContact: input.userContact,
          borgReplied: input.borgReplied,
          session: input.session,
        };
      },
      listObservedGroupAudienceEntityIdsForSpeaker: () => [...rec.observedGroupAudienceIds],
      listRecentVisibleOtherSessionEvents: (input: unknown) => {
        rec.lastVisibleActivityInput = input;
        return [...rec.visibleActivityEvents];
      },
    },
    creatorDirectives: {
      queue: (input: CreatorDirectiveQueueInput) => {
        rec.directiveQueueInputs.push(input);
        if (rec.directiveQueueError !== undefined) {
          throw rec.directiveQueueError;
        }
        const normalized = creatorDirectiveQueueInputSchema.parse(input);
        const directive = testDirective({
          kind: normalized.kind,
          created_by_entity_id: normalized.createdByEntityId,
          source_session_id: normalized.sourceSessionId,
          authorization_stream_entry_ids: normalized.authorizationStreamEntryIds,
          content_source_stream_entry_ids: normalized.contentSourceStreamEntryIds,
          subject_kind: normalized.subjectKind,
          subject_entity_id: normalized.subjectEntityId ?? null,
          semantic_slot: normalized.semanticSlot ?? null,
          canonical_fact: normalized.canonicalFact ?? normalized.semanticValue ?? null,
          operational_directive: normalized.operationalDirective ?? null,
          disclosure_policy: normalized.disclosurePolicy,
          activation_policy: normalized.activationPolicy ?? {
            scope: "same_as_disclosure",
            allowed_entity_ids: [],
            excluded_entity_ids: [],
          },
          priority: normalized.priority,
        });
        rec.creatorDirectives.push(directive);
        return directive;
      },
      get: (id: CreatorDirective["id"]) =>
        rec.creatorDirectives.find((directive) => directive.id === id) ?? null,
      list: (filter?: { status?: CreatorDirective["status"] }) =>
        rec.creatorDirectives.filter(
          (directive) => filter?.status === undefined || directive.status === filter.status,
        ),
      listApplicable: (options: unknown) => {
        rec.directiveApplicableOptions.push(options);
        return [...rec.directiveApplicable];
      },
      revoke: (id: CreatorDirective["id"], reason: string) => {
        if (rec.directiveRevokeError !== undefined) {
          throw rec.directiveRevokeError;
        }
        const index = rec.creatorDirectives.findIndex((directive) => directive.id === id);
        const current = rec.creatorDirectives[index];
        if (current === undefined || current.status !== "active") {
          return null;
        }
        const revoked = testDirective({
          ...current,
          status: "revoked",
          record_version: current.record_version + 1,
          revoked_reason: reason,
          updated_at: current.updated_at + 1,
        });
        rec.creatorDirectives[index] = revoked;
        return revoked;
      },
    },
    episodic: {
      // Real facade returns numeric counts.
      extract: async (options: unknown) => {
        rec.extractOptions.push(options);
        return { inserted: 1, updated: 0, skipped: 0 };
      },
      ingest: async (options?: { session?: string }) => {
        rec.ingestSessions.push(options?.session ?? "");
        return { ran: true, processedEntries: 2 };
      },
      search: async (
        _query: string,
        opts: {
          limit?: number;
          traceTurnId?: string;
          audienceEntityId?: EntityId | null;
          visibleAudienceEntityIds?: readonly EntityId[];
          recordRetrieval?: boolean;
        },
      ) => {
        rec.lastRecallLimit = opts.limit;
        rec.lastRecallTraceTurnId = opts.traceTurnId;
        rec.lastRecallOptions = { ...opts };
        rec.recallOptionsCalls.push({ ...opts });
        if (rec.recallError !== undefined) {
          throw rec.recallError;
        }
        if (rec.recallPromise !== undefined) {
          return rec.recallPromise;
        }
        return (
          rec.recallEpisodes ?? [testEpisode("ep_1" as Episode["id"], rec.episodeOverrides)]
        ).map((episode, index) => ({
          episode,
          score: 0.91,
          rawScore: 1.16 - index * 0.01,
        }));
      },
      searchWithTimeRangeFallback: async (
        _query: string,
        opts: {
          limit?: number;
          traceTurnId?: string;
          audienceEntityId?: EntityId | null;
          visibleAudienceEntityIds?: readonly EntityId[];
          timeRange: { start: number; end: number };
          recordRetrieval?: boolean;
        },
      ) => {
        rec.lastRecallLimit = opts.limit;
        rec.lastRecallTraceTurnId = opts.traceTurnId;
        rec.lastRecallOptions = { ...opts };
        rec.recallOptionsCalls.push({ ...opts });
        if (rec.recallError !== undefined) {
          throw rec.recallError;
        }
        if (rec.recallPromise !== undefined) {
          return rec.recallPromise;
        }
        const recalled = (
          rec.recallEpisodes ?? [testEpisode("ep_1" as Episode["id"], rec.episodeOverrides)]
        ).map((episode, index) => ({
          episode,
          score: 0.91,
          rawScore: 1.16 - index * 0.01,
        }));
        const strict = recalled.filter(
          (hit) =>
            hit.episode.start_time >= opts.timeRange.start &&
            hit.episode.start_time <= opts.timeRange.end,
        );

        return strict.length > 0
          ? { episodes: strict, timeRangeFallback: false }
          : { episodes: recalled, timeRangeFallback: true };
      },
      recordRetrieval: (episodeId: Episode["id"], score: number) => {
        rec.retrievalRecords.push({ episodeId, score });
      },
      listRecentForSession: async (options: unknown) => {
        rec.lastVenueOptions = options;
        return rec.venueEpisodes.map((episode) => ({
          episode,
          stats: {
            episode_id: episode.id,
            retrieval_count: 0,
            use_count: 0,
            last_retrieved: null,
            win_rate: 0,
            tier: "T1" as const,
            promoted_at: episode.created_at,
            promoted_from: null,
            gist: null,
            gist_generated_at: null,
            last_decayed_at: null,
            heat_multiplier: 1,
            valence_mean: 0,
            archived: false,
          },
          similarity: 0,
        }));
      },
      list: async (options?: { limit?: number; cursor?: string }) => {
        rec.lastListOptions = options;
        return {
          items: [testEpisode(undefined, rec.episodeOverrides)],
          nextCursor: "next-cursor",
        };
      },
      inspect: async (id: Episode["id"]) => {
        rec.inspectIds.push(id);
        return id === ("ep_missingmissing00" as Episode["id"])
          ? null
          : testEpisode(id, rec.episodeOverrides);
      },
    },
  } as unknown as Borg;
}

function recordingPool(): { pool: MemoryPool; rec: Recorder } {
  const selfEntity: EntityRecord = {
    id: createEntityId(),
    canonical_name: "team-agent",
    aliases: ["self"],
    kind: "self",
    borg_role: null,
    name_provenance: "config_default_user",
    created_at: 0,
  };
  const rec: Recorder = {
    tenants: [],
    exclusives: [],
    inspectIds: [],
    appendCalls: [],
    appendManyCalls: [],
    resolvedExternalSenders: [],
    resolvedExternalEntities: [],
    lookedUpExternalSenders: [],
    externalSenderIds: new Map(),
    externalEntityIds: new Map(),
    entities: [selfEntity],
    entityListCalls: 0,
    entityGetIds: [],
    selfEntityGetCalls: 0,
    episodeOverrides: {},
    ingestSessions: [],
    extractOptions: [],
    commitments: [],
    commitmentAdds: [],
    sessionEnsures: [],
    sessionTouches: [],
    activityRecords: [],
    activityProjectionInputs: [],
    activityObservationProjectionInputs: [],
    activityReplyProjectionInputs: [],
    observedGroupAudienceIds: [],
    visibleActivityEvents: [],
    recallOptionsCalls: [],
    retrievalRecords: [],
    venueEpisodes: [],
    creatorDirectives: [],
    directiveQueueInputs: [],
    directiveApplicable: [],
    directiveApplicableOptions: [],
  };
  const pool: MemoryPool = {
    listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
    async withTenant(tenantId, fn, opts) {
      rec.tenants.push(tenantId);
      rec.exclusives.push(opts?.exclusive);
      return fn(stubBorg(rec));
    },
  };
  return { pool, rec };
}

function start(
  pool: MemoryPool,
  token = TOKEN,
  handlerOptions: Omit<MemoryHandlerOptions, "pool" | "token"> = {},
): Promise<string> {
  const server = createServer(createMemoryHandler({ pool, token, ...handlerOptions }));
  servers.push(server);
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      resolve(`http://127.0.0.1:${port}`);
    });
  });
}

async function post(base: string, path: string, body: unknown, token?: string, raw?: string) {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (token !== undefined) {
    headers["x-borg-token"] = token;
  }
  return fetch(`${base}${path}`, { method: "POST", headers, body: raw ?? JSON.stringify(body) });
}

async function get(base: string, path: string, token?: string) {
  const headers: Record<string, string> = {};
  if (token !== undefined) {
    headers["x-borg-token"] = token;
  }
  return fetch(`${base}${path}`, { headers });
}

async function del(base: string, path: string, token?: string) {
  const headers: Record<string, string> = {};
  if (token !== undefined) {
    headers["x-borg-token"] = token;
  }
  return fetch(`${base}${path}`, { method: "DELETE", headers });
}

async function requestRaw(
  base: string,
  path: string,
  options: {
    method?: string;
    token?: string;
    body?: unknown;
  } = {},
): Promise<{ status: number; body: unknown; text: string }> {
  const baseUrl = new URL(base);
  const rawBody = options.body === undefined ? undefined : JSON.stringify(options.body);
  const headers: Record<string, string> = {};
  if (options.token !== undefined) {
    headers["x-borg-token"] = options.token;
  }
  if (rawBody !== undefined) {
    headers["content-type"] = "application/json";
    headers["content-length"] = String(Buffer.byteLength(rawBody));
  }

  return new Promise((resolve, reject) => {
    const req = httpRequest(
      {
        hostname: baseUrl.hostname,
        port: Number(baseUrl.port),
        method: options.method ?? "GET",
        path,
        headers,
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (chunk: Buffer) => chunks.push(chunk));
        res.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          let body: unknown = text;
          try {
            body = JSON.parse(text) as unknown;
          } catch {
            // Keep non-JSON bodies as text for diagnostics.
          }
          resolve({ status: res.statusCode ?? 0, body, text });
        });
      },
    );
    req.on("error", reject);
    req.end(rawBody);
  });
}

describe("memory sidecar handler", () => {
  it("serves /healthz without auth", async () => {
    const { pool } = recordingPool();
    const base = await start(pool);
    const res = await fetch(`${base}/healthz`);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true });
  });

  it("rejects missing/wrong token with 401 and does not touch the pool", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    expect((await post(base, "/memory/recall", { tenant: "acme", query: "q" })).status).toBe(401);
    expect(
      (await post(base, "/memory/recall", { tenant: "acme", query: "q" }, "nope")).status,
    ).toBe(401);
    expect(rec.tenants).toEqual([]);
  });

  it("fails closed when the configured token is empty", async () => {
    const { pool } = recordingPool();
    const base = await start(pool, "");
    expect((await post(base, "/memory/recall", { tenant: "acme", query: "q" }, "")).status).toBe(
      401,
    );
    expect(
      (await post(base, "/memory/recall", { tenant: "acme", query: "q" }, "anything")).status,
    ).toBe(401);
  });

  it("404s unknown routes and non-POST methods (after auth)", async () => {
    const { pool } = recordingPool();
    const base = await start(pool);
    expect((await post(base, "/memory/nope", { tenant: "acme" }, TOKEN)).status).toBe(404);
    const getRes = await fetch(`${base}/memory/recall`, { headers: { "x-borg-token": TOKEN } });
    expect(getRes.status).toBe(404);
  });

  it("uses the raw request path for auth and routing without dot-segment normalization", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);

    const authProbe = await requestRaw(base, "/foo/../healthz");
    expect(authProbe.status).toBe(401);
    expect(authProbe.body).toEqual({ error: "unauthorized" });

    const routeProbe = await requestRaw(base, "/memory/nope/../recall", {
      method: "POST",
      token: TOKEN,
      body: { tenant: "acme", query: "q" },
    });
    expect(routeProbe.status).toBe(404);
    expect(routeProbe.body).toEqual({ error: "not found" });

    const maintenanceProbe = await requestRaw(
      base,
      "/memory/nope/../maintenance?tenant=acme&mode=light&dryRun=0",
      { method: "POST", token: TOKEN },
    );
    expect(maintenanceProbe.status).toBe(404);
    expect(maintenanceProbe.body).toEqual({ error: "not found" });
    expect(rec.tenants).toEqual([]);
  });

  it("authenticates and accepts a detached maintenance run from query parameters", async () => {
    const { pool } = recordingPool();
    const runId = createMaintenanceRunId();
    const starts: unknown[] = [];
    const maintenanceCoordinator = {
      tryReserve: (input: unknown) => {
        starts.push(input);
        return {
          status: "accepted",
          runId,
          completion: new Promise(() => {}),
        };
      },
      startReserved: (tenant: string, startedRunId: string) => {
        starts.push({ scheduled: [tenant, startedRunId] });
        return true;
      },
      hasReservation: () => true,
      cancelReservation: () => true,
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });

    const unauthorized = await post(
      base,
      "/memory/maintenance?tenant=acme&mode=heavy&dryRun=1",
      {},
    );
    expect(unauthorized.status).toBe(401);
    expect(starts).toEqual([]);

    const accepted = await post(
      base,
      "/memory/maintenance?tenant=acme&mode=heavy&dryRun=1",
      {},
      TOKEN,
    );
    expect(accepted.status).toBe(202);
    expect(await accepted.json()).toEqual({ run_id: runId });
    expect(starts).toEqual([
      { tenant: "acme", mode: "heavy", dryRun: true },
      { scheduled: ["acme", runId] },
    ]);
  });

  it("reads back what the offline processes wrote to the self store", async () => {
    // The gap this closes: after a heavy maintenance run the only evidence was a
    // change count plus record ids in the audit log, so mis-voiced self-narrator
    // output was indistinguishable from clean output.
    const listOptions: unknown[] = [];
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant(_tenantId, fn) {
        return fn({
          self: {
            growthMarkers: {
              list: (opts: unknown) => {
                listOptions.push(opts);
                return [{ id: "grw_1", summary: "Nauczyłem się czytać logi ORA-" }];
              },
            },
            autobiographical: {
              listPeriods: () => [{ id: "abp_1", label: "2026-Q3" }],
            },
            openQuestions: {
              list: () => [{ id: "oq_1", question: "Kto jest właścicielem HK?" }],
            },
          },
        } as unknown as Borg);
      },
    };
    const base = await start(pool);

    const response = await fetch(`${base}/memory/self?tenant=acme&limit=25`, {
      headers: { "x-borg-token": TOKEN },
    });

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      ok: true,
      tenant: "acme",
      growth_markers: [{ id: "grw_1", summary: "Nauczyłem się czytać logi ORA-" }],
      periods: [{ id: "abp_1", label: "2026-Q3" }],
      open_questions: [{ id: "oq_1", question: "Kto jest właścicielem HK?" }],
    });
    expect(listOptions).toEqual([{ limit: 25 }]);
  });

  it("reads back semantic nodes without their embeddings", async () => {
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant(_tenantId, fn) {
        return fn({
          semantic: {
            nodes: {
              list: async () => [
                {
                  id: "semn_1",
                  kind: "concept",
                  label: "Notyfikator",
                  description: "System notyfikacji BSS.",
                  confidence: 0.8,
                  status: "active",
                  archived: false,
                  source_episode_ids: ["ep_1"],
                  created_at: 1_000,
                  embedding: new Float32Array([1, 2, 3, 4]),
                },
              ],
            },
          },
        } as unknown as Borg);
      },
    };
    const base = await start(pool);

    const response = await fetch(`${base}/memory/semantic?tenant=acme`, {
      headers: { "x-borg-token": TOKEN },
    });

    expect(response.status).toBe(200);
    const body = (await response.json()) as { nodes: Array<Record<string, unknown>> };
    expect(body.nodes).toHaveLength(1);
    expect(body.nodes[0]).toEqual({
      id: "semn_1",
      kind: "concept",
      label: "Notyfikator",
      description: "System notyfikacji BSS.",
      confidence: 0.8,
      status: "active",
      archived: false,
      source_episode_ids: ["ep_1"],
      created_at: 1_000,
    });
    expect(body.nodes[0]).not.toHaveProperty("embedding");
  });

  it("reads the review queue, where insights and flags actually live", async () => {
    // A reflector insight and an overseer flag are PROPOSALS: the semantic graph
    // shows nothing until a resolution accepts them, so the queue is the only place
    // their content is visible.
    const listArgs: unknown[] = [];
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant(_tenantId, fn) {
        return fn({
          review: {
            list: (opts: unknown) => {
              listArgs.push(opts);
              return [
                { id: 3, kind: "new_insight", refs: { nodeId: "semn_1" }, reason: "pattern" },
                { id: 11, kind: "misattribution", refs: { patch: {} }, reason: "ownership" },
              ];
            },
          },
        } as unknown as Borg);
      },
    };
    const base = await start(pool);

    const response = await fetch(`${base}/memory/review?tenant=acme&limit=1`, {
      headers: { "x-borg-token": TOKEN },
    });

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      ok: true,
      tenant: "acme",
      open_only: true,
      total: 2,
      items: [{ id: 3, kind: "new_insight", refs: { nodeId: "semn_1" }, reason: "pattern" }],
      truncated: true,
    });
    // openOnly defaults on; openOnly=0 opts into resolved items too.
    expect(listArgs).toEqual([{ openOnly: true }]);
  });

  it("requires an explicit tenant on the new read paths", async () => {
    const { pool } = recordingPool();
    const base = await start(pool);

    for (const path of ["/memory/self", "/memory/semantic", "/memory/review"]) {
      const response = await fetch(`${base}${path}`, { headers: { "x-borg-token": TOKEN } });
      expect(response.status).toBe(400);
    }
  });

  it("fans maintenance out across every discovered tenant when tenant is omitted", async () => {
    const { pool, rec } = recordingPool();
    pool.listTenantIds = () => Promise.resolve(["team-agent-ai", "team-agent-esb"]);
    const scheduled: Array<[string, string]> = [];
    const reserved: string[] = [];
    const maintenanceCoordinator = {
      tryReserve: (input: { tenant: string }) => {
        reserved.push(input.tenant);
        return {
          status: "accepted",
          runId: `run_${input.tenant}`,
          completion: new Promise(() => {}),
        };
      },
      startReserved: (tenant: string, runId: string) => {
        scheduled.push([tenant, runId]);
        return true;
      },
      hasReservation: () => true,
      cancelReservation: () => true,
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });

    const response = await post(base, "/memory/maintenance?mode=light&dryRun=0", {}, TOKEN);

    expect(response.status).toBe(202);
    expect(await response.json()).toEqual({
      runs: [
        { tenant: "team-agent-ai", run_id: "run_team-agent-ai" },
        { tenant: "team-agent-esb", run_id: "run_team-agent-esb" },
      ],
      skipped: [],
    });
    expect(reserved).toEqual(["team-agent-ai", "team-agent-esb"]);
    // Every reserved run is scheduled, and only after the response is handed off.
    expect(scheduled).toEqual([
      ["team-agent-ai", "run_team-agent-ai"],
      ["team-agent-esb", "run_team-agent-esb"],
    ]);
    // Readiness is established per tenant before acceptance.
    expect(rec.tenants).toEqual(["team-agent-ai", "team-agent-esb"]);
  });

  it("treats tenant=* as fan-out and keeps going past a tenant that is already running", async () => {
    const { pool } = recordingPool();
    pool.listTenantIds = () => Promise.resolve(["a-tenant", "busy-tenant", "z-tenant"]);
    const scheduled: string[] = [];
    const maintenanceCoordinator = {
      tryReserve: (input: { tenant: string }) =>
        input.tenant === "busy-tenant"
          ? { status: "conflict", runId: "run_busy" }
          : { status: "accepted", runId: `run_${input.tenant}`, completion: new Promise(() => {}) },
      startReserved: (tenant: string) => {
        scheduled.push(tenant);
        return true;
      },
      hasReservation: () => true,
      cancelReservation: () => true,
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });

    const response = await post(
      base,
      "/memory/maintenance?tenant=*&mode=heavy&dryRun=1",
      {},
      TOKEN,
    );

    expect(response.status).toBe(202);
    expect(await response.json()).toEqual({
      runs: [
        { tenant: "a-tenant", run_id: "run_a-tenant" },
        { tenant: "z-tenant", run_id: "run_z-tenant" },
      ],
      skipped: [{ tenant: "busy-tenant", reason: "already running", runId: "run_busy" }],
    });
    expect(scheduled).toEqual(["a-tenant", "z-tenant"]);
  });

  it("fails the fan-out request when no tenant run could be started", async () => {
    const { pool } = recordingPool();
    pool.listTenantIds = () => Promise.resolve(["one", "two"]);
    const maintenanceCoordinator = {
      tryReserve: () => ({ status: "conflict", runId: "run_existing" }),
      startReserved: () => true,
      hasReservation: () => true,
      cancelReservation: () => true,
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });

    const response = await post(base, "/memory/maintenance?mode=light&dryRun=0", {}, TOKEN);

    // Non-2xx so a `curl --fail` cron reports a failed job instead of a silent no-op.
    expect(response.status).toBe(409);
    expect(await response.json()).toEqual({
      error: "maintenance already running",
      skipped: [
        { tenant: "one", reason: "already running", runId: "run_existing" },
        { tenant: "two", reason: "already running", runId: "run_existing" },
      ],
    });
  });

  it("reports an empty volume instead of a clean no-op run", async () => {
    const { pool } = recordingPool();
    pool.listTenantIds = () => Promise.resolve([]);
    const maintenanceCoordinator = {
      tryReserve: () => {
        throw new Error("must not reserve without a tenant");
      },
      startReserved: () => true,
      hasReservation: () => true,
      cancelReservation: () => true,
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });

    const response = await post(base, "/memory/maintenance?mode=light&dryRun=0", {}, TOKEN);

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ error: "no tenants discovered" });
  });

  it("never runs discovery for an explicitly named tenant", async () => {
    const { pool } = recordingPool();
    let discoveries = 0;
    pool.listTenantIds = () => {
      discoveries += 1;
      return Promise.resolve(["acme", "other"]);
    };
    const runId = createMaintenanceRunId();
    const scheduled: Array<[string, string]> = [];
    const maintenanceCoordinator = {
      tryReserve: () => ({ status: "accepted", runId, completion: new Promise(() => {}) }),
      startReserved: (tenant: string, startedRunId: string) => {
        scheduled.push([tenant, startedRunId]);
        return true;
      },
      hasReservation: () => true,
      cancelReservation: () => true,
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });

    const response = await post(
      base,
      "/memory/maintenance?tenant=acme&mode=light&dryRun=0",
      {},
      TOKEN,
    );

    // Single-tenant contract is unchanged: bare {run_id}, no runs/skipped envelope.
    expect(response.status).toBe(202);
    expect(await response.json()).toEqual({ run_id: runId });
    expect(discoveries).toBe(0);
    expect(scheduled).toEqual([["acme", runId]]);
  });

  it("holds a synchronous reservation during readiness and rejects a racing POST", async () => {
    const runId = createMaintenanceRunId();
    let active = false;
    let scheduled = false;
    let readinessStarted!: () => void;
    const startedReadiness = new Promise<void>((resolve) => {
      readinessStarted = resolve;
    });
    let releaseReadiness!: () => void;
    const readinessGate = new Promise<void>((resolve) => {
      releaseReadiness = resolve;
    });
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant(_tenant, fn) {
        readinessStarted();
        await readinessGate;
        return fn({} as Borg);
      },
    };
    const maintenanceCoordinator = {
      tryReserve: () => {
        if (active) {
          return { status: "conflict" as const, runId };
        }
        active = true;
        return { status: "accepted" as const, runId, completion: new Promise(() => {}) };
      },
      startReserved: () => {
        scheduled = true;
        return true;
      },
      hasReservation: () => active,
      cancelReservation: () => {
        active = false;
        return true;
      },
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });
    const path = "/memory/maintenance?tenant=acme&mode=light&dryRun=0";

    const firstResponse = post(base, path, {}, TOKEN);
    await startedReadiness;
    const racing = await post(base, path, {}, TOKEN);

    expect(racing.status).toBe(409);
    expect(await racing.json()).toEqual({
      error: "maintenance already running",
      run_id: runId,
    });
    expect(scheduled).toBe(false);

    releaseReadiness();
    const accepted = await firstResponse;
    expect(accepted.status).toBe(202);
    expect(await accepted.json()).toEqual({ run_id: runId });
    expect(scheduled).toBe(true);
  });

  it("clears a reservation and returns 503 when tenant readiness fails", async () => {
    const runId = createMaintenanceRunId();
    const cancellations: unknown[] = [];
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant() {
        throw new Error("tenant config is invalid");
      },
    };
    const maintenanceCoordinator = {
      tryReserve: () => ({
        status: "accepted" as const,
        runId,
        completion: new Promise(() => {}),
      }),
      startReserved: () => true,
      hasReservation: () => true,
      cancelReservation: (tenant: string, cancelledRunId: string) => {
        cancellations.push([tenant, cancelledRunId]);
        return true;
      },
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });

    const response = await post(
      base,
      "/memory/maintenance?tenant=acme&mode=heavy&dryRun=1",
      {},
      TOKEN,
    );

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ error: "maintenance tenant unavailable" });
    expect(cancellations).toEqual([["acme", runId]]);
  });

  it("maps maintenance conflicts and disabled configuration to explicit statuses", async () => {
    const { pool } = recordingPool();
    const runId = createMaintenanceRunId();
    let outcome: "conflict" | "disabled" = "conflict";
    const maintenanceCoordinator = {
      tryReserve: () =>
        outcome === "conflict" ? { status: "conflict", runId } : { status: "disabled" },
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });
    const path = "/memory/maintenance?tenant=acme&mode=light&dryRun=0";

    const conflict = await post(base, path, {}, TOKEN);
    expect(conflict.status).toBe(409);
    expect(await conflict.json()).toEqual({
      error: "maintenance already running",
      run_id: runId,
    });

    outcome = "disabled";
    const disabled = await post(base, path, {}, TOKEN);
    expect(disabled.status).toBe(503);
    expect(await disabled.json()).toEqual({ error: "maintenance disabled" });
  });

  it("validates the maintenance query before starting a run", async () => {
    const { pool } = recordingPool();
    let starts = 0;
    const maintenanceCoordinator = {
      tryReserve: () => {
        starts += 1;
        return { status: "disabled" };
      },
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });

    expect((await post(base, "/memory/maintenance", {}, TOKEN)).status).toBe(400);
    expect(
      (await post(base, "/memory/maintenance?tenant=acme&mode=nope&dryRun=0", {}, TOKEN)).status,
    ).toBe(400);
    expect(
      (await post(base, "/memory/maintenance?tenant=acme&mode=light&dryRun=yes", {}, TOKEN)).status,
    ).toBe(400);
    expect(
      (
        await post(
          base,
          "/memory/maintenance?tenant=acme&tenant=other&mode=light&dryRun=0",
          {},
          TOKEN,
        )
      ).status,
    ).toBe(400);
    expect(starts).toBe(0);
  });

  it("returns coordinator status without opening a pooled being", async () => {
    const { pool, rec } = recordingPool();
    const runId = createMaintenanceRunId();
    const maintenanceCoordinator = {
      tryReserve: () => ({ status: "disabled" }),
      getStatus: (tenant: string) => ({
        current: { tenant, run_id: runId, state: "running" },
        last: null,
      }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });

    const response = await get(base, "/memory/maintenance/status?tenant=acme", TOKEN);
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      ok: true,
      tenant: "acme",
      current: { tenant: "acme", run_id: runId, state: "running" },
      last: null,
    });
    expect(rec.tenants).toEqual([]);
  });

  it("wraps maintenance audit listing and exclusive revert", async () => {
    const runId = createMaintenanceRunId();
    const calls: Array<{ tenant: string; exclusive: boolean | undefined }> = [];
    const auditCalls: unknown[] = [];
    const borg = {
      audit: {
        list: (options: unknown) => {
          auditCalls.push(["list", options]);
          return [{ id: 12, run_id: runId }];
        },
        revert: (auditId: number, revertedBy: string) => {
          auditCalls.push(["revert", auditId, revertedBy]);
          return Promise.resolve({ id: auditId, run_id: runId, reverted_by: revertedBy });
        },
      },
    } as unknown as Borg;
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant(tenant, fn, options) {
        calls.push({ tenant, exclusive: options?.exclusive });
        return fn(borg);
      },
    };
    const base = await start(pool);

    const listed = await get(base, `/memory/maintenance/audit?tenant=acme&run_id=${runId}`, TOKEN);
    expect(listed.status).toBe(200);
    expect(await listed.json()).toEqual({
      ok: true,
      tenant: "acme",
      run_id: runId,
      audit: [{ id: 12, run_id: runId }],
    });

    const reverted = await post(
      base,
      "/memory/maintenance/revert?tenant=acme&audit_id=12",
      {},
      TOKEN,
    );
    expect(reverted.status).toBe(200);
    expect(await reverted.json()).toEqual({
      ok: true,
      tenant: "acme",
      audit: { id: 12, run_id: runId, reverted_by: "memory-sidecar" },
    });
    expect(auditCalls).toEqual([
      ["list", { runId }],
      ["revert", 12, "memory-sidecar"],
    ]);
    expect(calls).toEqual([
      { tenant: "acme", exclusive: undefined },
      { tenant: "acme", exclusive: true },
    ]);
  });

  it("returns 404 for a missing maintenance audit and validates audit queries", async () => {
    const calls: Array<{ tenant: string; exclusive: boolean | undefined }> = [];
    const borg = {
      audit: {
        list: () => [],
        revert: () => Promise.resolve(null),
      },
    } as unknown as Borg;
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant(tenant, fn, options) {
        calls.push({ tenant, exclusive: options?.exclusive });
        return fn(borg);
      },
    };
    const base = await start(pool);

    expect(
      (await get(base, "/memory/maintenance/audit?tenant=acme&run_id=bad", TOKEN)).status,
    ).toBe(400);
    expect(
      (await post(base, "/memory/maintenance/revert?tenant=acme&audit_id=bad", {}, TOKEN)).status,
    ).toBe(400);
    const missing = await post(
      base,
      "/memory/maintenance/revert?tenant=acme&audit_id=99",
      {},
      TOKEN,
    );
    expect(missing.status).toBe(404);
    expect(await missing.json()).toEqual({ error: "audit record not found" });
    expect(calls).toEqual([{ tenant: "acme", exclusive: true }]);
  });

  it("recalls and maps episodes, routing by tenant, clamping the limit", async () => {
    const { pool, rec } = recordingPool();
    rec.episodeOverrides = { location: "AI Ninjas" };
    const base = await start(pool);
    const res = await post(
      base,
      "/memory/recall",
      { tenant: "acme", query: "who leads", limit: 999 },
      TOKEN,
    );
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: true,
      top_raw_score: 1.16,
      episodes: [
        {
          id: "ep_1",
          title: "Title",
          narrative: "Narrative",
          score: 0.91,
          raw_score: 1.16,
          location: "AI Ninjas",
          occurred_at: 10,
          participant_names: ["Ada"],
        },
      ],
    });
    expect(rec.tenants).toEqual(["acme"]);
    expect(rec.lastRecallLimit).toBe(50);
    expect(rec.lastRecallTraceTurnId).toBeUndefined();
    expect(rec.lastRecallOptions).not.toHaveProperty("recordRetrieval");
    expect(rec.retrievalRecords).toEqual([]);
  });

  it("skips entity lookups when recall returns no episodes", async () => {
    const entityAccesses: string[] = [];
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant(_tenantId, fn) {
        return fn({
          episodic: {
            search: async () => [],
          },
          entities: {
            get: () => {
              entityAccesses.push("get");
              return null;
            },
            getSelf: () => {
              entityAccesses.push("getSelf");
              return null;
            },
            list: () => {
              entityAccesses.push("list");
              return [];
            },
          },
        } as unknown as Borg);
      },
    };
    const base = await start(pool);
    const res = await post(base, "/memory/recall", { tenant: "acme", query: "q" }, TOKEN);

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: true,
      top_raw_score: null,
      episodes: [],
    });
    expect(entityAccesses).toEqual([]);
  });

  it("keeps recall response unchanged while passing a trace turn id when tracing is enabled", async () => {
    const { pool: offPool, rec: offRec } = recordingPool();
    const offBase = await start(offPool);
    const offRes = await post(
      offBase,
      "/memory/recall",
      { tenant: "acme", query: "who leads", limit: 3 },
      TOKEN,
    );
    const offBody = await offRes.json();

    const { pool: onPool, rec: onRec } = recordingPool();
    const onBase = await start(onPool, TOKEN, { traceRegistry: new MemoryTraceRegistry() });
    const onRes = await post(
      onBase,
      "/memory/recall",
      { tenant: "acme", query: "who leads", limit: 3 },
      TOKEN,
    );
    const onBody = await onRes.json();

    expect(onRes.status).toBe(200);
    expect(onBody).toEqual(offBody);
    expect(offRec.lastRecallTraceTurnId).toBeUndefined();
    expect(onRec.lastRecallTraceTurnId).toMatch(/^sidecar_recall:acme:/);
  });

  it("labels a partial recall as degraded so the caller can tell it from an empty memory", async () => {
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant(_tenantId, fn) {
        return fn({
          episodic: {
            search: async (
              _query: string,
              opts: { onDegraded?: (d: { subsystem: string; reason: string }) => void },
            ) => {
              opts.onDegraded?.({
                subsystem: "episodic_candidates",
                reason: "1/2 episodic intent lane(s) failed: EmbeddingError: stalled",
              });
              return [
                {
                  episode: testEpisode("ep_1" as Episode["id"]),
                  score: 0.91,
                  rawScore: 1.16,
                },
              ];
            },
          },
          entities: {
            get: () => null,
            getSelf: () => null,
          },
        } as unknown as Borg);
      },
    };
    const base = await start(pool);
    const res = await post(
      base,
      "/memory/recall",
      { tenant: "acme", query: "who leads", limit: 3 },
      TOKEN,
    );

    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      degraded?: boolean;
      degraded_reason?: string;
      episodes?: unknown[];
    };
    expect(body.degraded).toBe(true);
    expect(body.degraded_reason).toContain("episodic_candidates");
    expect(body.episodes).toHaveLength(1);
  });

  it("omits the degraded flag on a healthy recall", async () => {
    const { pool } = recordingPool();
    const base = await start(pool);
    const res = await post(
      base,
      "/memory/recall",
      { tenant: "acme", query: "who leads", limit: 3 },
      TOKEN,
    );

    const body = (await res.json()) as {
      degraded?: boolean;
      degraded_reason?: string;
      episodes?: unknown[];
    };
    expect(body.degraded).toBeUndefined();
    expect(body.degraded_reason).toBeUndefined();
  });

  it("answers within its deadline when the recall itself stalls", async () => {
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant(_tenantId, fn) {
        return fn({
          episodic: {
            // Never settles: the shape of a wedged provider call.
            search: () => new Promise(() => {}),
          },
        } as unknown as Borg);
      },
    };
    const base = await start(pool, TOKEN, { recallDeadlineMs: 25 });
    const res = await post(
      base,
      "/memory/recall",
      { tenant: "acme", query: "who leads", limit: 3 },
      TOKEN,
    );

    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      degraded?: boolean;
      degraded_reason?: string;
      episodes?: unknown[];
    };
    expect(body.degraded).toBe(true);
    expect(body.degraded_reason).toContain("deadline");
    expect(body.episodes).toEqual([]);
  });

  it("answers an embedding stall as a degradation instead of a 500", async () => {
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant(_tenantId, fn) {
        return fn({
          episodic: {
            search: async () => {
              throw new EmbeddingError("Embedding call stalled: 2 attempt(s) exceeded 1000ms each");
            },
          },
        } as unknown as Borg);
      },
    };
    const base = await start(pool);
    const res = await post(
      base,
      "/memory/recall",
      { tenant: "acme", query: "who leads", limit: 3 },
      TOKEN,
    );

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: true,
      episodes: [],
      degraded: true,
      degraded_reason: "embeddings: Embedding call stalled: 2 attempt(s) exceeded 1000ms each",
    });
  });

  it("abstains only when the configured threshold is above the top raw score", async () => {
    const { pool } = recordingPool();
    const abstainingBase = await start(pool, TOKEN, { recallAbstainThreshold: 1.17 });
    const abstained = await post(
      abstainingBase,
      "/memory/recall",
      { tenant: "acme", query: "who leads", limit: 3 },
      TOKEN,
    );

    expect(abstained.status).toBe(200);
    expect(await abstained.json()).toEqual({
      ok: true,
      episodes: [],
      abstained: true,
      abstain_reason: "low_relevance",
      top_raw_score: 1.16,
    });

    const exactThresholdBase = await start(recordingPool().pool, TOKEN, {
      recallAbstainThreshold: 1.16,
    });
    const retained = await post(
      exactThresholdBase,
      "/memory/recall",
      { tenant: "acme", query: "who leads", limit: 3 },
      TOKEN,
    );

    expect(await retained.json()).toEqual({
      ok: true,
      top_raw_score: 1.16,
      episodes: [
        {
          id: "ep_1",
          title: "Title",
          narrative: "Narrative",
          score: 0.91,
          raw_score: 1.16,
          location: null,
          occurred_at: 10,
          participant_names: ["Ada"],
        },
      ],
    });
  });

  it("lists episodes from the query tenant without a body, clamping limit and passing cursor", async () => {
    const { pool, rec } = recordingPool();
    rec.episodeOverrides = { location: "AI Ninjas" };
    const base = await start(pool);
    const res = await get(
      base,
      "/memory/episodes?tenant=acme&limit=999&cursor=opaque-cursor",
      TOKEN,
    );

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: true,
      episodes: [
        {
          id: "ep_aaaaaaaaaaaaaaaa",
          title: "Title",
          narrative: "Narrative",
          significance: 0.72,
          tags: ["planning", "admin"],
          source_stream_ids: ["strm_aaaaaaaaaaaaaaaa"],
          location: "AI Ninjas",
          occurred_at: 10,
          participant_names: ["Ada"],
        },
      ],
      nextCursor: "next-cursor",
    });
    expect(rec.tenants).toEqual(["acme"]);
    expect(rec.exclusives).toEqual([undefined]);
    expect(rec.lastListOptions).toEqual({ limit: 100, cursor: "opaque-cursor" });
  });

  it("returns a disabled empty trace response without opening a tenant", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const res = await get(base, "/memory/trace?tenant=acme", TOKEN);

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: true,
      tenant: "acme",
      events: [],
      disabled: true,
    });
    expect(rec.tenants).toEqual([]);
  });

  it("returns enabled trace events filtered by since without opening a tenant", async () => {
    let now = 5_000;
    const traceRegistry = new MemoryTraceRegistry({
      capacity: 5,
      now: () => now,
    });
    const tracer = traceRegistry.tracerFor("acme");
    tracer.emit("retrieval.started", {
      turnId: "turn_trace_1",
      query: "first",
    });
    now = 5_100;
    tracer.emit("retrieval.completed", {
      turnId: "turn_trace_2",
      episodeCount: 1,
      semanticHits: 0,
    });
    const since = traceRegistry.query("acme", 0).events[0]!.ts;
    const { pool, rec } = recordingPool();
    const base = await start(pool, TOKEN, { traceRegistry });
    const res = await get(base, `/memory/trace?tenant=acme&since=${since}`, TOKEN);

    expect(res.status).toBe(200);
    expect(await res.json()).toMatchObject({
      ok: true,
      tenant: "acme",
      events: [
        expect.objectContaining({
          turnId: "turn_trace_2",
          event: "retrieval.completed",
        }),
      ],
      nextSince: expect.any(Number),
      truncated: false,
    });
    expect(rec.tenants).toEqual([]);
  });

  it("rejects invalid trace tenant or since before touching the pool", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool, TOKEN, { traceRegistry: new MemoryTraceRegistry() });

    expect((await get(base, "/memory/trace", TOKEN)).status).toBe(400);
    expect((await get(base, "/memory/trace?tenant=UPPER", TOKEN)).status).toBe(400);
    expect((await get(base, "/memory/trace?tenant=acme&since=nope", TOKEN)).status).toBe(400);
    expect(rec.tenants).toEqual([]);
  });

  it("treats empty or whitespace list cursors as absent", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);

    expect((await get(base, "/memory/episodes?tenant=acme&cursor=", TOKEN)).status).toBe(200);
    expect(rec.lastListOptions).toEqual({ limit: 20 });

    expect((await get(base, "/memory/episodes?tenant=acme&cursor=%20%20", TOKEN)).status).toBe(200);
    expect(rec.lastListOptions).toEqual({ limit: 20 });
  });

  it("maps malformed list cursors to 400 client errors", async () => {
    const calls: string[] = [];
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant(tenantId, fn) {
        calls.push(tenantId);
        const invalidCursor = Object.assign(new Error("Invalid episode cursor"), {
          code: "EPISODE_CURSOR_INVALID",
        });
        return fn({
          episodic: {
            list: async () => {
              throw invalidCursor;
            },
          },
        } as unknown as Borg);
      },
    };
    const base = await start(pool);
    const res = await get(base, "/memory/episodes?tenant=acme&cursor=not-a-cursor", TOKEN);

    expect(res.status).toBe(400);
    expect(await res.json()).toEqual({ error: "invalid 'cursor'" });
    expect(calls).toEqual(["acme"]);
  });

  it("rejects missing or invalid query tenant for episode GET routes before touching the pool", async () => {
    const calls: string[] = [];
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant(tenantId, fn) {
        calls.push(tenantId);
        return fn({} as Borg);
      },
    };
    const base = await start(pool);

    expect((await get(base, "/memory/episodes", TOKEN)).status).toBe(400);
    expect((await get(base, "/memory/episodes?tenant=UPPER", TOKEN)).status).toBe(400);
    expect(
      (await get(base, "/memory/episodes/ep_aaaaaaaaaaaaaaaa?tenant=../evil", TOKEN)).status,
    ).toBe(400);
    expect(calls).toEqual([]);
  });

  it("inspects one episode by query tenant and excludes the embedding vector", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const res = await get(base, "/memory/episodes/ep_aaaaaaaaaaaaaaaa?tenant=acme", TOKEN);

    expect(res.status).toBe(200);
    const body = (await res.json()) as { episode: Record<string, unknown> };
    expect(body).toMatchObject({
      ok: true,
      episode: {
        id: "ep_aaaaaaaaaaaaaaaa",
        title: "Title",
        narrative: "Narrative",
        participants: ["Ada"],
        participant_names: ["Ada"],
        occurred_at: 10,
        source_stream_ids: ["strm_aaaaaaaaaaaaaaaa"],
        significance: 0.72,
        tags: ["planning", "admin"],
      },
    });
    expect(body.episode.created_at).toBe(1);
    expect(body.episode.participants).toEqual(["Ada"]);
    expect(body.episode).not.toHaveProperty("embedding");
    expect(rec.tenants).toEqual(["acme"]);
    expect(rec.exclusives).toEqual([undefined]);
    expect(rec.inspectIds).toEqual(["ep_aaaaaaaaaaaaaaaa"]);
  });

  it("adds event time and canonical participant names without changing inspect internals", async () => {
    const { pool, rec } = recordingPool();
    const selfEntity = rec.entities.find((entity) => entity.kind === "self");
    const participantEntityId = createEntityId();
    const unresolvedEntityId = createEntityId();

    expect(selfEntity).toBeDefined();
    Object.assign(selfEntity!, {
      canonical_name: "Current Borg",
      aliases: ["self", "Former Borg"],
    });
    rec.entities.push({
      id: participantEntityId,
      canonical_name: "Ada Lovelace",
      aliases: ["Ada"],
      kind: "person",
      borg_role: null,
      name_provenance: "transport_sender",
      created_at: 1,
    });
    const rawParticipants = [
      "Former Borg",
      episodeParticipantEntityIdTerm(participantEntityId),
      "Ada Lovelace",
      episodeParticipantEntityIdTerm(unresolvedEntityId),
      episodeParticipantEntityIdTerm(participantEntityId),
      "Current Borg",
    ];
    rec.episodeOverrides = {
      participants: rawParticipants,
      start_time: 1_725_000_123_456,
    };
    const base = await start(pool);
    const recall = await post(base, "/memory/recall", { tenant: "acme", query: "q" }, TOKEN);
    const listed = await get(base, "/memory/episodes?tenant=acme", TOKEN);
    const inspected = await get(base, "/memory/episodes/ep_aaaaaaaaaaaaaaaa?tenant=acme", TOKEN);
    const recallBody = (await recall.json()) as { episodes: Array<Record<string, unknown>> };
    const listBody = (await listed.json()) as { episodes: Array<Record<string, unknown>> };
    const inspectBody = (await inspected.json()) as { episode: Record<string, unknown> };

    for (const episode of [recallBody.episodes[0], listBody.episodes[0], inspectBody.episode]) {
      expect(episode).toMatchObject({
        occurred_at: 1_725_000_123_456,
        participant_names: ["Current Borg", "Ada Lovelace"],
      });
    }
    expect(inspectBody.episode.created_at).toBe(1);
    expect(inspectBody.episode.participants).toEqual(rawParticipants);
    expect(rec.entityListCalls).toBe(0);
    expect(rec.selfEntityGetCalls).toBe(3);
    expect(rec.entityGetIds).toEqual([
      participantEntityId,
      unresolvedEntityId,
      participantEntityId,
      unresolvedEntityId,
      participantEntityId,
      unresolvedEntityId,
    ]);
  });

  it("resolves bare participant entity ids and drops unresolved bare ids", async () => {
    const { pool, rec } = recordingPool();
    const participantEntityId = createEntityId();
    const unresolvedEntityId = createEntityId();
    rec.entities.push({
      id: participantEntityId,
      canonical_name: "Marcin Kowal",
      aliases: ["Kowal, Marcin"],
      kind: "person",
      borg_role: null,
      name_provenance: "transport_sender",
      created_at: 1,
    });
    rec.episodeOverrides = {
      participants: [participantEntityId, unresolvedEntityId],
    };
    const base = await start(pool);
    const response = await post(base, "/memory/recall", { tenant: "acme", query: "q" }, TOKEN);
    const body = (await response.json()) as {
      episodes: Array<{ participant_names: string[] }>;
    };

    expect(response.status).toBe(200);
    expect(body.episodes[0]?.participant_names).toEqual(["Marcin Kowal"]);
    expect(rec.entityGetIds).toEqual([participantEntityId, unresolvedEntityId]);
  });

  it("404s an unknown episode id and 400s an invalid episode id without touching the pool", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const missing = await get(base, "/memory/episodes/ep_missingmissing00?tenant=acme", TOKEN);

    expect(missing.status).toBe(404);
    expect(await missing.json()).toEqual({ ok: false });
    expect(rec.tenants).toEqual(["acme"]);
    expect(rec.inspectIds).toEqual(["ep_missingmissing00"]);

    const invalidCalls: string[] = [];
    const invalidPool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant(tenantId, fn) {
        invalidCalls.push(tenantId);
        return fn({} as Borg);
      },
    };
    const invalidBase = await start(invalidPool);
    expect(
      (await get(invalidBase, "/memory/episodes/not-an-episode?tenant=acme", TOKEN)).status,
    ).toBe(400);
    expect(invalidCalls).toEqual([]);
  });

  it("lists applicable active commitments with critical-first ordering and a bounded response", async () => {
    const { pool, rec } = recordingPool();
    const audienceId = createEntityId();
    const otherAudienceId = createEntityId();
    rec.externalSenderIds.set("known-audience", audienceId);
    rec.commitments.push(
      testCommitment({
        directive_family: "global_advisory",
        directive: "Global advisory",
        priority: 100,
        enforcement_class: "advisory",
      }),
      testCommitment({
        type: "boundary",
        kind: "boundary",
        directive_family: "global_critical",
        directive: "Global critical",
        priority: 1,
        enforcement_class: "critical",
        critical_domain: "privacy",
      }),
      testCommitment({
        type: "boundary",
        kind: "audience_rule",
        directive_family: "audience_critical",
        directive: "Audience critical",
        priority: 9,
        enforcement_class: "critical",
        critical_domain: "audience_scope",
        restricted_audience: audienceId,
      }),
      testCommitment({
        type: "boundary",
        kind: "audience_rule",
        directive_family: "other_audience",
        directive: "Other audience only",
        priority: 99,
        enforcement_class: "critical",
        critical_domain: "audience_scope",
        restricted_audience: otherAudienceId,
      }),
      testCommitment({
        directive_family: "retired",
        directive: "Retired",
        revoked_at: 200,
        revoked_reason: "done",
        revoke_provenance: { kind: "manual" },
      }),
    );
    const base = await start(pool);
    const response = await get(
      base,
      `/memory/commitments?tenant=acme&audience=${audienceId}`,
      TOKEN,
    );

    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      audience_entity_id: string;
      commitments: Array<Record<string, unknown>>;
      truncated: boolean;
    };
    expect(body.audience_entity_id).toBe(audienceId);
    expect(body.commitments.map((commitment) => commitment.family)).toEqual([
      "audience_critical",
      "global_critical",
      "global_advisory",
    ]);
    expect(body.commitments[0]).toMatchObject({
      type: "boundary",
      kind: "audience_rule",
      enforcement_class: "critical",
      critical_domain: "audience_scope",
      directive: "Audience critical",
      family: "audience_critical",
      priority: 9,
      audience_entity_id: audienceId,
    });
    expect(body.truncated).toBe(false);

    rec.commitments = Array.from({ length: 101 }, (_, index) =>
      testCommitment({
        directive_family: `bounded_${index}`,
        directive: `Bounded ${index}`,
        priority: index,
        created_at: index,
        last_reinforced_at: index,
      }),
    );
    const bounded = await get(base, "/memory/commitments?tenant=acme", TOKEN);
    const boundedBody = (await bounded.json()) as {
      commitments: unknown[];
      truncated: boolean;
    };
    expect(boundedBody.commitments).toHaveLength(100);
    expect(boundedBody.truncated).toBe(true);
  });

  it("resolves commitment audience scope from the append-turn sender external id", async () => {
    const { pool, rec } = recordingPool();
    const audienceId = createEntityId();
    const otherAudienceId = createEntityId();
    const externalId = "platform/user 42";
    rec.externalSenderIds.set(externalId, audienceId);
    rec.commitments.push(
      testCommitment({
        directive_family: "tenant_wide",
        directive: "Tenant-wide rule",
        priority: 10,
      }),
      testCommitment({
        directive_family: "sender_scoped",
        directive: "Sender-scoped rule",
        restricted_audience: audienceId,
      }),
      testCommitment({
        directive_family: "other_sender",
        directive: "Other sender rule",
        restricted_audience: otherAudienceId,
      }),
    );
    const base = await start(pool);
    const response = await get(
      base,
      `/memory/commitments?tenant=acme&audience_external_id=${encodeURIComponent(externalId)}`,
      TOKEN,
    );

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      ok: true,
      tenant: "acme",
      audience_entity_id: audienceId,
      audience_external_id: externalId,
      audience_resolved: true,
      commitments: [
        expect.objectContaining({ family: "tenant_wide" }),
        expect.objectContaining({ family: "sender_scoped" }),
      ],
      truncated: false,
    });
    expect(rec.lookedUpExternalSenders).toEqual([{ source: "team-agent.sender", externalId }]);
    expect(rec.resolvedExternalSenders).toEqual([]);
  });

  it("returns only tenant-wide commitments for an unknown external audience id", async () => {
    const { pool, rec } = recordingPool();
    rec.commitments.push(
      testCommitment({
        directive_family: "tenant_wide",
        directive: "Tenant-wide rule",
      }),
      testCommitment({
        directive_family: "known_sender_only",
        directive: "Known sender rule",
        restricted_audience: createEntityId(),
      }),
    );
    const base = await start(pool);
    const response = await get(
      base,
      "/memory/commitments?tenant=acme&audience_external_id=not-seen-yet",
      TOKEN,
    );

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      ok: true,
      tenant: "acme",
      audience_entity_id: null,
      audience_external_id: "not-seen-yet",
      audience_resolved: false,
      commitments: [expect.objectContaining({ family: "tenant_wide" })],
      truncated: false,
    });
    expect(rec.lookedUpExternalSenders).toEqual([
      { source: "team-agent.sender", externalId: "not-seen-yet" },
    ]);
  });

  it("rejects internal and external commitment audiences supplied together", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const response = await get(
      base,
      `/memory/commitments?tenant=acme&audience=${createEntityId()}&audience_external_id=sender-1`,
      TOKEN,
    );

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({
      error: "'audience' and 'audience_external_id' are mutually exclusive",
    });
    expect(rec.tenants).toEqual([]);
    expect(rec.lookedUpExternalSenders).toEqual([]);
  });

  it("strictly validates and creates an operator-set audience commitment", async () => {
    const { pool, rec } = recordingPool();
    const audienceId = createEntityId();
    rec.externalSenderIds.set("known-audience", audienceId);
    const base = await start(pool);
    const payload = {
      tenant: "acme",
      type: "boundary",
      kind: "audience_rule",
      enforcement_class: "critical",
      critical_domain: "privacy",
      directive: "Never disclose private launch notes.",
      family: "Private Launch Notes",
      priority: 12,
      audience_entity_id: audienceId,
    };
    const response = await post(base, "/memory/commitments", payload, TOKEN);

    expect(response.status).toBe(201);
    expect(await response.json()).toMatchObject({
      ok: true,
      commitment: {
        type: "boundary",
        kind: "audience_rule",
        enforcement_class: "critical",
        critical_domain: "privacy",
        directive: "Never disclose private launch notes.",
        family: "private_launch_notes",
        priority: 12,
        audience_entity_id: audienceId,
      },
    });
    expect(rec.commitmentAdds).toHaveLength(1);
    expect(rec.exclusives).toEqual([true]);

    const unknownField = await post(
      base,
      "/memory/commitments",
      { ...payload, unexpected: true },
      TOKEN,
    );
    expect(unknownField.status).toBe(400);
    expect(await unknownField.json()).toEqual({ error: "invalid commitment body" });

    const invalidClassification = await post(
      base,
      "/memory/commitments",
      {
        ...payload,
        type: "rule",
        kind: "process_norm",
        critical_domain: "safety",
      },
      TOKEN,
    );
    expect(invalidClassification.status).toBe(400);
  });

  it("retires an active commitment and rejects invalid or inactive ids", async () => {
    const { pool, rec } = recordingPool();
    const commitment = testCommitment();
    rec.commitments.push(commitment);
    const base = await start(pool);

    expect(
      (await del(base, "/memory/commitments?tenant=acme&id=not-a-commitment", TOKEN)).status,
    ).toBe(400);

    const retired = await del(base, `/memory/commitments?tenant=acme&id=${commitment.id}`, TOKEN);
    expect(retired.status).toBe(200);
    expect(await retired.json()).toMatchObject({
      ok: true,
      commitment: {
        id: commitment.id,
      },
    });
    expect(rec.commitments[0]?.revoked_reason).toBe("retired_by_operator");
    expect(rec.exclusives.at(-1)).toBe(true);

    expect(
      (await del(base, `/memory/commitments?tenant=acme&id=${commitment.id}`, TOKEN)).status,
    ).toBe(409);
  });

  it("creates, lists, and revokes operator directives with admin stream provenance", async () => {
    const { pool, rec } = recordingPool();
    const person = createEntityId();
    const group = createEntityId();
    rec.entities.push(
      {
        id: person,
        canonical_name: "Alice",
        aliases: [],
        kind: "person",
        borg_role: null,
        name_provenance: "transport_sender",
        created_at: 1,
      },
      {
        id: group,
        canonical_name: "AI Ninjas",
        aliases: [],
        kind: "group",
        borg_role: null,
        name_provenance: "transport_audience_label",
        created_at: 1,
      },
    );
    rec.externalEntityIds.set("team-agent.sender\u0000alice", person);
    rec.externalEntityIds.set("team-agent.conversation\u0000ninjas", group);
    const base = await start(pool);
    const created = await post(
      base,
      "/memory/directives",
      {
        tenant: "acme",
        kind: "response_policy",
        text: "Keep operational replies concise.",
        content_scope: "allow_list",
        allowed_external_ids: ["alice"],
        allowed_group_external_ids: ["ninjas"],
        topic_tags: ["operations"],
      },
      TOKEN,
    );
    const createdBody = (await created.json()) as {
      directive: { id: CreatorDirective["id"] };
    };

    expect(created.status).toBe(201);
    expect(createdBody.directive).toMatchObject({
      kind: "response_policy",
      text: "Keep operational replies concise.",
      content_scope: "allow_list",
      priority: 0,
      topic_tags: ["operations"],
    });
    expect(rec.directiveQueueInputs).toEqual([
      expect.objectContaining({
        operationalDirective: "Keep operational replies concise.",
        canonicalFact: null,
        subjectKind: "system",
        subjectEntityId: null,
        disclosurePolicy: expect.objectContaining({
          content_scope: "allow_list",
          allowed_entity_ids: [person, group],
          excluded_entity_ids: [],
          mention_policy: "answer_if_asked",
          denied_audience_behavior: "omit",
          boundary_prompt: "Keep operational replies concise.",
        }),
        activationPolicy: {
          scope: "same_as_disclosure",
          allowed_entity_ids: [],
          excluded_entity_ids: [],
        },
      }),
    ]);
    expect(rec.sessionEnsures).toEqual([
      expect.objectContaining({
        source_type: "memory_sidecar",
        audience_role: "operator",
      }),
    ]);
    expect(rec.appendCalls).toEqual([
      expect.objectContaining({
        input: expect.objectContaining({
          kind: "internal_event",
          content: expect.objectContaining({
            event: "memory_sidecar.operator_directive_queue_requested",
          }),
        }),
      }),
    ]);
    expect(rec.exclusives).toEqual([true]);

    const listed = await get(base, "/memory/directives?tenant=acme", TOKEN);
    expect(listed.status).toBe(200);
    expect(await listed.json()).toMatchObject({
      ok: true,
      directives: [{ id: createdBody.directive.id }],
    });

    const revoked = await requestRaw(
      base,
      `/memory/directives/${createdBody.directive.id}?tenant=acme`,
      {
        method: "DELETE",
        token: TOKEN,
        body: { reason: "replaced" },
      },
    );
    expect(revoked.status).toBe(200);
    expect(revoked.body).toMatchObject({
      ok: true,
      directive: { id: createdBody.directive.id, status: "revoked" },
    });
    expect(rec.appendCalls.at(-1)).toEqual(
      expect.objectContaining({
        input: expect.objectContaining({
          kind: "internal_event",
          content: {
            event: "memory_sidecar.operator_directive_revoke_requested",
            directive_id: createdBody.directive.id,
            reason: "replaced",
          },
        }),
      }),
    );
  });

  it("records a linked failure event when directive queueing fails", async () => {
    const { pool, rec } = recordingPool();
    rec.directiveQueueError = Object.assign(new Error("injected queue failure"), {
      code: "DIRECTIVE_QUEUE_FAILED",
    });
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);

    try {
      const base = await start(pool);
      const response = await post(
        base,
        "/memory/directives",
        {
          tenant: "acme",
          kind: "response_policy",
          text: "A rule that fails to queue.",
          content_scope: "public",
        },
        TOKEN,
      );

      expect(response.status).toBe(500);
      expect(await response.json()).toEqual({ error: "internal error" });
      const provenanceEntryId = "strm_bbbbbbbbbbbbbbbb";
      expect(rec.appendCalls.map((call) => (call.input as { content: unknown }).content)).toEqual([
        expect.objectContaining({
          event: "memory_sidecar.operator_directive_queue_requested",
        }),
        expect.objectContaining({
          event: "memory_sidecar.operator_directive_queue_failed",
          provenance_stream_entry_id: provenanceEntryId,
          failure_code: "DIRECTIVE_QUEUE_FAILED",
        }),
      ]);
      expect(rec.creatorDirectives).toEqual([]);
      expect(rec.sessionTouches).toHaveLength(2);
    } finally {
      consoleError.mockRestore();
    }
  });

  it("records a linked failure event when directive revocation fails", async () => {
    const { pool, rec } = recordingPool();
    const directive = testDirective();
    rec.creatorDirectives.push(directive);
    rec.directiveRevokeError = Object.assign(new Error("injected revoke failure"), {
      code: "DIRECTIVE_REVOKE_FAILED",
    });
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);

    try {
      const base = await start(pool);
      const response = await requestRaw(base, `/memory/directives/${directive.id}?tenant=acme`, {
        method: "DELETE",
        token: TOKEN,
        body: { reason: "bad replacement" },
      });

      expect(response.status).toBe(500);
      expect(response.body).toEqual({ error: "internal error" });
      expect(rec.appendCalls.map((call) => (call.input as { content: unknown }).content)).toEqual([
        {
          event: "memory_sidecar.operator_directive_revoke_requested",
          directive_id: directive.id,
          reason: "bad replacement",
        },
        {
          event: "memory_sidecar.operator_directive_revoke_failed",
          directive_id: directive.id,
          reason: "bad replacement",
          provenance_stream_entry_id: "strm_bbbbbbbbbbbbbbbb",
          failure_code: "DIRECTIVE_REVOKE_FAILED",
        },
      ]);
      expect(rec.creatorDirectives[0]?.status).toBe("active");
      expect(rec.sessionTouches).toHaveLength(2);
    } finally {
      consoleError.mockRestore();
    }
  });

  it("fails closed for unknown or contradictory directive external ids", async () => {
    const { pool, rec } = recordingPool();
    const person = createEntityId();
    rec.entities.push({
      id: person,
      canonical_name: "Alice",
      aliases: [],
      kind: "person",
      borg_role: null,
      name_provenance: "transport_sender",
      created_at: 1,
    });
    rec.externalEntityIds.set("team-agent.sender\u0000alice", person);
    const base = await start(pool);
    const unknown = await post(
      base,
      "/memory/directives",
      {
        tenant: "acme",
        kind: "response_policy",
        text: "Unknown target",
        content_scope: "allow_list",
        allowed_external_ids: ["unknown"],
      },
      TOKEN,
    );
    const contradictory = await post(
      base,
      "/memory/directives",
      {
        tenant: "acme",
        kind: "response_policy",
        text: "Contradictory target",
        content_scope: "allow_list",
        allowed_external_ids: ["alice"],
        excluded_external_ids: ["alice"],
      },
      TOKEN,
    );

    expect(unknown.status).toBe(400);
    expect(await unknown.json()).toEqual({ error: "unknown directive external id" });
    expect(contradictory.status).toBe(400);
    expect(await contradictory.json()).toEqual({
      error: "ambiguous directive external ids",
    });
    expect(rec.appendCalls).toEqual([]);
    expect(rec.directiveQueueInputs).toEqual([]);
  });

  it("maps subject-fact and self-identity directive subjects to deterministic defaults", async () => {
    const { pool, rec } = recordingPool();
    const alice = createEntityId();
    rec.entities.push({
      id: alice,
      canonical_name: "Alice",
      aliases: [],
      kind: "person",
      borg_role: null,
      name_provenance: "transport_sender",
      created_at: 1,
    });
    rec.externalEntityIds.set("team-agent.sender\u0000alice", alice);
    const base = await start(pool);

    const missingSubject = await post(
      base,
      "/memory/directives",
      {
        tenant: "acme",
        kind: "subject_fact",
        text: "Alice owns the launch decision.",
        content_scope: "subject_only",
      },
      TOKEN,
    );
    expect(missingSubject.status).toBe(400);

    const subjectFact = await post(
      base,
      "/memory/directives",
      {
        tenant: "acme",
        kind: "subject_fact",
        text: "Alice owns the launch decision.",
        content_scope: "subject_only",
        subject_external_id: "alice",
      },
      TOKEN,
    );
    const selfIdentity = await post(
      base,
      "/memory/directives",
      {
        tenant: "acme",
        kind: "self_identity",
        text: "The agent is the tenant's operations partner.",
        content_scope: "public",
      },
      TOKEN,
    );

    expect(subjectFact.status).toBe(201);
    expect(selfIdentity.status).toBe(201);
    expect(rec.directiveQueueInputs).toEqual([
      expect.objectContaining({
        subjectKind: "entity",
        subjectEntityId: alice,
        canonicalFact: "Alice owns the launch decision.",
        operationalDirective: null,
        disclosurePolicy: expect.objectContaining({
          content_scope: "subject_only",
          subject_may_know: true,
        }),
      }),
      expect.objectContaining({
        subjectKind: "borg_self",
        subjectEntityId: null,
        canonicalFact: "The agent is the tenant's operations partner.",
        operationalDirective: null,
      }),
    ]);
  });

  it("validates commitment audience ids before opening a tenant", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);

    expect((await get(base, "/memory/commitments?tenant=acme&audience=bad", TOKEN)).status).toBe(
      400,
    );
    expect(rec.tenants).toEqual([]);
  });

  it("remembers (append + extract), routing by tenant", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const res = await post(
      base,
      "/memory/remember",
      { tenant: "acme", content: "fact", author: "Bob" },
      TOKEN,
    );
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: true,
      extracted: { inserted: 1, updated: 0, skipped: 0 },
    });
    expect(rec.tenants).toEqual(["acme"]);
    expect(rec.extractOptions).toEqual([
      {
        sinceTs: 1000,
        bypassSalienceGate: true,
      },
    ]);
  });

  it("appends a raw turn and schedules background ingestion", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const rawSession = "tenant::user::conversation";
    const expectedSession = `sess_${createHash("sha256").update(rawSession).digest("hex").slice(0, 16)}`;
    const res = await post(
      base,
      "/memory/append-turn",
      {
        tenant: "acme",
        session: rawSession,
        user: "hello",
        assistant: "hi there",
      },
      TOKEN,
    );

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: true,
      session: expectedSession,
      entries: [
        { id: "strm_aaaaaaaaaaaaaaa0", kind: "user_msg" },
        { id: "strm_aaaaaaaaaaaaaaa1", kind: "agent_msg" },
      ],
    });
    expect(rec.appendMany).toEqual({
      session: expectedSession,
      inputs: [
        { kind: "user_msg", content: "hello" },
        { kind: "agent_msg", content: "hi there" },
      ],
    });
    expect(JSON.stringify(rec.appendMany?.inputs)).toBe(
      '[{"kind":"user_msg","content":"hello"},{"kind":"agent_msg","content":"hi there"}]',
    );
    expect(rec.tenants).toEqual(["acme", "acme"]);
    expect(rec.exclusives).toEqual([true, undefined]);
    expect(rec.ingestSessions).toEqual([expectedSession]);
  });

  it("rejects append-turn when both user and assistant are absent", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const response = await post(
      base,
      "/memory/append-turn",
      { tenant: "acme", session: "empty-turn" },
      TOKEN,
    );

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: "missing 'user' or 'assistant'" });
    expect(rec.tenants).toEqual([]);
  });

  it.each([
    {
      conversation: { type: "personal", name: "Alice" },
      persisted: { type: "personal", name: "Alice" },
    },
    {
      conversation: { type: "groupChat", name: "AI Ninjas" },
      persisted: { type: "groupChat", name: "AI Ninjas" },
    },
    {
      conversation: { type: "channel", name: "   " },
      persisted: { type: "channel", name: "" },
    },
  ])("persists a $conversation.type conversation on both turn entries", async (testCase) => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const response = await post(
      base,
      "/memory/append-turn",
      {
        tenant: "acme",
        session: "room",
        user: "hello",
        assistant: "hi",
        conversation: testCase.conversation,
      },
      TOKEN,
    );

    expect(response.status).toBe(200);
    await response.json();
    expect(rec.appendMany?.inputs).toEqual([
      { kind: "user_msg", content: "hello", conversation: testCase.persisted },
      { kind: "agent_msg", content: "hi", conversation: testCase.persisted },
    ]);
  });

  it("resolves optional sender identities and stamps only their user stream entries", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);

    for (const sender of [
      { external_id: "platform-alice", display_name: "Alice Nowak" },
      { external_id: "platform-bob", display_name: "Bob Chen" },
    ]) {
      const response = await post(
        base,
        "/memory/append-turn",
        {
          tenant: "acme",
          session: "shared-room",
          user: `message from ${sender.display_name}`,
          assistant: "acknowledged",
          sender,
        },
        TOKEN,
      );

      expect(response.status).toBe(200);
      await response.json();
    }

    const aliceId = rec.externalSenderIds.get("platform-alice");
    const bobId = rec.externalSenderIds.get("platform-bob");

    expect(aliceId).toBeDefined();
    expect(bobId).toBeDefined();
    expect(aliceId).not.toBe(bobId);
    expect(rec.resolvedExternalSenders).toEqual([
      {
        source: "team-agent.sender",
        externalId: "platform-alice",
        canonicalName: "Alice Nowak",
        kind: "person",
        provenance: "transport_sender",
      },
      {
        source: "team-agent.sender",
        externalId: "platform-bob",
        canonicalName: "Bob Chen",
        kind: "person",
        provenance: "transport_sender",
      },
    ]);
    expect(rec.appendManyCalls).toHaveLength(2);
    expect(rec.appendManyCalls[0]?.inputs).toEqual([
      { kind: "user_msg", content: "message from Alice Nowak", sender_entity_id: aliceId },
      { kind: "agent_msg", content: "acknowledged" },
    ]);
    expect(rec.appendManyCalls[1]?.inputs).toEqual([
      { kind: "user_msg", content: "message from Bob Chen", sender_entity_id: bobId },
      { kind: "agent_msg", content: "acknowledged" },
    ]);
  });

  it("enriches a personal append with session, audience, operator role, and activity", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const response = await post(
      base,
      "/memory/append-turn",
      {
        tenant: "acme",
        session: "tenant::alice::chat",
        user: "hello",
        assistant: "hi",
        sender: {
          external_id: "alice-external",
          display_name: "Alice",
          operator: true,
        },
        conversation: { type: "personal", name: "Alice" },
      },
      TOKEN,
    );

    expect(response.status).toBe(200);
    await response.json();
    const alice = rec.externalSenderIds.get("alice-external");
    const session = `sess_${createHash("sha256")
      .update("tenant::alice::chat")
      .digest("hex")
      .slice(0, 16)}`;

    expect(alice).toBeDefined();
    expect(rec.sessionEnsures).toEqual([
      expect.objectContaining({
        session_id: session,
        source_type: "team_agent",
        source_external_id: "tenant::alice::chat",
        audience_label: "Alice",
        audience_entity_id: alice,
        conversation_kind: "dm",
        audience_role: "operator",
      }),
    ]);
    expect(rec.appendMany?.inputs).toEqual([
      {
        kind: "user_msg",
        content: "hello",
        audience: alice,
        sender_entity_id: alice,
        conversation: { type: "personal", name: "Alice" },
      },
      {
        kind: "agent_msg",
        content: "hi",
        audience: alice,
        conversation: { type: "personal", name: "Alice" },
      },
    ]);
    expect(rec.activityRecords).toHaveLength(2);
    expect(rec.activityRecords[0]).toMatchObject({
      kind: "user_contact",
      speakerEntityId: alice,
      actorEntityId: alice,
      audienceEntityId: alice,
    });
    expect(rec.activityRecords[1]).toMatchObject({
      kind: "borg_replied",
      speakerEntityId: rec.entities[0]?.id,
      actorEntityId: rec.entities[0]?.id,
      audienceEntityId: alice,
    });
    expect(rec.sessionTouches).toEqual([
      {
        sessionId: session,
        update: { at: 1001, messageCountDelta: 1 },
      },
    ]);
  });

  it("records observation time as metadata and projects activity at append time", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const observedAt = Date.now() - 5_000;
    const response = await post(
      base,
      "/memory/append-turn",
      {
        tenant: "acme",
        session: "tenant::group::observed",
        user: "Alice told Bob about the launch.",
        observed_at: observedAt,
        sender: { external_id: "alice", display_name: "Alice" },
        conversation: {
          type: "groupChat",
          name: "AI Ninjas",
          external_id: "group-42",
        },
      },
      TOKEN,
    );
    const payload = (await response.json()) as {
      entries: Array<{ id: string; kind: string }>;
      session: string;
    };
    const alice = rec.externalSenderIds.get("alice");
    const group = rec.externalEntityIds.get("team-agent.conversation\u0000group-42");

    expect(response.status).toBe(200);
    expect(payload.entries).toEqual([{ id: "strm_bbbbbbbbbbbbbbbb", kind: "user_msg" }]);
    expect(rec.appendManyCalls).toEqual([]);
    expect(rec.appendCalls).toEqual([
      {
        session: payload.session,
        input: {
          kind: "user_msg",
          content: "Alice told Bob about the launch.",
          observed_at: observedAt,
          audience: group,
          sender_entity_id: alice,
          conversation: { type: "groupChat", name: "AI Ninjas" },
        },
      },
    ]);
    expect(rec.activityObservationProjectionInputs).toEqual([
      expect.objectContaining({
        userContact: expect.objectContaining({
          kind: "user_contact",
          occurredAt: 1000,
          speakerEntityId: alice,
          audienceEntityId: group,
          sourceStreamEntryIds: ["strm_bbbbbbbbbbbbbbbb"],
        }),
        touch: { at: 1000, messageCountDelta: 1 },
      }),
    ]);
    expect(rec.activityRecords).toHaveLength(1);
    expect(rec.sessionTouches).toEqual([
      {
        sessionId: payload.session,
        update: { at: 1000, messageCountDelta: 1 },
      },
    ]);
    expect(rec.ingestSessions).toEqual([payload.session]);
  });

  it("records a reply-only agent entry and projects the self reply for its group session", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const response = await post(
      base,
      "/memory/append-turn",
      {
        tenant: "acme",
        session: "tenant::group::reply-only",
        assistant: "A proactive update from the agent.",
        conversation: {
          type: "groupChat",
          name: "AI Ninjas",
          external_id: "group-42",
        },
      },
      TOKEN,
    );
    const payload = (await response.json()) as {
      entries: Array<{ id: string; kind: string }>;
      session: string;
    };
    const group = rec.externalEntityIds.get("team-agent.conversation\u0000group-42");
    const selfEntityId = rec.entities[0]?.id;

    expect(response.status).toBe(200);
    expect(payload.entries).toEqual([{ id: "strm_bbbbbbbbbbbbbbbb", kind: "agent_msg" }]);
    expect(rec.appendManyCalls).toEqual([]);
    expect(rec.appendCalls).toEqual([
      {
        session: payload.session,
        input: {
          kind: "agent_msg",
          content: "A proactive update from the agent.",
          audience: group,
          conversation: { type: "groupChat", name: "AI Ninjas" },
        },
      },
    ]);
    expect(
      rec.appendCalls.some((call) => (call.input as { kind?: string }).kind === "user_msg"),
    ).toBe(false);
    expect(rec.activityReplyProjectionInputs).toEqual([
      expect.objectContaining({
        session: expect.objectContaining({
          session_id: payload.session,
          source_type: "team_agent",
          source_external_id: "tenant::group::reply-only",
          conversation_kind: "thread",
          audience_entity_id: group,
          audience_role: "participant",
        }),
        borgReplied: {
          kind: "borg_replied",
          occurredAt: 1000,
          sessionId: payload.session,
          speakerEntityId: selfEntityId,
          actorEntityId: selfEntityId,
          audienceEntityId: group,
          participantEntityIds: [selfEntityId, group],
          sourceStreamEntryIds: ["strm_bbbbbbbbbbbbbbbb"],
        },
        touch: { at: 1000, messageCountDelta: 1 },
      }),
    ]);
    expect(rec.activityRecords).toEqual([
      expect.objectContaining({
        kind: "borg_replied",
        speakerEntityId: selfEntityId,
        actorEntityId: selfEntityId,
      }),
    ]);
    expect(rec.sessionTouches).toEqual([
      {
        sessionId: payload.session,
        update: { at: 1000, messageCountDelta: 1 },
      },
    ]);
    expect(rec.ingestSessions).toEqual([payload.session]);
  });

  it("returns a durable enhanced append when the awareness projection fails atomically", async () => {
    const { pool, rec } = recordingPool();
    const traceRegistry = new MemoryTraceRegistry();
    rec.activityProjectionError = Object.assign(new Error("injected projection failure"), {
      code: "SQLITE_BUSY",
    });
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);

    try {
      const base = await start(pool, TOKEN, { traceRegistry });
      const response = await post(
        base,
        "/memory/append-turn",
        {
          tenant: "acme",
          session: "tenant::alice::projection-failure",
          user: "durable user message",
          assistant: "durable assistant message",
          sender: { external_id: "alice", display_name: "Alice" },
          conversation: { type: "personal", name: "Alice" },
        },
        TOKEN,
      );

      expect(response.status).toBe(200);
      expect(await response.json()).toMatchObject({
        ok: true,
        entries: [
          { id: "strm_aaaaaaaaaaaaaaa0", kind: "user_msg" },
          { id: "strm_aaaaaaaaaaaaaaa1", kind: "agent_msg" },
        ],
      });
      expect(rec.appendManyCalls).toHaveLength(1);
      expect(rec.activityProjectionInputs).toHaveLength(1);
      expect(rec.sessionEnsures).toEqual([]);
      expect(rec.activityRecords).toEqual([]);
      expect(rec.sessionTouches).toEqual([]);
      expect(traceRegistry.query("acme").events).toEqual([
        expect.objectContaining({
          event: "sidecar.append_projection.degraded",
          reason: "awareness_projection_failed",
          error_code: "SQLITE_BUSY",
          source_stream_entry_ids: ["strm_aaaaaaaaaaaaaaa0", "strm_aaaaaaaaaaaaaaa1"],
        }),
      ]);
      expect(consoleError).toHaveBeenCalledWith(
        'memory-sidecar: append-turn awareness projection failed for tenant "acme"',
        rec.activityProjectionError,
      );
    } finally {
      consoleError.mockRestore();
    }
  });

  it("uses a separate stable group identity for enhanced group and channel appends", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);

    for (const conversation of [
      { type: "groupChat" as const, name: "AI Ninjas", external_id: "group-42" },
      { type: "channel" as const, name: "", external_id: "channel-7" },
    ]) {
      const response = await post(
        base,
        "/memory/append-turn",
        {
          tenant: "acme",
          session: `session-${conversation.external_id}`,
          user: "hello",
          assistant: "hi",
          sender: { external_id: "alice", display_name: "Alice" },
          conversation,
        },
        TOKEN,
      );
      expect(response.status).toBe(200);
      await response.json();
    }

    const group = rec.externalEntityIds.get("team-agent.conversation\u0000group-42");
    const channel = rec.externalEntityIds.get("team-agent.conversation\u0000channel-7");
    expect(group).toBeDefined();
    expect(channel).toBeDefined();
    expect(group).not.toBe(rec.externalSenderIds.get("alice"));
    expect(rec.entities.find((entity) => entity.id === group)).toMatchObject({
      canonical_name: "AI Ninjas",
      kind: "group",
    });
    expect(rec.entities.find((entity) => entity.id === channel)).toMatchObject({
      canonical_name: "channel:channel-7",
      kind: "group",
    });
    expect(rec.sessionEnsures).toEqual([
      expect.objectContaining({ conversation_kind: "thread", audience_entity_id: group }),
      expect.objectContaining({ conversation_kind: "channel", audience_entity_id: channel }),
    ]);
  });

  it("keeps the exact legacy append path when a group sender lacks conversation.external_id", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const response = await post(
      base,
      "/memory/append-turn",
      {
        tenant: "acme",
        session: "legacy-group",
        user: "hello",
        assistant: "hi",
        sender: { external_id: "alice", display_name: "Alice", operator: "legacy-value" },
        conversation: { type: "groupChat", name: "AI Ninjas" },
      },
      TOKEN,
    );

    expect(response.status).toBe(200);
    await response.json();
    const alice = rec.externalSenderIds.get("alice");
    expect(rec.appendMany?.inputs).toEqual([
      {
        kind: "user_msg",
        content: "hello",
        sender_entity_id: alice,
        conversation: { type: "groupChat", name: "AI Ninjas" },
      },
      {
        kind: "agent_msg",
        content: "hi",
        conversation: { type: "groupChat", name: "AI Ninjas" },
      },
    ]);
    expect(rec.sessionEnsures).toEqual([]);
    expect(rec.activityRecords).toEqual([]);
    expect(
      rec.resolvedExternalEntities.some(
        (input) => (input as { source?: string }).source === "team-agent.conversation",
      ),
    ).toBe(false);
  });

  it("validates sender.operator only when complete enhanced identity is available", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const response = await post(
      base,
      "/memory/append-turn",
      {
        tenant: "acme",
        session: "personal",
        user: "hello",
        assistant: "hi",
        sender: { external_id: "alice", display_name: "Alice", operator: "not-boolean" },
        conversation: { type: "personal", name: "Alice" },
      },
      TOKEN,
    );

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({
      error: "invalid 'sender.operator'; expected boolean",
    });
    expect(rec.tenants).toEqual([]);
    expect(rec.appendManyCalls).toEqual([]);
  });

  it("assembles personal context from only the person and observed group audiences", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool, TOKEN, {
      recentActivityWindowMs: 24 * 60 * 60_000,
      recentActivityLimit: 12,
    });
    const alice = createEntityId();
    const group = createEntityId();
    const bob = createEntityId();
    rec.entities.push(
      {
        id: alice,
        canonical_name: "Alice",
        aliases: [],
        kind: "person",
        borg_role: null,
        name_provenance: "transport_sender",
        created_at: 1,
      },
      {
        id: group,
        canonical_name: "AI Ninjas",
        aliases: [],
        kind: "group",
        borg_role: null,
        name_provenance: "transport_audience_label",
        created_at: 1,
      },
      {
        id: bob,
        canonical_name: "Bob",
        aliases: [],
        kind: "person",
        borg_role: null,
        name_provenance: "transport_sender",
        created_at: 1,
      },
    );
    rec.externalEntityIds.set("team-agent.sender\u0000alice", alice);
    rec.externalSenderIds.set("alice", alice);
    rec.observedGroupAudienceIds = [group];
    const occurredAt = Date.now() - 60_000;
    rec.visibleActivityEvents = [
      {
        kind: "user_contact",
        occurredAt,
        sessionId: parseSessionId("sess_bbbbbbbbbbbbbbbb"),
        audienceEntityId: group,
        conversationKind: "thread",
        conversationName: "AI Ninjas",
        participantLabel: "Alice",
        sourceStreamEntryIds: [parseStreamEntryId("strm_bbbbbbbbbbbbbbbb")],
      },
    ];
    rec.commitments = [
      testCommitment({ directive: "Global rule", restricted_audience: null, priority: 10 }),
      testCommitment({ directive: "Alice rule", restricted_audience: alice, priority: 9 }),
      testCommitment({ directive: "Group rule", restricted_audience: group, priority: 8 }),
    ];
    const visibleDirective = testDirective({
      operational_directive: "Visible operator rule",
      disclosure_policy: {
        content_scope: "operator_only",
        allowed_entity_ids: [],
        excluded_entity_ids: [],
        subject_may_know: null,
        mention_policy: "answer_if_asked",
        denied_audience_behavior: "omit",
        boundary_prompt: "Private rule applies.",
        topic_tags: ["ops"],
      },
    });
    const omittedDirective = testDirective({ operational_directive: "Hidden rule" });
    rec.directiveApplicable = [
      {
        directive: visibleDirective,
        recipient_entity_ids: [alice],
        activation: { active: true, reason: "operator_only" },
        disclosure: { render_mode: "content", reason: "operator_only" },
        render_mode: "content",
        reason: "operator_only",
      },
      {
        directive: omittedDirective,
        recipient_entity_ids: [alice],
        activation: { active: true, reason: "public" },
        disclosure: { render_mode: "omit", reason: "unauthorized_omit" },
        render_mode: "omit",
        reason: "unauthorized_omit",
      },
      {
        directive: testDirective({ operational_directive: "Inactive rule" }),
        recipient_entity_ids: [alice],
        activation: { active: false, reason: "unauthorized_omit" },
        disclosure: { render_mode: "content", reason: "public" },
        render_mode: "content",
        reason: "public",
      },
    ];
    rec.recallEpisodes = [
      testEpisode("ep_public0000000000" as Episode["id"], {
        audience_entity_id: null,
        origin_audience_entity_ids: [],
        shared: true,
      }),
      testEpisode("ep_alice00000000000" as Episode["id"], {
        audience_entity_id: alice,
        origin_audience_entity_ids: [alice],
        shared: false,
      }),
      testEpisode("ep_group00000000000" as Episode["id"], {
        audience_entity_id: group,
        origin_audience_entity_ids: [group],
        shared: false,
      }),
      testEpisode("ep_bob0000000000000" as Episode["id"], {
        audience_entity_id: bob,
        origin_audience_entity_ids: [bob],
        shared: false,
      }),
    ];

    const response = await post(
      base,
      "/memory/context",
      {
        tenant: "acme",
        session: "tenant::alice::personal",
        sender: { external_id: "alice", display_name: "Alice", operator: true },
        conversation: { type: "personal", name: "Alice" },
        query: "What matters now?",
      },
      TOKEN,
    );
    const payload = (await response.json()) as Record<string, unknown>;

    expect(response.status).toBe(200);
    expect(payload.audience).toEqual({
      entity_id: alice,
      kind: "person",
      name: "Alice",
      role: "operator",
    });
    expect((payload.episodes as Array<{ id: string }>).map((episode) => episode.id)).toEqual([
      "ep_public0000000000",
      "ep_alice00000000000",
      "ep_group00000000000",
    ]);
    expect(payload.hidden_episode_count).toBe(1);
    expect(payload.commitments).toEqual([
      expect.objectContaining({ directive: "Global rule" }),
      expect.objectContaining({ directive: "Alice rule" }),
    ]);
    expect(payload.directives).toEqual([
      expect.objectContaining({
        id: visibleDirective.id,
        render_mode: "content",
        text: "Visible operator rule",
      }),
    ]);
    expect(payload.recent_activity).toEqual([
      expect.objectContaining({
        kind: "user_contact",
        relative_age: "1m ago",
        conversation: { type: "groupChat", name: "AI Ninjas" },
        participant_name: "Alice",
        text: 'Alice contacted the agent 1m ago in group chat "AI Ninjas".',
      }),
    ]);
    expect(rec.lastRecallOptions).toMatchObject({
      audienceEntityId: alice,
      visibleAudienceEntityIds: [alice, group],
      limit: 8,
    });
    expect(rec.lastVisibleActivityInput).toMatchObject({
      audienceEntityIds: [alice, group],
      limit: 12,
    });
    expect(rec.directiveApplicableOptions).toEqual([
      {
        currentAudienceEntityId: alice,
        participantEntityIds: [alice],
        allowListAudienceEntityIds: [alice],
        sessionRole: "operator",
        trustedTenantOperator: true,
      },
    ]);
    expect(rec.exclusives).toEqual([true, undefined, undefined]);
  });

  it("does not widen group context through the current speaker's other memberships", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const sender = createEntityId();
    const currentGroup = createEntityId();
    const otherGroup = createEntityId();
    rec.entities.push(
      {
        id: sender,
        canonical_name: "Alice",
        aliases: [],
        kind: "person",
        borg_role: null,
        name_provenance: "transport_sender",
        created_at: 1,
      },
      {
        id: currentGroup,
        canonical_name: "Current Group",
        aliases: [],
        kind: "group",
        borg_role: null,
        name_provenance: "transport_audience_label",
        created_at: 1,
      },
      {
        id: otherGroup,
        canonical_name: "Other Group",
        aliases: [],
        kind: "group",
        borg_role: null,
        name_provenance: "transport_audience_label",
        created_at: 1,
      },
    );
    rec.externalEntityIds.set("team-agent.sender\u0000alice", sender);
    rec.externalEntityIds.set("team-agent.conversation\u0000current", currentGroup);
    rec.externalSenderIds.set("alice", sender);
    rec.observedGroupAudienceIds = [otherGroup];
    rec.recallEpisodes = [
      testEpisode("ep_current000000000" as Episode["id"], {
        audience_entity_id: currentGroup,
        origin_audience_entity_ids: [currentGroup],
        shared: false,
      }),
      testEpisode("ep_other00000000000" as Episode["id"], {
        audience_entity_id: otherGroup,
        origin_audience_entity_ids: [otherGroup],
        shared: false,
      }),
    ];

    const response = await post(
      base,
      "/memory/context",
      {
        tenant: "acme",
        session: "group-session",
        sender: { external_id: "alice", display_name: "Alice" },
        conversation: {
          type: "groupChat",
          name: "Current Group",
          external_id: "current",
        },
        query: "group context",
        sections: ["episodes"],
      },
      TOKEN,
    );
    const payload = (await response.json()) as {
      episodes: Array<{ id: string }>;
      hidden_episode_count: number;
    };

    expect(payload.episodes.map((episode) => episode.id)).toEqual(["ep_current000000000"]);
    expect(payload.hidden_episode_count).toBe(1);
    expect(rec.lastRecallOptions).toMatchObject({
      audienceEntityId: currentGroup,
      visibleAudienceEntityIds: [currentGroup],
    });
  });

  it("strictly scopes episode time ranges and marks an automatic unscoped fallback", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const inRange = testEpisode("ep_timerangeinside1" as Episode["id"], {
      title: "Today in range",
      start_time: 150,
      end_time: 160,
      shared: true,
    });
    const outside = testEpisode("ep_timerangeoutside" as Episode["id"], {
      title: "Older fallback",
      start_time: 50,
      end_time: 60,
      shared: true,
    });
    rec.recallEpisodes = [outside, inRange];

    const strictResponse = await post(
      base,
      "/memory/context",
      {
        tenant: "acme",
        session: "personal-time-range",
        sender: { external_id: "alice", display_name: "Alice" },
        conversation: { type: "personal", name: "Alice" },
        query: "today's discussion",
        limit: 2,
        sections: ["episodes"],
        time_range: { start: 100, end: 200 },
      },
      TOKEN,
    );
    const strictPayload = (await strictResponse.json()) as Record<string, unknown>;

    expect(strictResponse.status).toBe(200);
    expect((strictPayload.episodes as Array<{ id: string }>).map((episode) => episode.id)).toEqual([
      inRange.id,
    ]);
    expect(strictPayload).not.toHaveProperty("episodes_time_range_fallback");
    expect(rec.lastRecallOptions).toMatchObject({
      limit: 2,
      timeRange: { start: 100, end: 200 },
    });

    rec.recallEpisodes = [outside];
    const fallbackResponse = await post(
      base,
      "/memory/recall",
      {
        tenant: "acme",
        query: "today's discussion",
        limit: 2,
        time_range: { start: 100, end: 200 },
      },
      TOKEN,
    );
    const fallbackPayload = (await fallbackResponse.json()) as Record<string, unknown>;

    expect(fallbackResponse.status).toBe(200);
    expect(fallbackPayload.episodes_time_range_fallback).toBe(true);
    expect(
      (fallbackPayload.episodes as Array<{ id: string }>).map((episode) => episode.id),
    ).toEqual([outside.id]);
  });

  it("applies episode exclusions before the requested limit on recall and context", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const excludedTitle = testEpisode("ep_excludedtitle001" as Episode["id"], {
      title: "OUTCOME rollup: launch",
      shared: true,
    });
    const excludedNarrative = testEpisode("ep_excludedmarker01" as Episode["id"], {
      title: "Decision digest",
      narrative: "OUTCOME fp=123 decision=ship",
      shared: true,
    });
    const firstIncluded = testEpisode("ep_includedfirst001" as Episode["id"], {
      title: "Python discussion",
      shared: true,
    });
    const secondIncluded = testEpisode("ep_includedsecond01" as Episode["id"], {
      title: "TanStack discussion",
      shared: true,
    });
    const overflow = testEpisode("ep_includedoverflow" as Episode["id"], {
      title: "Overflow discussion",
      shared: true,
    });
    rec.recallEpisodes = [
      excludedTitle,
      excludedNarrative,
      firstIncluded,
      secondIncluded,
      overflow,
    ];
    const exclude = {
      title_prefixes: ["OUTCOME rollup"],
      narrative_markers: ["OUTCOME fp="],
    };

    const recallResponse = await post(
      base,
      "/memory/recall",
      { tenant: "acme", query: "technology", limit: 2, exclude },
      TOKEN,
    );
    const recallPayload = (await recallResponse.json()) as {
      episodes: Array<{ id: string }>;
    };

    expect(recallResponse.status).toBe(200);
    expect(recallPayload.episodes.map((episode) => episode.id)).toEqual([
      firstIncluded.id,
      secondIncluded.id,
    ]);
    expect(rec.lastRecallLimit).toBe(6);
    expect(rec.lastRecallOptions).toMatchObject({ recordRetrieval: false });
    expect(rec.retrievalRecords).toEqual([
      { episodeId: firstIncluded.id, score: 0.91 },
      { episodeId: secondIncluded.id, score: 0.91 },
    ]);

    rec.retrievalRecords.length = 0;

    const contextResponse = await post(
      base,
      "/memory/context",
      {
        tenant: "acme",
        session: "personal-exclusions",
        sender: { external_id: "alice", display_name: "Alice" },
        conversation: { type: "personal", name: "Alice" },
        query: "technology",
        limit: 2,
        sections: ["episodes"],
        exclude,
      },
      TOKEN,
    );
    const contextPayload = (await contextResponse.json()) as {
      episodes: Array<{ id: string }>;
    };

    expect(contextResponse.status).toBe(200);
    expect(contextPayload.episodes.map((episode) => episode.id)).toEqual([
      firstIncluded.id,
      secondIncluded.id,
    ]);
    expect(rec.lastRecallOptions).toMatchObject({ recordRetrieval: false });
    expect(rec.retrievalRecords).toEqual([
      { episodeId: firstIncluded.id, score: 0.91 },
      { episodeId: secondIncluded.id, score: 0.91 },
    ]);
  });

  it("returns excluded-filtered recent episodes for only the current venue", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const venueSince = 1_725_000_000_000;
    const excluded = testEpisode("ep_venueexcluded001" as Episode["id"], {
      title: "OUTCOME rollup: room",
      start_time: venueSince + 3_000,
      shared: true,
    });
    const newest = testEpisode("ep_venuenewest00001" as Episode["id"], {
      title: "Newest room memory",
      start_time: venueSince + 2_000,
      shared: true,
    });
    const older = testEpisode("ep_venueolder000001" as Episode["id"], {
      title: "Older room memory",
      start_time: venueSince + 1_000,
      shared: true,
    });
    rec.venueEpisodes = [excluded, newest, older];

    const missingSince = await post(
      base,
      "/memory/context",
      {
        tenant: "acme",
        session: "venue-room",
        sender: { external_id: "alice", display_name: "Alice" },
        conversation: {
          type: "groupChat",
          name: "AI Ninjas",
          external_id: "group-42",
        },
        sections: ["venue_recent"],
      },
      TOKEN,
    );
    expect(missingSince.status).toBe(400);

    const response = await post(
      base,
      "/memory/context",
      {
        tenant: "acme",
        session: "venue-room",
        sender: { external_id: "alice", display_name: "Alice" },
        conversation: {
          type: "groupChat",
          name: "AI Ninjas",
          external_id: "group-42",
        },
        sections: ["venue_recent"],
        venue_since: venueSince,
        exclude: { title_prefixes: ["OUTCOME rollup"], narrative_markers: [] },
      },
      TOKEN,
    );
    const payload = (await response.json()) as {
      venue_recent: Array<{ id: string; score: number; raw_score: number }>;
    };
    const expectedSession = `sess_${createHash("sha256")
      .update("venue-room")
      .digest("hex")
      .slice(0, 16)}`;

    expect(response.status).toBe(200);
    expect(payload.venue_recent.map((episode) => episode.id)).toEqual([newest.id, older.id]);
    expect(payload.venue_recent).toEqual([
      expect.objectContaining({
        id: newest.id,
        score: 0,
        raw_score: 0,
        disclosure: expect.anything(),
      }),
      expect.objectContaining({
        id: older.id,
        score: 0,
        raw_score: 0,
        disclosure: expect.anything(),
      }),
    ]);
    expect(rec.lastVenueOptions).toEqual({
      sessionId: expectedSession,
      sinceMs: venueSince,
      audienceEntityId: rec.externalEntityIds.get("team-agent.conversation\u0000group-42"),
      limit: 36,
    });
    expect(rec.recallOptionsCalls).toEqual([]);
  });

  it("strictly validates observation and episode-scoping extension fields", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);

    expect(
      (
        await post(
          base,
          "/memory/append-turn",
          { tenant: "acme", session: "observed", user: "message", observed_at: "now" },
          TOKEN,
        )
      ).status,
    ).toBe(400);
    const serverNow = Date.now();
    expect(
      (
        await post(
          base,
          "/memory/append-turn",
          {
            tenant: "acme",
            session: "observed",
            user: "message",
            observed_at: serverNow - 10 * 60_000,
          },
          TOKEN,
        )
      ).status,
    ).toBe(400);
    expect(
      (
        await post(
          base,
          "/memory/append-turn",
          {
            tenant: "acme",
            session: "observed",
            user: "message",
            observed_at: serverNow + 2 * 60_000,
          },
          TOKEN,
        )
      ).status,
    ).toBe(400);
    expect(
      (
        await post(
          base,
          "/memory/recall",
          { tenant: "acme", query: "query", time_range: { start: 200, end: 100 } },
          TOKEN,
        )
      ).status,
    ).toBe(400);
    expect(
      (
        await post(
          base,
          "/memory/recall",
          {
            tenant: "acme",
            query: "query",
            exclude: { title_prefixes: [], narrative_markers: [], extra: true },
          },
          TOKEN,
        )
      ).status,
    ).toBe(400);
    expect(
      (
        await post(
          base,
          "/memory/context",
          {
            tenant: "acme",
            session: "venue",
            sender: { external_id: "alice", display_name: "Alice" },
            conversation: { type: "personal", name: "Alice" },
            sections: ["venue_recent"],
            venue_since: 100,
            venue_limit: 51,
          },
          TOKEN,
        )
      ).status,
    ).toBe(400);
    expect(
      (
        await post(
          base,
          "/memory/context",
          {
            tenant: "acme",
            session: "participants",
            sender: { external_id: "alice", display_name: "Alice" },
            conversation: { type: "personal", name: "Alice" },
            participants: [{ external_id: "bob", display_name: "Bob", operator: "not-a-boolean" }],
            sections: ["audience"],
          },
          TOKEN,
        )
      ).status,
    ).toBe(400);
    expect(
      (
        await post(
          base,
          "/memory/context",
          {
            tenant: "acme",
            session: "participants",
            sender: { external_id: "alice", display_name: "Alice" },
            conversation: { type: "personal", name: "Alice" },
            participants: Array.from({ length: 33 }, (_, index) => ({
              external_id: `person-${index}`,
              display_name: `Person ${index}`,
              operator: false,
            })),
            sections: ["audience"],
          },
          TOKEN,
        )
      ).status,
    ).toBe(400);
    expect(rec.tenants).toEqual([]);
  });

  it("validates context sections and preserves non-episode sections on embedding degradation", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);

    expect(
      (
        await post(
          base,
          "/memory/context",
          {
            tenant: "acme",
            session: "personal",
            sender: { external_id: "alice", display_name: "Alice" },
            conversation: { type: "personal", name: "Alice" },
          },
          TOKEN,
        )
      ).status,
    ).toBe(400);
    expect(
      (
        await post(
          base,
          "/memory/context",
          {
            tenant: "acme",
            session: "personal",
            sender: { external_id: "alice", display_name: "Alice" },
            conversation: { type: "personal", name: "Alice" },
            sections: ["unknown"],
          },
          TOKEN,
        )
      ).status,
    ).toBe(400);
    expect(
      (
        await post(
          base,
          "/memory/context",
          {
            tenant: "acme",
            session: "group",
            sender: { external_id: "alice", display_name: "Alice" },
            conversation: { type: "channel", name: "General" },
            query: "context",
          },
          TOKEN,
        )
      ).status,
    ).toBe(400);

    rec.recallError = new EmbeddingError("gateway stalled");
    const response = await post(
      base,
      "/memory/context",
      {
        tenant: "acme",
        session: "personal",
        sender: { external_id: "alice", display_name: "Alice" },
        conversation: { type: "personal", name: "Alice" },
        query: "context",
        sections: ["audience", "episodes"],
      },
      TOKEN,
    );
    const payload = (await response.json()) as Record<string, unknown>;

    expect(response.status).toBe(200);
    expect(payload.audience).toEqual(expect.objectContaining({ name: "Alice" }));
    expect(payload.episodes).toEqual([]);
    expect(payload.degraded).toBe(true);
    expect(payload.degraded_reason).toBe("embeddings: gateway stalled");
  });

  it("preserves non-episode context when audience-scoped recall exceeds its deadline", async () => {
    const { pool, rec } = recordingPool();
    rec.recallPromise = new Promise(() => {});
    const base = await start(pool, TOKEN, { recallDeadlineMs: 25 });
    const response = await post(
      base,
      "/memory/context",
      {
        tenant: "acme",
        session: "personal",
        sender: { external_id: "alice", display_name: "Alice" },
        conversation: { type: "personal", name: "Alice" },
        query: "context",
        sections: ["audience", "episodes"],
      },
      TOKEN,
    );
    const payload = (await response.json()) as Record<string, unknown>;

    expect(response.status).toBe(200);
    expect(payload.audience).toEqual(expect.objectContaining({ name: "Alice" }));
    expect(payload.episodes).toEqual([]);
    expect(payload.hidden_episode_count).toBe(0);
    expect(payload.degraded).toBe(true);
    expect(payload.degraded_reason).toContain("deadline");
  });

  it("rejects malformed optional sender objects before touching the tenant", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const response = await post(
      base,
      "/memory/append-turn",
      {
        tenant: "acme",
        session: "room",
        user: "hello",
        assistant: "hi",
        sender: { external_id: "platform-alice", display_name: "" },
      },
      TOKEN,
    );

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({
      error: "invalid 'sender'; expected non-empty 'external_id' and 'display_name'",
    });
    expect(rec.tenants).toEqual([]);
  });

  it.each([
    null,
    [],
    "personal",
    { type: "directMessage", name: "Alice" },
    { type: "personal", name: 42 },
    { type: 42, name: "Alice" },
    { type: "channel" },
  ])(
    "rejects malformed optional conversation %# before touching the tenant",
    async (conversation) => {
      const { pool, rec } = recordingPool();
      const base = await start(pool);
      const response = await post(
        base,
        "/memory/append-turn",
        {
          tenant: "acme",
          session: "room",
          user: "hello",
          assistant: "hi",
          conversation,
        },
        TOKEN,
      );

      expect(response.status).toBe(400);
      expect(await response.json()).toEqual({
        error:
          "invalid 'conversation'; expected type 'personal', 'groupChat', or 'channel' and string 'name'",
      });
      expect(rec.tenants).toEqual([]);
    },
  );

  it("does not serialize later append-turn requests behind pending ingestion", async () => {
    const appendSessions: string[] = [];
    let ingestionStarted = false;
    let releaseIngestion!: () => void;
    const ingestion = new Promise<{ ran: boolean; processedEntries: number }>((resolve) => {
      releaseIngestion = () => resolve({ ran: true, processedEntries: 2 });
    });
    const borg = {
      stream: {
        appendMany: async (_inputs: unknown[], options?: { session?: string }) => {
          appendSessions.push(options?.session ?? "");
          return [
            { id: "strm_aaaaaaaaaaaaaaaa", kind: "user_msg" },
            { id: "strm_bbbbbbbbbbbbbbbb", kind: "agent_msg" },
          ];
        },
      },
      episodic: {
        ingest: async () => {
          ingestionStarted = true;
          return ingestion;
        },
      },
    } as unknown as Borg;
    let exclusiveTail: Promise<unknown> = Promise.resolve();
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      withTenant<T>(
        _tenantId: string,
        fn: (borg: Borg) => T | Promise<T>,
        opts?: { exclusive?: boolean },
      ) {
        if (opts?.exclusive === true) {
          const run = exclusiveTail.then(() => fn(borg));
          exclusiveTail = run.then(
            () => undefined,
            () => undefined,
          );
          return run;
        }
        return Promise.resolve(fn(borg));
      },
    };
    const base = await start(pool);

    const first = await post(
      base,
      "/memory/append-turn",
      { tenant: "acme", session: "first", user: "u1", assistant: "a1" },
      TOKEN,
    );
    expect(first.status).toBe(200);
    await first.json();
    expect(ingestionStarted).toBe(true);

    const secondStatus = await Promise.race([
      post(
        base,
        "/memory/append-turn",
        { tenant: "acme", session: "second", user: "u2", assistant: "a2" },
        TOKEN,
      ).then(async (res) => {
        await res.json();
        return res.status;
      }),
      new Promise<number>((resolve) => {
        setTimeout(() => resolve(599), 50);
      }),
    ]);
    releaseIngestion();

    expect(secondStatus).toBe(200);
    expect(appendSessions).toHaveLength(2);
  });

  it("accepts an already-valid borg session id for append-turn", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const session = "sess_aaaaaaaaaaaaaaaa";
    const res = await post(
      base,
      "/memory/append-turn",
      { tenant: "acme", session, user: "u", assistant: "a" },
      TOKEN,
    );

    expect(res.status).toBe(200);
    expect(rec.appendMany?.session).toBe(session);
  });

  it("validates required fields and accepts observation and reply-only appends", async () => {
    const { pool } = recordingPool();
    const base = await start(pool);
    expect((await post(base, "/memory/recall", { query: "q" }, TOKEN)).status).toBe(400); // no tenant
    expect((await post(base, "/memory/recall", { tenant: "acme" }, TOKEN)).status).toBe(400); // no query
    expect((await post(base, "/memory/remember", { tenant: "acme" }, TOKEN)).status).toBe(400); // no content
    expect(
      (
        await post(
          base,
          "/memory/append-turn",
          { tenant: "acme", user: "u", assistant: "a" },
          TOKEN,
        )
      ).status,
    ).toBe(400);
    expect(
      (
        await post(
          base,
          "/memory/append-turn",
          { tenant: "acme", session: "s", assistant: "a" },
          TOKEN,
        )
      ).status,
    ).toBe(200);
    expect(
      (await post(base, "/memory/append-turn", { tenant: "acme", session: "s", user: "u" }, TOKEN))
        .status,
    ).toBe(200);
  });

  it("400s on invalid or non-object JSON", async () => {
    const { pool } = recordingPool();
    const base = await start(pool);
    expect((await post(base, "/memory/recall", undefined, TOKEN, "{not json")).status).toBe(400);
    expect((await post(base, "/memory/recall", undefined, TOKEN, "null")).status).toBe(400);
    expect((await post(base, "/memory/recall", undefined, TOKEN, "42")).status).toBe(400);
  });

  it("413s an oversized body", async () => {
    const { pool } = recordingPool();
    const base = await start(pool);
    const big = "x".repeat(70 * 1024); // > 64KB default
    const res = await post(
      base,
      "/memory/remember",
      undefined,
      TOKEN,
      JSON.stringify({ tenant: "acme", content: big }),
    );
    expect(res.status).toBe(413);
  });

  it("rejects a malformed tenant id with 400 before touching the pool", async () => {
    const calls: string[] = [];
    const pool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant(tenantId, fn) {
        calls.push(tenantId);
        return fn({} as Borg);
      },
    };
    const base = await start(pool);
    expect(
      (await post(base, "/memory/recall", { tenant: "../evil", query: "q" }, TOKEN)).status,
    ).toBe(400);
    expect(
      (await post(base, "/memory/recall", { tenant: "UPPER", query: "q" }, TOKEN)).status,
    ).toBe(400);
    expect(calls).toEqual([]); // pool never reached for an invalid tenant
  });

  it("does not leak internals on an unexpected error (generic 500)", async () => {
    const boomPool: MemoryPool = {
      listTenantIds: () => Promise.resolve([...STUB_TENANTS]),
      async withTenant() {
        throw new Error("sqlite path /secret/db.sqlite is locked");
      },
    };
    const base = await start(boomPool);
    const res = await post(base, "/memory/recall", { tenant: "acme", query: "q" }, TOKEN);
    expect(res.status).toBe(500);
    expect(await res.json()).toEqual({ error: "internal error" }); // no internal detail leaked
  });

  it("rolls back a failed real append projection and keeps an identical source-id retry idempotent", async () => {
    const root = mkdtempSync(join(tmpdir(), "borg-memory-projection-http-"));
    const traceRegistry = new MemoryTraceRegistry();
    const pool = new BorgPool({
      root,
      openOptions: {
        embeddingDimensions: 4,
        embeddingClient: new FakeEmbeddingClient(4),
        llmClient: new FakeLLMClient(),
        liveExtraction: false,
        liveCommitmentExtraction: false,
      },
      initializeBeing: (_tenantId, borg) => {
        borg.entities.ensureSelf("Sol", { provenance: "config_default_user" });
      },
    });
    let restoreRecord: (() => void) | undefined;
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);

    try {
      await pool.withTenant(
        "acme",
        (borg) => {
          const deps = (borg as unknown as { deps: BorgDependencies }).deps;
          const originalRecord = deps.activityRepository.record.bind(deps.activityRepository);
          let recordCalls = 0;
          const recordSpy = vi
            .spyOn(deps.activityRepository, "record")
            .mockImplementation((input) => {
              const recorded = originalRecord(input);
              recordCalls += 1;
              if (recordCalls === 2) {
                throw Object.assign(new Error("injected second activity failure"), {
                  code: "ACTIVITY_PROJECTION_INJECTED",
                });
              }
              return recorded;
            });
          restoreRecord = () => recordSpy.mockRestore();
        },
        { exclusive: true },
      );

      const base = await start(pool, TOKEN, { traceRegistry });
      const response = await post(
        base,
        "/memory/append-turn",
        {
          tenant: "acme",
          session: "teams::personal::projection-rollback",
          user: "Durable input",
          assistant: "Durable reply",
          sender: { external_id: "alice", display_name: "Alice" },
          conversation: { type: "personal", name: "Alice" },
        },
        TOKEN,
      );
      const payload = (await response.json()) as {
        session: SessionId;
        entries: Array<{ id: Episode["source_stream_ids"][number]; kind: string }>;
      };
      restoreRecord?.();
      restoreRecord = undefined;

      expect(response.status).toBe(200);
      expect(payload.entries).toHaveLength(2);
      const afterFailure = await pool.withTenant("acme", (borg) => {
        const deps = (borg as unknown as { deps: BorgDependencies }).deps;
        return {
          streamEntries: borg.stream.tail(10, { session: payload.session }),
          session: borg.sessions.get(payload.session),
          activityCount: Number(
            deps.sqlite.prepare("SELECT COUNT(*) AS count FROM activity_events").get()?.count ?? 0,
          ),
        };
      });
      expect(afterFailure.streamEntries).toHaveLength(2);
      expect(afterFailure.session).toBeNull();
      expect(afterFailure.activityCount).toBe(0);
      expect(traceRegistry.query("acme").events).toEqual([
        expect.objectContaining({
          event: "sidecar.append_projection.degraded",
          error_code: "ACTIVITY_PROJECTION_INJECTED",
        }),
      ]);

      const afterRetries = await pool.withTenant(
        "acme",
        (borg) => {
          const deps = (borg as unknown as { deps: BorgDependencies }).deps;
          const entries = borg.stream.tail(10, { session: payload.session });
          const userEntry = entries.find((entry) => entry.kind === "user_msg");
          const assistantEntry = entries.find((entry) => entry.kind === "agent_msg");
          const alice = borg.entities.findByExternalId("team-agent.sender", "alice");
          const selfEntity = borg.entities.getSelf();

          if (
            userEntry === undefined ||
            assistantEntry === undefined ||
            alice === null ||
            selfEntity === null
          ) {
            throw new Error("expected complete persisted turn identity");
          }

          const projection = {
            session: {
              session_id: payload.session,
              source_type: "team_agent",
              source_external_id: "teams::personal::projection-rollback",
              label: "Alice",
              audience_label: "Alice",
              audience_entity_id: alice,
              conversation_kind: "dm" as const,
              audience_role: "participant" as const,
              status: "active" as const,
              created_at: userEntry.timestamp,
              last_activity_at: userEntry.timestamp,
            },
            userContact: {
              kind: "user_contact" as const,
              occurredAt: userEntry.timestamp,
              sessionId: payload.session,
              speakerEntityId: alice,
              actorEntityId: alice,
              audienceEntityId: alice,
              participantEntityIds: [alice],
              sourceStreamEntryIds: [userEntry.id],
            },
            borgReplied: {
              kind: "borg_replied" as const,
              occurredAt: assistantEntry.timestamp,
              sessionId: payload.session,
              speakerEntityId: selfEntity.id,
              actorEntityId: selfEntity.id,
              audienceEntityId: alice,
              participantEntityIds: [selfEntity.id, alice],
              sourceStreamEntryIds: [assistantEntry.id],
            },
            touch: { at: assistantEntry.timestamp, messageCountDelta: 1 },
          };

          borg.activity.projectCompletedTurn(projection);
          borg.activity.projectCompletedTurn(projection);

          return {
            session: borg.sessions.get(payload.session),
            activityCount: Number(
              deps.sqlite.prepare("SELECT COUNT(*) AS count FROM activity_events").get()?.count ??
                0,
            ),
          };
        },
        { exclusive: true },
      );
      expect(afterRetries.session?.message_count).toBe(1);
      expect(afterRetries.activityCount).toBe(2);
    } finally {
      restoreRecord?.();
      consoleError.mockRestore();
      await pool.closeAll();
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("persists observation time as metadata and awareness at append time through a real BorgPool", async () => {
    const root = mkdtempSync(join(tmpdir(), "borg-memory-observation-http-"));
    const llmClient = new FakeLLMClient();
    const pool = new BorgPool({
      root,
      openOptions: {
        embeddingDimensions: 4,
        embeddingClient: new FakeEmbeddingClient(4),
        llmClient,
        liveExtraction: true,
        liveCommitmentExtraction: false,
      },
      initializeBeing: (_tenantId, borg) => {
        borg.entities.ensureSelf("Sol", { provenance: "config_default_user" });
      },
    });

    try {
      const base = await start(pool);
      const observedAt = Date.now() - 5_000;
      const requestedAt = Date.now();
      const response = await post(
        base,
        "/memory/append-turn",
        {
          tenant: "acme",
          session: "teams::group::observation",
          user: "An observed room message",
          observed_at: observedAt,
          sender: { external_id: "alice", display_name: "Alice" },
          conversation: {
            type: "groupChat",
            name: "Observation Room",
            external_id: "observation-room",
          },
        },
        TOKEN,
      );
      const payload = (await response.json()) as {
        session: SessionId;
        entries: Array<{ id: string; kind: string }>;
      };

      expect(response.status).toBe(200);
      expect(payload.entries).toHaveLength(1);
      expect(payload.entries[0]?.kind).toBe("user_msg");

      llmClient.pushResponse({
        text: "",
        input_tokens: 10,
        output_tokens: 20,
        stop_reason: "tool_use",
        tool_calls: [
          {
            id: "toolu_observation_episode",
            name: "EmitEpisodeCandidates",
            input: {
              episodes: [
                {
                  title: "Observed room message",
                  narrative: "Alice posted an observed message in the Observation Room.",
                  source_stream_ids: [payload.entries[0]?.id],
                  participants: ["Alice"],
                  location: "Observation Room",
                  tags: ["observation"],
                  emotional_arc: null,
                  confidence: 0.9,
                  significance: 0.6,
                },
              ],
              relational_slot_updates: [],
            },
          },
        ],
      });
      await pool.withTenant("acme", async (borg) => {
        const deps = (borg as unknown as { deps: BorgDependencies }).deps;
        await deps.streamIngestionCoordinator?.flush(payload.session);
      });

      const persisted = await pool.withTenant("acme", async (borg) => {
        const deps = (borg as unknown as { deps: BorgDependencies }).deps;
        return {
          entries: borg.stream.tail(10, { session: payload.session }),
          episodes: await borg.episodic.listAll(),
          session: borg.sessions.get(payload.session),
          activities: deps.sqlite
            .prepare(
              `SELECT kind, occurred_at, source_stream_entry_ids
               FROM activity_events
               ORDER BY occurred_at ASC, id ASC`,
            )
            .all() as Array<{
            kind: string;
            occurred_at: number;
            source_stream_entry_ids: string;
          }>,
        };
      });

      expect(persisted.entries).toEqual([
        expect.objectContaining({
          id: payload.entries[0]?.id,
          kind: "user_msg",
          observed_at: observedAt,
        }),
      ]);
      expect(persisted.entries[0]?.timestamp).toBeGreaterThanOrEqual(requestedAt);
      expect(persisted.entries.some((entry) => entry.kind === "agent_msg")).toBe(false);
      expect(persisted.session).toMatchObject({
        session_id: payload.session,
        message_count: 1,
        last_activity_at: persisted.entries[0]?.timestamp,
      });
      expect(persisted.activities).toEqual([
        expect.objectContaining({
          kind: "user_contact",
          occurred_at: persisted.entries[0]?.timestamp,
        }),
      ]);
      expect(persisted.episodes).toEqual([
        expect.objectContaining({
          source_stream_ids: [payload.entries[0]?.id],
          start_time: observedAt,
          end_time: observedAt,
        }),
      ]);
    } finally {
      await pool.closeAll();
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("persists a reply-only self activity and session count through a real BorgPool", async () => {
    const root = mkdtempSync(join(tmpdir(), "borg-memory-reply-only-http-"));
    const pool = new BorgPool({
      root,
      openOptions: {
        embeddingDimensions: 4,
        embeddingClient: new FakeEmbeddingClient(4),
        llmClient: new FakeLLMClient(),
        liveExtraction: false,
        liveCommitmentExtraction: false,
      },
      initializeBeing: (_tenantId, borg) => {
        borg.entities.ensureSelf("Sol", { provenance: "config_default_user" });
      },
    });

    try {
      const base = await start(pool);
      const response = await post(
        base,
        "/memory/append-turn",
        {
          tenant: "acme",
          session: "teams::group::reply-only",
          assistant: "A proactive update from Sol.",
          conversation: {
            type: "groupChat",
            name: "AI Ninjas",
            external_id: "reply-only-room",
          },
        },
        TOKEN,
      );
      const payload = (await response.json()) as {
        session: SessionId;
        entries: Array<{ id: string; kind: string }>;
      };

      expect(response.status).toBe(200);
      expect(payload.entries).toHaveLength(1);
      expect(payload.entries[0]?.kind).toBe("agent_msg");

      const persisted = await pool.withTenant("acme", (borg) => {
        const deps = (borg as unknown as { deps: BorgDependencies }).deps;
        const selfEntity = borg.entities.getSelf();
        const groupEntityId = borg.entities.findByExternalId(
          "team-agent.conversation",
          "reply-only-room",
        );

        return {
          selfEntity,
          groupEntityId,
          entries: borg.stream.tail(10, { session: payload.session }),
          session: borg.sessions.get(payload.session),
          activities: deps.sqlite
            .prepare(
              `SELECT kind, speaker_entity_id, actor_entity_id, audience_entity_id,
                      source_stream_entry_ids
               FROM activity_events
               ORDER BY occurred_at ASC, id ASC`,
            )
            .all() as Array<{
            kind: string;
            speaker_entity_id: string | null;
            actor_entity_id: string | null;
            audience_entity_id: string | null;
            source_stream_entry_ids: string;
          }>,
        };
      });

      expect(persisted.entries).toEqual([
        expect.objectContaining({
          id: payload.entries[0]?.id,
          kind: "agent_msg",
          content: "A proactive update from Sol.",
          audience: persisted.groupEntityId,
          conversation: { type: "groupChat", name: "AI Ninjas" },
        }),
      ]);
      expect(persisted.entries.some((entry) => entry.kind === "user_msg")).toBe(false);
      expect(persisted.session).toMatchObject({
        session_id: payload.session,
        audience_entity_id: persisted.groupEntityId,
        message_count: 1,
      });
      expect(persisted.activities).toEqual([
        {
          kind: "borg_replied",
          speaker_entity_id: persisted.selfEntity?.id,
          actor_entity_id: persisted.selfEntity?.id,
          audience_entity_id: persisted.groupEntityId,
          source_stream_entry_ids: JSON.stringify([payload.entries[0]?.id]),
        },
      ]);
    } finally {
      await pool.closeAll();
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("applies group and participant directive allows with participant exclusions fail-closed", async () => {
    const root = mkdtempSync(join(tmpdir(), "borg-memory-directive-group-http-"));
    const pool = new BorgPool({
      root,
      openOptions: {
        embeddingDimensions: 4,
        embeddingClient: new FakeEmbeddingClient(4),
        llmClient: new FakeLLMClient(),
        liveExtraction: false,
        liveCommitmentExtraction: false,
      },
      initializeBeing: (_tenantId, borg) => {
        borg.entities.ensureSelf("Sol", { provenance: "config_default_user" });
      },
    });

    try {
      const base = await start(pool);
      const appendResponse = await post(
        base,
        "/memory/append-turn",
        {
          tenant: "acme",
          session: "teams::group::directive-room",
          user: "Hello room",
          assistant: "Hello Alice",
          sender: { external_id: "alice", display_name: "Alice" },
          conversation: {
            type: "groupChat",
            name: "Directive Room",
            external_id: "directive-room",
          },
        },
        TOKEN,
      );
      expect(appendResponse.status).toBe(200);
      await appendResponse.json();

      const participantIdentityResponse = await post(
        base,
        "/memory/context",
        {
          tenant: "acme",
          session: "teams::group::directive-room",
          sender: { external_id: "alice", display_name: "Alice" },
          conversation: {
            type: "groupChat",
            name: "Directive Room",
            external_id: "directive-room",
          },
          participants: [
            { external_id: "bob", display_name: "Bob", operator: true },
            { external_id: "bob", display_name: "Duplicate Bob", operator: false },
          ],
          sections: ["audience"],
        },
        TOKEN,
      );
      expect(participantIdentityResponse.status).toBe(200);
      await expect(participantIdentityResponse.json()).resolves.toMatchObject({
        audience: { role: "participant" },
      });

      const definitions = [
        {
          key: "groupAllowed",
          body: {
            tenant: "acme",
            kind: "response_policy",
            text: "Rule allowed for this room.",
            content_scope: "allow_list",
            allowed_group_external_ids: ["directive-room"],
          },
        },
        {
          key: "groupExcluded",
          body: {
            tenant: "acme",
            kind: "response_policy",
            text: "Rule excluded from this room.",
            content_scope: "all_except",
            excluded_group_external_ids: ["directive-room"],
          },
        },
        {
          key: "participantExcluded",
          body: {
            tenant: "acme",
            kind: "response_policy",
            text: "Room rule suppressed while Bob is present.",
            content_scope: "allow_list",
            allowed_group_external_ids: ["directive-room"],
            excluded_external_ids: ["bob"],
          },
        },
        {
          key: "participantAllowed",
          body: {
            tenant: "acme",
            kind: "response_policy",
            text: "Rule allowed when Bob is a current participant.",
            content_scope: "allow_list",
            allowed_external_ids: ["bob"],
          },
        },
      ] as const;
      const ids = new Map<string, CreatorDirective["id"]>();

      for (const definition of definitions) {
        const response = await post(base, "/memory/directives", definition.body, TOKEN);
        const payload = (await response.json()) as {
          directive: { id: CreatorDirective["id"] };
        };
        expect(response.status).toBe(201);
        ids.set(definition.key, payload.directive.id);
      }

      const contextResponse = await post(
        base,
        "/memory/context",
        {
          tenant: "acme",
          session: "teams::group::directive-room",
          sender: { external_id: "alice", display_name: "Alice" },
          conversation: {
            type: "groupChat",
            name: "Directive Room",
            external_id: "directive-room",
          },
          participants: [{ external_id: "bob", display_name: "Bob", operator: true }],
          sections: ["directives"],
        },
        TOKEN,
      );
      const context = (await contextResponse.json()) as {
        directives: Array<{ id: CreatorDirective["id"] }>;
      };
      const visibleDirectiveIds = context.directives.map((directive) => directive.id);

      expect(contextResponse.status).toBe(200);
      expect(visibleDirectiveIds).toContain(ids.get("groupAllowed"));
      expect(visibleDirectiveIds).toContain(ids.get("participantAllowed"));
      expect(visibleDirectiveIds).not.toContain(ids.get("groupExcluded"));
      expect(visibleDirectiveIds).not.toContain(ids.get("participantExcluded"));
    } finally {
      await pool.closeAll();
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("integrates enriched turns, scoped context, and directive provenance through a real BorgPool", async () => {
    const root = mkdtempSync(join(tmpdir(), "borg-memory-context-http-"));
    const embeddingClient = new FakeEmbeddingClient(4);
    const llmClient = new FakeLLMClient();
    const pool = new BorgPool({
      root,
      openOptions: {
        embeddingDimensions: 4,
        embeddingClient,
        llmClient,
        liveExtraction: false,
        liveCommitmentExtraction: false,
      },
      initializeBeing: (_tenantId, borg) => {
        borg.entities.ensureSelf("Sol", { provenance: "config_default_user" });
      },
    });

    try {
      const base = await start(pool);
      const turns = [
        {
          key: "alice-personal",
          session: "teams::personal::alice",
          sender: { external_id: "alice", display_name: "Alice", operator: true },
          conversation: { type: "personal", name: "Alice" },
        },
        {
          key: "alice-group",
          session: "teams::group::ai-ninjas",
          sender: { external_id: "alice", display_name: "Alice" },
          conversation: {
            type: "groupChat",
            name: "AI Ninjas",
            external_id: "ai-ninjas",
          },
        },
        {
          key: "bob-personal",
          session: "teams::personal::bob",
          sender: { external_id: "bob", display_name: "Bob" },
          conversation: { type: "personal", name: "Bob" },
        },
      ] as const;
      const sessions = new Map<string, SessionId>();

      for (const turn of turns) {
        const response = await post(
          base,
          "/memory/append-turn",
          {
            tenant: "acme",
            session: turn.session,
            user: `Message from ${turn.key}`,
            assistant: `Reply to ${turn.key}`,
            sender: turn.sender,
            conversation: turn.conversation,
          },
          TOKEN,
        );
        const payload = (await response.json()) as { session: SessionId };

        expect(response.status).toBe(200);
        sessions.set(turn.key, payload.session);
      }

      const identity = await pool.withTenant("acme", (borg) => {
        const alice = borg.entities.findByExternalId("team-agent.sender", "alice");
        const bob = borg.entities.findByExternalId("team-agent.sender", "bob");
        const group = borg.entities.findByExternalId("team-agent.conversation", "ai-ninjas");

        if (alice === null || bob === null || group === null) {
          throw new Error("expected sidecar identities to be persisted");
        }

        return {
          alice,
          bob,
          group,
          sessions: borg.sessions.list(),
          observedGroups: borg.activity.listObservedGroupAudienceEntityIdsForSpeaker(alice),
        };
      });
      expect(identity.observedGroups).toEqual([identity.group]);
      expect(identity.sessions).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            session_id: sessions.get("alice-personal"),
            source_type: "team_agent",
            source_external_id: "teams::personal::alice",
            conversation_kind: "dm",
            audience_entity_id: identity.alice,
            audience_role: "operator",
            message_count: 1,
          }),
          expect.objectContaining({
            session_id: sessions.get("alice-group"),
            source_type: "team_agent",
            source_external_id: "teams::group::ai-ninjas",
            conversation_kind: "thread",
            audience_entity_id: identity.group,
            message_count: 1,
          }),
        ]),
      );

      const persistedEpisodes = await pool.withTenant(
        "acme",
        async (borg) => {
          const deps = (borg as unknown as { deps: BorgDependencies }).deps;
          const alicePersonalEntries = borg.stream.tail(10, {
            session: sessions.get("alice-personal"),
          });
          const aliceGroupEntries = borg.stream.tail(10, {
            session: sessions.get("alice-group"),
          });
          const bobPersonalEntries = borg.stream.tail(10, {
            session: sessions.get("bob-personal"),
          });
          const alicePersonalSource = alicePersonalEntries.find(
            (entry) => entry.kind === "user_msg",
          );
          const aliceGroupSource = aliceGroupEntries.find((entry) => entry.kind === "user_msg");
          const bobPersonalSource = bobPersonalEntries.find((entry) => entry.kind === "user_msg");

          if (
            alicePersonalSource?.audience === undefined ||
            aliceGroupSource?.audience === undefined ||
            bobPersonalSource?.audience === undefined
          ) {
            throw new Error("expected audience-stamped source entries");
          }
          expect(alicePersonalSource.audience).toBe(identity.alice);
          expect(aliceGroupSource.audience).toBe(identity.group);
          expect(bobPersonalSource.audience).toBe(identity.bob);

          const queryEmbedding = await embeddingClient.embed("situational awareness");
          const now = Date.now();
          const episodes = [
            testEpisode(createEpisodeId(), {
              title: "Public episode",
              narrative: "Public context visible to every audience.",
              participants: [episodeParticipantEntityIdTerm(identity.alice)],
              source_stream_ids: [alicePersonalSource.id],
              audience_entity_id: null,
              origin_audience_entity_ids: [],
              shared: true,
              embedding: queryEmbedding,
              start_time: now - 4_000,
              end_time: now - 3_900,
              created_at: now - 4_000,
              updated_at: now - 4_000,
            }),
            testEpisode(createEpisodeId(), {
              title: "Alice personal episode",
              narrative: "Private context from Alice's personal chat.",
              participants: [episodeParticipantEntityIdTerm(identity.alice)],
              source_stream_ids: [alicePersonalSource.id],
              audience_entity_id: alicePersonalSource.audience as EntityId,
              origin_audience_entity_ids: [alicePersonalSource.audience as EntityId],
              shared: false,
              embedding: queryEmbedding,
              start_time: now - 3_000,
              end_time: now - 2_900,
              created_at: now - 3_000,
              updated_at: now - 3_000,
            }),
            testEpisode(createEpisodeId(), {
              title: "Observed group episode",
              narrative: "Context from a group where Alice spoke.",
              participants: [episodeParticipantEntityIdTerm(identity.alice)],
              source_stream_ids: [aliceGroupSource.id],
              audience_entity_id: aliceGroupSource.audience as EntityId,
              origin_audience_entity_ids: [aliceGroupSource.audience as EntityId],
              shared: false,
              embedding: queryEmbedding,
              start_time: now - 2_000,
              end_time: now - 1_900,
              created_at: now - 2_000,
              updated_at: now - 2_000,
            }),
            testEpisode(createEpisodeId(), {
              title: "Bob personal episode",
              narrative: "Private context from Bob's personal chat.",
              participants: [episodeParticipantEntityIdTerm(identity.bob)],
              source_stream_ids: [bobPersonalSource.id],
              audience_entity_id: bobPersonalSource.audience as EntityId,
              origin_audience_entity_ids: [bobPersonalSource.audience as EntityId],
              shared: false,
              embedding: queryEmbedding,
              start_time: now - 1_000,
              end_time: now - 900,
              created_at: now - 1_000,
              updated_at: now - 1_000,
            }),
          ];

          for (const episode of episodes) {
            await deps.episodicRepository.createEpisode(episode);
          }

          return Object.fromEntries(episodes.map((episode) => [episode.title, episode.id]));
        },
        { exclusive: true },
      );

      const commitmentResponse = await post(
        base,
        "/memory/commitments",
        {
          tenant: "acme",
          type: "boundary",
          kind: "audience_rule",
          enforcement_class: "critical",
          critical_domain: "privacy",
          directive: "Keep Alice's private context in Alice's audience.",
          family: "Alice private context",
          priority: 20,
          audience_entity_id: identity.alice,
        },
        TOKEN,
      );
      expect(commitmentResponse.status).toBe(201);
      await commitmentResponse.json();

      const directiveResponse = await post(
        base,
        "/memory/directives",
        {
          tenant: "acme",
          kind: "response_policy",
          text: "Keep operational answers concise.",
          content_scope: "allow_list",
          allowed_external_ids: ["alice"],
          topic_tags: ["operations"],
        },
        TOKEN,
      );
      expect(directiveResponse.status).toBe(201);
      const directivePayload = (await directiveResponse.json()) as {
        directive: { id: CreatorDirective["id"] };
      };
      const recallExpansionResponse = {
        text: "",
        input_tokens: 0,
        output_tokens: 0,
        stop_reason: "tool_use" as const,
        tool_calls: [
          {
            id: "toolu_memory_context_expansion",
            name: "EmitRecallExpansion",
            input: { facets: [], named_terms: [] },
          },
        ],
      };
      llmClient.pushResponse(recallExpansionResponse);
      llmClient.pushResponse(recallExpansionResponse);

      const contextResponse = await post(
        base,
        "/memory/context",
        {
          tenant: "acme",
          session: "teams::personal::alice",
          sender: { external_id: "alice", display_name: "Alice", operator: true },
          conversation: { type: "personal", name: "Alice" },
          participants: [{ external_id: "bob", display_name: "Bob", operator: true }],
          query: "situational awareness",
          sections: ["audience", "episodes", "recent_activity", "commitments", "directives"],
        },
        TOKEN,
      );
      const context = (await contextResponse.json()) as {
        audience: { entity_id: EntityId; kind: string; role: string };
        recent_activity: Array<{
          conversation: { type: string; name: string };
          participant_name: string;
          text: string;
        }>;
        commitments: Array<{ directive: string; audience_entity_id: EntityId | null }>;
        directives: Array<{ id: CreatorDirective["id"]; text: string; render_mode: string }>;
        episodes: Array<{ id: Episode["id"] }>;
        hidden_episode_count: number;
        degraded: boolean;
      };

      expect(contextResponse.status).toBe(200);
      expect(context.degraded).toBe(false);
      expect(context.audience).toEqual({
        entity_id: identity.alice,
        kind: "person",
        role: "operator",
        name: "Alice",
      });
      expect(context.recent_activity).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            conversation: { type: "groupChat", name: "AI Ninjas" },
            participant_name: "Alice",
            text: expect.stringContaining('in group chat "AI Ninjas".'),
          }),
        ]),
      );
      expect(context.recent_activity.map((activity) => activity.participant_name)).not.toContain(
        "Bob",
      );
      expect(context.commitments).toEqual([
        expect.objectContaining({
          directive: "Keep Alice's private context in Alice's audience.",
          audience_entity_id: identity.alice,
        }),
      ]);
      expect(context.directives).toEqual([
        expect.objectContaining({
          id: directivePayload.directive.id,
          text: "Keep operational answers concise.",
          render_mode: "content",
        }),
      ]);
      const aliceEpisodeIds = context.episodes.map((episode) => episode.id);
      expect(aliceEpisodeIds).toEqual(
        expect.arrayContaining([
          persistedEpisodes["Public episode"],
          persistedEpisodes["Alice personal episode"],
          persistedEpisodes["Observed group episode"],
        ]),
      );
      expect(aliceEpisodeIds).not.toContain(persistedEpisodes["Bob personal episode"]);
      expect(context.hidden_episode_count).toBe(0);

      const groupContextResponse = await post(
        base,
        "/memory/context",
        {
          tenant: "acme",
          session: "teams::group::ai-ninjas",
          sender: { external_id: "alice", display_name: "Alice" },
          conversation: {
            type: "groupChat",
            name: "AI Ninjas",
            external_id: "ai-ninjas",
          },
          query: "situational awareness",
          sections: ["episodes"],
        },
        TOKEN,
      );
      const groupContext = (await groupContextResponse.json()) as {
        episodes: Array<{ id: Episode["id"] }>;
        hidden_episode_count: number;
      };
      const groupEpisodeIds = groupContext.episodes.map((episode) => episode.id);

      expect(groupContextResponse.status).toBe(200);
      expect(groupEpisodeIds).toEqual(
        expect.arrayContaining([
          persistedEpisodes["Public episode"],
          persistedEpisodes["Observed group episode"],
        ]),
      );
      expect(groupEpisodeIds).not.toContain(persistedEpisodes["Alice personal episode"]);
      expect(groupEpisodeIds).not.toContain(persistedEpisodes["Bob personal episode"]);
      expect(groupContext.hidden_episode_count).toBe(0);

      const retrievalCountsBefore = await pool.withTenant("acme", (borg) =>
        Object.fromEntries(
          Object.values(persistedEpisodes).map((episodeId) => [
            episodeId,
            borg.episodic.getStats(episodeId)?.retrieval_count ?? 0,
          ]),
        ),
      );
      llmClient.pushResponse(recallExpansionResponse);
      const excludedContextResponse = await post(
        base,
        "/memory/context",
        {
          tenant: "acme",
          session: "teams::personal::alice",
          sender: { external_id: "alice", display_name: "Alice", operator: true },
          conversation: { type: "personal", name: "Alice" },
          query: "situational awareness",
          limit: 1,
          sections: ["episodes"],
          exclude: {
            title_prefixes: ["Public episode"],
            narrative_markers: [],
          },
        },
        TOKEN,
      );
      const excludedContext = (await excludedContextResponse.json()) as {
        episodes: Array<{ id: Episode["id"] }>;
      };
      const returnedEpisodeId = excludedContext.episodes[0]?.id;

      expect(excludedContextResponse.status).toBe(200);
      expect(excludedContext.episodes).toHaveLength(1);
      expect(returnedEpisodeId).not.toBe(persistedEpisodes["Public episode"]);

      const retrievalCountsAfter = await pool.withTenant("acme", (borg) =>
        Object.fromEntries(
          Object.values(persistedEpisodes).map((episodeId) => [
            episodeId,
            borg.episodic.getStats(episodeId)?.retrieval_count ?? 0,
          ]),
        ),
      );

      for (const episodeId of Object.values(persistedEpisodes)) {
        expect(retrievalCountsAfter[episodeId]).toBe(
          (retrievalCountsBefore[episodeId] ?? 0) + (episodeId === returnedEpisodeId ? 1 : 0),
        );
      }

      const adminProvenance = await pool.withTenant("acme", (borg) => {
        const adminSession = borg.sessions
          .list()
          .find((session) => session.source_external_id === "memory-sidecar::admin-api");

        if (adminSession === undefined) {
          throw new Error("expected directive admin session");
        }

        return {
          adminSession,
          entries: borg.stream.tail(10, { session: adminSession.session_id }),
        };
      });
      expect(adminProvenance.adminSession).toMatchObject({
        source_type: "memory_sidecar",
        conversation_kind: "dm",
        audience_role: "operator",
      });
      expect(adminProvenance.entries).toEqual([
        expect.objectContaining({
          kind: "internal_event",
          content: expect.objectContaining({
            event: "memory_sidecar.operator_directive_queue_requested",
          }),
        }),
      ]);
    } finally {
      await pool.closeAll();
      rmSync(root, { recursive: true, force: true });
    }
  });
});
