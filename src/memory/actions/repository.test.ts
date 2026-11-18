import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  createActionId,
  createSharedStateEntryId,
  createEntityId,
  createStreamEntryId,
  type ActionId,
} from "../../util/ids.js";
import { actionMigrations } from "./migrations.js";
import {
  ActionRepository,
  STALE_OPEN_DRAW_LIMIT,
  createActionRecordsTableSchema,
} from "./repository.js";
import type { ActionRecord } from "./types.js";
import { actionActivityCutoff, actionActivityTier } from "./activity.js";
import { listActionCandidatesForCognition } from "../../cognition/evidence-ledger/action-threads.js";
import {
  presentActionCandidates,
  retrieveActionCandidates,
} from "../../cognition/actions/action-candidates.js";

class MapEmbeddingClient implements EmbeddingClient {
  readonly embedTexts: string[] = [];
  readonly embedBatchTexts: string[][] = [];

  constructor(private readonly vectors: ReadonlyMap<string, readonly number[]>) {}

  async embed(text: string): Promise<Float32Array> {
    this.embedTexts.push(text);
    const vector = this.vectors.get(text);

    if (vector === undefined) {
      throw new Error(`No scripted embedding for ${text}`);
    }

    return Float32Array.from(vector);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    this.embedBatchTexts.push([...texts]);
    return Promise.all(texts.map((text) => this.embed(text)));
  }

  clearCalls(): void {
    this.embedTexts.length = 0;
    this.embedBatchTexts.length = 0;
  }
}

function makeAction(overrides: Partial<ActionRecord> = {}): ActionRecord {
  const nowMs = overrides.created_at ?? 1_000;

  return {
    id: overrides.id ?? createActionId(),
    description: overrides.description ?? "Review Atlas rollout",
    actor: overrides.actor ?? "borg",
    audience_entity_id: overrides.audience_entity_id ?? null,
    goal_id: overrides.goal_id ?? null,
    open_question_id: overrides.open_question_id ?? null,
    state: overrides.state ?? "committed_to_do",
    confidence: overrides.confidence ?? 0.8,
    provenance_episode_ids: overrides.provenance_episode_ids ?? [],
    provenance_stream_entry_ids: overrides.provenance_stream_entry_ids ?? [createStreamEntryId()],
    created_at: nowMs,
    updated_at: overrides.updated_at ?? nowMs,
    considering_at: overrides.considering_at ?? null,
    committed_at: overrides.committed_at ?? null,
    scheduled_at: overrides.scheduled_at ?? null,
    completed_at: overrides.completed_at ?? null,
    not_done_at: overrides.not_done_at ?? null,
    declined_at: null,
    expired_at: overrides.expired_at ?? null,
    archived_at: overrides.archived_at ?? null,
    unknown_at: overrides.unknown_at ?? null,
    canonicalized_by_artifact_entry_id: overrides.canonicalized_by_artifact_entry_id ?? null,
    session_scope: overrides.session_scope ?? null,
    session_anchor_id: overrides.session_anchor_id ?? null,
    last_referenced_at_ms: overrides.last_referenced_at_ms ?? nowMs,
    last_referenced_turn_counter: overrides.last_referenced_turn_counter ?? null,
    last_referenced_turn_global: overrides.last_referenced_turn_global ?? null,
  };
}

describe("ActionRepository", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  async function openFixture(embeddingClient?: EmbeddingClient) {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-actions-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const table = await store.openTable({
      name: "action_records",
      schema: createActionRecordsTableSchema(4),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: actionMigrations,
    });
    const repository = new ActionRepository({
      db,
      table,
      embeddingClient,
      clock: new FixedClock(5_000),
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    return repository;
  }

  it("returns the effective stored lifecycle counter when ensuring global turn values", () => {
    const db = openDatabase(":memory:", {
      migrations: actionMigrations,
    });
    const repository = new ActionRepository({
      db,
      clock: new FixedClock(1_000),
    });
    const setCounter = (value: number) => {
      db.prepare(
        `
          INSERT INTO action_lifecycle_turn_counter (id, value)
            VALUES ('global', ?)
          ON CONFLICT (id) DO UPDATE SET value = excluded.value
        `,
      ).run(value);
    };
    const counterValue = () =>
      (
        db.prepare("SELECT value FROM action_lifecycle_turn_counter WHERE id = 'global'").get() as {
          value: number;
        }
      ).value;

    try {
      setCounter(100);

      expect(repository.ensureLifecycleTurnGlobal(10)).toBe(100);
      expect(counterValue()).toBe(100);

      setCounter(50);

      expect(repository.ensureLifecycleTurnGlobal(200)).toBe(200);
      expect(counterValue()).toBe(200);
    } finally {
      db.close();
    }
  });

  it("orders the page draw by activity before truncation, preserving four open rows per actor without writes", () => {
    const nowMs = 100 * 86_400_000;
    const cutoff = actionActivityCutoff(nowMs);
    const db = openDatabase(":memory:", { migrations: actionMigrations });
    const repository = new ActionRepository({ db });
    const speaker = createEntityId();
    const otherParticipant = createEntityId();
    try {
      const records = [
        ...Array.from({ length: 300 }, (_, i) =>
          makeAction({ actor: "user", updated_at: cutoff - 1 - i }),
        ),
        ...["borg", speaker, otherParticipant].flatMap((actor) =>
          Array.from({ length: 5 }, (_, i) =>
            makeAction({
              actor: actor as ActionRecord["actor"],
              updated_at: 1 + i,
              last_referenced_at_ms: 10 - i,
            }),
          ),
        ),
        makeAction({
          actor: "user",
          state: "considering",
          updated_at: cutoff,
          last_referenced_at_ms: 1,
        }),
        makeAction({
          actor: "user",
          state: "unknown",
          updated_at: 1,
          last_referenced_at_ms: nowMs,
        }),
        ...["completed", "declined", "not_done", "expired", "archived"].map((state, i) =>
          makeAction({
            actor: "user",
            state: state as ActionRecord["state"],
            updated_at: nowMs - i,
            last_referenced_at_ms: 1,
          }),
        ),
        makeAction({
          actor: "user",
          state: "completed",
          updated_at: cutoff - 1,
          last_referenced_at_ms: 1,
        }),
      ];
      for (const record of records) repository.add({ ...record, created_at: 0 });
      const before = repository.list();
      const draw = listActionCandidatesForCognition({
        actionRepository: repository,
        audienceEntityId: speaker,
        rankParticipantEntityIds: [speaker, otherParticipant],
        nowMs,
        limit: 256,
      }).map((candidate) => candidate.record);
      expect(draw).toHaveLength(12 + STALE_OPEN_DRAW_LIMIT + 8);
      expect(draw.slice(0, 7).map((row) => row.state)).toEqual([
        "unknown",
        "considering",
        "completed",
        "declined",
        "not_done",
        "expired",
        "archived",
      ]);
      for (const actor of ["borg", speaker, otherParticipant]) {
        expect(
          draw.filter((row) => row.actor === actor).map((row) => row.last_referenced_at_ms),
        ).toEqual([10, 9, 8, 7]);
      }
      const all = repository.list({ activityNowMs: nowMs });
      expect(all.map((row) => actionActivityTier(row, nowMs))).toEqual([
        ...Array(2).fill(0),
        ...Array(5).fill(1),
        ...Array(315).fill(2),
        3,
      ]);
      expect(repository.countStaleOpen(nowMs)).toBe(315);
      expect(repository.list()).toEqual(before);
    } finally {
      db.close();
    }
  });

  it("keeps actor reservations when the current audience's live rows saturate the draw", () => {
    const nowMs = 100 * 86_400_000;
    const db = openDatabase(":memory:", { migrations: actionMigrations });
    const repository = new ActionRepository({ db });
    const audience = createEntityId();
    const speaker = createEntityId();
    const participant = createEntityId();
    const elsewhere = createEntityId();
    try {
      const actorRows = ["borg", speaker, participant].flatMap((actor) =>
        Array.from({ length: 4 }, (_, i) =>
          makeAction({
            actor: actor as ActionRecord["actor"],
            audience_entity_id: elsewhere,
            created_at: 0,
            updated_at: i,
          }),
        ),
      );
      for (const row of actorRows) repository.add(row);
      for (let i = 0; i < 300; i++)
        repository.add(
          makeAction({ actor: "user", audience_entity_id: audience, updated_at: nowMs - i }),
        );
      const queries = vi.spyOn(db, "prepare");
      const draw = listActionCandidatesForCognition({
        actionRepository: repository,
        audienceEntityId: audience,
        rankParticipantEntityIds: [speaker, participant],
        nowMs,
        limit: 256,
      });
      // Three actor queries plus the audience prefix; a full draw needs no global fill query.
      expect(queries).toHaveBeenCalledTimes(4);
      queries.mockRestore();
      expect(draw).toHaveLength(256);
      expect(draw.map((candidate) => candidate.record.id)).toEqual(
        expect.arrayContaining(actorRows.map((row) => row.id)),
      );
      expect(
        draw.filter((candidate) => candidate.record.audience_entity_id === audience),
      ).toHaveLength(244);
    } finally {
      db.close();
    }
  });

  it("exempts actor reservations from the stale draw cap while continuing through older closed rows", () => {
    const nowMs = 100 * 86_400_000;
    const db = openDatabase(":memory:", { migrations: actionMigrations });
    const repository = new ActionRepository({ db });
    const actors = Array.from({ length: 5 }, () => createEntityId());
    try {
      const reserved = actors.flatMap((actor) =>
        Array.from({ length: 4 }, (_, i) => makeAction({ actor, updated_at: 2_000 + i })),
      );
      const rest = Array.from({ length: 280 }, (_, i) =>
        makeAction({ actor: "user", updated_at: 1_000 + i }),
      );
      const closed = Array.from({ length: 20 }, (_, i) =>
        makeAction({ actor: "user", state: "completed", updated_at: 1_000 + i }),
      );
      for (const row of [...reserved, ...rest, ...closed]) repository.add(row);
      const draw = repository.list({ activityNowMs: nowMs, reserveActors: actors, limit: 256 });
      expect(draw.filter((row) => row.state !== "completed")).toHaveLength(
        reserved.length + STALE_OPEN_DRAW_LIMIT,
      );
      expect(draw.filter((row) => row.state === "completed")).toHaveLength(20);
      expect(draw.map((row) => row.id)).toEqual(
        expect.arrayContaining(reserved.map((row) => row.id)),
      );
      expect(draw.map((row) => row.id)).toEqual(
        expect.arrayContaining(rest.slice(-STALE_OPEN_DRAW_LIMIT).map((row) => row.id)),
      );
      expect(repository.countStaleOpen(nowMs)).toBe(300);
      expect(repository.list()).toHaveLength(320);
    } finally {
      db.close();
    }
  });

  it("still offers a stale row found by vector similarity outside the extractor's recent pool", async () => {
    const nowMs = 100 * 86_400_000;
    const embeddingClient = new MapEmbeddingClient(
      new Map([
        ["Old promise", [1, 0, 0, 0]],
        ["Current row", [0, 1, 0, 0]],
        ["Query", [1, 0, 0, 0]],
      ]),
    );
    const repository = await openFixture(embeddingClient);
    const old = makeAction({
      description: "Old promise",
      actor: "user",
      created_at: 0,
      updated_at: 1,
    });
    repository.add(old);
    const storedOld = repository.get(old.id);
    for (let i = 0; i < 85; i++)
      repository.add(
        makeAction({ description: "Current row", actor: "user", updated_at: nowMs - i }),
      );
    await repository.waitForPendingEmbeddings();
    expect(
      repository.list({ prioritizeActive: true, limit: 80 }).some((row) => row.id === old.id),
    ).toBe(false);
    const result = await retrieveActionCandidates({
      repository,
      text: "Query",
      audienceEntityId: null,
      onSearchFailure: (error) => {
        throw error;
      },
    });
    expect(
      presentActionCandidates(result.records, result.total).shown.some((row) => row.id === old.id),
    ).toBe(true);
    expect(repository.get(old.id)).toEqual(storedOld);
  });

  it("adds, gets, updates, and lists action records by state actor and audience", async () => {
    const repository = await openFixture();
    const audienceEntityId = createEntityId();
    const first = makeAction({
      actor: "borg",
      audience_entity_id: audienceEntityId,
      state: "committed_to_do",
    });
    const second = makeAction({
      description: "User reviewed the incident summary",
      actor: "user",
      state: "completed",
      completed_at: 2_000,
      updated_at: 2_000,
    });
    const thirdAudienceEntityId = createEntityId();
    const third = makeAction({
      description: "Prepare private channel follow-up",
      actor: "borg",
      audience_entity_id: thirdAudienceEntityId,
      state: "scheduled",
      scheduled_at: 3_000,
      updated_at: 3_000,
    });

    repository.add(first);
    repository.add(second);
    repository.add(third);
    repository.update(first.id, {
      state: "completed",
      confidence: 0.95,
    });

    expect(repository.get(first.id)).toMatchObject({
      description: "Review Atlas rollout",
      state: "completed",
      confidence: 0.95,
      updated_at: 5_000,
      completed_at: 5_000,
    });
    expect(repository.list({ state: "completed" }).map((record) => record.id)).toEqual([
      first.id,
      second.id,
    ]);
    expect(repository.list({ actor: "user" }).map((record) => record.id)).toEqual([second.id]);
    expect(repository.list({ audienceEntityId }).map((record) => record.id as ActionId)).toEqual([
      first.id,
    ]);
    expect(repository.list({ audienceEntityId: null }).map((record) => record.id)).toEqual([
      second.id,
    ]);
    expect(repository.list({ recallAllAudiences: true }).map((record) => record.id)).toEqual([
      first.id,
      third.id,
      second.id,
    ]);
  });

  it("lists action records within a created-at range", async () => {
    // The day summarizer feeds a closed day's own action records to the
    // narrative writer; the range must not leak neighboring days' attempts.
    const repository = await openFixture();
    const before = makeAction({ created_at: 1_000 });
    const inside = makeAction({ description: "Attempt inside the day", created_at: 5_000 });
    const after = makeAction({ description: "Attempt after the day", created_at: 9_000 });

    repository.add(before);
    repository.add(inside);
    repository.add(after);

    expect(
      repository
        .list({ recallAllAudiences: true, createdSinceMs: 2_000, createdUntilMs: 8_000 })
        .map((record) => record.id),
    ).toEqual([inside.id]);
  });

  it("counts active and canonicalized action records", async () => {
    const repository = await openFixture();

    repository.add(
      makeAction({
        state: "considering",
        considering_at: 1_000,
      }),
      { creationSource: "extractor" },
    );
    repository.add(
      makeAction({
        description: "Schedule Atlas rollout",
        state: "scheduled",
        scheduled_at: 1_000,
      }),
      { creationSource: "api" },
    );
    repository.add(
      makeAction({
        description: "Complete Atlas rollout",
        state: "completed",
        completed_at: 1_000,
        canonicalized_by_artifact_entry_id: createSharedStateEntryId(),
      }),
      { creationSource: "reflector" },
    );
    repository.add(
      makeAction({
        description: "Unknown Atlas action",
        state: "unknown",
        unknown_at: 1_000,
      }),
    );

    expect(repository.countActive()).toBe(3);
    expect(repository.countCanonicalized()).toBe(1);
    expect(repository.getCreationCountsBySource()).toEqual({
      extractor: 1,
      reflector: 1,
      api: 1,
      tool: 0,
      unknown: 1,
    });
  });

  it("finds action records by description vector similarity", async () => {
    const repository = await openFixture(
      new MapEmbeddingClient(
        new Map([
          ["Review Atlas rollout", [1, 0, 0, 0]],
          ["Draft billing follow-up", [0, 1, 0, 0]],
          ["Atlas rollout", [1, 0, 0, 0]],
        ]),
      ),
    );
    const atlas = makeAction({
      description: "Review Atlas rollout",
      state: "completed",
      completed_at: 2_000,
    });
    const billing = makeAction({
      description: "Draft billing follow-up",
      state: "completed",
      completed_at: 3_000,
    });

    repository.add(atlas);
    repository.add(billing);
    await repository.waitForPendingEmbeddings();

    await expect(repository.findByDescription("Atlas rollout", 1)).resolves.toEqual([atlas]);
  });

  it("finds description similarity pairs above threshold with one embedding batch", async () => {
    const repository = await openFixture(
      new MapEmbeddingClient(
        new Map([
          ["Review Atlas rollout", [1, 0, 0, 0]],
          ["Check Atlas deployment", [0.9, 0.1, 0, 0]],
          ["Draft billing follow-up", [0, 1, 0, 0]],
        ]),
      ),
    );
    const review = makeAction({
      description: "Review Atlas rollout",
    });
    const check = makeAction({
      description: "Check Atlas deployment",
    });
    const billing = makeAction({
      description: "Draft billing follow-up",
    });

    await expect(
      repository.findSimilarDescriptionPairs([review, check, billing], 0.85),
    ).resolves.toEqual([
      {
        leftId: review.id,
        rightId: check.id,
        similarity: expect.any(Number),
      },
    ]);
  });

  it("uses stored action embeddings for description similarity pairs", async () => {
    const embeddingClient = new MapEmbeddingClient(
      new Map([
        ["Review Atlas rollout", [1, 0, 0, 0]],
        ["Check Atlas deployment", [0.9, 0.1, 0, 0]],
        ["Draft billing follow-up", [0, 1, 0, 0]],
      ]),
    );
    const repository = await openFixture(embeddingClient);
    const review = makeAction({
      description: "Review Atlas rollout",
    });
    const check = makeAction({
      description: "Check Atlas deployment",
    });
    const billing = makeAction({
      description: "Draft billing follow-up",
    });

    repository.add(review);
    repository.add(check);
    repository.add(billing);
    await repository.waitForPendingEmbeddings();
    embeddingClient.clearCalls();

    await expect(
      repository.findSimilarDescriptionPairs([review, check, billing], 0.85),
    ).resolves.toEqual([
      {
        leftId: review.id,
        rightId: check.id,
        similarity: expect.any(Number),
      },
    ]);
    expect(embeddingClient.embedBatchTexts).toEqual([]);
  });
});
