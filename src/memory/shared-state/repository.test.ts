import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  composeMigrations,
  openDatabase,
  type SqliteDatabase,
} from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { IdentityCasMismatchError, StorageError } from "../../util/errors.js";
import { createEntityId, createSharedStateEntryId, createStreamEntryId } from "../../util/ids.js";
import { serializeJsonValue } from "../../util/json-value.js";
import { sharedStateMigrations } from "./migrations.js";
import { SharedStateRepository, type SharedStateOperation } from "./repository.js";

describe("SharedStateRepository", () => {
  let db: SqliteDatabase;
  let repository: SharedStateRepository;
  const clock = new FixedClock(1_000);

  beforeEach(() => {
    db = openDatabase(":memory:", {
      migrations: composeMigrations(sharedStateMigrations),
    });
    repository = new SharedStateRepository({
      db,
      clock,
    });
  });

  afterEach(() => {
    db.close();
  });

  function expectSourceTrustRejection(
    write: () => unknown,
    expected: {
      streamEntryId: string;
      field?: string;
      reason?: string;
    },
  ) {
    let thrown: unknown;

    try {
      write();
    } catch (error) {
      thrown = error;
    }

    expect(thrown).toBeInstanceOf(StorageError);
    expect(thrown).toMatchObject({
      code: "SHARED_STATE_SOURCE_NOT_TRUSTED",
    });
    expect((thrown as Error).cause).toMatchObject({
      streamEntryId: expected.streamEntryId,
      ...(expected.field === undefined ? {} : { field: expected.field }),
      ...(expected.reason === undefined ? {} : { reason: expected.reason }),
    });
  }

  it("returns null for an empty artifact", () => {
    expect(repository.get(createEntityId())).toBeNull();
  });

  it("creates a parent artifact and entries with provenance citations", () => {
    const audience = createEntityId();
    const owner = createEntityId();
    const source = createStreamEntryId();
    const artifact = repository.upsert(
      audience,
      [
        {
          type: "add",
          state_key: "decision.route",
          kind: "locked",
          text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3",
          owner_entity_id: owner,
          provenance_stream_entry_ids: [source],
        },
      ],
      {
        lastCompiledStreamEntryId: source,
      },
    );

    expect(artifact).toMatchObject({
      audience_entity_id: audience,
      record_version: 1,
      last_compiled_stream_entry_id: source,
    });
    expect(artifact?.entries).toHaveLength(1);
    expect(artifact?.entries[0]).toMatchObject({
      kind: "locked",
      state_key: "decision.route",
      text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3",
      owner_entity_id: owner,
      provenance_stream_entry_ids: [source],
      last_updated_stream_entry_ids: [source],
      superseded_by_id: null,
    });
  });

  it("reads legacy entries without a state key as null", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const entryId = createSharedStateEntryId();

    db.prepare(
      `
        INSERT INTO shared_state_artifacts (
          audience_entity_id, record_version, created_at, updated_at,
          last_compiled_at, last_compiled_stream_entry_id
        ) VALUES (?, ?, ?, ?, ?, ?)
      `,
    ).run(audience, 1, clock.now(), clock.now(), null, null);
    db.prepare(
      `
        INSERT INTO shared_state_entries (
          id, audience_entity_id, state_key, kind, text, owner_entity_id,
          provenance_stream_entry_ids, last_updated_stream_entry_ids,
          created_at, last_updated_at, superseded_by_id, rank, canonicalizes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `,
    ).run(
      entryId,
      audience,
      null,
      "live",
      "Legacy-shaped live entry",
      null,
      serializeJsonValue([source]),
      serializeJsonValue([source]),
      clock.now(),
      clock.now(),
      null,
      0,
      serializeJsonValue({
        goal_ids: [],
        commitment_ids: [],
        action_ids: [],
        open_question_ids: [],
      }),
    );

    const artifact = repository.get(audience);

    expect(artifact?.entries[0]?.state_key).toBeNull();
  });

  it("reads legacy pending entries from storage", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const entryId = createSharedStateEntryId();

    db.prepare(
      `
        INSERT INTO shared_state_artifacts (
          audience_entity_id, record_version, created_at, updated_at,
          last_compiled_at, last_compiled_stream_entry_id
        ) VALUES (?, ?, ?, ?, ?, ?)
      `,
    ).run(audience, 1, clock.now(), clock.now(), null, null);
    db.prepare(
      `
        INSERT INTO shared_state_entries (
          id, audience_entity_id, state_key, kind, text, owner_entity_id,
          provenance_stream_entry_ids, last_updated_stream_entry_ids,
          created_at, last_updated_at, superseded_by_id, rank, canonicalizes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `,
    ).run(
      entryId,
      audience,
      "decision.awaiting_verification",
      "pending",
      "Legacy pending shared-state entry",
      null,
      serializeJsonValue([source]),
      serializeJsonValue([source]),
      clock.now(),
      clock.now(),
      null,
      0,
      serializeJsonValue({
        goal_ids: [],
        commitment_ids: [],
        action_ids: [],
        open_question_ids: [],
      }),
    );

    expect(repository.get(audience)?.entries[0]).toMatchObject({
      id: entryId,
      kind: "pending",
      state_key: "decision.awaiting_verification",
      text: "Legacy pending shared-state entry",
    });
  });

  it("rejects add writes without a state key and accepts keyed adds", () => {
    const rejectedAudience = createEntityId();
    const acceptedAudience = createEntityId();
    const rejectedSource = createStreamEntryId();
    const acceptedSource = createStreamEntryId();
    const missingStateKeyOperation = {
      type: "add",
      state_key: null,
      kind: "live",
      text: "Unkeyed live entry",
      provenance_stream_entry_ids: [rejectedSource],
    } as unknown as SharedStateOperation;

    let thrown: unknown;

    try {
      repository.upsert(rejectedAudience, [missingStateKeyOperation]);
    } catch (error) {
      thrown = error;
    }

    expect(thrown).toBeInstanceOf(StorageError);
    expect(thrown).toMatchObject({
      code: "SHARED_STATE_STATE_KEY_REQUIRED",
    });

    const artifact = repository.upsert(acceptedAudience, [
      {
        type: "add",
        state_key: "x",
        kind: "live",
        text: "Keyed live entry",
        provenance_stream_entry_ids: [acceptedSource],
      },
    ]);

    expect(artifact?.entries[0]?.state_key).toBe("x");
  });

  it("accepts internal live lifecycle kinds and transitions kind without changing update metadata", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const entryId = createSharedStateEntryId();
    const initial = repository.upsert(
      audience,
      [
        {
          type: "add",
          id: entryId,
          state_key: "state.fixture",
          kind: "low_salience_live",
          text: "Placeholder state remains represented by its key",
          provenance_stream_entry_ids: [source],
          last_updated_stream_entry_ids: [source],
          created_at: 100,
          last_updated_at: 100,
        },
      ],
      {
        now: 1_000,
      },
    );

    expect(initial?.entries[0]).toMatchObject({
      kind: "low_salience_live",
      last_updated_at: 100,
      last_updated_stream_entry_ids: [source],
    });

    const transitioned = repository.upsert(
      audience,
      [
        {
          type: "transition_kind",
          id: entryId,
          kind: "dormant_live",
        },
      ],
      {
        now: 2_000,
      },
    );

    expect(transitioned?.entries[0]).toMatchObject({
      kind: "dormant_live",
      last_updated_at: 100,
      last_updated_stream_entry_ids: [source],
      last_updated_turn_global: null,
    });
  });

  it("writes a new state key on update while keeping the entry identity", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const entryId = createSharedStateEntryId();

    const initial = repository.upsert(audience, [
      {
        type: "add",
        id: entryId,
        state_key: "audit.inference.supersede_as_durable_correction_path",
        kind: "locked",
        text: "Supersede was recorded as the durable correction path.",
        provenance_stream_entry_ids: [source],
        created_at: 1_000,
      },
    ]);

    expect(initial?.entries[0]?.created_at).toBe(1_000);

    // No text field: the key alone is the change, and the store applies it to the same row.
    const renamed = repository.upsert(audience, [
      {
        type: "update",
        id: entryId,
        state_key: "audit.inference.supersede_binds_old_body_to_new_fate",
        last_updated_stream_entry_ids: [source],
        last_updated_at: 2_000,
      },
    ]);

    expect(renamed?.entries).toHaveLength(1);
    expect(renamed?.entries[0]).toMatchObject({
      id: entryId,
      state_key: "audit.inference.supersede_binds_old_body_to_new_fate",
      kind: "locked",
      text: "Supersede was recorded as the durable correction path.",
      created_at: 1_000,
      last_updated_at: 2_000,
      superseded_by_id: null,
    });
  });

  it("persists last updated global turn on add update and supersede but not kind transitions", () => {
    const audience = createEntityId();
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const thirdSource = createStreamEntryId();
    const entryId = createSharedStateEntryId();

    const initial = repository.upsert(
      audience,
      [
        {
          type: "add",
          id: entryId,
          state_key: "decision.route",
          kind: "live",
          text: "Route is still open",
          provenance_stream_entry_ids: [firstSource],
        },
      ],
      {
        lastUpdatedTurnGlobal: 10,
      },
    );

    expect(initial?.entries[0]?.last_updated_turn_global).toBe(10);

    const updated = repository.upsert(
      audience,
      [
        {
          type: "update",
          id: entryId,
          state_key: "decision.route",
          text: "Route is settled",
          last_updated_stream_entry_ids: [secondSource],
        },
      ],
      {
        lastUpdatedTurnGlobal: 12,
      },
    );

    expect(updated?.entries[0]?.last_updated_turn_global).toBe(12);

    const transitioned = repository.upsert(
      audience,
      [
        {
          type: "transition_kind",
          id: entryId,
          kind: "low_salience_live",
        },
      ],
      {
        lastUpdatedTurnGlobal: 99,
      },
    );

    expect(transitioned?.entries[0]?.last_updated_turn_global).toBe(12);

    const superseded = repository.upsert(
      audience,
      [
        {
          type: "supersede",
          id: entryId,
          replacement: {
            state_key: "decision.route",
            kind: "locked",
            text: "Final route is locked",
            provenance_stream_entry_ids: [thirdSource],
          },
          last_updated_stream_entry_ids: [thirdSource],
        },
      ],
      {
        lastUpdatedTurnGlobal: 15,
      },
    );

    const oldEntry = superseded?.entries.find((entry) => entry.id === entryId);
    const replacement = superseded?.entries.find((entry) => entry.id !== entryId);

    expect(oldEntry?.last_updated_turn_global).toBe(15);
    expect(replacement?.last_updated_turn_global).toBe(15);
  });

  it("rejects add, update, and supersede writes with quarantined source ids when a trust validator is configured", () => {
    const audience = createEntityId();
    const allowedSource = createStreamEntryId();
    const quarantinedSource = createStreamEntryId();
    const trustedRepository = new SharedStateRepository({
      db,
      clock,
      sourceTrustValidator: (streamEntryId) =>
        streamEntryId === quarantinedSource
          ? { allowed: false, reason: "quarantined" }
          : { allowed: true },
    });
    const expectUntrustedSource = (write: () => unknown) => {
      let thrown: unknown;

      try {
        write();
      } catch (error) {
        thrown = error;
      }

      expect(thrown).toBeInstanceOf(StorageError);
      expect(thrown).toMatchObject({
        code: "SHARED_STATE_SOURCE_NOT_TRUSTED",
      });
      expect((thrown as Error).cause).toMatchObject({
        streamEntryId: quarantinedSource,
        reason: "quarantined",
      });
    };

    const initial = trustedRepository.upsert(audience, [
      {
        type: "add",
        state_key: "decision.workstream",
        kind: "locked",
        text: "Canonical workstream decision",
        provenance_stream_entry_ids: [allowedSource],
      },
    ]);
    const entryId = initial?.entries[0]?.id;

    expect(entryId).toBeDefined();
    expect(trustedRepository.get(audience)?.entries[0]?.provenance_stream_entry_ids).toEqual([
      allowedSource,
    ]);
    expectUntrustedSource(() =>
      trustedRepository.upsert(audience, [
        {
          type: "add",
          state_key: "decision.workstream",
          kind: "locked",
          text: "Untrusted canonical decision",
          provenance_stream_entry_ids: [quarantinedSource],
        },
      ]),
    );
    expectUntrustedSource(() =>
      trustedRepository.upsert(audience, [
        {
          type: "update",
          id: entryId!,
          state_key: "decision.workstream",
          text: "Updated canonical workstream decision",
          add_provenance_stream_entry_ids: [quarantinedSource],
          last_updated_stream_entry_ids: [quarantinedSource],
        },
      ]),
    );
    expectUntrustedSource(() =>
      trustedRepository.upsert(audience, [
        {
          type: "supersede",
          id: entryId!,
          replacement: {
            state_key: "decision.workstream",
            kind: "locked",
            text: "Replacement canonical workstream decision",
            provenance_stream_entry_ids: [quarantinedSource],
          },
          last_updated_stream_entry_ids: [quarantinedSource],
        },
      ]),
    );
  });

  it("rejects add writes with quarantined last-updated ids when provenance is trusted", () => {
    const audience = createEntityId();
    const allowedSource = createStreamEntryId();
    const quarantinedSource = createStreamEntryId();
    const trustedRepository = new SharedStateRepository({
      db,
      clock,
      sourceTrustValidator: (streamEntryId) =>
        streamEntryId === quarantinedSource
          ? { allowed: false, reason: "quarantined" }
          : { allowed: true },
    });

    expectSourceTrustRejection(
      () =>
        trustedRepository.upsert(audience, [
          {
            type: "add",
            state_key: "decision.workstream",
            kind: "locked",
            text: "Canonical workstream decision",
            provenance_stream_entry_ids: [allowedSource],
            last_updated_stream_entry_ids: [quarantinedSource],
          },
        ]),
      {
        streamEntryId: quarantinedSource,
        field: "last_updated_stream_entry_ids",
        reason: "quarantined",
      },
    );
  });

  it("rejects update writes with quarantined last-updated ids when provenance is trusted", () => {
    const audience = createEntityId();
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const quarantinedSource = createStreamEntryId();
    const trustedRepository = new SharedStateRepository({
      db,
      clock,
      sourceTrustValidator: (streamEntryId) =>
        streamEntryId === quarantinedSource
          ? { allowed: false, reason: "quarantined" }
          : { allowed: true },
    });
    const initial = trustedRepository.upsert(audience, [
      {
        type: "add",
        state_key: "decision.workstream",
        kind: "locked",
        text: "Canonical workstream decision",
        provenance_stream_entry_ids: [firstSource],
      },
    ]);
    const entryId = initial?.entries[0]?.id;
    const before = trustedRepository.get(audience);

    expect(entryId).toBeDefined();
    expectSourceTrustRejection(
      () =>
        trustedRepository.upsert(audience, [
          {
            type: "update",
            id: entryId!,
            state_key: "decision.workstream",
            text: "Updated canonical workstream decision",
            add_provenance_stream_entry_ids: [secondSource],
            last_updated_stream_entry_ids: [quarantinedSource],
          },
        ]),
      {
        streamEntryId: quarantinedSource,
        field: "last_updated_stream_entry_ids",
        reason: "quarantined",
      },
    );
    expect(trustedRepository.get(audience)).toEqual(before);
  });

  it("rejects supersede writes with quarantined last-updated ids when replacement provenance is trusted", () => {
    const audience = createEntityId();
    const firstSource = createStreamEntryId();
    const replacementSource = createStreamEntryId();
    const quarantinedSource = createStreamEntryId();
    const trustedRepository = new SharedStateRepository({
      db,
      clock,
      sourceTrustValidator: (streamEntryId) =>
        streamEntryId === quarantinedSource
          ? { allowed: false, reason: "quarantined" }
          : { allowed: true },
    });
    const initial = trustedRepository.upsert(audience, [
      {
        type: "add",
        state_key: "decision.workstream",
        kind: "locked",
        text: "Canonical workstream decision",
        provenance_stream_entry_ids: [firstSource],
      },
    ]);
    const entryId = initial?.entries[0]?.id;
    const before = trustedRepository.get(audience);

    expect(entryId).toBeDefined();
    expectSourceTrustRejection(
      () =>
        trustedRepository.upsert(audience, [
          {
            type: "supersede",
            id: entryId!,
            replacement: {
              state_key: "decision.workstream",
              kind: "locked",
              text: "Replacement canonical workstream decision",
              provenance_stream_entry_ids: [replacementSource],
            },
            last_updated_stream_entry_ids: [quarantinedSource],
          },
        ]),
      {
        streamEntryId: quarantinedSource,
        field: "last_updated_stream_entry_ids",
        reason: "quarantined",
      },
    );
    expect(trustedRepository.get(audience)).toEqual(before);
  });

  it("increments the parent record version on update", () => {
    const audience = createEntityId();
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const initial = repository.upsert(audience, [
      {
        type: "add",
        state_key: "question.granada_pacing",
        kind: "live",
        text: "Question: Granada pacing",
        provenance_stream_entry_ids: [firstSource],
      },
    ]);
    const entryId = initial?.entries[0]?.id;

    expect(entryId).toBeDefined();
    const updated = repository.upsert(
      audience,
      [
        {
          type: "update",
          id: entryId!,
          state_key: "question.granada_pacing",
          text: "Question: Granada pacing and accommodation type",
          add_provenance_stream_entry_ids: [secondSource],
          last_updated_stream_entry_ids: [secondSource],
        },
      ],
      {
        expectedVersion: initial?.record_version,
      },
    );

    expect(updated?.record_version).toBe(2);
    expect(updated?.entries[0]).toMatchObject({
      text: "Question: Granada pacing and accommodation type",
      provenance_stream_entry_ids: [firstSource, secondSource],
      last_updated_stream_entry_ids: [secondSource],
    });
  });

  it("creates an empty parent artifact when an empty operation set carries compile metadata", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();

    const artifact = repository.upsert(audience, [], {
      lastCompiledAt: 2_000,
      lastCompiledStreamEntryId: source,
      now: 2_000,
    });

    expect(artifact).toMatchObject({
      audience_entity_id: audience,
      record_version: 1,
      last_compiled_at: 2_000,
      last_compiled_stream_entry_id: source,
      entries: [],
    });
  });

  it("updates compile metadata for an empty operation set and bumps record version", () => {
    const audience = createEntityId();
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const initial = repository.upsert(
      audience,
      [
        {
          type: "add",
          state_key: "question.granada_pacing",
          kind: "live",
          text: "Question: Granada pacing",
          provenance_stream_entry_ids: [firstSource],
        },
      ],
      {
        lastCompiledStreamEntryId: firstSource,
      },
    );

    const updated = repository.upsert(audience, [], {
      expectedVersion: initial?.record_version,
      lastCompiledAt: 2_000,
      lastCompiledStreamEntryId: secondSource,
      now: 2_000,
    });

    expect(updated?.record_version).toBe((initial?.record_version ?? 0) + 1);
    expect(updated?.last_compiled_at).toBe(2_000);
    expect(updated?.last_compiled_stream_entry_id).toBe(secondSource);
    expect(updated?.entries).toEqual(initial?.entries);
  });

  it("rejects a stale marker-only write with the existing CAS mismatch error", () => {
    const audience = createEntityId();
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const staleSource = createStreamEntryId();
    const initial = repository.upsert(audience, [
      {
        type: "add",
        state_key: "question.granada_pacing",
        kind: "live",
        text: "Question: Granada pacing",
        provenance_stream_entry_ids: [firstSource],
      },
    ]);

    repository.upsert(audience, [], {
      expectedVersion: initial?.record_version,
      lastCompiledAt: 2_000,
      lastCompiledStreamEntryId: secondSource,
      now: 2_000,
    });

    const staleWrite = () =>
      repository.upsert(audience, [], {
        expectedVersion: initial?.record_version,
        lastCompiledAt: 1_500,
        lastCompiledStreamEntryId: staleSource,
        now: 1_500,
      });

    let thrown: unknown;

    try {
      staleWrite();
    } catch (error) {
      thrown = error;
    }

    expect(thrown).toBeInstanceOf(IdentityCasMismatchError);
    expect(thrown).toMatchObject({
      code: "IDENTITY_CAS_MISMATCH",
    });
    expect(repository.get(audience)?.last_compiled_stream_entry_id).toBe(secondSource);
  });

  it("throws a CAS mismatch between concurrent repository instances", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-decision-artifact-"));
    const dbPath = join(tempDir, "borg.db");
    const firstDb = openDatabase(dbPath, {
      migrations: composeMigrations(sharedStateMigrations),
    });
    const secondDb = openDatabase(dbPath, {
      migrations: composeMigrations(sharedStateMigrations),
    });
    const firstWriter = new SharedStateRepository({
      db: firstDb,
      clock,
    });
    const secondWriter = new SharedStateRepository({
      db: secondDb,
      clock,
    });
    const audience = createEntityId();
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const thirdSource = createStreamEntryId();

    try {
      firstWriter.upsert(audience, [
        {
          type: "add",
          state_key: "question.toledo_placement",
          kind: "live",
          text: "Question: Toledo placement",
          provenance_stream_entry_ids: [firstSource],
        },
      ]);
      const firstSnapshot = firstWriter.get(audience);
      const secondSnapshot = secondWriter.get(audience);
      const entryId = firstSnapshot?.entries[0]?.id;

      expect(entryId).toBeDefined();
      expect(firstSnapshot?.record_version).toBe(secondSnapshot?.record_version);

      firstWriter.upsert(
        audience,
        [
          {
            type: "update",
            id: entryId!,
            state_key: "question.toledo_placement",
            text: "Question: Toledo placement before Madrid",
            add_provenance_stream_entry_ids: [secondSource],
            last_updated_stream_entry_ids: [secondSource],
          },
        ],
        {
          expectedVersion: firstSnapshot?.record_version,
        },
      );

      expect(() =>
        secondWriter.upsert(
          audience,
          [
            {
              type: "update",
              id: entryId!,
              state_key: "question.toledo_placement",
              text: "Question: Toledo placement after Madrid",
              add_provenance_stream_entry_ids: [thirdSource],
              last_updated_stream_entry_ids: [thirdSource],
            },
          ],
          {
            expectedVersion: secondSnapshot?.record_version,
          },
        ),
      ).toThrow(IdentityCasMismatchError);
    } finally {
      firstDb.close();
      secondDb.close();
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  // Pins the `removal_basis=destructive` line the render carries. An update overwrites `text` in
  // place and a prune is a bare DELETE, so neither leaves the replaced body anywhere in the database
  // -- there is no history table for these rows and nothing writes an audit event against them. If a
  // later change starts retaining either, the render's claim becomes an under-claim and this fails.
  it("keeps no prior body for an updated or pruned entry while supersede retains its predecessor", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const initial = repository.upsert(audience, [
      {
        type: "add",
        state_key: "decision.route",
        kind: "locked",
        text: "original body",
        provenance_stream_entry_ids: [source],
      },
      {
        type: "add",
        state_key: "decision.window",
        kind: "live",
        text: "doomed body",
        provenance_stream_entry_ids: [source],
      },
    ]);
    const updatedId = initial?.entries.find((entry) => entry.state_key === "decision.route")?.id;
    const prunedId = initial?.entries.find((entry) => entry.state_key === "decision.window")?.id;

    expect(updatedId).toBeDefined();
    expect(prunedId).toBeDefined();

    repository.upsert(audience, [
      {
        type: "update",
        id: updatedId!,
        state_key: "decision.route",
        text: "replacement body",
        last_updated_stream_entry_ids: [source],
      },
      {
        type: "prune",
        id: prunedId!,
      },
    ]);

    const rowCountFor = (needle: string): number =>
      (
        db
          .prepare("SELECT COUNT(*) AS count FROM shared_state_entries WHERE text LIKE ?")
          .get(`%${needle}%`) as { count: number }
      ).count;

    expect(rowCountFor("replacement body")).toBe(1);
    expect(rowCountFor("original body")).toBe(0);
    expect(rowCountFor("doomed body")).toBe(0);
    expect(repository.get(audience)?.entries.some((entry) => entry.id === prunedId)).toBe(false);

    const superseded = repository.upsert(audience, [
      {
        type: "supersede",
        id: updatedId!,
        replacement: {
          state_key: "decision.route",
          kind: "locked",
          text: "successor body",
          provenance_stream_entry_ids: [source],
        },
        last_updated_stream_entry_ids: [source],
      },
    ]);

    expect(superseded?.entries.find((entry) => entry.id === updatedId)?.text).toBe(
      "replacement body",
    );
    expect(rowCountFor("replacement body")).toBe(1);
  });

  it("rejects pruning a replacement entry while superseded entries still point to it", () => {
    const audience = createEntityId();
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const initial = repository.upsert(audience, [
      {
        type: "add",
        state_key: "decision.route",
        kind: "locked",
        text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 2",
        provenance_stream_entry_ids: [firstSource],
      },
    ]);
    const oldEntryId = initial?.entries[0]?.id;

    expect(oldEntryId).toBeDefined();
    const superseded = repository.upsert(audience, [
      {
        type: "supersede",
        id: oldEntryId!,
        replacement: {
          state_key: "decision.route",
          kind: "locked",
          text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3",
          provenance_stream_entry_ids: [secondSource],
        },
        last_updated_stream_entry_ids: [secondSource],
      },
    ]);
    const replacementId = superseded?.entries.find((entry) => entry.id !== oldEntryId)?.id;

    expect(replacementId).toBeDefined();
    expect(() =>
      repository.upsert(audience, [
        {
          type: "prune",
          id: replacementId!,
        },
      ]),
    ).toThrow();
    expect(
      repository.get(audience)?.entries.find((entry) => entry.id === oldEntryId),
    ).toMatchObject({
      superseded_by_id: replacementId,
    });
  });

  it("advances record_version once per compile marker regardless of operation count", () => {
    const audience = createEntityId();
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const thirdSource = createStreamEntryId();

    const created = repository.upsert(
      audience,
      [
        {
          type: "add",
          state_key: "decision.route",
          kind: "locked",
          text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3",
          provenance_stream_entry_ids: [firstSource],
        },
      ],
      { lastCompiledStreamEntryId: firstSource },
    );

    expect(created).toMatchObject({ record_version: 1 });

    // A compile that applied nothing still advances the counter, as long as it
    // carried a compile marker -- so a version bump is not evidence of a write.
    const afterNoOp = repository.upsert(audience, [], {
      lastCompiledAt: clock.now(),
      lastCompiledStreamEntryId: secondSource,
    });

    expect(afterNoOp).toMatchObject({
      record_version: 2,
      last_compiled_stream_entry_id: secondSource,
    });
    expect(afterNoOp?.entries).toHaveLength(1);

    // Four applied operations advance it by exactly one, the same as zero did.
    const afterFour = repository.upsert(
      audience,
      [
        {
          type: "add",
          state_key: "decision.flight",
          kind: "locked",
          text: "Locked flight: SS to SVQ at 4:15pm",
          provenance_stream_entry_ids: [thirdSource],
        },
        {
          type: "add",
          state_key: "decision.hotel",
          kind: "live",
          text: "Hotel shortlist is down to two",
          provenance_stream_entry_ids: [thirdSource],
        },
        {
          type: "add",
          state_key: "decision.budget",
          kind: "tentative",
          text: "Budget ceiling may be raised",
          provenance_stream_entry_ids: [thirdSource],
        },
        {
          type: "add",
          state_key: "decision.transfer",
          kind: "live",
          text: "Airport transfer still unbooked",
          provenance_stream_entry_ids: [thirdSource],
        },
      ],
      { lastCompiledStreamEntryId: thirdSource },
    );

    expect(afterFour).toMatchObject({ record_version: 3 });
    expect(afterFour?.entries).toHaveLength(5);

    // No operations and no compile marker is the only non-advancing shape.
    const afterInert = repository.upsert(audience, []);

    expect(afterInert).toMatchObject({
      record_version: 3,
      last_compiled_stream_entry_id: thirdSource,
    });
  });

  it("deletes the parent and cascades entries", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();

    repository.upsert(audience, [
      {
        type: "add",
        state_key: "decision.flight",
        kind: "locked",
        text: "Locked flight: SS to SVQ at 4:15pm",
        provenance_stream_entry_ids: [source],
      },
    ]);

    repository.delete(audience);

    expect(repository.get(audience)).toBeNull();
    const row = db.prepare("SELECT COUNT(*) AS count FROM shared_state_entries").get() as {
      count: number;
    };
    expect(row.count).toBe(0);
  });
});
