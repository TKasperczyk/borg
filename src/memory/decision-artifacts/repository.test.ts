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
import { createEntityId, createStreamEntryId } from "../../util/ids.js";
import { sharedStateMigrations } from "./migrations.js";
import { SharedStateRepository } from "./repository.js";

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
      text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3",
      owner_entity_id: owner,
      provenance_stream_entry_ids: [source],
      last_updated_stream_entry_ids: [source],
      superseded_by_id: null,
    });
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

  it("rejects pruning a replacement entry while superseded entries still point to it", () => {
    const audience = createEntityId();
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const initial = repository.upsert(audience, [
      {
        type: "add",
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

  it("deletes the parent and cascades entries", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();

    repository.upsert(audience, [
      {
        type: "add",
        kind: "locked",
        text: "Locked flight: SS to SVQ at 4:15pm",
        provenance_stream_entry_ids: [source],
      },
    ]);

    repository.delete(audience);

    expect(repository.get(audience)).toBeNull();
    const row = db.prepare("SELECT COUNT(*) AS count FROM decision_artifact_entries").get() as {
      count: number;
    };
    expect(row.count).toBe(0);
  });
});
