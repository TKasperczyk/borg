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
import { IdentityCasMismatchError } from "../../util/errors.js";
import { createEntityId, createStreamEntryId } from "../../util/ids.js";
import { decisionArtifactMigrations } from "./migrations.js";
import { DecisionArtifactRepository } from "./repository.js";

describe("DecisionArtifactRepository", () => {
  let db: SqliteDatabase;
  let repository: DecisionArtifactRepository;
  const clock = new FixedClock(1_000);

  beforeEach(() => {
    db = openDatabase(":memory:", {
      migrations: composeMigrations(decisionArtifactMigrations),
    });
    repository = new DecisionArtifactRepository({
      db,
      clock,
    });
  });

  afterEach(() => {
    db.close();
  });

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

  it("throws a CAS mismatch between concurrent repository instances", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-decision-artifact-"));
    const dbPath = join(tempDir, "borg.db");
    const firstDb = openDatabase(dbPath, {
      migrations: composeMigrations(decisionArtifactMigrations),
    });
    const secondDb = openDatabase(dbPath, {
      migrations: composeMigrations(decisionArtifactMigrations),
    });
    const firstWriter = new DecisionArtifactRepository({
      db: firstDb,
      clock,
    });
    const secondWriter = new DecisionArtifactRepository({
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
