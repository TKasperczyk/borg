import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";

import { selfMigrations } from "./migrations.js";
import { TraitsRepository, ValuesRepository } from "./repository.js";

describe("self migrations", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("upgrades self evidence-event provenance checks to allow online writes", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "self.db");
    tempDirs.push(tempDir);

    const initialDb = openDatabase(dbPath, {
      migrations: selfMigrations.filter((migration) => migration.id < 264),
    });
    initialDb.close();

    const db = openDatabase(dbPath, {
      migrations: selfMigrations,
    });
    const clock = new FixedClock(12_000);
    const values = new ValuesRepository({
      db,
      clock,
    });
    const traits = new TraitsRepository({
      db,
      clock,
    });

    try {
      const value = values.add({
        label: "clarity",
        description: "Prefer explicit state.",
        priority: 1,
        provenance: {
          kind: "manual",
        },
      });
      values.reinforce(value.id, {
        kind: "online",
        process: "reflector",
      });
      values.recordContradiction({
        valueId: value.id,
        provenance: {
          kind: "online",
          process: "overseer",
        },
      });

      traits.reinforce({
        label: "patient",
        delta: 0.2,
        provenance: {
          kind: "manual",
        },
      });
      const trait = traits.reinforce({
        label: "patient",
        delta: 0.1,
        provenance: {
          kind: "online",
          process: "reflector",
        },
      });
      traits.recordContradiction({
        label: "patient",
        provenance: {
          kind: "online",
          process: "overseer",
        },
      });

      expect(values.listReinforcementEvents(value.id)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            provenance: {
              kind: "online",
              process: "reflector",
            },
          }),
        ]),
      );
      expect(values.listContradictionEvents(value.id)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            provenance: {
              kind: "online",
              process: "overseer",
            },
          }),
        ]),
      );
      expect(traits.listReinforcementEvents(trait.id)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            provenance: {
              kind: "online",
              process: "reflector",
            },
          }),
        ]),
      );
      expect(traits.listContradictionEvents(trait.id)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            provenance: {
              kind: "online",
              process: "overseer",
            },
          }),
        ]),
      );
    } finally {
      db.close();
    }
  });
});
