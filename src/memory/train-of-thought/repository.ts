import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import type { EntityId } from "../../util/ids.js";
import { trainOfThoughtSchema, type TrainOfThought } from "./types.js";

export type TrainOfThoughtUpsertInput = {
  text: string;
  selfEntityId: EntityId;
  now?: number;
};

export type TrainOfThoughtRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function mapTrainOfThoughtRow(row: Record<string, unknown>): TrainOfThought {
  const parsed = trainOfThoughtSchema.safeParse({
    self_entity_id: row.self_entity_id,
    text: row.text,
    disclosure_class: row.disclosure_class,
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
  });

  if (!parsed.success) {
    throw new StorageError("Train of thought row failed validation", {
      cause: parsed.error,
      code: "TRAIN_OF_THOUGHT_ROW_INVALID",
    });
  }

  return parsed.data;
}

export class TrainOfThoughtRepository {
  private readonly clock: Clock;

  constructor(private readonly options: TrainOfThoughtRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  get(): TrainOfThought | null {
    const row = this.db
      .prepare(
        `
          SELECT self_entity_id, text, disclosure_class, created_at, updated_at
          FROM train_of_thought
          WHERE id = 1
        `,
      )
      .get() as Record<string, unknown> | undefined;

    return row === undefined ? null : mapTrainOfThoughtRow(row);
  }

  upsert(input: TrainOfThoughtUpsertInput): TrainOfThought {
    const now = input.now ?? this.clock.now();

    this.db
      .prepare(
        `
          INSERT INTO train_of_thought (
            id, self_entity_id, text, disclosure_class, created_at, updated_at
          ) VALUES (1, ?, ?, 'self_private', ?, ?)
          ON CONFLICT(id) DO UPDATE SET
            self_entity_id = excluded.self_entity_id,
            text = excluded.text,
            disclosure_class = 'self_private',
            updated_at = excluded.updated_at
        `,
      )
      .run(input.selfEntityId, input.text, now, now);

    const stored = this.get();

    if (stored === null) {
      throw new StorageError("Train of thought row was not stored", {
        code: "TRAIN_OF_THOUGHT_STORE_FAILED",
      });
    }

    return stored;
  }
}
