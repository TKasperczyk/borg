import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import type { EntityId } from "../../util/ids.js";
import {
  trainOfThoughtJournalEntrySchema,
  trainOfThoughtSchema,
  type TrainOfThought,
  type TrainOfThoughtJournalEntry,
} from "./types.js";

export type TrainOfThoughtAppendInput = {
  text: string;
  selfEntityId: EntityId;
  sourceTurnId?: string | null;
  markerStreamEntryId?: string | null;
  now?: number;
};

export type TrainOfThoughtListOptions = {
  limit?: number;
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

function mapTrainOfThoughtJournalEntryRow(
  row: Record<string, unknown>,
): TrainOfThoughtJournalEntry {
  const parsed = trainOfThoughtJournalEntrySchema.safeParse({
    id: Number(row.id),
    self_entity_id: row.self_entity_id,
    text: row.text,
    disclosure_class: row.disclosure_class,
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
    source_turn_id: row.source_turn_id,
    marker_stream_entry_id: row.marker_stream_entry_id,
  });

  if (!parsed.success) {
    throw new StorageError("Train of thought journal row failed validation", {
      cause: parsed.error,
      code: "TRAIN_OF_THOUGHT_JOURNAL_ROW_INVALID",
    });
  }

  return parsed.data;
}

function trainOfThoughtFromJournalEntry(entry: TrainOfThoughtJournalEntry): TrainOfThought {
  return mapTrainOfThoughtRow(entry);
}

export class TrainOfThoughtRepository {
  private readonly clock: Clock;

  constructor(private readonly options: TrainOfThoughtRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  latest(): TrainOfThoughtJournalEntry | null {
    const row = this.db
      .prepare(
        `
          SELECT
            id,
            self_entity_id,
            text,
            disclosure_class,
            created_at,
            updated_at,
            source_turn_id,
            marker_stream_entry_id
          FROM train_of_thought_journal_entries
          ORDER BY updated_at DESC, id DESC
          LIMIT 1
        `,
      )
      .get() as Record<string, unknown> | undefined;

    return row === undefined ? null : mapTrainOfThoughtJournalEntryRow(row);
  }

  list(options: TrainOfThoughtListOptions = {}): TrainOfThoughtJournalEntry[] {
    const limit = options.limit ?? 20;

    if (!Number.isInteger(limit) || limit <= 0) {
      throw new StorageError("Train of thought journal list limit must be a positive integer", {
        code: "TRAIN_OF_THOUGHT_LIST_LIMIT_INVALID",
      });
    }

    const rows = this.db
      .prepare(
        `
          SELECT
            id,
            self_entity_id,
            text,
            disclosure_class,
            created_at,
            updated_at,
            source_turn_id,
            marker_stream_entry_id
          FROM train_of_thought_journal_entries
          ORDER BY updated_at DESC, id DESC
          LIMIT ?
        `,
      )
      .all(limit) as Record<string, unknown>[];

    return rows.map(mapTrainOfThoughtJournalEntryRow);
  }

  get(): TrainOfThought | null {
    const latest = this.latest();

    return latest === null ? null : trainOfThoughtFromJournalEntry(latest);
  }

  append(input: TrainOfThoughtAppendInput): TrainOfThoughtJournalEntry {
    const now = input.now ?? this.clock.now();

    const result = this.db
      .prepare(
        `
          INSERT INTO train_of_thought_journal_entries (
            self_entity_id,
            text,
            disclosure_class,
            created_at,
            updated_at,
            source_turn_id,
            marker_stream_entry_id
          ) VALUES (?, ?, 'self_private', ?, ?, ?, ?)
        `,
      )
      .run(
        input.selfEntityId,
        input.text,
        now,
        now,
        input.sourceTurnId ?? null,
        input.markerStreamEntryId ?? null,
      );

    const row = this.db
      .prepare(
        `
          SELECT
            id,
            self_entity_id,
            text,
            disclosure_class,
            created_at,
            updated_at,
            source_turn_id,
            marker_stream_entry_id
          FROM train_of_thought_journal_entries
          WHERE id = ?
        `,
      )
      .get(Number(result.lastInsertRowid)) as Record<string, unknown> | undefined;

    if (row === undefined) {
      throw new StorageError("Train of thought journal entry was not stored", {
        code: "TRAIN_OF_THOUGHT_STORE_FAILED",
      });
    }

    return mapTrainOfThoughtJournalEntryRow(row);
  }

  upsert(input: TrainOfThoughtAppendInput): TrainOfThought {
    return trainOfThoughtFromJournalEntry(this.append(input));
  }
}
