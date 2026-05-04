import { parseJsonArray, type JsonArrayCodecOptions } from "../../storage/codecs.js";
import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createRelationalSlotId,
  parseRelationalSlotId,
  type EntityId,
  type RelationalSlotId,
  type StreamEntryId,
} from "../../util/ids.js";
import {
  RELATIONAL_SLOT_STATES,
  relationalSlotAssertionSchema,
  relationalSlotNegationSchema,
  relationalSlotSchema,
  relationalSlotStateSchema,
  type RelationalSlot,
  type RelationalSlotAlternateValue,
  type RelationalSlotAssertion,
  type RelationalSlotNegation,
  type RelationalSlotState,
} from "./types.js";

const RELATIONAL_SLOT_JSON_CODEC = {
  errorCode: "RELATIONAL_SLOT_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse relational slot ${label}`,
} satisfies JsonArrayCodecOptions;

export type RelationalSlotRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

export type RelationalSlotListOptions = {
  subjectEntityId?: EntityId;
  states?: readonly RelationalSlotState[];
  limit?: number;
};

export type RelationalSlotApplyResult = {
  previous: RelationalSlot | null;
  slot: RelationalSlot;
  constrained: boolean;
  neutral_phrase: string;
  values_to_neutralize: string[];
};

function uniqueTrimmed(values: readonly string[]): string[] {
  const unique: string[] = [];

  for (const value of values) {
    const trimmed = value.trim();

    if (trimmed.length === 0) {
      continue;
    }

    if (unique.some((existing) => existing === trimmed)) {
      continue;
    }

    unique.push(trimmed);
  }

  return unique;
}

function uniqueStreamEntryIds(values: readonly StreamEntryId[]): StreamEntryId[] {
  const unique: StreamEntryId[] = [];

  for (const value of values) {
    if (unique.some((existing) => existing === value)) {
      continue;
    }

    unique.push(value);
  }

  return unique;
}

function mapRelationalSlotRow(row: Record<string, unknown>): RelationalSlot {
  const parsed = relationalSlotSchema.safeParse({
    id: row.id,
    subject_entity_id: row.subject_entity_id,
    slot_key: row.slot_key,
    value: row.value,
    state: row.state,
    evidence_stream_entry_ids: parseJsonArray<StreamEntryId>(
      String(row.evidence_stream_entry_ids ?? "[]"),
      "evidence_stream_entry_ids",
      RELATIONAL_SLOT_JSON_CODEC,
    ),
    contradicted_by_stream_entry_ids: parseJsonArray<StreamEntryId>(
      String(row.contradicted_by_stream_entry_ids ?? "[]"),
      "contradicted_by_stream_entry_ids",
      RELATIONAL_SLOT_JSON_CODEC,
    ),
    alternate_values: parseJsonArray<RelationalSlotAlternateValue>(
      String(row.alternate_values ?? "[]"),
      "alternate_values",
      RELATIONAL_SLOT_JSON_CODEC,
    ),
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
  });

  if (!parsed.success) {
    throw new StorageError("Relational slot row failed validation", {
      cause: parsed.error,
      code: "RELATIONAL_SLOT_ROW_INVALID",
    });
  }

  return parsed.data;
}

function containsState(
  states: readonly RelationalSlotState[],
  state: RelationalSlotState,
): boolean {
  return states.some((candidate) => candidate === state);
}

function neutralPhraseForSlotKey(slotKey: string): string {
  switch (slotKey) {
    case "partner.name":
    case "partner.role":
      return "your partner";
    case "dog.name":
      return "your dog";
    default:
      return "that relation";
  }
}

function allSlotValues(slot: RelationalSlot): string[] {
  return uniqueTrimmed([slot.value, ...slot.alternate_values.map((alternate) => alternate.value)]);
}

function addAlternateValue(
  alternates: readonly RelationalSlotAlternateValue[],
  value: string,
  evidenceStreamEntryIds: readonly StreamEntryId[],
): RelationalSlotAlternateValue[] {
  let found = false;
  const next = alternates.map((alternate) => {
    if (alternate.value !== value) {
      return alternate;
    }

    found = true;
    return {
      ...alternate,
      evidence_stream_entry_ids: uniqueStreamEntryIds([
        ...alternate.evidence_stream_entry_ids,
        ...evidenceStreamEntryIds,
      ]),
    };
  });

  if (found) {
    return next;
  }

  return [
    ...next,
    {
      value,
      evidence_stream_entry_ids: uniqueStreamEntryIds([...evidenceStreamEntryIds]),
    },
  ];
}

function hasKnownValue(slot: RelationalSlot, value: string): boolean {
  if (slot.value === value) {
    return true;
  }

  return slot.alternate_values.some((alternate) => alternate.value === value);
}

function distinctValueCount(slot: RelationalSlot): number {
  return allSlotValues(slot).length;
}

function buildResult(
  previous: RelationalSlot | null,
  slot: RelationalSlot,
): RelationalSlotApplyResult {
  const constrained = slot.state === "contested" || slot.state === "quarantined";

  return {
    previous,
    slot,
    constrained,
    neutral_phrase: neutralPhraseForSlotKey(slot.slot_key),
    values_to_neutralize: constrained ? allSlotValues(slot) : [],
  };
}

export class RelationalSlotRepository {
  private readonly clock: Clock;

  constructor(private readonly options: RelationalSlotRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private runSlotWrite<T>(callback: () => T): T {
    if (this.db.raw.inTransaction) {
      return callback();
    }

    return this.db.transaction(callback).immediate();
  }

  private upsert(slot: RelationalSlot): void {
    this.db
      .prepare(
        `
          INSERT INTO relational_slots (
            id, subject_entity_id, slot_key, value, state,
            evidence_stream_entry_ids, contradicted_by_stream_entry_ids,
            alternate_values, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(subject_entity_id, slot_key) DO UPDATE SET
            value = excluded.value,
            state = excluded.state,
            evidence_stream_entry_ids = excluded.evidence_stream_entry_ids,
            contradicted_by_stream_entry_ids = excluded.contradicted_by_stream_entry_ids,
            alternate_values = excluded.alternate_values,
            updated_at = excluded.updated_at
        `,
      )
      .run(
        slot.id,
        slot.subject_entity_id,
        slot.slot_key,
        slot.value,
        slot.state,
        serializeJsonValue(slot.evidence_stream_entry_ids),
        serializeJsonValue(slot.contradicted_by_stream_entry_ids),
        serializeJsonValue(slot.alternate_values),
        slot.created_at,
        slot.updated_at,
      );
  }

  get(id: RelationalSlotId): RelationalSlot | null {
    const row = this.db.prepare("SELECT * FROM relational_slots WHERE id = ?").get(id) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : mapRelationalSlotRow(row);
  }

  findBySubjectAndKey(subjectEntityId: EntityId, slotKey: string): RelationalSlot | null {
    const row = this.db
      .prepare(
        `
          SELECT *
          FROM relational_slots
          WHERE subject_entity_id = ? AND slot_key = ?
        `,
      )
      .get(subjectEntityId, slotKey.trim()) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapRelationalSlotRow(row);
  }

  list(options: RelationalSlotListOptions = {}): RelationalSlot[] {
    const filters: string[] = [];
    const values: unknown[] = [];

    if (options.subjectEntityId !== undefined) {
      filters.push("subject_entity_id = ?");
      values.push(options.subjectEntityId);
    }

    if (options.states !== undefined && options.states.length > 0) {
      for (const state of options.states) {
        relationalSlotStateSchema.parse(state);
      }

      filters.push(`state IN (${options.states.map(() => "?").join(", ")})`);
      values.push(...options.states);
    }

    const limit = options.limit ?? 100;
    const whereClause = filters.length === 0 ? "" : `WHERE ${filters.join(" AND ")}`;
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM relational_slots
          ${whereClause}
          ORDER BY updated_at DESC, id ASC
          LIMIT ?
        `,
      )
      .all(...values, limit) as Record<string, unknown>[];

    return rows.map((row) => mapRelationalSlotRow(row));
  }

  listConstrained(options: Omit<RelationalSlotListOptions, "states"> = {}): RelationalSlot[] {
    return this.list({
      ...options,
      states: ["contested", "quarantined"],
    });
  }

  applyAssertion(assertion: RelationalSlotAssertion): RelationalSlotApplyResult {
    const parsed = relationalSlotAssertionSchema.parse({
      ...assertion,
      slot_key: assertion.slot_key.trim(),
      asserted_value: assertion.asserted_value.trim(),
      source_stream_entry_ids: uniqueStreamEntryIds(assertion.source_stream_entry_ids),
    });

    return this.runSlotWrite(() => {
      const current = this.findBySubjectAndKey(parsed.subject_entity_id, parsed.slot_key);
      const nowMs = this.clock.now();

      if (current === null) {
        const slot = relationalSlotSchema.parse({
          id: createRelationalSlotId(),
          subject_entity_id: parsed.subject_entity_id,
          slot_key: parsed.slot_key,
          value: parsed.asserted_value,
          state: "established",
          evidence_stream_entry_ids: parsed.source_stream_entry_ids,
          contradicted_by_stream_entry_ids: [],
          alternate_values: [],
          created_at: nowMs,
          updated_at: nowMs,
        });

        this.upsert(slot);
        return buildResult(null, slot);
      }

      let next = current;

      if (parsed.asserted_value === current.value) {
        next = relationalSlotSchema.parse({
          ...current,
          evidence_stream_entry_ids: uniqueStreamEntryIds([
            ...current.evidence_stream_entry_ids,
            ...parsed.source_stream_entry_ids,
          ]),
          updated_at: nowMs,
        });
      } else {
        const alternateValues = addAlternateValue(
          current.alternate_values,
          parsed.asserted_value,
          parsed.source_stream_entry_ids,
        );
        const candidate = relationalSlotSchema.parse({
          ...current,
          state:
            current.state === "established"
              ? "contested"
              : current.state === "revoked"
                ? "established"
                : current.state,
          alternate_values: alternateValues,
          contradicted_by_stream_entry_ids: uniqueStreamEntryIds([
            ...current.contradicted_by_stream_entry_ids,
            ...parsed.source_stream_entry_ids,
          ]),
          updated_at: nowMs,
        });
        next = relationalSlotSchema.parse({
          ...candidate,
          state: distinctValueCount(candidate) >= 3 ? "quarantined" : candidate.state,
        });
      }

      this.upsert(next);
      return buildResult(current, next);
    });
  }

  applyNegation(negation: RelationalSlotNegation): RelationalSlotApplyResult | null {
    const parsed = relationalSlotNegationSchema.parse({
      ...negation,
      slot_key: negation.slot_key.trim(),
      rejected_value: negation.rejected_value?.trim() ?? null,
      source_stream_entry_ids: uniqueStreamEntryIds(negation.source_stream_entry_ids),
    });

    return this.runSlotWrite(() => {
      const current = this.findBySubjectAndKey(parsed.subject_entity_id, parsed.slot_key);

      if (current === null) {
        return null;
      }

      if (parsed.rejected_value !== null && !hasKnownValue(current, parsed.rejected_value)) {
        return null;
      }

      const next = relationalSlotSchema.parse({
        ...current,
        state: containsState(RELATIONAL_SLOT_STATES, current.state) ? "quarantined" : current.state,
        contradicted_by_stream_entry_ids: uniqueStreamEntryIds([
          ...current.contradicted_by_stream_entry_ids,
          ...parsed.source_stream_entry_ids,
        ]),
        updated_at: this.clock.now(),
      });

      this.upsert(next);
      return buildResult(current, next);
    });
  }

  restore(slot: RelationalSlot): RelationalSlot {
    const parsed = relationalSlotSchema.parse(slot);
    this.runSlotWrite(() => {
      this.upsert(parsed);
    });
    return parsed;
  }

  setState(id: RelationalSlotId | string, state: RelationalSlotState): RelationalSlot | null {
    const slotId = typeof id === "string" ? parseRelationalSlotId(id) : id;
    const current = this.get(slotId);

    if (current === null) {
      return null;
    }

    const next = relationalSlotSchema.parse({
      ...current,
      state,
      updated_at: this.clock.now(),
    });
    this.upsert(next);
    return next;
  }
}

export { neutralPhraseForSlotKey };
