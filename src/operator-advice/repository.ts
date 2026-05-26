import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { StorageError } from "../util/errors.js";
import {
  createOperatorAdviceId,
  operatorAdviceIdHelpers,
  type OperatorAdviceId,
} from "../util/ids.js";

import {
  operatorAdviceIdSchema,
  operatorAdviceListFilterSchema,
  operatorAdviceMarkConsumedInputSchema,
  operatorAdviceQueueInputSchema,
  operatorAdviceRecordSchema,
  type OperatorAdviceListFilter,
  type OperatorAdviceMarkConsumedInput,
  type OperatorAdviceQueueInput,
  type OperatorAdviceRecord,
} from "./types.js";

const DEFAULT_TTL_MS = 24 * 60 * 60 * 1_000;
const DEFAULT_LIST_LIMIT = 100;
const MAX_LIST_LIMIT = 1_000;

type OperatorAdviceRow = {
  id: string;
  session_id: string | null;
  audience_entity_id: string | null;
  text: string;
  created_at: number;
  expires_at: number | null;
  consumed_at: number | null;
  consumed_by_turn_id: string | null;
  canceled_at: number | null;
};

function boundedLimit(limit: number | undefined): number {
  return Math.min(Math.max(1, limit ?? DEFAULT_LIST_LIMIT), MAX_LIST_LIMIT);
}

function mapOperatorAdviceRow(row: OperatorAdviceRow): OperatorAdviceRecord {
  const parsed = operatorAdviceRecordSchema.safeParse(row);

  if (!parsed.success) {
    throw new StorageError("Operator advice row failed validation", {
      cause: parsed.error,
      code: "OPERATOR_ADVICE_ROW_INVALID",
    });
  }

  return parsed.data;
}

export class OperatorAdviceRepository {
  constructor(
    private readonly db: SqliteDatabase,
    private readonly clock: Clock = new SystemClock(),
  ) {}

  queue(input: OperatorAdviceQueueInput): OperatorAdviceRecord {
    const parsed = operatorAdviceQueueInputSchema.safeParse(input);
    if (!parsed.success) {
      throw new StorageError("Invalid operator advice input", {
        cause: parsed.error,
        code: "OPERATOR_ADVICE_INVALID",
      });
    }

    const now = this.clock.now();
    const id = createOperatorAdviceId();
    const expiresAt = parsed.data.expires_at ?? now + DEFAULT_TTL_MS;

    this.db
      .prepare(
        `
          INSERT INTO operator_advice (
            id, session_id, audience_entity_id, text, created_at, expires_at,
            consumed_at, consumed_by_turn_id, canceled_at
          ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
        `,
      )
      .run(
        id,
        parsed.data.session_id ?? null,
        parsed.data.audience_entity_id ?? null,
        parsed.data.text,
        now,
        expiresAt,
      );

    const record = this.get(id);
    if (record === null) {
      throw new StorageError(`Operator advice ${id} was not stored`, {
        code: "OPERATOR_ADVICE_QUEUE_FAILED",
      });
    }

    return record;
  }

  get(id: OperatorAdviceId): OperatorAdviceRecord | null {
    const parsedId = operatorAdviceIdSchema.parse(id);
    const row = this.db
      .prepare(
        `
          SELECT
            id, session_id, audience_entity_id, text, created_at, expires_at,
            consumed_at, consumed_by_turn_id, canceled_at
          FROM operator_advice
          WHERE id = ?
        `,
      )
      .get(parsedId) as OperatorAdviceRow | undefined;

    return row === undefined ? null : mapOperatorAdviceRow(row);
  }

  list(filter?: OperatorAdviceListFilter): OperatorAdviceRecord[] {
    const parsed =
      filter === undefined ? undefined : operatorAdviceListFilterSchema.parse(filter);
    const filters: string[] = [];
    const values: unknown[] = [];
    const now = this.clock.now();

    if (parsed?.pendingOnly === true) {
      filters.push("consumed_at IS NULL");
      filters.push("canceled_at IS NULL");
      filters.push("(expires_at IS NULL OR expires_at > ?)");
      values.push(now);
    }

    const sessionId = parsed?.session_id ?? null;
    const audienceEntityId = parsed?.audience_entity_id ?? null;
    if (sessionId !== null && audienceEntityId !== null) {
      filters.push("(session_id = ? OR audience_entity_id = ?)");
      values.push(sessionId, audienceEntityId);
    } else if (sessionId !== null) {
      filters.push("session_id = ?");
      values.push(sessionId);
    } else if (audienceEntityId !== null) {
      filters.push("audience_entity_id = ?");
      values.push(audienceEntityId);
    }

    values.push(boundedLimit(parsed?.limit));

    const where = filters.length === 0 ? "" : `WHERE ${filters.join(" AND ")}`;
    const orderBy =
      parsed?.pendingOnly === true
        ? "created_at ASC, id ASC"
        : parsed?.pendingOnly === false
          ? [
              "COALESCE(consumed_at, canceled_at, expires_at, created_at) DESC",
              "created_at DESC",
              "id DESC",
            ].join(", ")
          : "created_at ASC, id ASC";
    const rows = this.db
      .prepare(
        `
          SELECT
            id, session_id, audience_entity_id, text, created_at, expires_at,
            consumed_at, consumed_by_turn_id, canceled_at
          FROM operator_advice
          ${where}
          ORDER BY ${orderBy}
          LIMIT ?
        `,
      )
      .all(...values) as OperatorAdviceRow[];

    return rows.map(mapOperatorAdviceRow);
  }

  cancel(id: OperatorAdviceId): OperatorAdviceRecord | null {
    const parsedId = operatorAdviceIdSchema.parse(id);
    const current = this.get(parsedId);
    if (current === null) {
      return null;
    }

    if (
      current.consumed_at === null &&
      current.canceled_at === null &&
      (current.expires_at === null || current.expires_at > this.clock.now())
    ) {
      this.db
        .prepare(
          `
            UPDATE operator_advice
            SET canceled_at = ?
            WHERE id = ?
          `,
        )
        .run(this.clock.now(), parsedId);
    }

    return this.get(parsedId);
  }

  markConsumed(
    ids: readonly OperatorAdviceId[],
    input: OperatorAdviceMarkConsumedInput,
  ): OperatorAdviceRecord[] {
    const parsed = operatorAdviceMarkConsumedInputSchema.parse(input);
    const uniqueIds = [...new Set(ids)];
    if (uniqueIds.length === 0) {
      return [];
    }

    for (const id of uniqueIds) {
      if (!operatorAdviceIdHelpers.is(id)) {
        throw new StorageError(`Invalid operator advice id: ${id}`, {
          code: "OPERATOR_ADVICE_INVALID_ID",
        });
      }
    }

    const now = parsed.now ?? this.clock.now();
    const updateOne = this.db.prepare(
      `
        UPDATE operator_advice
        SET consumed_at = ?, consumed_by_turn_id = ?
        WHERE id = ?
          AND consumed_at IS NULL
          AND canceled_at IS NULL
          AND (expires_at IS NULL OR expires_at > ?)
      `,
    );
    const getOne = this.db.prepare(
      `
        SELECT
          id, session_id, audience_entity_id, text, created_at, expires_at,
          consumed_at, consumed_by_turn_id, canceled_at
        FROM operator_advice
        WHERE id = ?
      `,
    );
    const consume = this.db.transaction((adviceIds: readonly OperatorAdviceId[]) => {
      const records: OperatorAdviceRecord[] = [];
      for (const adviceId of adviceIds) {
        const result = updateOne.run(now, parsed.turn_id, adviceId, now);
        if (result.changes === 0) {
          continue;
        }

        const row = getOne.get(adviceId) as OperatorAdviceRow | undefined;
        if (row !== undefined) {
          records.push(mapOperatorAdviceRow(row));
        }
      }
      return records;
    });

    return consume(uniqueIds);
  }

  unconsume(ids: readonly OperatorAdviceId[], input: { turn_id: string }): void {
    const turnId = input.turn_id;
    if (turnId.trim().length === 0) {
      throw new StorageError("Operator advice unconsume requires turn_id", {
        code: "OPERATOR_ADVICE_INVALID_TURN_ID",
      });
    }

    const uniqueIds = [...new Set(ids)];
    if (uniqueIds.length === 0) {
      return;
    }

    for (const id of uniqueIds) {
      if (!operatorAdviceIdHelpers.is(id)) {
        throw new StorageError(`Invalid operator advice id: ${id}`, {
          code: "OPERATOR_ADVICE_INVALID_ID",
        });
      }
    }

    const updateOne = this.db.prepare(
      `
        UPDATE operator_advice
        SET consumed_at = NULL, consumed_by_turn_id = NULL
        WHERE id = ?
          AND consumed_by_turn_id = ?
      `,
    );
    const rollback = this.db.transaction((adviceIds: readonly OperatorAdviceId[]) => {
      for (const adviceId of adviceIds) {
        updateOne.run(adviceId, turnId);
      }
    });

    rollback(uniqueIds);
  }
}
