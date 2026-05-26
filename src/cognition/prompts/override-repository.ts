import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";

import { isPromptKey, type PromptKey } from "./registry.js";

export type PromptOverrideRecord = {
  prompt_key: PromptKey;
  override_text: string;
  updated_at: number;
};

type PromptOverrideRow = {
  prompt_key: string;
  override_text: string;
  updated_at: number;
};

export class PromptOverrideRepository {
  constructor(
    private readonly db: SqliteDatabase,
    private readonly clock: Clock = new SystemClock(),
  ) {}

  get(key: PromptKey): string | null {
    const row = this.db
      .prepare("SELECT override_text FROM prompt_overrides WHERE prompt_key = ?")
      .get(key) as { override_text: string } | undefined;
    return row?.override_text ?? null;
  }

  list(): PromptOverrideRecord[] {
    const rows = this.db
      .prepare(
        "SELECT prompt_key, override_text, updated_at FROM prompt_overrides ORDER BY prompt_key ASC",
      )
      .all() as PromptOverrideRow[];

    return rows
      .filter((row) => isPromptKey(row.prompt_key))
      .map((row) => ({
        prompt_key: row.prompt_key as PromptKey,
        override_text: row.override_text,
        updated_at: row.updated_at,
      }));
  }

  set(key: PromptKey, text: string): PromptOverrideRecord {
    const updatedAt = this.clock.now();
    this.db
      .prepare(
        `INSERT INTO prompt_overrides (prompt_key, override_text, updated_at)
         VALUES (?, ?, ?)
         ON CONFLICT(prompt_key) DO UPDATE SET
           override_text = excluded.override_text,
           updated_at = excluded.updated_at`,
      )
      .run(key, text, updatedAt);

    return { prompt_key: key, override_text: text, updated_at: updatedAt };
  }

  clear(key: PromptKey): boolean {
    const result = this.db
      .prepare("DELETE FROM prompt_overrides WHERE prompt_key = ?")
      .run(key);
    return result.changes > 0;
  }
}
