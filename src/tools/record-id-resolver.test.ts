import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { openDatabase, type SqliteDatabase } from "../storage/sqlite/index.js";
import {
  createRecordIdResolver,
  MAX_ID_PREFIX_CANDIDATES,
  RECORD_ID_STORES,
  type RecordIdKind,
} from "./record-id-resolver.js";

describe("stored record ID resolution", () => {
  let db: SqliteDatabase;
  beforeEach(() => {
    db = openDatabase(":memory:");
    for (const store of Object.values(RECORD_ID_STORES)) {
      const { table, column } = store;
      const timestamp =
        "timestamp" in store ? `, "${store.timestamp}" INTEGER DEFAULT 1700000000000` : "";
      db.exec(
        `CREATE TABLE "${table}" ("${column}" TEXT PRIMARY KEY, archived INTEGER, status TEXT, audience_entity_id TEXT, session_scope TEXT${timestamp})`,
      );
    }
  });
  afterEach(() => db.close());

  function insert(kind: RecordIdKind, suffix: string) {
    const { helper, table, column } = RECORD_ID_STORES[kind];
    const id = helper.prefix + suffix;
    db.prepare(
      `INSERT INTO "${table}" ("${column}", archived, status, audience_entity_id, session_scope) VALUES (?, 1, 'superseded', 'ent_else000000000000', 'sess_else000000000000')`,
    ).run(id);
    return id;
  }

  it.each(Object.keys(RECORD_ID_STORES) as RecordIdKind[])(
    "resolves %s globally, including archived/superseded rows, using only its ID index",
    (kind) => {
      const id = insert(kind, "test000000000001");
      const resolver = createRecordIdResolver(db);
      const prepare = vi.spyOn(db, "prepare");
      expect(resolver.resolve(id, [kind])).toEqual({ status: "unchanged", id });
      expect(prepare).not.toHaveBeenCalled();
      const handle = RECORD_ID_STORES[kind].helper.prefix + "test";
      expect(resolver.resolve(handle, [kind])).toEqual({ status: "resolved", id });
      expect(resolver.resolve(id.slice(0, -1), [kind])).toEqual({ status: "resolved", id });
      const queries = prepare.mock.calls.map(([sql]) => sql);
      prepare.mockRestore();
      for (const query of queries) {
        expect(query).not.toMatch(/archived|status|audience|session_scope/);
        const plan = db.prepare(`EXPLAIN QUERY PLAN ${query}`).all(handle, handle + "z", 11);
        expect(plan).toHaveLength(1);
        expect(plan[0]!.detail).toMatch(/SEARCH .* USING COVERING INDEX .*\(.*>\? AND .*<\?\)/);
        expect(plan[0]!.detail).not.toContain("SCAN");
      }
    },
  );

  it("fails closed on collisions, with bounded sorted full IDs and an exact omitted count", () => {
    const candidates = Array.from({ length: MAX_ID_PREFIX_CANDIDATES + 3 }, (_, i) =>
      insert("goal", `same${String(i).padStart(12, "0")}`),
    );
    const resolver = createRecordIdResolver(db);
    try {
      resolver.resolve("goal_same", ["goal"]);
      expect.fail("ambiguous prefix accepted");
    } catch (error) {
      expect(error).toMatchObject({ code: "ambiguous_id_prefix" });
      expect(JSON.parse((error as Error).message)).toEqual({
        code: "ambiguous_id_prefix",
        input: "goal_same",
        candidates: candidates
          .slice(0, MAX_ID_PREFIX_CANDIDATES)
          .map((id) => ({ id, timestamp: "2023-11-14T22:13:20.000Z" })),
        omitted_count: 3,
        message: "Retry with a longer prefix or inspect a candidate with the matching read tool.",
      });
    }
    expect(resolver.resolve(candidates[0]!, ["goal"])).toEqual({
      status: "unchanged",
      id: candidates[0],
    });
    const countPlan = db
      .prepare("EXPLAIN QUERY PLAN SELECT COUNT(*) FROM goals WHERE id >= ? AND id < ?")
      .all("goal_same", "goal_samf");
    expect(countPlan[0]!.detail).toContain("SEARCH goals USING COVERING INDEX");
  });

  it.each(Object.keys(RECORD_ID_STORES) as RecordIdKind[])(
    "%s ambiguity includes only IDs and available UTC timestamps via indexed point reads",
    (kind) => {
      const first = insert(kind, "same100000000000");
      const second = insert(kind, "same200000000000");
      const store = RECORD_ID_STORES[kind];
      const prepare = vi.spyOn(db, "prepare");
      let detail: { candidates: unknown[] } | undefined;
      try {
        createRecordIdResolver(db).resolve(store.helper.prefix + "same", [kind]);
      } catch (error) {
        detail = JSON.parse((error as Error).message);
      }
      expect(detail?.candidates).toEqual(
        [first, second].map((id) => ({
          id,
          ...("timestamp" in store ? { timestamp: "2023-11-14T22:13:20.000Z" } : {}),
        })),
      );
      const queries = prepare.mock.calls.map(([sql]) => sql);
      prepare.mockRestore();
      for (const query of queries.filter((sql) => sql.includes("AS timestamp"))) {
        const plan = db.prepare(`EXPLAIN QUERY PLAN ${query}`).all(first);
        expect(plan[0]!.detail).toMatch(/SEARCH .* USING INDEX .*\(.*=\?\)/);
        expect(plan[0]!.detail).not.toContain("SCAN");
      }
    },
  );

  it.each(["end9", "endz"])("keeps the exclusive prefix boundary after %s", (suffix) => {
    const id = insert("goal", suffix + "000000000001");
    for (const neighbor of [
      "end8000000000001",
      "enda000000000001",
      "endy000000000001",
      "ene0000000000001",
    ])
      insert("goal", neighbor);
    const prefix = `goal_${suffix}`;
    expect(createRecordIdResolver(db).resolve(prefix, ["goal"])).toEqual({
      status: "resolved",
      id,
    });
    insert("goal", suffix + "zzzzzzzzzzzz");
    expect(() => createRecordIdResolver(db).resolve(prefix, ["goal"])).toThrow('"omitted_count":0');
  });

  it("a longer prefix disambiguates a handle and a small collision reports zero omitted", () => {
    const first = insert("open_question", "same100000000000");
    insert("open_question", "same200000000000");
    const resolver = createRecordIdResolver(db);
    expect(() => resolver.resolve("oq_same", ["open_question"])).toThrow('"omitted_count":0');
    expect(resolver.resolve("oq_same1", ["open_question"])).toEqual({
      status: "resolved",
      id: first,
    });
  });

  it("never repairs wrong suffixes or searches another record type", () => {
    insert("goal", "test000000000001");
    const resolver = createRecordIdResolver(db);
    expect(resolver.resolve("goal_none", ["goal"])).toEqual({
      status: "not_found",
      id: "goal_none",
    });
    expect(resolver.resolve("goal_test000000000099", ["goal"])).toEqual({
      status: "unchanged",
      id: "goal_test000000000099",
    });
    expect(resolver.resolve("goal_test", ["open_question"])).toEqual({
      status: "unchanged",
      id: "goal_test",
    });
    for (const input of ["goal_", "goal_t", "goal_te", "goal_tes"]) {
      expect(() => resolver.resolve(input, ["goal"])).toThrow('"code":"invalid_id_prefix"');
    }
    for (const input of ["goal_test%", "goal_test_", "goal_test'", "goal_test000000000001z"]) {
      expect(resolver.resolve(input, ["goal"])).toEqual({ status: "unchanged", id: input });
    }
    expect(resolver.resolve("default", ["session"])).toEqual({
      status: "unchanged",
      id: "default",
    });
  });
});
