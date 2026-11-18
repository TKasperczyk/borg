import { afterEach, describe, expect, it, vi } from "vitest";
import { openDatabase, type SqliteDatabase } from "../../storage/sqlite/index.js";
import { identityMigrations } from "./migrations.js";
import { IdentityEventRepository } from "./repository.js";
import type { AnnotationTarget } from "./annotations.js";

describe("audited annotation reads", () => {
  let db: SqliteDatabase;
  afterEach(() => db?.close());

  function setup() {
    db = openDatabase(":memory:", { migrations: identityMigrations });
    return new IdentityEventRepository({ db });
  }

  it("joins exact typed targets, independently of actions and the 50-event window", () => {
    const repo = setup();
    const episode: AnnotationTarget = { record_type: "episode", record_id: "synthetic-target" };
    const node: AnnotationTarget = { record_type: "semantic_node", record_id: episode.record_id };
    for (const [target, text] of [
      [episode, "Episode correction"],
      [node, "Node correction"],
    ] as const) {
      repo.record({
        ...target,
        action: "arbitrary_audit_action",
        ts: 1,
        provenance: { kind: "manual" },
        new_value: {
          stream_entry_id: `synthetic-${target.record_type}`,
          annotation: {
            event: "episode_annotation",
            target_ids: [target.record_id],
            correction: text,
            authority: "Operator",
          },
        },
      });
    }
    for (let i = 0; i < 60; i++)
      repo.record({
        ...episode,
        action: "episode_annotation",
        ts: i + 2,
        provenance: { kind: "manual" },
        new_value: { note: "Not an annotation payload" },
      });
    repo.record({
      ...episode,
      action: "unrelated",
      provenance: { kind: "manual" },
      new_value: {
        stream_entry_id: "synthetic-wrong-target",
        annotation: {
          event: "episode_annotation",
          target_ids: ["synthetic-target-extra"],
          correction: "Wrong target",
        },
      },
    });
    const results = repo.listAnnotationsForTargets([
      episode,
      node,
      episode,
      { ...episode, record_id: "synthetic-target-extra" },
    ]);
    expect(results.map((target) => target.annotations.map((entry) => entry.text))).toEqual([
      ["Episode correction"],
      ["Node correction"],
      [],
    ]);
    expect(repo.listAnnotationsForTargets([])).toEqual([]);
  });

  it("preserves the legacy episode_id/note shape and multilingual text byte for byte", () => {
    const repo = setup();
    const target = { record_type: "episode" as const, record_id: "synthetic-legacy" };
    const note = "旧记录有误。\nZażółć gęślą jaźń. تصحيح 🧭 e\u0301";
    repo.record({
      ...target,
      action: "historical_operator_action",
      ts: 100,
      provenance: { kind: "manual" },
      new_value: {
        stream_entry_id: "synthetic-old-entry",
        annotation: { event: "episode_annotation", episode_id: target.record_id, note },
      },
    });
    const annotations = repo.listAnnotationsForTargets([target])[0]!.annotations;
    expect(annotations).toEqual([
      { stream_entry_id: "synthetic-old-entry", text: note, written_at: 100, authority: null },
    ]);
    expect(Buffer.from(annotations[0]!.text)).toEqual(Buffer.from(note));
  });

  it("retracts only the named entry, deduplicates bindings and orders newest first", () => {
    const repo = setup();
    const target = { record_type: "episode" as const, record_id: "synthetic-retracted" };
    for (const [entry, ts] of [
      ["first", 1],
      ["second", 2],
      ["third", 2],
      ["third", 2],
    ] as const) {
      repo.record({
        ...target,
        action: "write",
        ts,
        provenance: { kind: "manual" },
        new_value: {
          stream_entry_id: entry,
          annotation: {
            event: "episode_annotation",
            target_ids: [target.record_id],
            correction: entry,
          },
        },
      });
    }
    for (const entry of ["first", "absent"])
      repo.record({
        ...target,
        action: "unrelated_repair_name",
        ts: 3,
        provenance: { kind: "manual" },
        new_value: {
          stream_entry_id: `retraction-${entry}`,
          annotation: {
            event: "episode_annotation_retracted",
            target_ids: [target.record_id],
            reverses_stream_entry_id: entry,
          },
        },
      });
    expect(
      repo
        .listAnnotationsForTargets([target])[0]!
        .annotations.map((annotation) => annotation.stream_entry_id),
    ).toEqual(["third", "second"]);
  });

  it("reads across parameter batches without limiting annotations per target", () => {
    const repo = setup();
    const targets: AnnotationTarget[] = Array.from({ length: 405 }, (_, i) => ({
      record_type: "episode",
      record_id: `synthetic-batch-${i}`,
    }));
    for (const target of [targets[0]!, targets[404]!])
      for (let i = 0; i < 5; i++)
        repo.record({
          ...target,
          action: "write",
          ts: i,
          provenance: { kind: "manual" },
          new_value: {
            stream_entry_id: `synthetic-entry-${i}`,
            annotation: {
              event: "episode_annotation",
              target_ids: [target.record_id],
              correction: String(i),
            },
          },
        });
    const results = repo.listAnnotationsForTargets(targets);
    expect(results).toHaveLength(405);
    expect(results[0]!.annotations).toHaveLength(5);
    expect(results[404]!.annotations).toHaveLength(5);
    expect(results[1]!.annotations).toEqual([]);
  });
  it("skips malformed history once per read without losing valid targets or applying malformed retractions", () => {
    setup();
    const emit = vi.fn();
    const repo = new IdentityEventRepository({
      db,
      tracer: { enabled: true, includePayloads: false, emit },
    });
    const target = { record_type: "episode" as const, record_id: "synthetic-valid" };
    const other = { record_type: "semantic_node" as const, record_id: "synthetic-other" };
    for (const item of [target, other])
      repo.record({
        ...item,
        action: "synthetic",
        ts: 1,
        provenance: { kind: "manual" },
        new_value: {
          stream_entry_id: item.record_id,
          annotation: {
            event: "episode_annotation",
            target_ids: [item.record_id],
            correction: "Valid multilingual 更正",
          },
        },
      });
    const invalid = [
      "{",
      JSON.stringify({
        stream_entry_id: "synthetic-partial",
        annotation: { event: "episode_annotation", target_ids: [target.record_id] },
      }),
      JSON.stringify({
        stream_entry_id: null,
        annotation: {
          event: "episode_annotation_retracted",
          target_ids: [target.record_id],
          reverses_stream_entry_id: target.record_id,
        },
      }),
    ];
    for (const value of invalid)
      db.prepare(
        "INSERT INTO identity_events (record_type, record_id, action, new_value_json, provenance_kind, ts) VALUES (?, ?, 'synthetic', ?, 'manual', 2)",
      ).run(target.record_type, target.record_id, value);
    for (const timestamp of [1e100, 9_223_372_036_854_775_807n]) {
      db.prepare(
        "UPDATE identity_events SET ts = ? WHERE record_id = ? AND new_value_json NOT IN (?, ?, ?)",
      ).run(timestamp, target.record_id, ...invalid);
      const result = repo.listAnnotationsForTargets([target, other]);
      expect(result.map((item) => item.annotations.length)).toEqual([1, 1]);
      expect(result[0]!.annotations[0]).not.toHaveProperty("written_at");
      expect(result[1]!.annotations[0]!.written_at).toBe(1);
      expect(emit).toHaveBeenCalledExactlyOnceWith("memory.annotations.skipped", {
        turnId: "memory-annotation-read",
        counts_by_reason: {
          invalid_timestamp: 1,
          invalid_json: 1,
          missing_text: 1,
          invalid_payload: 1,
        },
      });
      emit.mockClear();
    }
    repo.listAnnotationsForTargets([other]);
    expect(emit).not.toHaveBeenCalled();
  });
});
