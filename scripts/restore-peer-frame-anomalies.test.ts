import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { backfillStreamEntryIndex } from "../src/borg/reconciliation.js";
import { createMigrations } from "../src/borg/storage-setup.js";
import { openDatabase } from "../src/storage/sqlite/index.js";
import { SessionsRepository } from "../src/sessions/repository.js";
import { EntityRepository } from "../src/memory/commitments/repository.js";
import {
  StreamEntryIndexRepository,
  StreamReader,
  StreamWriter,
  getSessionStreamPath,
  filterActiveStreamEntries,
} from "../src/stream/index.js";
import { createSessionId } from "../src/util/ids.js";
import { restorePeerFrameAnomalies } from "./restore-peer-frame-anomalies.js";

const cleanup: (() => void)[] = [];
afterEach(() => {
  while (cleanup.length) cleanup.pop()!();
});

describe("restorePeerFrameAnomalies", () => {
  it("dry-runs, applies and replays idempotently by source type, retaining aborts and unrelated inactive rows", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-peer-restoration-"));
    const db = openDatabase(join(dataDir, "borg.db"), { migrations: createMigrations() });
    const sessions = new SessionsRepository({ db });
    const entities = new EntityRepository({ db });
    const audience = entities.resolve("Fixture peer");
    const index = new StreamEntryIndexRepository({ db, dataDir });
    const peerSession = createSessionId();
    const nonPeerSession = createSessionId();
    for (const [id, source, label] of [
      [peerSession, "kira", "opaque"],
      [nonPeerSession, "demo", "kira claude_code"],
    ] as const) {
      sessions.ensure({
        session_id: id,
        source_type: source,
        label,
        audience_entity_id: audience,
        audience_label: "Fixture peer",
        conversation_kind: "demo",
      });
    }
    const peer = new StreamWriter({ dataDir, sessionId: peerSession, entryIndex: index });
    const nonPeer = new StreamWriter({ dataDir, sessionId: nonPeerSession, entryIndex: index });
    cleanup.push(() => {
      peer.close();
      nonPeer.close();
      db.close();
      rmSync(dataDir, { recursive: true, force: true });
    });
    const source = await peer.append({ kind: "user_msg", content: "Restorable private report." });
    const aborted = await peer.append({
      kind: "user_msg",
      content: "Aborted report.",
      turn_id: "failed",
    });
    const invalid = await peer.append({
      kind: "user_msg",
      content: "Otherwise invalid.",
      turn_status: "aborted",
    });
    const unrelated = await peer.append({
      kind: "user_msg",
      content: "Independently invalidated, without a quarantine marker.",
    });
    await peer.append({
      kind: "internal_event",
      content: { event: "aborted_turn", aborted_stream_entry_ids: [unrelated.id] },
    });
    const nonPeerSource = await nonPeer.append({
      kind: "user_msg",
      content: "Non-peer stays quarantined.",
    });
    for (const [writer, entries] of [
      [peer, [source, aborted, invalid]],
      [nonPeer, [nonPeerSource]],
    ] as const) {
      await writer.append({
        kind: "internal_event",
        content: {
          event: "quarantined_user_entry",
          source_stream_entry_id: entries[0].id,
          cited_stream_entry_ids: entries.map((entry) => entry.id),
          kind: "system_prompt_claim",
          confidence: 0.92,
          rationale: "原文を保った分類理由。",
          turn_id: "classification-turn",
        },
      });
    }
    await peer.append({
      kind: "internal_event",
      content: { event: "aborted_turn", turn_id: "failed", aborted_stream_entry_ids: [aborted.id] },
    });
    const peerPath = getSessionStreamPath(dataDir, peerSession);
    const nonPeerPath = getSessionStreamPath(dataDir, nonPeerSession);
    const before = readFileSync(peerPath, "utf8");
    const otherBefore = readFileSync(nonPeerPath, "utf8");
    const dbBefore = readFileSync(join(dataDir, "borg.db"));
    const dry = await restorePeerFrameAnomalies({ dataDir });
    expect(dry).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ entry_id: source.id, status: "restore", source_type: "kira" }),
        expect.objectContaining({ entry_id: aborted.id, status: "kept_inactive" }),
        expect.objectContaining({ entry_id: invalid.id, status: "kept_inactive" }),
      ]),
    );
    expect(dry).toHaveLength(3);
    expect(readFileSync(peerPath, "utf8")).toBe(before);
    expect(readFileSync(join(dataDir, "borg.db"))).toEqual(dbBefore);
    expect(index.lookup(source.id)?.active).toBe(false);
    await restorePeerFrameAnomalies({ dataDir, apply: true });
    expect(index.lookup(source.id)?.active).toBe(true);
    expect(index.quarantinedSharedStateArtifactRefs().has(source.id)).toBe(false);
    for (const entry of [aborted, invalid, nonPeerSource, unrelated])
      expect(index.lookup(entry.id)?.active).toBe(false);
    expect(readFileSync(nonPeerPath, "utf8")).toBe(otherBefore);
    expect(index.frameAnomalyAnnotations(source.id)).toEqual([
      expect.objectContaining({
        rationale: "原文を保った分類理由。",
        classifier_model: null,
        disclosure_label: {
          disclosure_class: "relationship_private",
          origin_audience_entity_ids: [audience],
          private_to_entity_ids: [audience],
          public_to_entity_ids: [],
        },
      }),
    ]);
    const after = readFileSync(peerPath, "utf8");
    // Crash after JSONL commit / before SQLite projection: rerun repairs only the projection.
    db.exec(
      "DELETE FROM stream_frame_anomaly_annotations; DELETE FROM stream_quarantine_restoration_applications;",
    );
    db.prepare("UPDATE stream_entry_index SET active = 0 WHERE entry_id = ?").run(source.id);
    await restorePeerFrameAnomalies({ dataDir, apply: true });
    expect(readFileSync(peerPath, "utf8")).toBe(after);
    expect(index.lookup(source.id)?.active).toBe(true);
    expect(index.lookup(unrelated.id)?.active).toBe(false);
    expect(index.frameAnomalyAnnotations(source.id)).toHaveLength(1);
    const activeFlags = db
      .prepare("SELECT entry_id, active FROM stream_entry_index ORDER BY entry_id")
      .all();
    await backfillStreamEntryIndex({ dataDir, entryIndex: index });
    expect(
      db.prepare("SELECT entry_id, active FROM stream_entry_index ORDER BY entry_id").all(),
    ).toEqual(activeFlags);
    db.exec(
      "DELETE FROM stream_frame_anomaly_annotations; DELETE FROM stream_quarantine_refs; DELETE FROM stream_entry_index; DELETE FROM stream_quarantine_restoration_applications;",
    );
    await backfillStreamEntryIndex({ dataDir, entryIndex: index });
    expect(
      db.prepare("SELECT entry_id, active FROM stream_entry_index ORDER BY entry_id").all(),
    ).toEqual(activeFlags);
    expect(index.lookup(source.id)?.active).toBe(true);
    expect(index.quarantinedSharedStateArtifactRefs().has(source.id)).toBe(false);
    for (const entry of [aborted, invalid, nonPeerSource, unrelated])
      expect(index.lookup(entry.id)?.active).toBe(false);
    const recalled = filterActiveStreamEntries(
      new StreamReader({ dataDir, sessionId: peerSession }).tail(100),
    );
    expect(recalled.find((entry) => entry.id === source.id)?.frame_anomalies).toHaveLength(1);
    expect(recalled.map((entry) => entry.id)).not.toContain(aborted.id);
    // A later index-only deactivation must not be undone by an old restoration.
    db.prepare("UPDATE stream_entry_index SET active = 0 WHERE entry_id = ?").run(source.id);
    await index.backfillSession(peerSession);
    expect(index.lookup(source.id)?.active).toBe(false);
    const rerun = await restorePeerFrameAnomalies({ dataDir, apply: true });
    expect(rerun.some((row) => row.entry_id === source.id)).toBe(false);
    expect(index.lookup(source.id)?.active).toBe(false);
    expect(readFileSync(peerPath, "utf8")).toBe(after);
    expect(
      db.prepare("SELECT COUNT(*) AS count FROM stream_quarantine_restoration_applications").get(),
    ).toEqual({ count: 1 });
    // Even another annotation event naming the same pair is not a new reversal.
    const originalAnnotation = index.frameAnomalyAnnotations(source.id)[0]!;
    await peer.append({ kind: "internal_event", content: originalAnnotation });
    expect(index.lookup(source.id)?.active).toBe(false);
    await index.backfillSession(peerSession);
    expect(index.lookup(source.id)?.active).toBe(false);
  });

  it("uses configured peer source types and does not guess missing classifier rationale", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-peer-restoration-config-"));
    const db = openDatabase(join(dataDir, "borg.db"), { migrations: createMigrations() });
    const sessionId = createSessionId();
    new SessionsRepository({ db }).ensure({
      session_id: sessionId,
      source_type: "custom_peer",
      label: "irrelevant",
      audience_label: "private",
      conversation_kind: "demo",
    });
    const index = new StreamEntryIndexRepository({ db, dataDir });
    const writer = new StreamWriter({ dataDir, sessionId, entryIndex: index });
    cleanup.push(() => {
      writer.close();
      db.close();
      rmSync(dataDir, { recursive: true, force: true });
    });
    const indexOnly = await writer.append({
      kind: "user_msg",
      content: "No quarantine or durable invalidation marker.",
    });
    db.prepare("UPDATE stream_entry_index SET active = 0 WHERE entry_id = ?").run(indexOnly.id);
    const source = await writer.append({ kind: "user_msg", content: "Original." });
    await writer.append({
      kind: "internal_event",
      content: {
        event: "quarantined_user_entry",
        cited_stream_entry_ids: [source.id],
        kind: "frame_assignment_claim",
        confidence: 0.9,
      },
    });
    expect(await restorePeerFrameAnomalies({ dataDir })).toEqual([]);
    writeFileSync(
      join(dataDir, "config.json"),
      JSON.stringify({ frameAnomaly: { peerChannelSourceTypes: ["custom_peer"] } }),
    );
    expect(await restorePeerFrameAnomalies({ dataDir, apply: true })).toEqual([
      expect.objectContaining({ entry_id: source.id, status: "invalid_annotation_source" }),
    ]);
    expect(index.lookup(source.id)?.active).toBe(false);
    expect(index.lookup(indexOnly.id)?.active).toBe(false);
    expect(index.frameAnomalyAnnotations(indexOnly.id)).toEqual([]);
    await backfillStreamEntryIndex({ dataDir, entryIndex: index });
    expect(index.lookup(indexOnly.id)?.active).toBe(false);
  });
});
