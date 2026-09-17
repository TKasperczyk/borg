import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createMigrations } from "../src/borg/storage-setup.js";
import { FakeEmbeddingClient } from "../src/embeddings/index.js";
import { EntityRepository, CommitmentRepository } from "../src/memory/commitments/repository.js";
import {
  EpisodicRepository,
  createEpisodesTableSchema,
} from "../src/memory/episodic/repository.js";
import { episodeParticipantEntityIdTerm } from "../src/memory/episodic/participant-terms.js";
import { IdentityEventRepository } from "../src/memory/identity/repository.js";
import { GoalsRepository } from "../src/memory/self/goals-repository.js";
import {
  SemanticNodeRepository,
  SemanticEdgeRepository,
  createSemanticNodesTableSchema,
} from "../src/memory/semantic/repository.js";
import { TrainOfThoughtRepository } from "../src/memory/train-of-thought/repository.js";
import { SessionsRepository } from "../src/sessions/repository.js";
import { StreamWriter, StreamEntryIndexRepository } from "../src/stream/index.js";
import { LanceDbStore } from "../src/storage/lancedb/index.js";
import { openDatabase } from "../src/storage/sqlite/index.js";
import { ManualClock } from "../src/util/clock.js";
import { createEpisodeId, createSemanticNodeId, DEFAULT_SESSION_ID } from "../src/util/ids.js";
import { writeJsonFileAtomic } from "../src/util/atomic-write.js";
import {
  applyEntityGraphRepair,
  planEntityGraphRepair,
  parseRepairArgs,
  resumeEntityGraphRepair,
  formatRepairPlan,
} from "./repair-entity-graph.js";
import { digest, remap } from "./repair-entity-graph-storage.js";

const cleanup: Array<() => Promise<void>> = [];
afterEach(async () => {
  vi.restoreAllMocks();
  while (cleanup.length) await cleanup.pop()!();
});

async function fixture() {
  const dataDir = mkdtempSync(join(tmpdir(), "borg-entity-repair-"));
  const db = openDatabase(join(dataDir, "borg.db"), { migrations: createMigrations() });
  const store = new LanceDbStore({ uri: join(dataDir, "lancedb") });
  cleanup.push(async () => {
    await store.close();
    db.close();
    rmSync(dataDir, { recursive: true, force: true });
  });
  const nodesTable = await store.openTable({
    name: "semantic_nodes",
    schema: createSemanticNodesTableSchema(4),
  });
  const episodesTable = await store.openTable({
    name: "episodes",
    schema: createEpisodesTableSchema(4),
  });
  const clock = new ManualClock(1000);
  const entities = new EntityRepository({ db, clock });
  const self = entities.add({
    canonicalName: "Seed identity",
    kind: "self",
    provenance: "assistant_seeded",
  });
  const namedId = entities.resolveExternal({
    source: "fixture",
    externalId: "declared",
    canonicalName: "Chosen identity",
    provenance: "user_declared",
  });
  const named = entities.addAlias(namedId, "Earlier identity")!;
  const room = entities.add({
    canonicalName: "transport:placeholder",
    provenance: "transport_audience_label",
  });
  const participant = entities.add({
    canonicalName: "Participant placeholder",
    provenance: "user_declared",
  });
  const nodes = new SemanticNodeRepository({ db, table: nodesTable, clock });
  const edges = new SemanticEdgeRepository({ db, clock });
  const episodeId = createEpisodeId();
  const nodeInput = {
    kind: "entity",
    label: "Chosen identity",
    description: "Identity description",
    aliases: [named.id],
    confidence: 0.8,
    source_episode_ids: [episodeId],
    created_at: 1000,
    updated_at: 1000,
    last_verified_at: 1000,
    embedding: new Float32Array([1, 0, 0, 0]),
    archived: false,
    superseded_by: null,
  };
  const survivor = await nodes.insert({ ...nodeInput, id: createSemanticNodeId() });
  const duplicate = await nodes.insert({
    ...nodeInput,
    id: createSemanticNodeId(),
    aliases: ["Alternate handle", room.id],
  });
  const other = await nodes.insert({
    ...nodeInput,
    id: createSemanticNodeId(),
    label: room.id,
    aliases: [],
  });
  db.transaction(() => {
    nodes.adjustConfidenceTransactional({ id: duplicate.id, adjust: () => 0.9 });
    nodes.adjustConfidenceTransactional({ id: survivor.id, adjust: () => 0.9 });
  })();
  const edgeInput = {
    to_node_id: other.id,
    relation: "supports" as const,
    confidence: 0.8,
    evidence_episode_ids: [episodeId],
    created_at: 1000,
    last_verified_at: 1000,
  };
  edges.addEdge({ ...edgeInput, from_node_id: survivor.id });
  edges.addEdge({ ...edgeInput, from_node_id: duplicate.id });
  const index = new StreamEntryIndexRepository({ db, dataDir });
  const writer = new StreamWriter({ dataDir, entryIndex: index, clock });
  const entry = await writer.append({
    kind: "internal_event",
    sender_entity_id: named.id,
    content: {
      event: "fixture",
      entity_id: named.id,
      entity_ids: [named.id, self.id],
      prose: `A quoted ${named.id} remains historical text.`,
    },
  });
  const nextEntry = await writer.append({ kind: "internal_event", content: { event: "next" } });
  await writer.close();
  writeJsonFileAtomic(join(dataDir, "working", "default.json"), {
    audience: room.id,
    nodes: [duplicate.id],
  });
  const episodes = new EpisodicRepository({ db, table: episodesTable, clock });
  await episodes.createEpisode({
    id: episodeId,
    title: "Episode placeholder",
    narrative: "Episode narrative",
    participants: [
      episodeParticipantEntityIdTerm(named.id),
      episodeParticipantEntityIdTerm(self.id),
    ],
    location: null,
    start_time: 1000,
    end_time: 2000,
    source_stream_ids: [entry.id],
    significance: 0.8,
    tags: [],
    confidence: 0.9,
    lineage: { derived_from: [], supersedes: [] },
    embedding: new Float32Array([1, 0, 0, 0]),
    emotional_arc: null,
    audience_entity_id: room.id,
    origin_audience_entity_ids: [room.id, participant.id],
    shared: false,
    created_at: 1000,
    updated_at: 1000,
  });
  const identity = new IdentityEventRepository({ db, clock });
  const goals = new GoalsRepository({ db, clock, identityEventRepository: identity });
  const goal = goals.add({
    description: "Goal placeholder",
    priority: 5,
    provenance: { kind: "manual" },
    audienceEntityId: room.id,
  });
  clock.advance(1000);
  goals.update(goal.id, { target_at: 5000 }, { kind: "manual" });
  db.prepare(
    "DELETE FROM identity_events WHERE record_type = 'goal' AND record_id = ? AND action <> 'create'",
  ).run(goal.id);
  clock.advance(1000);
  goals.update(goal.id, { priority: 6 }, { kind: "manual" });
  const commitments = new CommitmentRepository({ db, clock, identityEventRepository: identity });
  const commitment = commitments.add({
    type: "promise",
    directiveFamily: "placeholder",
    directive: "Fixture commitment",
    priority: 5,
    madeToEntity: room.id,
    aboutEntity: named.id,
    committedByEntityId: named.id,
    restrictedAudience: room.id,
    provenance: { kind: "manual" },
  });
  const sessions = new SessionsRepository({ db, clock });
  sessions.ensure({
    session_id: DEFAULT_SESSION_ID,
    source_type: "demo",
    label: "Session placeholder",
    audience_label: room.canonical_name,
    audience_entity_id: room.id,
    conversation_kind: "demo",
  });
  const journal = new TrainOfThoughtRepository({ db, clock }).append({
    selfEntityId: named.id,
    text: "Journal placeholder",
  });
  // An extension table proves discovery is not a brittle list of known columns.
  db.exec("CREATE TABLE extension_fixture (id TEXT PRIMARY KEY, payload TEXT, entity_id TEXT)");
  db.prepare("INSERT INTO extension_fixture VALUES (?, ?, ?)").run(
    "extension",
    JSON.stringify({ [named.id]: { audience: [room.id, participant.id], node: duplicate.id } }),
    named.id,
  );
  const input = {
    dataDir,
    merges: [
      { from: named.id, into: self.id, renameInto: named.canonical_name },
      { from: room.id, into: participant.id },
    ],
    nodeMerges: [{ from: duplicate.id, into: survivor.id }],
  };
  return {
    dataDir,
    db,
    store,
    clock,
    entities,
    nodes,
    episodes,
    nodesTable,
    self,
    named,
    room,
    participant,
    survivor,
    duplicate,
    other,
    goal,
    commitment,
    journal,
    entry,
    nextEntry,
    input,
  };
}

describe("offline entity graph repair", () => {
  it("requires explicit inputs and defaults to a dry run", () => {
    expect(() => parseRepairArgs([])).toThrow("--data-dir");
    expect(() =>
      parseRepairArgs(["--data-dir", "/tmp/placeholder", "--rename-into", "Placeholder"]),
    ).toThrow("must follow");
    expect(parseRepairArgs(["--data-dir", "/tmp/placeholder", "--merge", "a=b"])).toMatchObject({
      apply: false,
      resume: false,
    });
  });

  it("plans read-only, then repairs SQL, nested JSON, streams, mirrors, graph dependencies, and goal history", async () => {
    const f = await fixture();
    const beforeStream = readFileSync(join(f.dataDir, "stream", "default.jsonl"), "utf8");
    const beforeDb = digest(readFileSync(join(f.dataDir, "borg.db")).toString("base64"));
    const plan = await planEntityGraphRepair(f.input);
    expect(plan.goals).toHaveLength(1);
    expect(plan.goals[0]?.version).toBe(2);
    expect(plan.inventory.sqlite.extension_fixture).toContain("payload");
    expect(formatRepairPlan(plan)).toContain("semantic_reembeddings");
    expect(readFileSync(join(f.dataDir, "stream", "default.jsonl"), "utf8")).toBe(beforeStream);
    expect(digest(readFileSync(join(f.dataDir, "borg.db")).toString("base64"))).toBe(beforeDb);
    expect(readdirSync(f.dataDir)).not.toContain("entity-graph-repair");
    const embeddings = new FakeEmbeddingClient(4);
    await applyEntityGraphRepair(plan, embeddings);
    expect(f.entities.get(f.named.id)).toBeNull();
    expect(f.entities.findByExternalId("fixture", "declared")).toBe(f.self.id);
    expect(f.entities.getSelf()).toMatchObject({
      id: f.self.id,
      canonical_name: "Chosen identity",
      name_provenance: "user_declared",
    });
    expect(f.entities.resolveAudienceLabel("transport:placeholder")).toBe(f.participant.id);
    expect(
      f.db.prepare("SELECT * FROM commitments WHERE id = ?").get(f.commitment.id),
    ).toMatchObject({
      made_to_entity: f.participant.id,
      restricted_audience: f.participant.id,
      about_entity: f.self.id,
      committed_by_entity_id: f.self.id,
    });
    expect(
      f.db
        .prepare("SELECT self_entity_id FROM train_of_thought_journal_entries WHERE id = ?")
        .get(f.journal.id),
    ).toEqual({ self_entity_id: f.self.id });
    const extension = f.db.prepare("SELECT * FROM extension_fixture").get() as {
      payload: string;
      entity_id: string;
    };
    expect(extension.entity_id).toBe(f.self.id);
    expect(JSON.parse(extension.payload)).toEqual({
      [f.self.id]: { audience: [f.participant.id], node: f.survivor.id },
    });
    expect(f.db.prepare("SELECT count(*) AS n FROM semantic_edges").get()).toEqual({ n: 1 });
    expect(f.db.prepare("SELECT count(*) AS n FROM semantic_belief_dependencies").get()).toEqual({
      n: 1,
    });
    expect(
      f.db.prepare("SELECT * FROM semantic_nodes WHERE id = ?").get(f.duplicate.id),
    ).toMatchObject({ archived: 1, status: "superseded", superseded_by: f.survivor.id });
    expect(f.db.prepare("SELECT * FROM semantic_node_vector_sync_outbox").all()).toEqual([]);
    await f.nodesTable.checkoutLatest();
    const survivor = await f.nodes.get(f.survivor.id);
    expect(survivor?.aliases).toEqual(["Alternate handle"]);
    expect(Array.from(survivor!.embedding)).not.toEqual([1, 0, 0, 0]);
    const stream = readFileSync(join(f.dataDir, "stream", "default.jsonl"));
    for (const id of [f.entry.id, f.nextEntry.id]) {
      const row = f.db
        .prepare("SELECT byte_offset, sender_entity_id FROM stream_entry_index WHERE entry_id = ?")
        .get(id) as { byte_offset: number; sender_entity_id: string };
      expect(JSON.parse(stream.subarray(row.byte_offset).toString().split("\n")[0]!).id).toBe(id);
      if (id === f.entry.id) expect(row.sender_entity_id).toBe(f.self.id);
    }
    expect(stream.toString()).toContain(`A quoted ${f.named.id} remains historical text.`);
    expect(
      f.db
        .prepare(
          "SELECT count(*) AS n FROM identity_events WHERE record_type = 'goal' AND record_id = ?",
        )
        .get(f.goal.id),
    ).toEqual({ n: 3 });
    const audit = f.db
      .prepare(
        "SELECT reversal FROM maintenance_audit WHERE action = 'entity_graph_sql_rows' AND json_extract(targets, '$.table') = 'entities'",
      )
      .get() as { reversal: string };
    expect(audit.reversal).toContain(f.named.id);
    const repeated = await planEntityGraphRepair(f.input);
    expect(repeated.sql).toEqual([]);
    expect(repeated.lance).toEqual([]);
    expect(repeated.files).toEqual([]);
    expect(repeated.goals).toEqual([]);
    expect(f.db.prepare("PRAGMA foreign_key_check").all()).toEqual([]);
  });

  it("resumes after SQLite committed but a file write could not complete", async () => {
    const f = await fixture();
    const plan = await planEntityGraphRepair(f.input);
    const path = join(f.dataDir, "working", "default.json");
    const before = readFileSync(path, "utf8");
    writeFileSync(path, "{}");
    await expect(applyEntityGraphRepair(plan, new FakeEmbeddingClient(4))).rejects.toThrow(
      "File changed since planning",
    );
    expect(f.entities.get(f.named.id)).toBeNull();
    writeFileSync(path, before);
    await resumeEntityGraphRepair(f.dataDir);
    expect(f.db.prepare("SELECT * FROM semantic_node_vector_sync_outbox").all()).toEqual([]);
    expect(
      f.db
        .prepare(
          "SELECT count(*) AS n FROM maintenance_audit WHERE action = 'entity_graph_complete'",
        )
        .get(),
    ).toEqual({ n: 1 });
  });

  it("rejects stale plans atomically and embedding failures before mutation", async () => {
    const f = await fixture();
    const plan = await planEntityGraphRepair(f.input);
    await expect(
      applyEntityGraphRepair(plan, {
        embed: async () => {
          throw new Error("embedding offline");
        },
        embedBatch: async () => [],
      }),
    ).rejects.toThrow("embedding offline");
    expect(f.entities.get(f.named.id)).not.toBeNull();
    expect(readdirSync(f.dataDir)).not.toContain("entity-graph-repair");
    const fresh = await planEntityGraphRepair(f.input);
    f.entities.addAlias(f.self.id, "New handle");
    await expect(applyEntityGraphRepair(fresh, new FakeEmbeddingClient(4))).rejects.toThrow(
      "SQL row changed",
    );
    expect(f.entities.get(f.named.id)).not.toBeNull();
    expect(f.db.prepare("SELECT * FROM maintenance_audit").all()).toEqual([]);
  });

  it("rejects cycles and conflicting survivor names without guessing identity", async () => {
    const f = await fixture();
    await expect(
      planEntityGraphRepair({
        ...f.input,
        merges: [...f.input.merges, { from: f.self.id, into: f.named.id }],
      }),
    ).rejects.toThrow("chains/cycles");
    await expect(
      planEntityGraphRepair({
        ...f.input,
        merges: [{ from: f.named.id, into: f.self.id, renameInto: "Unobserved name" }],
      }),
    ).rejects.toThrow("canonical name");
    expect(remap(`Prose ${f.named.id}`, new Map([[f.named.id, f.self.id]]))).toBe(
      `Prose ${f.named.id}`,
    );
    const pasted = { content: JSON.stringify({ entity_id: f.named.id }) };
    expect(remap(pasted, new Map([[f.named.id, f.self.id]]))).toEqual(pasted);
  });
});
