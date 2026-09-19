import {
  ActionRepository,
  createActionRecordsTableSchema,
} from "../src/memory/actions/repository.js";
import { actionRecordSchema } from "../src/memory/actions/types.js";
import { ActivityRepository } from "../src/memory/activity/repository.js";
import {
  ObservedEventRepository,
  createObservedEventsTableSchema,
} from "../src/memory/observed-events/repository.js";
import { buildObservedEventRecurrenceKey } from "../src/memory/observed-events/emission.js";
import {
  OpenQuestionsRepository,
  buildOpenQuestionDedupeKey,
} from "../src/memory/self/open-questions.js";
import { SharedStateRepository } from "../src/memory/shared-state/repository.js";
import { SocialRepository } from "../src/memory/social/repository.js";
import {
  RecallStateRepository,
  createEmptyRecallState,
  recallEvidenceHandleKey,
} from "../src/retrieval/recall-state.js";
import { ReviewQueueRepository } from "../src/memory/review-queue/review-queue.js";
import { WorkingMemoryStore } from "../src/memory/working/store.js";
import { createWorkingMemory } from "../src/memory/working/types.js";
import { ProceduralEvidenceRepository } from "../src/memory/procedural/repository.js";
import { GrowthMarkersRepository } from "../src/memory/self/growth-markers.js";
import { LivedExperienceDaySummaryRepository } from "../src/memory/activity/lived-experience-day-summary.js";
import { TraitsRepository } from "../src/memory/self/traits-repository.js";
import { ValuesRepository } from "../src/memory/self/values-repository.js";
import { relationshipPrivateMemoryDisclosureLabel } from "../src/memory/common/disclosure-label.js";
import { createActionId } from "../src/util/ids.js";
import { createAttachmentId, createImagePerceptionId } from "../src/util/ids.js";
import { AttachmentRepository } from "../src/attachments/repository.js";
import {
  ImagePerceptionRepository,
  createImagePerceptionTableSchema,
  type ImagePerceptionRecord,
} from "../src/attachments/perception.js";
import { OperatorAttentionRepository } from "../src/memory/operator-attention/repository.js";
import { RetrievalPipeline } from "../src/retrieval/pipeline.js";
import * as atomicWrite from "../src/util/atomic-write.js";
import { resolveTurnAudienceEntityId } from "../demo/server/src/app.js";
import type { Borg } from "../src/index.js";
import {
  resolveActiveParticipants,
  resolveEpisodeSourceParticipants,
} from "../src/cognition/participants.js";
import { StreamReader, hydrateStreamEntriesById } from "../src/stream/index.js";
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
import { LanceDbStore, LanceDbTable } from "../src/storage/lancedb/index.js";
import { openDatabase } from "../src/storage/sqlite/index.js";
import { ManualClock } from "../src/util/clock.js";
import {
  createEpisodeId,
  createSemanticNodeId,
  createSessionId,
  DEFAULT_SESSION_ID,
} from "../src/util/ids.js";
import { writeJsonFileAtomic } from "../src/util/atomic-write.js";
import {
  applyEntityGraphRepair,
  planEntityGraphRepair as rawPlan,
  planEntityGraphReversal,
  parseRepairArgs,
  resumeEntityGraphRepair,
  formatRepairPlan,
} from "./repair-entity-graph.js";
import { digest, remap, sqlInventory } from "./repair-entity-graph-storage.js";

const planEntityGraphRepair = (input: Parameters<typeof rawPlan>[0]) =>
  rawPlan(input, new FakeEmbeddingClient(4));

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
  const survivingEdge = edges.addEdge({ ...edgeInput, from_node_id: survivor.id });
  const duplicateEdge = edges.addEdge({ ...edgeInput, from_node_id: duplicate.id });
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
  const episodes = new EpisodicRepository({
    db,
    table: episodesTable,
    clock,
    entityRepository: entities,
  });
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
    edges,
    survivingEdge,
    duplicateEdge,
    episodeId,
    episodesTable,
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

async function addReferenceClasses(f: Awaited<ReturnType<typeof fixture>>) {
  const { db, clock, room, named, participant, duplicate, survivor, episodeId, entry } = f;
  const actionsTable = await f.store.openTable({
    name: "action_records",
    schema: createActionRecordsTableSchema(4),
  });
  const observedTable = await f.store.openTable({
    name: "observed_events",
    schema: createObservedEventsTableSchema(4),
  });
  const actions = new ActionRepository({
    db,
    clock,
    table: actionsTable,
    embeddingClient: new FakeEmbeddingClient(4),
  });
  const action = actionRecordSchema.parse({
    id: createActionId(),
    description: "Action placeholder",
    actor: named.id,
    audience_entity_id: room.id,
    state: "scheduled",
    confidence: 0.8,
    provenance_episode_ids: [episodeId],
    provenance_stream_entry_ids: [entry.id],
    created_at: 1000,
    updated_at: 1000,
    considering_at: null,
    committed_at: null,
    scheduled_at: 1000,
    completed_at: null,
    not_done_at: null,
    unknown_at: null,
  });
  actions.add(action);
  await actions.waitForPendingEmbeddings();
  const activity = new ActivityRepository({ db, clock }).record({
    kind: "user_contact",
    occurredAt: 1000,
    sessionId: DEFAULT_SESSION_ID,
    speakerEntityId: named.id,
    actorEntityId: named.id,
    audienceEntityId: room.id,
    participantEntityIds: [named.id, room.id],
    sourceStreamEntryIds: [entry.id],
  });
  const observed = new ObservedEventRepository({
    db,
    clock,
    table: observedTable,
    embeddingClient: new FakeEmbeddingClient(4),
  });
  const eventInput = {
    occurredAt: 1000,
    sessionId: DEFAULT_SESSION_ID,
    stance: "rejected_frame" as const,
    taint: "quarantined" as const,
    beliefEffect: "unchanged" as const,
    classificationKind: "frame_assignment_claim",
    disclosureClass: "social_observed" as const,
    interactionText: "Observed placeholder",
    speakerEntityId: room.id,
    audienceEntityId: room.id,
    sourceEntityId: named.id,
    sourceStreamEntryIds: [entry.id],
  };
  const event = observed.record({
    ...eventInput,
    recurrenceKey: buildObservedEventRecurrenceKey(eventInput),
  });
  await observed.waitForPendingEmbeddings();
  const questions = new OpenQuestionsRepository({ db, clock });
  const question = questions.add({
    question: "Question placeholder",
    urgency: 0.5,
    source: "user",
    audience_entity_id: room.id,
    related_semantic_node_ids: [duplicate.id, survivor.id],
    related_episode_ids: [episodeId],
  });
  const shared = new SharedStateRepository({ db, clock });
  const artifact = shared.upsert(room.id, [
    {
      type: "add",
      state_key: "placeholder",
      kind: "locked",
      text: "State placeholder",
      owner_entity_id: named.id,
      provenance_stream_entry_ids: [entry.id],
    },
  ]);
  const social = new SocialRepository({ db, clock, entityRepository: f.entities });
  social.recordInteraction(room.id, { provenance: { kind: "manual" } });
  const recall = new RecallStateRepository({ db, clock });
  const nodeKey = recallEvidenceHandleKey({ source: "semantic_node", nodeId: duplicate.id });
  const survivorKey = recallEvidenceHandleKey({ source: "semantic_node", nodeId: survivor.id });
  const edgeKey = recallEvidenceHandleKey({ source: "semantic_edge", edgeId: f.duplicateEdge.id });
  recall.save({
    ...createEmptyRecallState({ scopeKey: room.id, nowMs: 1000 }),
    suppressedHandles: { [nodeKey]: 4, [survivorKey]: 8, [edgeKey]: 6 },
  });
  const pair = new ReviewQueueRepository({ db, clock }).enqueue({
    kind: "duplicate",
    refs: { node_ids: [duplicate.id, survivor.id] },
    reason: "Pair placeholder",
  });
  const selfEdge = f.edges.addEdge({
    from_node_id: duplicate.id,
    to_node_id: survivor.id,
    relation: "supports",
    confidence: 0.8,
    evidence_episode_ids: [episodeId],
    created_at: 1000,
    last_verified_at: 1000,
  });
  const attempt = {
    problem_text: "Problem placeholder",
    approach_summary: "Approach placeholder",
    selected_skill_id: null,
    source_stream_ids: [entry.id],
    turn_counter: 1,
    audience_entity_id: room.id,
  };
  const evidence = new ProceduralEvidenceRepository({ db, clock }).insert({
    pendingAttemptSnapshot: attempt,
    classification: "unclear",
    evidenceText: "Evidence placeholder",
    audienceEntityId: room.id,
    resolvedEpisodeIds: [episodeId],
  });
  const disclosure = relationshipPrivateMemoryDisclosureLabel([room.id]);
  const marker = new GrowthMarkersRepository({ db, clock }).add({
    ts: 1000,
    category: "understanding",
    what_changed: "Change placeholder",
    evidence_episode_ids: [episodeId],
    disclosure_label: disclosure,
    confidence: 0.8,
    source_process: "fixture",
    provenance: { kind: "manual" },
  });
  const summary = new LivedExperienceDaySummaryRepository({ db, clock }).upsert({
    selfEntityId: named.id,
    utcDay: "1970-01-01",
    dayStartMs: 0,
    dayEndMs: 86400000,
    gist: "Summary placeholder",
    countsSnapshot: {},
    sourceEpisodeIds: [episodeId],
    disclosureLabel: disclosure,
    provenance: { kind: "manual" },
  });
  new TraitsRepository({ db, clock }).reinforce({
    label: "Trait placeholder",
    delta: 0.1,
    provenance: { kind: "episodes", episode_ids: [episodeId] },
  });
  new ValuesRepository({ db, clock }).add({
    label: "Value placeholder",
    description: "Value description",
    priority: 1,
    provenance: { kind: "episodes", episode_ids: [episodeId] },
  });
  const working = new WorkingMemoryStore({
    dataDir: f.dataDir,
    clock,
    entityRepository: f.entities,
  });
  working.save({
    ...createWorkingMemory(DEFAULT_SESSION_ID, 1000),
    pending_social_attribution: {
      entity_id: room.id,
      interaction_id: 1,
      agent_response_summary: null,
      turn_completed_ts: 1000,
    },
    pending_trait_attribution: {
      trait_label: "Trait placeholder",
      strength_delta: 0.1,
      source_stream_entry_ids: [entry.id],
      turn_completed_ts: 1000,
      audience_entity_id: room.id,
    },
    pending_procedural_attempts: [attempt],
  });
  return {
    action,
    actionsTable,
    activity,
    event,
    eventInput,
    observed,
    observedTable,
    question,
    questions,
    artifact,
    shared,
    social,
    recall,
    pair,
    selfEdge,
    evidence,
    marker,
    summary,
    working,
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

  it("plans read-only, then repairs SQL, nested JSON, mirrors, graph dependencies, and goal history while preserving streams", async () => {
    const f = await fixture();
    const beforeStream = readFileSync(join(f.dataDir, "stream", "default.jsonl"), "utf8");
    const beforeDb = digest(readFileSync(join(f.dataDir, "borg.db")).toString("base64"));
    const plan = await planEntityGraphRepair(f.input);
    expect(plan.goals).toHaveLength(1);
    expect(plan.goals[0]?.version).toBe(2);
    expect(formatRepairPlan(plan)).toContain("semantic_reembeddings");
    expect(readFileSync(join(f.dataDir, "stream", "default.jsonl"), "utf8")).toBe(beforeStream);
    expect(digest(readFileSync(join(f.dataDir, "borg.db")).toString("base64"))).toBe(beforeDb);
    expect(readdirSync(f.dataDir)).not.toContain("entity-graph-repair");
    await applyEntityGraphRepair(plan);
    expect(f.entities.get(f.named.id)?.id).toBe(f.self.id);
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
    expect(
      f.db.prepare("SELECT count(*) AS n FROM semantic_edges WHERE valid_to IS NULL").get(),
    ).toEqual({ n: 1 });
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
      if (id === f.entry.id) expect(row.sender_entity_id).toBe(f.named.id);
    }
    expect(stream.toString()).toBe(beforeStream);
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
    expect(repeated.runId).toBeNull();
    expect(formatRepairPlan(repeated)).toContain('"journal":null');
    const filesBefore = readdirSync(join(f.dataDir, "entity-graph-repair"));
    const auditBefore = f.db.prepare("SELECT * FROM maintenance_audit").all();
    const hashesBefore = ["borg.db", "borg.db-wal"].map((file) =>
      digest(readFileSync(join(f.dataDir, file))),
    );
    await applyEntityGraphRepair(repeated);
    expect(readdirSync(join(f.dataDir, "entity-graph-repair"))).toEqual(filesBefore);
    expect(f.db.prepare("SELECT * FROM maintenance_audit").all()).toEqual(auditBefore);
    expect(
      ["borg.db", "borg.db-wal"].map((file) => digest(readFileSync(join(f.dataDir, file)))),
    ).toEqual(hashesBefore);
    expect(f.db.prepare("PRAGMA foreign_key_check").all()).toEqual([]);
  });

  it("plans the normal redirect migration and preserves SQLite sequence high-water marks", async () => {
    const f = await fixture();
    f.db.exec(
      "DROP VIEW resolved_entities; DROP INDEX entities_single_self_idx; DROP INDEX entities_merged_into_idx; ALTER TABLE entities DROP COLUMN merged_into;",
    );
    f.db.prepare("DELETE FROM _migrations WHERE name = 'entity_identity_redirects'").run();
    f.db.prepare("UPDATE sqlite_sequence SET seq = 9000 WHERE name = 'identity_events'").run();
    const plan = await planEntityGraphRepair(f.input);
    expect(plan.migrations).toEqual([{ id: 7_000_004, name: "entity_identity_redirects" }]);
    expect(plan.schemaChanges.map((row) => row.name)).toContain("resolved_entities");
    expect(plan.additionalSql.find((row) => row.table === "identity_events")?.rowid).toBe(9001);
    await applyEntityGraphRepair(plan);
    expect(f.entities.get(f.named.id)?.id).toBe(f.self.id);
    expect(f.db.prepare("SELECT * FROM identity_events WHERE id = 9001").get()).toBeDefined();
    await applyEntityGraphRepair(await planEntityGraphReversal(f.dataDir, plan.runId!));
    expect(f.entities.get(f.named.id)?.id).toBe(f.named.id);
    expect(f.db.prepare("SELECT * FROM identity_events WHERE id = 9001").get()).toBeUndefined();
    expect(
      f.db.prepare("SELECT * FROM _migrations WHERE name = 'entity_identity_redirects'").get(),
    ).toBeDefined();
  });

  it("covers derived keys, every reference class, immutable provenance, tuple retirement and untouched vectors", async () => {
    const f = await fixture();
    const refs = await addReferenceClasses(f);
    const original = sqlInventory(f.db);
    const stream = readFileSync(join(f.dataDir, "stream", "default.jsonl"));
    const working = readFileSync(join(f.dataDir, "working", "default.json"));
    const actionVector = (await refs.actionsTable.list())[0]!.embedding;
    const episodeVector = (await f.episodesTable.list())[0]!.embedding;
    const observedVector = (await refs.observedTable.list())[0]!.embedding;
    const plan = await planEntityGraphRepair(f.input);
    expect(plan.sql.some((row) => row.table === "stream_entry_index")).toBe(false);
    expect(plan.files).toEqual([]);
    const lines = formatRepairPlan(plan)
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    for (const row of [...plan.sql, ...plan.additionalSql, ...plan.completionSql, ...plan.drainSql])
      expect(
        lines.some(
          (line) =>
            line.table === row.table &&
            line.rowid === row.rowid &&
            JSON.stringify(line.after) === JSON.stringify(row.after),
        ),
      ).toBe(true);
    await applyEntityGraphRepair(plan);
    const get = (table: string, id: string | number) =>
      f.db.prepare(`SELECT * FROM "${table}" WHERE id = ?`).get(id)!;
    expect(get("action_records", refs.action.id)).toMatchObject({
      actor: f.self.id,
      audience_entity_id: f.participant.id,
      state: "scheduled",
      updated_at: 1000,
    });
    expect(get("activity_events", refs.activity.id)).toMatchObject({
      speaker_entity_id: f.self.id,
      actor_entity_id: f.self.id,
      audience_entity_id: f.participant.id,
    });
    expect(
      JSON.parse(String(get("activity_events", refs.activity.id).participant_entity_ids)),
    ).toEqual([f.self.id, f.participant.id]);
    expect(get("observed_events", refs.event.id)).toMatchObject({
      speaker_entity_id: f.participant.id,
      audience_entity_id: f.participant.id,
      source_entity_id: f.self.id,
      recurrence_key: buildObservedEventRecurrenceKey({
        ...refs.eventInput,
        speakerEntityId: f.participant.id,
      }),
    });
    expect(get("open_questions", refs.question.id)).toMatchObject({
      audience_entity_id: f.participant.id,
      dedupe_key: buildOpenQuestionDedupeKey({
        question: refs.question.question,
        relatedEpisodeIds: refs.question.related_episode_ids,
        relatedSemanticNodeIds: [f.survivor.id, f.survivor.id],
        audienceEntityId: f.participant.id,
      }),
    });
    expect(refs.shared.get(f.participant.id)?.entries[0]).toMatchObject({
      owner_entity_id: f.self.id,
      audience_entity_id: f.participant.id,
    });
    expect(refs.social.getProfile(f.room.id)?.entity_id).toBe(f.participant.id);
    expect(refs.social.listEvents(f.room.id)[0]?.entity_id).toBe(f.participant.id);
    expect(refs.recall.load(f.participant.id)?.suppressedHandles).toEqual({
      [recallEvidenceHandleKey({ source: "semantic_node", nodeId: f.survivor.id })]: 8,
      [recallEvidenceHandleKey({ source: "semantic_edge", edgeId: f.survivingEdge.id })]: 6,
    });
    expect(get("review_queue", refs.pair.id)).toMatchObject({ resolution: "dismiss" });
    expect(JSON.parse(String(get("review_queue", refs.pair.id).refs)).node_ids).toEqual([
      f.survivor.id,
      f.survivor.id,
    ]);
    expect(get("semantic_edges", refs.selfEdge.id)).toMatchObject({
      from_node_id: f.duplicate.id,
      to_node_id: f.survivor.id,
    });
    expect(get("semantic_edges", refs.selfEdge.id).valid_to).not.toBeNull();
    expect(get("procedural_evidence", refs.evidence.id).audience_entity_id).toBe(f.participant.id);
    expect(
      JSON.parse(String(get("procedural_evidence", refs.evidence.id).pending_attempt_snapshot))
        .audience_entity_id,
    ).toBe(f.participant.id);
    expect(String(get("growth_markers", refs.marker.id).disclosure_label)).toContain(
      f.participant.id,
    );
    expect(get("lived_experience_day_summaries", refs.summary.id).self_entity_id).toBe(f.self.id);
    for (const table of ["traits", "values", "trait_reinforcement_events", "value_sources"])
      expect(sqlInventory(f.db, [table]).get(table)).toEqual(original.get(table));
    expect(refs.working.load(DEFAULT_SESSION_ID).pending_social_attribution?.entity_id).toBe(
      f.participant.id,
    );
    expect(
      refs.working.load(DEFAULT_SESSION_ID).pending_trait_attribution?.audience_entity_id,
    ).toBe(f.participant.id);
    expect(
      refs.working.load(DEFAULT_SESSION_ID).pending_procedural_attempts[0]?.audience_entity_id,
    ).toBe(f.participant.id);
    const index = new StreamEntryIndexRepository({
      db: f.db,
      dataDir: f.dataDir,
      entityRepository: f.entities,
    });
    expect(index.lookup(f.entry.id)?.sender_entity_id).toBe(f.self.id);
    expect(
      new StreamReader({ dataDir: f.dataDir, entryIndex: index }).tail(2)[0]?.sender_entity_id,
    ).toBe(f.self.id);
    expect(
      (
        await hydrateStreamEntriesById({
          dataDir: f.dataDir,
          sessionId: DEFAULT_SESSION_ID,
          streamEntryIds: [f.entry.id],
          entryIndex: index,
        })
      ).get(f.entry.id)?.sender_entity_id,
    ).toBe(f.self.id);
    expect(sqlInventory(f.db, ["stream_entry_index"]).get("stream_entry_index")).toEqual(
      original.get("stream_entry_index"),
    );
    expect(readFileSync(join(f.dataDir, "stream", "default.jsonl"))).toEqual(stream);
    expect(readFileSync(join(f.dataDir, "working", "default.json"))).toEqual(working);
    for (const table of [refs.actionsTable, refs.observedTable, f.episodesTable])
      await table.checkoutLatest();
    expect((await refs.actionsTable.list())[0]!.embedding).toEqual(actionVector);
    expect((await refs.observedTable.list())[0]!.embedding).toEqual(observedVector);
    expect((await f.episodesTable.list())[0]!.embedding).toEqual(episodeVector);
    await applyEntityGraphRepair(await planEntityGraphReversal(f.dataDir, plan.runId!));
    for (const [table, rows] of original)
      if (table !== "maintenance_audit")
        expect(sqlInventory(f.db, [table]).get(table), table).toEqual(rows);
  });

  it("rejects parent and derived-key collisions without losing children", async () => {
    const f = await fixture();
    const refs = await addReferenceClasses(f);
    refs.shared.upsert(f.participant.id, [
      {
        type: "add",
        state_key: "other",
        kind: "locked",
        text: "Other state",
        provenance_stream_entry_ids: [f.entry.id],
      },
    ]);
    const before = sqlInventory(f.db);
    await expect(planEntityGraphRepair(f.input)).rejects.toThrow("Parent collision/deletion");
    expect(sqlInventory(f.db)).toEqual(before);
    expect(refs.shared.get(f.room.id)?.entries).toHaveLength(1);
  });

  it("repairs attachment, image-perception and operator-attention audience references", async () => {
    const f = await fixture();
    const attachments = new AttachmentRepository(f.db);
    const attachmentId = createAttachmentId();
    attachments.insert({
      attachment_id: attachmentId,
      sha256: "fixture",
      media_type: "image/png",
      byte_size: 12,
      width: 1,
      height: 1,
      storage_ref: "blobs/fixture.png",
      thumbnail_ref: null,
      perception_id: null,
      text_embedding_ref: null,
      visual_embedding_ref: null,
      active: true,
      audience: "transport:placeholder",
      audience_entity_id: f.room.id,
      created_turn_global: 1,
      parent_entry_id: f.entry.id,
      stream_entry_id: f.nextEntry.id,
      parent_turn_id: null,
      created_at: 1000,
    });
    const table = await f.store.openTable({
      name: "image_perception_embeddings",
      schema: createImagePerceptionTableSchema(4),
    });
    const perceptions = new ImagePerceptionRepository(f.db, table);
    const record: ImagePerceptionRecord = {
      perception_id: createImagePerceptionId(),
      payload_id: createImagePerceptionId(),
      attachment_id: attachmentId,
      parent_entry_id: f.entry.id,
      stream_entry_id: f.nextEntry.id,
      parent_turn_id: null,
      sha256: "fixture",
      media_type: "image/png",
      perception_prompt_version: "fixture",
      model: "fixture",
      caption: "Image placeholder",
      image_kind: "other",
      visible_text: [],
      objects: [],
      people_or_roles: [],
      scene: "",
      colors_and_visual_attributes: [],
      spatial_relationships: [],
      possible_user_relevant_details: [],
      search_terms: [],
      uncertainties: [],
      audience: "transport:placeholder",
      audience_entity_id: f.room.id,
      active: true,
      created_turn_global: 1,
      created_at: 1000,
      text_embedding_ref: null,
      embedding_text: "Image placeholder",
      embedding_status: "complete",
    };
    perceptions.insertPayloadAndUpsertArtifact(record);
    await perceptions.upsertEmbedding(record, new Float32Array([1, 0, 0, 0]));
    const attention = new OperatorAttentionRepository({ db: f.db });
    attention.record({
      record_key: "fixture:1",
      filed_at: 1000,
      filer_entity_id: f.named.id,
      subject: "Fixture",
    });
    const before = sqlInventory(f.db);
    const vector = await table.list();
    const plan = await planEntityGraphRepair(f.input);
    await applyEntityGraphRepair(plan);
    expect(attachments.get(attachmentId)?.audience_entity_id).toBe(f.participant.id);
    expect(perceptions.get(record.perception_id)?.audience_entity_id).toBe(f.participant.id);
    expect(f.db.prepare("SELECT filer_entity_id FROM operator_attention_records").get()).toEqual({
      filer_entity_id: f.self.id,
    });
    await table.checkoutLatest();
    expect(await table.list()).toEqual(vector);
    await applyEntityGraphRepair(await planEntityGraphReversal(f.dataDir, plan.runId!));
    for (const name of [
      "stream_attachments",
      "image_perception_artifacts",
      "image_perception_payloads",
      "operator_attention_records",
    ])
      expect(sqlInventory(f.db, [name]).get(name), name).toEqual(before.get(name));
  });

  it("preflights mismatched sessions and keeps ingress/ensure bindings before and after redirect", async () => {
    const f = await fixture();
    f.db.prepare("UPDATE sessions SET audience_entity_id = ?").run(f.participant.id);
    const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
    const sessions = new SessionsRepository({
      db: f.db,
      clock: f.clock,
      entityRepository: f.entities,
    });
    const session = sessions.get(DEFAULT_SESSION_ID)!;
    const borg = { entities: f.entities } as unknown as Borg;
    expect(
      resolveTurnAudienceEntityId(borg, {
        audienceLabel: session.audience_label,
        existingAudienceEntityId: f.participant.id,
        explicitAudienceEntityId: f.room.id,
      }),
    ).toBe(f.participant.id);
    expect(sessions.ensure({ ...session, audience_entity_id: f.room.id }).audience_entity_id).toBe(
      f.participant.id,
    );
    expect(warning).toHaveBeenCalled();
    const plan = await planEntityGraphRepair(f.input);
    expect(
      plan.reports.find((report) => report.category === "session_audience_preflight")?.rows,
    ).toEqual([
      expect.objectContaining({
        mismatch: true,
        bound: expect.objectContaining({
          entity: expect.objectContaining({ id: f.participant.id }),
        }),
        label: expect.objectContaining({
          entity: expect.objectContaining({ id: f.room.id }),
          origin_episodes: 1,
        }),
        projected_label: f.participant.id,
        projected_binding: f.participant.id,
      }),
    ]);
    await applyEntityGraphRepair(plan);
    const pipeline = new RetrievalPipeline({
      dataDir: f.dataDir,
      episodicRepository: f.episodes,
      entityRepository: f.entities,
      embeddingClient: new FakeEmbeddingClient(4),
      clock: f.clock,
    });
    for (const id of [f.room.id, f.participant.id]) {
      expect(
        resolveTurnAudienceEntityId(borg, {
          audienceLabel: session.audience_label,
          explicitAudienceEntityId: id,
        }),
      ).toBe(f.participant.id);
      expect(sessions.ensure({ ...session, audience_entity_id: id }).audience_entity_id).toBe(
        f.participant.id,
      );
      expect(sessions.get(DEFAULT_SESSION_ID)?.audience_entity_id).toBe(f.participant.id);
      expect(sessions.getMany([DEFAULT_SESSION_ID])[0]?.audience_entity_id).toBe(f.participant.id);
      expect(sessions.list()[0]?.audience_entity_id).toBe(f.participant.id);
      expect((await pipeline.getEpisode(f.episodeId, { audienceEntityId: id }))?.episode.id).toBe(
        f.episodeId,
      );
      expect(
        (await pipeline.getEpisode(f.episodeId, { visibleAudienceEntityIds: [id] }))?.episode.id,
      ).toBe(f.episodeId);
      expect(
        (await f.episodes.listRecentForDisclosure({ audienceEntityId: id })).map(
          (row) => row.episode.id,
        ),
      ).toContain(f.episodeId);
    }
    f.db.prepare("UPDATE sessions SET audience_entity_id = ?").run(f.room.id);
    expect(sessions.get(DEFAULT_SESSION_ID)?.audience_entity_id).toBe(f.participant.id);
    expect(sessions.getMany([DEFAULT_SESSION_ID])[0]?.audience_entity_id).toBe(f.participant.id);
    expect(sessions.list()[0]?.audience_entity_id).toBe(f.participant.id);
    expect(
      resolveActiveParticipants({
        audienceEntityId: f.room.id,
        senderEntityId: f.participant.id,
        streamEntries: [],
        entityRepository: f.entities,
      }),
    ).toEqual([expect.objectContaining({ entityId: f.participant.id, role: "speaker" })]);
    expect(
      resolveEpisodeSourceParticipants({
        episodes: [
          {
            participants: [`entity_id:${f.room.id}`, `entity_id:${f.participant.id}`],
            source_stream_ids: [],
          },
        ],
        entryIndex: { lookupMany: () => new Map() },
        entityRepository: f.entities,
      }),
    ).toEqual([expect.objectContaining({ entityId: f.participant.id })]);
  });

  it("canonicalizes activity identities on inputs, outputs and disclosure comparisons", async () => {
    const f = await fixture();
    f.db.prepare("UPDATE entities SET kind = 'group' WHERE id = ?").run(f.participant.id);
    const activity = new ActivityRepository({
      db: f.db,
      clock: f.clock,
      entityRepository: f.entities,
    });
    f.entities.merge(f.room.id, f.participant.id);
    f.entities.merge(f.named.id, f.self.id);
    const row = activity.record({
      kind: "user_contact",
      occurredAt: 1000,
      sessionId: DEFAULT_SESSION_ID,
      speakerEntityId: f.named.id,
      audienceEntityId: f.room.id,
      sourceStreamEntryIds: [f.entry.id],
    });
    expect(row).toMatchObject({
      speaker_entity_id: f.self.id,
      audience_entity_id: f.participant.id,
    });
    f.db
      .prepare(
        "UPDATE activity_events SET speaker_entity_id = ?, audience_entity_id = ? WHERE id = ?",
      )
      .run(f.named.id, f.room.id, row.id);
    for (const id of [f.named.id, f.self.id])
      expect(activity.listObservedGroupAudienceEntityIdsForSpeaker(id)).toEqual([f.participant.id]);
    for (const id of [f.room.id, f.participant.id])
      expect(
        activity.listRecentVisibleOtherSessionEvents({
          currentSessionId: createSessionId(),
          audienceEntityIds: [id],
          sinceMs: 0,
          limit: 10,
        }),
      ).toEqual([expect.objectContaining({ audienceEntityId: f.participant.id })]);
    expect(activity.get(row.id)).toMatchObject({
      speaker_entity_id: f.self.id,
      audience_entity_id: f.participant.id,
    });
  });

  it("reconciles an entity blocker across retained stream-index handles in either direction", async () => {
    const f = await fixture();
    const goals = new GoalsRepository({ db: f.db, clock: f.clock });
    f.entities.merge(f.room.id, f.participant.id);
    for (const [blocker, sender] of [
      [f.room.id, f.participant.id],
      [f.participant.id, f.room.id],
    ] as const) {
      const goal = goals.add({
        description: "Block placeholder",
        priority: 5,
        provenance: { kind: "manual" },
      });
      goals.block(
        goal.id,
        {
          blocker: { kind: "entity", entity_id: blocker },
          attempt_status: "attempted_unavailable",
          reason: "Pending reply",
        },
        { kind: "manual" },
      );
      f.clock.advance(10);
      const index = new StreamEntryIndexRepository({ db: f.db, dataDir: f.dataDir });
      const writer = new StreamWriter({ dataDir: f.dataDir, entryIndex: index, clock: f.clock });
      await writer.append({
        kind: "user_msg",
        sender_entity_id: sender,
        content: "Reply placeholder",
      });
      await writer.close();
      const before = sqlInventory(f.db, ["stream_entry_index"]);
      goals.reconcileBlocks();
      expect(goals.get(goal.id)?.status).toBe("active");
      expect(sqlInventory(f.db, ["stream_entry_index"])).toEqual(before);
    }
  });

  it.each(["metadata", "embedding"])(
    "rejects stale Lance %s before reversal writes, even after a reversal plan was made",
    async (field) => {
      const f = await fixture();
      const plan = await planEntityGraphRepair(f.input);
      await applyEntityGraphRepair(plan);
      const reverse = await planEntityGraphReversal(f.dataDir, plan.runId!);
      const before = sqlInventory(f.db);
      const files = readdirSync(join(f.dataDir, "entity-graph-repair"));
      await f.episodesTable.checkoutLatest();
      const row = (await f.episodesTable.list())[0]!;
      await f.episodesTable.upsert(
        [
          {
            ...row,
            ...(field === "embedding"
              ? { embedding: [0, 1, 0, 0] }
              : { narrative: "Intervening change" }),
          },
        ],
        { on: "id" },
      );
      await expect(planEntityGraphReversal(f.dataDir, plan.runId!)).rejects.toThrow(
        "Lance row changed",
      );
      await expect(applyEntityGraphRepair(reverse)).rejects.toThrow("Lance row changed");
      expect(sqlInventory(f.db)).toEqual(before);
      expect(readdirSync(join(f.dataDir, "entity-graph-repair"))).toEqual(files);
    },
  );

  it("syncs journal reachability before SQLite, and fails without a commit if parent sync fails", async () => {
    const f = await fixture();
    const plan = await planEntityGraphRepair(f.input);
    const before = sqlInventory(f.db);
    const sync = vi.spyOn(atomicWrite, "syncDirectory").mockImplementation((path) => {
      if (path === f.dataDir) throw new Error("parent sync failed");
    });
    await expect(applyEntityGraphRepair(plan)).rejects.toThrow("parent sync failed");
    expect(sync).toHaveBeenCalledWith(f.dataDir);
    expect(sqlInventory(f.db)).toEqual(before);
    expect(readdirSync(join(f.dataDir, "entity-graph-repair"))).toEqual([]);
  });

  it("keeps new-insight reviews active when node references collapse", async () => {
    const f = await fixture();
    const reviews = new ReviewQueueRepository({ db: f.db, clock: f.clock });
    const item = reviews.enqueue({
      kind: "new_insight",
      reason: "Insight placeholder",
      refs: {
        node_ids: [f.duplicate.id, f.survivor.id],
        episode_ids: [f.episodeId],
        evidence_cluster_key: "fixture",
        evidence_cluster_size: 1,
        reflector_pending_insight: {
          target: {
            mode: "update",
            node_id: f.survivor.id,
            patch: {
              description: "Insight placeholder",
              confidence: 0.8,
              source_episode_ids: [f.episodeId],
              last_verified_at: 1000,
              embedding: [1, 0, 0, 0],
              archived: false,
            },
          },
          candidate_support_edges: [],
          evidence_cluster: { key: "fixture", episode_ids: [f.episodeId], size: 1 },
        },
      },
    });
    await applyEntityGraphRepair(await planEntityGraphRepair(f.input));
    expect(reviews.get(item.id)).toMatchObject({
      resolved_at: null,
      resolution: null,
      refs: { node_ids: [f.survivor.id, f.survivor.id] },
    });
  });

  it("rejects open-question key collisions", async () => {
    const f = await fixture();
    const questions = new OpenQuestionsRepository({ db: f.db, clock: f.clock });
    questions.add({
      question: "Collision placeholder",
      urgency: 0.5,
      source: "user",
      audience_entity_id: f.room.id,
      related_semantic_node_ids: [f.duplicate.id],
    });
    questions.add({
      question: "Collision placeholder",
      urgency: 0.5,
      source: "user",
      audience_entity_id: f.participant.id,
      related_semantic_node_ids: [f.survivor.id],
    });
    await expect(planEntityGraphRepair(f.input)).rejects.toThrow(/collision|Conflicting/);
    expect(f.db.prepare("SELECT count(*) AS n FROM open_questions").get()).toEqual({ n: 2 });
  });

  it("resumes a failure between an outbox Lance commit and its SQLite acknowledgement", async () => {
    const f = await fixture();
    const plan = await planEntityGraphRepair(f.input);
    const originalUpsert = LanceDbTable.prototype.upsert;
    let armed = false;
    const spy = vi.spyOn(LanceDbTable.prototype, "upsert").mockImplementation(async function (
      this: LanceDbTable,
      ...args: Parameters<LanceDbTable["upsert"]>
    ) {
      await originalUpsert.apply(this, args);
      if (armed) {
        armed = false;
        throw new Error("interrupted after mirror commit");
      }
    });
    await expect(
      applyEntityGraphRepair(plan, {
        checkpoint: (point) => {
          if (point === "after_lance:semantic_nodes") armed = true;
        },
      }),
    ).rejects.toThrow("Vector outbox not drained");
    expect(
      f.db.prepare("SELECT * FROM semantic_node_vector_sync_outbox WHERE attempts > 0").all(),
    ).toHaveLength(1);
    spy.mockRestore();
    await resumeEntityGraphRepair(f.dataDir);
    expect(f.db.prepare("SELECT * FROM semantic_node_vector_sync_outbox").all()).toEqual([]);
    await applyEntityGraphRepair(await planEntityGraphReversal(f.dataDir, plan.runId!));
    expect(f.entities.get(f.named.id)?.id).toBe(f.named.id);
  });

  it("reports unknown changed schemas and rejects malformed active review tuples", async () => {
    const f = await fixture();
    f.db.exec("CREATE TABLE extension_fixture (entity_id TEXT)");
    f.db.prepare("INSERT INTO extension_fixture VALUES (?)").run(f.named.id);
    await expect(planEntityGraphRepair(f.input)).rejects.toThrow("No repository validator");
    f.db.exec("DROP TABLE extension_fixture");
    f.db
      .prepare(
        "INSERT INTO review_queue (kind,refs,reason,created_at) VALUES ('duplicate',?,'Fixture',1000)",
      )
      .run(JSON.stringify({ node_ids: [f.duplicate.id] }));
    await expect(planEntityGraphRepair(f.input)).rejects.toThrow("Invalid projected review_queue");
    expect(f.entities.get(f.named.id)?.id).toBe(f.named.id);
  });

  it("rejects recurrence-key collisions before any mutation", async () => {
    const f = await fixture();
    const events = new ObservedEventRepository({ db: f.db, clock: f.clock });
    for (const entity of [f.room, f.participant]) {
      const input = {
        occurredAt: 1000,
        sessionId: DEFAULT_SESSION_ID,
        stance: "rejected_frame" as const,
        taint: "quarantined" as const,
        beliefEffect: "unchanged" as const,
        classificationKind: "frame_assignment_claim",
        disclosureClass: "social_observed" as const,
        interactionText: "Fixture",
        speakerEntityId: entity.id,
        audienceEntityId: entity.id,
        sourceStreamEntryIds: [f.entry.id],
      };
      events.record({ ...input, recurrenceKey: buildObservedEventRecurrenceKey(input) });
    }
    await expect(planEntityGraphRepair(f.input)).rejects.toThrow(/Conflicting|collision/);
    expect(f.db.prepare("SELECT count(*) AS n FROM observed_events").get()).toEqual({ n: 2 });
  });

  it.each([
    "after_journal",
    "after_sqlite",
    "after_lance:episodes",
    "after_lance:semantic_nodes",
    "during_outbox",
    "before_completion",
    "after_completion",
    "before_archive",
    "after_archive",
  ])("recovers at %s, then reverses from audit rows", async (boundary) => {
    const f = await fixture();
    const before = sqlInventory(f.db);
    const stream = readFileSync(join(f.dataDir, "stream", "default.jsonl"));
    const working = readFileSync(join(f.dataDir, "working", "default.json"));
    const originalVectors = await f.nodesTable.list();
    const plan = await planEntityGraphRepair(f.input);
    await expect(
      applyEntityGraphRepair(plan, {
        checkpoint: (point) => {
          if (
            point === boundary ||
            (boundary === "during_outbox" && point.startsWith("during_outbox:"))
          )
            throw new Error("interrupted");
        },
      }),
    ).rejects.toThrow("interrupted");
    await resumeEntityGraphRepair(f.dataDir);
    expect(f.entities.get(f.named.id)?.id).toBe(f.self.id);
    expect(f.db.prepare("SELECT * FROM semantic_node_vector_sync_outbox").all()).toEqual([]);
    const reverse = await planEntityGraphReversal(f.dataDir, plan.runId!);
    await applyEntityGraphRepair(reverse);
    for (const [table, rows] of before) {
      if (table === "maintenance_audit") continue;
      expect(sqlInventory(f.db).get(table), table).toEqual(rows);
    }
    await f.nodesTable.checkoutLatest();
    expect(await f.nodesTable.list()).toEqual(originalVectors);
    expect(readFileSync(join(f.dataDir, "stream", "default.jsonl"))).toEqual(stream);
    expect(readFileSync(join(f.dataDir, "working", "default.json"))).toEqual(working);
  });

  it("rejects stale plans atomically and embedding failures before mutation", async () => {
    const f = await fixture();
    await expect(
      rawPlan(f.input, {
        embed: async () => {
          throw new Error("embedding offline");
        },
        embedBatch: async () => [],
      }),
    ).rejects.toThrow("embedding offline");
    expect(readdirSync(f.dataDir)).not.toContain("entity-graph-repair");
    const fresh = await planEntityGraphRepair(f.input);
    f.entities.addAlias(f.self.id, "New handle");
    await expect(applyEntityGraphRepair(fresh)).rejects.toThrow("SQL row changed");
    expect(f.entities.get(f.named.id)?.id).toBe(f.named.id);
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
