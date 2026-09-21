import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FakeEmbeddingClient } from "../src/embeddings/index.js";
import type { LLMClient, LLMCompleteOptions } from "../src/llm/index.js";
import * as clients from "../src/borg/clients.js";
import { EntityRepository } from "../src/memory/commitments/repository.js";
import {
  EpisodicRepository,
  createEpisodesTableSchema,
} from "../src/memory/episodic/repository.js";
import {
  SemanticNodeRepository,
  SemanticEdgeRepository,
  createSemanticNodesTableSchema,
} from "../src/memory/semantic/repository.js";
import { LanceDbStore, LanceDbTable } from "../src/storage/lancedb/index.js";
import { createEpisodeId, createSemanticNodeId, createStreamEntryId } from "../src/util/ids.js";
import { fingerprint } from "./audited-data-repair.js";
import { repairFixture, repairBackupFixture } from "./audited-data-repair-test-support.js";
import * as graph from "./repair-entity-graph.js";
import { lanceRowMatches, openLance, sqlInventory } from "./repair-entity-graph-storage.js";
import {
  buildSelfLabelDossier,
  DOSSIER_EPISODE_LIMIT,
  DOSSIER_EXCERPT_CHARS,
  parseVerdictFile,
  selfLabelInventory,
  type SelfLabelDossier,
  type VerdictFile,
} from "./judge-self-label-nodes-dossier.js";
import {
  applySelfLabelNodes,
  judgeSelfLabelNodes,
  loadSelfLabelPlan,
  main,
  planSelfLabelNodes,
  readVerdictFile,
  writeVerdictFile,
} from "./judge-self-label-nodes.js";

const cleanup: Array<() => void | Promise<void>> = [];
afterEach(async () => {
  vi.restoreAllMocks();
  while (cleanup.length) await cleanup.pop()!();
});

async function fixture(liveShape = false) {
  const f = repairFixture(liveShape);
  cleanup.push(() => f.cleanup());
  const store = new LanceDbStore({ uri: join(f.dataDir, "lancedb") });
  cleanup.push(async () => {
    await store.close();
  });
  const table = await store.openTable({
    name: "semantic_nodes",
    schema: createSemanticNodesTableSchema(4),
  });
  const episodeTable = await store.openTable({
    name: "episodes",
    schema: createEpisodesTableSchema(4),
  });
  const entities = new EntityRepository({ db: f.db, clock: f.clock });
  const self = entities.add({ kind: "self", canonicalName: "Écho" });
  entities.addAlias(self.id, "別名");
  const duplicateEntity = entities.add({ canonicalName: "éCHO", kind: "person" });
  const nodes = new SemanticNodeRepository({ db: f.db, table, clock: f.clock });
  const edges = new SemanticEdgeRepository({ db: f.db, clock: f.clock });
  const episodes = new EpisodicRepository({ db: f.db, table: episodeTable, clock: f.clock });
  const episode = await episodes.createEpisode({
    id: createEpisodeId(),
    title: "記憶の出典",
    narrative: "Une description de mon expérience.\n" + "続き🌱".repeat(500),
    participants: [],
    location: null,
    start_time: 1000,
    end_time: 1000,
    source_stream_ids: [createStreamEntryId()],
    significance: 0.8,
    tags: [],
    confidence: 0.9,
    lineage: { derived_from: [], supersedes: [] },
    embedding: new Float32Array([1, 0, 0, 0]),
    emotional_arc: null,
    shared: false,
    created_at: 1000,
    updated_at: 1000,
  });
  const addNode = async (kind: string, label: string, extra: Record<string, unknown> = {}) =>
    nodes.insert({
      id: createSemanticNodeId(),
      kind,
      label,
      description: "Description de cette identité.",
      domain: "identity",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 1000,
      updated_at: 1000,
      last_verified_at: 1000,
      embedding: new Float32Array([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
      ...extra,
    });
  const canonical = await addNode("self", "Anchor with a different label");
  const same = await addNode("entity", " éCHO ", { aliases: ["先の名前"] });
  const other = await addNode("person", "別名", {
    description: "Une autre personne portant ce nom.",
  });
  const unsure = await addNode("entity", "Écho", { description: "Contexte insuffisant." });
  const neighbor = await addNode("concept", "Projet");
  const addEdge = (from: string, to: string) =>
    edges.addEdge({
      from_node_id: from as typeof same.id,
      to_node_id: to as typeof same.id,
      relation: "supports",
      confidence: 0.8,
      evidence_episode_ids: [episode.id],
      created_at: 1000,
      last_verified_at: 1000,
    });
  const outgoing = addEdge(same.id, neighbor.id);
  const incoming = addEdge(neighbor.id, same.id);
  return {
    ...f,
    store,
    table,
    episodeTable,
    entities,
    self,
    duplicateEntity,
    nodes,
    edges,
    episodes,
    episode,
    addNode,
    addEdge,
    canonical,
    same,
    other,
    unsure,
    neighbor,
    outgoing,
    incoming,
  };
}
type Fixture = Awaited<ReturnType<typeof fixture>>;
function backup(f: Fixture) {
  const result = repairBackupFixture(f.dataDir);
  cleanup.push(result.cleanup);
  return result.path;
}
function llm(f: Fixture, allOther = false) {
  let active = 0;
  let maxActive = 0;
  const complete = vi.fn(async (options: LLMCompleteOptions) => {
    active++;
    maxActive = Math.max(maxActive, active);
    await new Promise<void>((done) => setImmediate(done));
    const dossier = JSON.parse(options.messages[0]!.content) as SelfLabelDossier;
    active--;
    return {
      text: 'Ignored prose: {"verdict":"self"}',
      input_tokens: 10,
      output_tokens: 10,
      stop_reason: "tool_use",
      tool_calls: [
        {
          id: "judgment",
          name: "JudgeSelfLabelNode",
          input: {
            verdict: allOther
              ? "other"
              : dossier.node.id === f.same.id
                ? "self"
                : dossier.node.id === f.other.id
                  ? "other"
                  : "unsure",
            reason: "根拠に基づく判断。別の名の持ち主かもしれない。",
            confidence: 0.75,
          },
        },
      ],
    };
  });
  const client: LLMClient = { complete, converse: vi.fn() };
  return { client, complete, maxActive: () => maxActive };
}
async function judged(
  f: Fixture,
  allOther = false,
  liveStoreConfirmed?: string,
  selfNode?: string,
) {
  return (
    await judgeSelfLabelNodes({
      dataDir: f.dataDir,
      llm: llm(f, allOther).client,
      liveStoreConfirmed,
      selfNode,
    })
  ).file;
}
async function planned(
  f: Fixture,
  file?: VerdictFile,
  backupPath = backup(f),
  liveStoreConfirmed?: string,
  selfNode?: string,
) {
  return planSelfLabelNodes({
    dataDir: f.dataDir,
    verdicts: file ?? (await judged(f, false, liveStoreConfirmed, selfNode)),
    backupVerified: backupPath,
    embeddingClient: new FakeEmbeddingClient(4),
    liveStoreConfirmed,
    selfNode,
  });
}
function rows(f: Fixture) {
  return Object.fromEntries(sqlInventory(f.db));
}

describe("judged self-label repair", () => {
  it("selects trimmed case-insensitive names and aliases; excludes archived, superseded and other kinds", async () => {
    const f = await fixture();
    await f.addNode("entity", "Écho", { archived: true });
    await f.addNode("entity", "Écho", { status: "superseded" });
    await f.addNode("entity", "Écho", { superseded_by: f.canonical.id });
    await f.addNode("concept", "Écho");
    await f.addNode("entity", "Unrelated", { aliases: ["Écho"] });
    f.db.prepare("UPDATE semantic_nodes SET label = ? WHERE id = ?").run("Écho", f.canonical.id);
    const inventory = selfLabelInventory(f.db);
    expect(inventory.candidates.map((row) => row.id).sort()).toEqual(
      [f.same.id, f.other.id, f.unsure.id].sort(),
    );
    expect(inventory.canonical.id).toBe(f.canonical.id);
    expect(selfLabelInventory(f.db, f.canonical.id).anchor).toEqual(inventory.anchor);
    expect(inventory.matchingEntities.map((row) => row.id)).toEqual([f.duplicateEntity.id]);
  });

  it("refuses a missing structural self anchor, never inferring it from a label", async () => {
    const f = await fixture();
    f.db.prepare("UPDATE semantic_nodes SET kind = 'entity' WHERE id = ?").run(f.canonical.id);
    expect(() => selfLabelInventory(f.db)).toThrow(
      "No active kind=self semantic nodes. Candidates: []",
    );
  });

  it("requires --self-node for ambiguous anchors and lists IDs, labels, dates and both edge/source counts", async () => {
    const f = await fixture();
    f.entities.addAlias(f.self.id, "self");
    const observation = await f.addNode("self", "self", {
      created_at: 900,
      source_episode_ids: [f.episode.id, createEpisodeId()],
    });
    await f.addNode("self", "Inactive anchor", { archived: true });
    f.addEdge(observation.id, f.neighbor.id);
    f.addEdge(f.neighbor.id, observation.id);
    f.addEdge(observation.id, f.same.id);
    const candidates = [
      {
        id: f.canonical.id,
        label: f.canonical.label,
        created_at: 1000,
        edge_count: 0,
        source_count: 1,
      },
      { id: observation.id, label: "self", created_at: 900, edge_count: 3, source_count: 2 },
    ].sort((a, b) => a.id.localeCompare(b.id));
    const message = `Multiple active kind=self semantic nodes; --self-node is required. Candidates: ${JSON.stringify(candidates)}`;
    const before = rows(f),
      client = llm(f),
      out = join(f.dataDir, "ambiguous.json");
    expect(() => selfLabelInventory(f.db)).toThrow(message);
    await expect(
      judgeSelfLabelNodes({ dataDir: f.dataDir, out, llm: client.client }),
    ).rejects.toThrow(message);
    expect(client.complete).not.toHaveBeenCalled();
    expect(existsSync(out)).toBe(false);
    expect(rows(f)).toEqual(before);
  });

  it.each([
    "non-self",
    "archived",
    "superseded-status",
    "superseded-pointer",
    "quarantined",
    "missing",
  ])(
    "rejects a --self-node selecting a %s node even when one valid anchor exists",
    async (kind) => {
      const f = await fixture();
      const invalid = await f.addNode(kind === "non-self" ? "entity" : "self", "Écho", {
        archived: kind === "archived" || kind === "missing",
        status:
          kind === "superseded-status"
            ? "superseded"
            : kind === "quarantined"
              ? "quarantined"
              : "active",
        superseded_by: kind === "superseded-pointer" ? f.canonical.id : null,
      });
      const selfNode = kind === "missing" ? createSemanticNodeId() : invalid.id;
      const client = llm(f),
        out = join(f.dataDir, "invalid-anchor.json"),
        before = rows(f);
      await expect(
        judgeSelfLabelNodes({ dataDir: f.dataDir, selfNode, out, llm: client.client }),
      ).rejects.toThrow(
        "--self-node must name an active, unarchived, unsuperseded kind=self semantic node",
      );
      expect(client.complete).not.toHaveBeenCalled();
      expect(existsSync(out)).toBe(false);
      expect(rows(f)).toEqual(before);
    },
  );

  it("uses the explicit anchor throughout judging, merge planning and recovery, keeping other self nodes report-only", async () => {
    const f = await fixture();
    f.entities.addAlias(f.self.id, "self");
    const observation = await f.addNode("self", "self", { created_at: 900 });
    f.addEdge(observation.id, f.same.id);
    const rawObservation = f.db
      .prepare("SELECT * FROM semantic_nodes WHERE id = ?")
      .get(observation.id);
    const client = llm(f),
      path = backup(f),
      anchor = { id: f.canonical.id, label: f.canonical.label };
    const result = await judgeSelfLabelNodes({
      dataDir: f.dataDir,
      selfNode: anchor.id,
      llm: client.client,
    });
    expect(result.file.self_node).toEqual(anchor);
    expect(readVerdictFile(result.path).self_node).toEqual(anchor);
    expect(client.complete).toHaveBeenCalledTimes(3);
    for (const [request] of client.complete.mock.calls) {
      const dossier = JSON.parse(request.messages[0]!.content) as SelfLabelDossier;
      expect(dossier.self.node_id).toBe(anchor.id);
      expect([f.same.id, f.other.id, f.unsure.id]).toContain(dossier.node.id);
    }
    // The report is refreshed when planning, independently of the judged candidates.
    f.addEdge(f.neighbor.id, observation.id);
    await expect(planned(f, result.file, path)).rejects.toThrow("--self-node is required");
    await expect(planned(f, result.file, path, undefined, observation.id)).rejects.toThrow(
      "selected self node since judging",
    );
    const plan = await planned(f, result.file, path, undefined, anchor.id);
    expect(plan.selfLabelJudgments.self_node).toEqual(anchor);
    expect(plan.input.nodeMerges).toEqual([{ from: f.same.id, into: anchor.id }]);
    expect(
      plan.reports.find((row) => row.category === "other_self_nodes_report_only")!.rows,
    ).toEqual([
      {
        id: observation.id,
        label: observation.label,
        created_at: 900,
        edge_count: 2,
        source_count: 1,
      },
    ]);
    expect(
      plan.sql.some((row) => row.table === "semantic_nodes" && row.after?.id === observation.id),
    ).toBe(false);
    for (const row of [...plan.additionalSql, ...plan.completionSql].filter(
      (row) => row.table === "maintenance_audit",
    ))
      expect(JSON.parse(String(row.after!.targets)).self_node).toEqual(anchor);
    await expect(applySelfLabelNodes(plan, { backupVerified: path })).rejects.toThrow(
      "--self-node is required",
    );
    await expect(
      applySelfLabelNodes(plan, { backupVerified: path, selfNode: observation.id }),
    ).rejects.toThrow("selected self node since judging");
    expect(f.audit.list()).toHaveLength(0);
    expect(existsSync(graph.pendingPath(f.dataDir))).toBe(false);
    await expect(
      applySelfLabelNodes(plan, {
        backupVerified: path,
        selfNode: anchor.id,
        checkpoint: (point) => {
          if (point === "after_sqlite") throw new Error("interrupted");
        },
      }),
    ).rejects.toThrow("interrupted");
    await expect(
      applySelfLabelNodes(loadSelfLabelPlan(f.dataDir), { backupVerified: path, resume: true }),
    ).rejects.toThrow("--self-node is required");
    await applySelfLabelNodes(loadSelfLabelPlan(f.dataDir), {
      backupVerified: path,
      selfNode: anchor.id,
      resume: true,
    });
    expect(f.db.prepare("SELECT * FROM semantic_nodes WHERE id = ?").get(observation.id)).toEqual(
      rawObservation,
    );
    expect(
      f.audit.list().every((row) => fingerprint(row.targets.self_node) === fingerprint(anchor)),
    ).toBe(true);
    expect(f.audit.list().filter((row) => row.action.startsWith("judge-self-label:"))).toHaveLength(
      3,
    );
  });

  it("resumes judging with the same selected anchor and refuses a different anchor without overwriting the file", async () => {
    const f = await fixture(),
      otherAnchor = await f.addNode("self", "別名"),
      client = llm(f);
    const first = await judgeSelfLabelNodes({
      dataDir: f.dataDir,
      selfNode: otherAnchor.id,
      llm: client.client,
    });
    expect(first.file.self_node).toEqual({ id: otherAnchor.id, label: otherAnchor.label });
    const original = readFileSync(first.path, "utf8");
    client.complete.mockClear();
    await expect(
      judgeSelfLabelNodes({
        dataDir: f.dataDir,
        selfNode: f.canonical.id,
        out: first.path,
        llm: client.client,
      }),
    ).rejects.toThrow("Selected self node differs from the verdict file");
    expect(readFileSync(first.path, "utf8")).toBe(original);
    const resumed = await judgeSelfLabelNodes({
      dataDir: f.dataDir,
      selfNode: otherAnchor.id,
      out: first.path,
      llm: client.client,
    });
    expect(resumed.file.verdicts).toEqual(first.file.verdicts);
    expect(client.complete).not.toHaveBeenCalled();
  });

  it("refuses incidental structural changes to a report-only self node", async () => {
    const f = await fixture(),
      observation = await f.addNode("self", "別名");
    f.db
      .prepare("UPDATE semantic_nodes SET corrected_by = ? WHERE id = ?")
      .run(f.same.id, observation.id);
    const file = await judged(f, false, undefined, f.canonical.id),
      before = rows(f);
    await expect(planned(f, file, backup(f), undefined, f.canonical.id)).rejects.toThrow(
      "Merge would modify non-merge node",
    );
    expect(rows(f)).toEqual(before);
  });

  it("builds dossiers with both edge directions, capped repository excerpts and explicit missing sources", async () => {
    const f = await fixture();
    const sourceIds = [f.episode.id, ...Array.from({ length: 8 }, () => createEpisodeId())];
    f.db
      .prepare("UPDATE semantic_nodes SET source_episode_ids = ? WHERE id = ?")
      .run(JSON.stringify(sourceIds), f.same.id);
    const get = vi.spyOn(f.episodes, "get");
    const dossier = await buildSelfLabelDossier(f.db, f.episodes, f.same.id);
    expect(dossier.self).toMatchObject({
      entity_id: f.self.id,
      canonical_name: "Écho",
      aliases: ["別名"],
    });
    expect(dossier.node).toMatchObject({
      kind: "entity",
      label: " éCHO ",
      domain: "identity",
      confidence: 0.8,
      created_at: 1000,
      aliases: ["先の名前"],
    });
    expect(dossier.edges.map((edge) => edge.direction).sort()).toEqual(["incoming", "outgoing"]);
    expect(
      dossier.edges.every(
        (edge) =>
          edge.other.label === "Projet" &&
          edge.other.kind === "concept" &&
          edge.relation === "supports",
      ),
    ).toBe(true);
    expect(dossier.episodes).toHaveLength(DOSSIER_EPISODE_LIMIT);
    expect(dossier.episodes_omitted).toBe(3);
    expect(dossier.episodes[0]!.excerpt!.text.length).toBeLessThanOrEqual(DOSSIER_EXCERPT_CHARS);
    expect(dossier.episodes[0]!.excerpt!.truncated).toBe(true);
    expect(dossier.episodes[1]!.missing).toBe(true);
    expect(get).toHaveBeenCalledWith(f.episode.id, { includeArchived: true });
    expect(get).toHaveBeenCalledTimes(DOSSIER_EPISODE_LIMIT);
    f.db
      .prepare("UPDATE episode_stats SET gist = ?, archived = 1 WHERE episode_id = ?")
      .run("短い要約", f.episode.id);
    const withGist = await buildSelfLabelDossier(f.db, f.episodes, f.same.id);
    expect(withGist.episodes[0]).toMatchObject({
      excerpt: { text: "短い要約" },
      excerpt_source: "gist",
      archived: true,
      missing: false,
    });
  });

  it("judges once per node at concurrency two, uses extraction, and only writes a round-trippable verdict file", async () => {
    const f = await fixture();
    const before = rows(f),
      vectors = await f.table.list();
    writeFileSync(
      join(f.dataDir, "config.json"),
      JSON.stringify({
        anthropic: {
          models: { extraction: "fixture-extraction-model", background: "different-model" },
        },
      }),
    );
    const model = llm(f);
    const result = await judgeSelfLabelNodes({ dataDir: f.dataDir, llm: model.client });
    expect(model.complete).toHaveBeenCalledTimes(3);
    expect(model.maxActive()).toBe(2);
    expect(result.file.model_role).toBe("extraction");
    expect(result.file.model).toBe("fixture-extraction-model");
    for (const [request] of model.complete.mock.calls) {
      expect(request.model).toBe(result.file.model);
      expect(request.tool_choice).toEqual({ type: "tool", name: "JudgeSelfLabelNode" });
      expect(request.system).toContain("label match is not evidence");
      expect(request.system).toContain("Unsure is a valid");
    }
    expect(readVerdictFile(result.path)).toEqual(result.file);
    expect(rows(f)).toEqual(before);
    expect(await f.table.list()).toEqual(vectors);
    expect(existsSync(graph.pendingPath(f.dataDir))).toBe(false);
    expect(() => writeVerdictFile(result.path, result.file)).toThrow("already exists");
  });

  it("never recovers a verdict from message text or manufactures one when the tool is malformed", async () => {
    const f = await fixture();
    const client = llm(f);
    client.complete.mockResolvedValue({
      text: '{"verdict":"self","reason":"x","confidence":1}',
      input_tokens: 1,
      output_tokens: 1,
      stop_reason: "end_turn",
      tool_calls: [],
    });
    const result = await judgeSelfLabelNodes({ dataDir: f.dataDir, llm: client.client });
    expect(result.file.verdicts).toEqual([]);
    expect(result.file.failures).toHaveLength(3);
    expect(result.file.failures.every((row) => row.error_class === "StructuredToolCallError")).toBe(
      true,
    );
    expect(readVerdictFile(result.path)).toEqual(result.file);
    expect(client.complete).toHaveBeenCalledTimes(3);
    expect(f.audit.list()).toHaveLength(0);
  });

  it("rejects conflicting judgment calls instead of accepting the first verdict", async () => {
    const f = await fixture(),
      client = llm(f),
      before = rows(f);
    client.complete.mockResolvedValue({
      text: "",
      input_tokens: 1,
      output_tokens: 1,
      stop_reason: "tool_use",
      tool_calls: ["self", "unsure"].map((verdict, index) => ({
        id: `judgment_${index}`,
        name: "JudgeSelfLabelNode",
        input: { verdict, reason: "Conflicting fixture judgments", confidence: 0.75 },
      })),
    });
    const result = await judgeSelfLabelNodes({ dataDir: f.dataDir, llm: client.client });
    expect(result.file.verdicts).toEqual([]);
    expect(result.file.failures).toHaveLength(3);
    expect(
      result.file.failures.every(
        (row) =>
          row.error_class === "StructuredToolCallError" &&
          row.cause_message === "Expected exactly one JudgeSelfLabelNode call; received 2",
      ),
    ).toBe(true);
    expect(rows(f)).toEqual(before);
    expect(readVerdictFile(result.path)).toEqual(result.file);
  });

  it("keeps successful results through a partial failure and retries only the failed node", async () => {
    const f = await fixture(),
      client = llm(f),
      before = rows(f);
    const complete = client.complete.getMockImplementation()!;
    client.complete.mockImplementation(async (request) => {
      const dossier = JSON.parse(request.messages[0]!.content) as SelfLabelDossier;
      if (dossier.node.id === f.other.id) throw new TypeError("fixture transport failure");
      return await complete(request);
    });
    const out = join(f.dataDir, "partial.json");
    const partial = await judgeSelfLabelNodes({ dataDir: f.dataDir, out, llm: client.client });
    expect(client.complete).toHaveBeenCalledTimes(3);
    expect(partial.file.verdicts).toHaveLength(2);
    expect(partial.file.failures).toMatchObject([
      {
        status: "failed",
        node_id: f.other.id,
        error_class: "StructuredToolCallError",
        cause_class: "TypeError",
      },
    ]);
    expect(readVerdictFile(out)).toEqual(partial.file);
    await expect(planned(f, partial.file)).rejects.toThrow(`${f.other.id} (failed)`);
    const retry = llm(f);
    const resumed = await judgeSelfLabelNodes({ dataDir: f.dataDir, out, llm: retry.client });
    expect(retry.complete).toHaveBeenCalledTimes(1);
    expect(JSON.parse(retry.complete.mock.calls[0]![0].messages[0]!.content).node.id).toBe(
      f.other.id,
    );
    expect(resumed.file.failures).toEqual([]);
    expect(resumed.file.verdicts).toHaveLength(3);
    for (const record of partial.file.verdicts)
      expect(resumed.file.verdicts.find((row) => row.node_id === record.node_id)).toEqual(record);
    const again = llm(f);
    await judgeSelfLabelNodes({ dataDir: f.dataDir, out, llm: again.client });
    expect(again.complete).not.toHaveBeenCalled();
    expect(rows(f)).toEqual(before);
  });

  it("persists a completed node while its peer is pending and lists every missing node on plan", async () => {
    const f = await fixture(),
      client = llm(f),
      out = join(f.dataDir, "in-flight.json");
    const ids = selfLabelInventory(f.db).candidates.map((row) => String(row.id));
    let release!: () => void;
    const gate = new Promise<void>((done) => {
      release = done;
    });
    const complete = client.complete.getMockImplementation()!;
    client.complete.mockImplementation(async (request) => {
      if (JSON.parse(request.messages[0]!.content).node.id === ids[1]) await gate;
      return await complete(request);
    });
    const running = judgeSelfLabelNodes({ dataDir: f.dataDir, out, llm: client.client });
    try {
      await vi.waitFor(() =>
        expect(readVerdictFile(out).verdicts.map((row) => row.node_id)).toContain(ids[0]),
      );
      const saved = readVerdictFile(out);
      expect(saved.completed_at).toBeNull();
      expect(saved.candidate_node_ids).toEqual(ids);
      expect(saved.verdicts).toHaveLength(1);
      await expect(planned(f, saved)).rejects.toThrow(`${ids[1]} (missing), ${ids[2]} (missing)`);
      expect(client.complete).toHaveBeenCalledTimes(2);
    } finally {
      release();
      await running;
    }
    expect(readVerdictFile(out).verdicts).toHaveLength(3);
  });

  it("resume-judging rejudges only changed dossiers and preserves unchanged verdicts", async () => {
    const f = await fixture(),
      out = join(f.dataDir, "changed.json");
    const original = await judgeSelfLabelNodes({ dataDir: f.dataDir, out, llm: llm(f).client });
    f.db.prepare("UPDATE semantic_nodes SET confidence = 0.5 WHERE id = ?").run(f.unsure.id);
    const retry = llm(f);
    const next = await judgeSelfLabelNodes({ dataDir: f.dataDir, out, llm: retry.client });
    expect(retry.complete).toHaveBeenCalledTimes(1);
    expect(next.file.verdicts.find((row) => row.node_id === f.unsure.id)!.dossier_hash).not.toBe(
      original.file.verdicts.find((row) => row.node_id === f.unsure.id)!.dossier_hash,
    );
    for (const record of original.file.verdicts.filter((row) => row.node_id !== f.unsure.id))
      expect(next.file.verdicts.find((row) => row.node_id === record.node_id)).toEqual(record);
  });

  it.each([1, 2] as const)(
    "resumes complete version-%s files without repeating judgments",
    async (version) => {
      const f = await fixture(),
        original = await judged(f),
        out = join(f.dataDir, "legacy-verdicts.json");
      const {
        self_node: _anchor,
        failures: _failures,
        candidate_node_ids: _candidates,
        ...old
      } = original;
      const legacy =
        version === 1
          ? { ...old, version }
          : {
              ...old,
              version,
              failures: original.failures,
              candidate_node_ids: original.candidate_node_ids,
            };
      writeFileSync(out, JSON.stringify(legacy));
      const plan = await planned(f, readVerdictFile(out));
      expect(plan.selfLabelJudgments.self_node).toEqual(original.self_node);
      const client = llm(f);
      const next = await judgeSelfLabelNodes({ dataDir: f.dataDir, out, llm: client.client });
      expect(client.complete).not.toHaveBeenCalled();
      expect(next.file.version).toBe(3);
      expect(next.file.self_node).toEqual(original.self_node);
      expect(next.file.verdicts).toEqual(original.verdicts);
      expect(readVerdictFile(out)).toEqual(next.file);
    },
  );

  it("does not overwrite an external verdict-file edit while a judgment is running", async () => {
    const f = await fixture(),
      client = llm(f),
      out = join(f.dataDir, "edited.json");
    const complete = client.complete.getMockImplementation()!;
    let external: string | undefined;
    client.complete.mockImplementation(async (request) => {
      if (!external) {
        const edited = readVerdictFile(out);
        edited.completed_at = 123;
        external = JSON.stringify(edited);
        writeFileSync(out, external);
      }
      return await complete(request);
    });
    await expect(
      judgeSelfLabelNodes({ dataDir: f.dataDir, out, llm: client.client }),
    ).rejects.toThrow("verdict file changed while judging");
    expect(readFileSync(out, "utf8")).toBe(external);
    expect(f.audit.list()).toHaveLength(0);
  });

  it("rejects malformed, duplicated or tampered verdict files", async () => {
    const f = await fixture(),
      file = await judged(f);
    const changed = structuredClone(file);
    changed.verdicts[0]!.dossier.node.description.text = "tampered";
    expect(() => parseVerdictFile(changed)).toThrow("Invalid or duplicate");
    expect(() =>
      parseVerdictFile({ ...file, verdicts: [...file.verdicts, file.verdicts[0]] }),
    ).toThrow("Invalid or duplicate");
    expect(() =>
      parseVerdictFile({ ...file, verdicts: [{ ...file.verdicts[0], confidence: 2 }] }),
    ).toThrow();
    expect(() => parseVerdictFile({ ...file, self_node: undefined })).toThrow(
      "requires the selected self node",
    );
    expect(() =>
      parseVerdictFile({ ...file, self_node: { id: f.other.id, label: f.other.label } }),
    ).toThrow("Invalid or duplicate");
  });

  it.each(["node", "edge", "target", "episode"])(
    "refuses %s evidence changed since judgment before making a journal or audit",
    async (change) => {
      const f = await fixture(),
        file = await judged(f);
      if (change === "node")
        f.db.prepare("UPDATE semantic_nodes SET confidence = 0.4 WHERE id = ?").run(f.same.id);
      if (change === "edge")
        f.db.prepare("UPDATE semantic_edges SET confidence = 0.4 WHERE id = ?").run(f.outgoing.id);
      if (change === "target")
        f.db
          .prepare("UPDATE semantic_nodes SET description = 'changed' WHERE id = ?")
          .run(f.canonical.id);
      if (change === "episode")
        f.db
          .prepare("UPDATE episode_stats SET gist = 'changed' WHERE episode_id = ?")
          .run(f.episode.id);
      const before = rows(f);
      await expect(planned(f, file)).rejects.toThrow("Dossier hash mismatch");
      expect(rows(f)).toEqual(before);
      expect(existsSync(graph.pendingPath(f.dataDir))).toBe(false);
    },
  );

  it("plans all three judgments and delegates only self merges to the existing graph machinery", async () => {
    const f = await fixture(),
      file = await judged(f),
      before = rows(f);
    const spy = vi.spyOn(graph, "planEntityGraphRepair");
    const plan = await planned(f, file);
    expect(spy).toHaveBeenCalledWith(
      {
        dataDir: f.dataDir,
        merges: [],
        scope: "explicit-node-merges",
        nodeMerges: [{ from: f.same.id, into: f.canonical.id }],
      },
      expect.anything(),
      expect.anything(),
    );
    expect(plan.goals).toEqual([]);
    expect(plan.migrations).toEqual([]);
    expect(
      plan.sql
        .filter((row) => row.table === "semantic_nodes")
        .map((row) => row.after!.id)
        .sort(),
    ).toEqual([f.same.id, f.canonical.id].sort());
    const audits = plan.additionalSql.filter((row) =>
      String(row.after?.action).startsWith("judge-self-label:"),
    );
    expect(audits.map((row) => row.after!.action).sort()).toEqual([
      "judge-self-label:other",
      "judge-self-label:self",
      "judge-self-label:unsure",
    ]);
    expect(plan.reports.at(-1)!.rows[0]!.id).toBe(f.duplicateEntity.id);
    expect(rows(f)).toEqual(before);
    expect(f.audit.list()).toHaveLength(0);
  });

  it.each(["entity", "person"])(
    "applies a %s merge through the existing path with embeddings, edge collisions and unchanged non-merges",
    async (kind) => {
      const f = await fixture();
      await f.nodes.update(f.same.id, { kind });
      const collision = f.addEdge(f.canonical.id, f.neighbor.id);
      const coalesced = f.addEdge(f.same.id, f.canonical.id);
      const rawOther = f.db.prepare("SELECT * FROM semantic_nodes WHERE id = ?").get(f.other.id);
      const file = await judged(f),
        path = backup(f),
        plan = await planned(f, file, path);
      const apply = vi.spyOn(graph, "applyEntityGraphRepair");
      await applySelfLabelNodes(plan, { backupVerified: path });
      expect(apply).toHaveBeenCalledOnce();
      expect(
        f.db
          .prepare("SELECT archived, superseded_by, status FROM semantic_nodes WHERE id = ?")
          .get(f.same.id),
      ).toEqual({ archived: 1, superseded_by: f.canonical.id, status: "superseded" });
      expect(
        f.db.prepare("SELECT to_node_id FROM semantic_edges WHERE id = ?").get(f.incoming.id)!
          .to_node_id,
      ).toBe(f.canonical.id);
      expect(
        f.db.prepare("SELECT valid_to FROM semantic_edges WHERE id = ?").get(collision.id)!
          .valid_to,
      ).toBeNull();
      expect(
        f.db.prepare("SELECT valid_to FROM semantic_edges WHERE id = ?").get(f.outgoing.id)!
          .valid_to,
      ).not.toBeNull();
      expect(
        f.db
          .prepare("SELECT from_node_id, to_node_id, valid_to FROM semantic_edges WHERE id = ?")
          .get(coalesced.id),
      ).toMatchObject({
        from_node_id: f.same.id,
        to_node_id: f.canonical.id,
        valid_to: expect.any(Number),
      });
      expect(f.db.prepare("SELECT * FROM semantic_nodes WHERE id = ?").get(f.other.id)).toEqual(
        rawOther,
      );
      expect(
        f.db.prepare("SELECT * FROM entities WHERE id = ?").get(f.duplicateEntity.id),
      ).toBeDefined();
      expect(await f.nodes.get(f.same.id)).toMatchObject({
        archived: true,
        superseded_by: f.canonical.id,
      });
      const connection = (await openLance(f.dataDir))!;
      const mirror = new LanceDbTable(await connection.openTable("semantic_nodes"));
      try {
        const current = await mirror.list();
        for (const change of plan.lance.filter((row) => row.table === "semantic_nodes")) {
          expect(
            lanceRowMatches(current.find((row) => row.id === change.id)!, change, "after"),
          ).toBe(true);
        }
        expect(
          JSON.parse(String(current.find((row) => row.id === f.canonical.id)!.aliases)),
        ).toContain("éCHO");
      } finally {
        mirror.close();
        connection.close();
      }
      expect(f.db.prepare("SELECT * FROM semantic_node_vector_sync_outbox").all()).toEqual([]);
      for (const record of file.verdicts) {
        const audit = f.audit
          .list({ run_id: plan.runId! })
          .find((row) => row.action === `judge-self-label:${record.verdict}`)!;
        expect(audit.process).toBe("correction");
        expect(audit.targets.judgment).toEqual({
          verdict: record.verdict,
          reason: record.reason,
          confidence: record.confidence,
          model: file.model,
          dossier_hash: record.dossier_hash,
        });
        if (record.verdict !== "self")
          expect(audit.reversal).toEqual({ kind: "none", data_changes: false });
      }
      expect(existsSync(graph.pendingPath(f.dataDir))).toBe(false);
    },
  );

  it("an all-nonmerge file changes only maintenance audits, including on resume", async () => {
    const f = await fixture(),
      file = await judged(f, true),
      path = backup(f);
    const before = rows(f),
      vectors = await f.table.list();
    const plan = await planned(f, file, path);
    expect(plan.sql).toEqual([]);
    expect(plan.lance).toEqual([]);
    await applySelfLabelNodes(plan, { backupVerified: path });
    const after = rows(f);
    delete before.maintenance_audit;
    delete after.maintenance_audit;
    expect(after).toEqual(before);
    expect(await f.table.list()).toEqual(vectors);
    const audits = f.audit.list();
    expect(audits.filter((row) => row.action === "judge-self-label:other")).toHaveLength(3);
    await applySelfLabelNodes(loadSelfLabelPlan(f.dataDir), { backupVerified: path, resume: true });
    expect(f.audit.list()).toEqual(audits);
  });

  it("refuses an incidental structural rewrite of a node judged other or unsure", async () => {
    const f = await fixture();
    f.db
      .prepare("UPDATE semantic_nodes SET corrected_by = ? WHERE id = ?")
      .run(f.same.id, f.other.id);
    const file = await judged(f),
      before = rows(f);
    await expect(planned(f, file)).rejects.toThrow("Merge would modify non-merge node");
    expect(rows(f)).toEqual(before);
    expect(existsSync(graph.pendingPath(f.dataDir))).toBe(false);
  });

  it("reapplying unchanged other/unsure judgments produces no changes or duplicate audits", async () => {
    const f = await fixture(),
      file = await judged(f, true),
      path = backup(f);
    file.verdicts.find((row) => row.node_id === f.unsure.id)!.verdict = "unsure";
    const first = await planned(f, file, path),
      preparedBeforeApply = await planned(f, file, path);
    await applySelfLabelNodes(first, { backupVerified: path });
    const before = rows(f),
      audits = f.audit.list();
    for (const record of file.verdicts) {
      expect(
        audits.find((row) => row.targets.node_id === record.node_id)!.targets.judgment_id,
      ).toBe(
        fingerprint({
          node_id: record.node_id,
          dossier_hash: record.dossier_hash,
          verdict: record.verdict,
        }),
      );
    }
    expect(await applySelfLabelNodes(first, { backupVerified: path })).toEqual({
      changed: false,
      run_id: null,
    });
    expect(await applySelfLabelNodes(preparedBeforeApply, { backupVerified: path })).toEqual({
      changed: false,
      run_id: null,
    });
    const again = await planned(f, file, path);
    expect(again.runId).toBeNull();
    expect(again.sql).toEqual([]);
    expect(again.lance).toEqual([]);
    expect(again.additionalSql).toEqual([]);
    expect(again.completionSql).toEqual([]);
    expect(
      again.reports
        .find((report) => report.category === "judgment_audits")!
        .rows.every((row) => row.status === "already_recorded"),
    ).toBe(true);
    expect(await applySelfLabelNodes(again, { backupVerified: path })).toEqual({
      changed: false,
      run_id: null,
    });
    expect(rows(f)).toEqual(before);
    expect(existsSync(graph.pendingPath(f.dataDir))).toBe(false);
  });

  it("recognizes older judgment audits without the explicit stable identity field", async () => {
    const f = await fixture(),
      file = await judged(f, true),
      path = backup(f);
    await applySelfLabelNodes(await planned(f, file, path), { backupVerified: path });
    f.db
      .prepare(
        "UPDATE maintenance_audit SET targets = json_remove(targets, '$.judgment_id') WHERE action LIKE 'judge-self-label:%'",
      )
      .run();
    const before = rows(f);
    const again = await planned(f, file, path);
    expect(again.runId).toBeNull();
    await applySelfLabelNodes(again, { backupVerified: path });
    expect(rows(f)).toEqual(before);
  });

  it("records a new audit when the dossier or verdict changes and still refuses stale evidence", async () => {
    const f = await fixture(),
      path = backup(f);
    const original = await judgeSelfLabelNodes({ dataDir: f.dataDir, llm: llm(f, true).client });
    await applySelfLabelNodes(await planned(f, original.file, path), { backupVerified: path });
    f.db.prepare("UPDATE semantic_nodes SET confidence = 0.6 WHERE id = ?").run(f.unsure.id);
    await expect(planned(f, original.file, path)).rejects.toThrow("Dossier hash mismatch");
    const updated = await judgeSelfLabelNodes({
      dataDir: f.dataDir,
      out: original.path,
      llm: llm(f, true).client,
    });
    const plan = await planned(f, updated.file, path);
    expect(
      plan.additionalSql.filter((row) => String(row.after?.action).startsWith("judge-self-label:")),
    ).toHaveLength(1);
    await applySelfLabelNodes(plan, { backupVerified: path });
    updated.file.verdicts.find((row) => row.node_id === f.other.id)!.verdict = "unsure";
    const differentVerdict = await planned(f, updated.file, path);
    expect(
      differentVerdict.additionalSql.filter((row) =>
        String(row.after?.action).startsWith("judge-self-label:"),
      ),
    ).toHaveLength(1);
    await applySelfLabelNodes(differentVerdict, { backupVerified: path });
    const judgments = f.audit.list().filter((row) => row.action.startsWith("judge-self-label:"));
    expect(judgments).toHaveLength(5);
    expect(new Set(judgments.map((row) => row.targets.judgment_id)).size).toBe(5);
  });

  it("refuses a partially outdated audit plan before writing a journal", async () => {
    const f = await fixture(),
      file = await judged(f, true),
      path = backup(f),
      plan = await planned(f, file, path);
    const change = plan.additionalSql.find((row) =>
      String(row.after?.action).startsWith("judge-self-label:"),
    )!.after!;
    f.audit.record({
      run_id: plan.runId!,
      process: "correction",
      action: String(change.action),
      targets: JSON.parse(String(change.targets)),
      reversal: JSON.parse(String(change.reversal)),
    });
    const before = rows(f);
    await expect(applySelfLabelNodes(plan, { backupVerified: path })).rejects.toThrow(
      "judgment audits changed since planning; re-plan",
    );
    expect(rows(f)).toEqual(before);
    expect(existsSync(graph.pendingPath(f.dataDir))).toBe(false);
  });

  it.each([
    "after_journal",
    "after_sqlite",
    "after_lance:semantic_nodes",
    "after_completion",
    "after_archive",
  ] as const)("resumes %s with exactly one judgment per node", async (point) => {
    const f = await fixture(),
      path = backup(f),
      plan = await planned(f, undefined, path);
    await expect(
      applySelfLabelNodes(plan, {
        backupVerified: path,
        checkpoint: (at) => {
          if (at === point) throw new Error("interrupted");
        },
      }),
    ).rejects.toThrow("interrupted");
    await applySelfLabelNodes(loadSelfLabelPlan(f.dataDir), { backupVerified: path, resume: true });
    expect(f.audit.list().filter((row) => row.action.startsWith("judge-self-label:"))).toHaveLength(
      3,
    );
    expect(
      f.db.prepare("SELECT archived FROM semantic_nodes WHERE id = ?").get(f.same.id)!.archived,
    ).toBe(1);
    expect(existsSync(graph.pendingPath(f.dataDir))).toBe(false);
  });

  it.each(["after_journal", "after_sqlite", "after_archive"] as const)(
    "legacy CLI refuses judged recovery at %s without changing the store or journal",
    async (point) => {
      const f = await fixture(),
        path = backup(f),
        plan = await planned(f, undefined, path);
      await expect(
        applySelfLabelNodes(plan, {
          backupVerified: path,
          checkpoint: (at) => {
            if (at === point) throw new Error("interrupted");
          },
        }),
      ).rejects.toThrow("interrupted");
      const journalPath =
        point === "after_archive"
          ? join(f.dataDir, "entity-graph-repair", `${plan.runId}.json`)
          : graph.pendingPath(f.dataDir);
      const journal = readFileSync(journalPath, "utf8"),
        before = rows(f);
      expect(JSON.parse(journal).kind).toBe("judge-self-label-nodes");
      await expect(graph.main(["--data-dir", f.dataDir, "--apply", "--resume"])).rejects.toThrow(
        "must resume through scripts/judge-self-label-nodes.ts",
      );
      expect(rows(f)).toEqual(before);
      expect(readFileSync(journalPath, "utf8")).toBe(journal);
      await applySelfLabelNodes(loadSelfLabelPlan(f.dataDir), {
        backupVerified: path,
        resume: true,
      });
      expect(
        f.audit.list().filter((row) => row.action.startsWith("judge-self-label:")),
      ).toHaveLength(3);
    },
  );

  it("legacy recovery also refuses pre-kind judged journals and mismatched kind/origin", async () => {
    const f = await fixture(),
      path = backup(f),
      plan = await planned(f, undefined, path);
    await expect(
      applySelfLabelNodes(plan, {
        backupVerified: path,
        checkpoint: () => {
          throw new Error("interrupted");
        },
      }),
    ).rejects.toThrow("interrupted");
    const pending = graph.pendingPath(f.dataDir),
      journal = JSON.parse(readFileSync(pending, "utf8"));
    delete journal.kind;
    writeFileSync(pending, JSON.stringify(journal));
    await expect(graph.resumeEntityGraphRepair(f.dataDir)).rejects.toThrow("must resume through");
    journal.kind = "entity-graph-repair";
    writeFileSync(pending, JSON.stringify(journal));
    await expect(graph.resumeEntityGraphRepair(f.dataDir)).rejects.toThrow("kind/origin mismatch");
    expect(f.audit.list()).toHaveLength(0);
  });

  it("rechecks hashes after planning and before apply, including pending precommit resume", async () => {
    const f = await fixture(),
      path = backup(f),
      plan = await planned(f, undefined, path);
    await expect(
      applySelfLabelNodes(plan, {
        backupVerified: path,
        checkpoint: () => {
          throw new Error("interrupted");
        },
      }),
    ).rejects.toThrow("interrupted");
    const journal = readFileSync(graph.pendingPath(f.dataDir), "utf8");
    f.db.prepare("UPDATE semantic_nodes SET description = 'changed' WHERE id = ?").run(f.unsure.id);
    await expect(
      applySelfLabelNodes(loadSelfLabelPlan(f.dataDir), { backupVerified: path, resume: true }),
    ).rejects.toThrow("Dossier hash mismatch");
    expect(readFileSync(graph.pendingPath(f.dataDir), "utf8")).toBe(journal);
    expect(f.audit.list()).toHaveLength(0);
  });

  it("rejects a node changed after planning before any journal or audit write", async () => {
    const f = await fixture(),
      path = backup(f),
      plan = await planned(f, undefined, path);
    f.db.prepare("UPDATE semantic_nodes SET description = 'modified' WHERE id = ?").run(f.same.id);
    const before = rows(f);
    await expect(applySelfLabelNodes(plan, { backupVerified: path })).rejects.toThrow(
      "Dossier hash mismatch",
    );
    expect(rows(f)).toEqual(before);
    expect(existsSync(graph.pendingPath(f.dataDir))).toBe(false);
  });

  it("rejects changed postcommit evidence and a tampered journal without further writes", async () => {
    const f = await fixture(),
      path = backup(f),
      plan = await planned(f, undefined, path);
    await expect(
      applySelfLabelNodes(plan, {
        backupVerified: path,
        checkpoint: (point) => {
          if (point === "after_sqlite") throw new Error("interrupted");
        },
      }),
    ).rejects.toThrow("interrupted");
    const pending = graph.pendingPath(f.dataDir),
      journal = readFileSync(pending, "utf8");
    f.db
      .prepare("UPDATE semantic_nodes SET description = 'modified' WHERE id = ?")
      .run(f.unsure.id);
    const before = rows(f);
    await expect(
      applySelfLabelNodes(loadSelfLabelPlan(f.dataDir), { backupVerified: path, resume: true }),
    ).rejects.toThrow("Dossier hash mismatch");
    expect(rows(f)).toEqual(before);
    expect(readFileSync(pending, "utf8")).toBe(journal);
    const edited = JSON.parse(journal);
    edited.plan.selfLabelJudgments.verdictFile.verdicts[0].reason = "edited judgment";
    writeFileSync(pending, JSON.stringify(edited));
    expect(() => loadSelfLabelPlan(f.dataDir)).toThrow("Invalid judged self-label journal");
    expect(rows(f)).toEqual(before);
  });

  it("refuses missing or changed backups", async () => {
    const f = await fixture(),
      path = backup(f),
      plan = await planned(f, undefined, path);
    await expect(applySelfLabelNodes(plan)).rejects.toThrow("--backup-verified");
    await expect(applySelfLabelNodes(plan, { backupVerified: f.dataDir })).rejects.toThrow(
      "independent",
    );
    await expect(applySelfLabelNodes(plan, { backupVerified: backup(f) })).rejects.toThrow(
      "verified backup",
    );
    expect(existsSync(graph.pendingPath(f.dataDir))).toBe(false);
    expect(f.audit.list()).toHaveLength(0);
  });

  it.each(["pending", "unknown"])("refuses %s migrations in both judge and apply", async (kind) => {
    const f = await fixture(),
      path = backup(f),
      plan = await planned(f, undefined, path);
    if (kind === "pending")
      f.db.prepare("DELETE FROM _migrations WHERE id = (SELECT MAX(id) FROM _migrations)").run();
    else
      f.db
        .prepare(
          "UPDATE _migrations SET name = 'unknown' WHERE id = (SELECT MAX(id) FROM _migrations)",
        )
        .run();
    const client = llm(f);
    await expect(judgeSelfLabelNodes({ dataDir: f.dataDir, llm: client.client })).rejects.toThrow(
      "Pending or unknown migrations",
    );
    expect(client.complete).not.toHaveBeenCalled();
    await expect(applySelfLabelNodes(plan, { backupVerified: path })).rejects.toThrow(
      "Pending or unknown migrations",
    );
    expect(existsSync(graph.pendingPath(f.dataDir))).toBe(false);
    expect(f.audit.list()).toHaveLength(0);
  });

  it("rejects partial files, candidate changes and bank mismatches before merge planning", async () => {
    const f = await fixture(),
      file = await judged(f),
      spy = vi.spyOn(graph, "planEntityGraphRepair");
    await expect(planned(f, { ...file, verdicts: file.verdicts.slice(1) })).rejects.toThrow(
      "Failed or missing judgments",
    );
    await expect(planned(f, { ...file, data_dir: "/unrelated-bank" })).rejects.toThrow(
      "verdict store",
    );
    await f.addNode("person", "別名");
    await expect(planned(f, file)).rejects.toThrow("Failed or missing judgments");
    expect(spy).not.toHaveBeenCalled();
  });

  it("a zero-candidate file creates no apply journal, audits, embeddings or model requests", async () => {
    const f = await fixture();
    f.db.prepare("UPDATE semantic_nodes SET archived = 1 WHERE kind IN ('entity', 'person')").run();
    const client = llm(f),
      path = backup(f);
    const { file } = await judgeSelfLabelNodes({ dataDir: f.dataDir, llm: client.client });
    const before = rows(f);
    expect(client.complete).not.toHaveBeenCalled();
    expect(file.self_node).toEqual({ id: f.canonical.id, label: f.canonical.label });
    const plan = await planned(f, file, path);
    expect(plan.runId).toBeNull();
    await applySelfLabelNodes(plan, { backupVerified: path });
    expect(rows(f)).toEqual(before);
    expect(existsSync(graph.pendingPath(f.dataDir))).toBe(false);
  });

  it("guards a new verdict path lexically before creating a synthetic live-shaped directory", async () => {
    const f = await fixture(),
      file = await judged(f);
    const out = join(f.dataDir, "demo", "server", ".borg-data", "new.json");
    expect(() => writeVerdictFile(out, file)).toThrow("live demo store is forbidden");
    expect(existsSync(join(f.dataDir, "demo"))).toBe(false);
  });

  it("judge and new plans refuse a pending repair; resume refuses another repair's journal", async () => {
    const f = await fixture(),
      path = backup(f),
      plan = await planned(f, undefined, path);
    await expect(
      applySelfLabelNodes(plan, {
        backupVerified: path,
        checkpoint: () => {
          throw new Error("interrupted");
        },
      }),
    ).rejects.toThrow("interrupted");
    await expect(judged(f)).rejects.toThrow("Entity graph repair is pending");
    await expect(planned(f, plan.selfLabelJudgments.verdictFile, path)).rejects.toThrow(
      "Entity graph repair is pending",
    );
    const pending = graph.pendingPath(f.dataDir),
      journal = JSON.parse(readFileSync(pending, "utf8"));
    delete journal.plan.selfLabelJudgments;
    writeFileSync(pending, JSON.stringify(journal));
    expect(() => loadSelfLabelPlan(f.dataDir)).toThrow("do not resume another repair");
  });

  it("requires live confirmation on synthetic guarded paths and repeats it on resume", async () => {
    const f = await fixture(true),
      path = backup(f);
    await expect(judged(f)).rejects.toThrow("live demo store is forbidden");
    const file = await judged(f, false, path),
      plan = await planned(f, file, path, path);
    await expect(
      applySelfLabelNodes(plan, {
        backupVerified: path,
        liveStoreConfirmed: path,
        checkpoint: (point) => {
          if (point === "after_sqlite") throw new Error("interrupted");
        },
      }),
    ).rejects.toThrow("interrupted");
    await expect(applySelfLabelNodes(plan, { backupVerified: path, resume: true })).rejects.toThrow(
      "live demo store is forbidden",
    );
    await applySelfLabelNodes(loadSelfLabelPlan(f.dataDir, { liveStoreConfirmed: path }), {
      backupVerified: path,
      liveStoreConfirmed: path,
      resume: true,
    });
    expect(
      f.audit
        .list()
        .every(
          (row) =>
            fingerprint(row.targets.live_store_override) === fingerprint({ backup_path: path }),
        ),
    ).toBe(true);
  });

  it.each([false, true])(
    "CLI judge/plan/apply/resume with multiple self nodes = %s",
    async (multiple) => {
      const f = await fixture();
      if (multiple) await f.addNode("self", "別名");
      const selection = multiple ? ["--self-node", f.canonical.id] : [];
      vi.spyOn(clients, "createLlmFactory").mockReturnValue(() => llm(f, true).client);
      const stdout = vi.spyOn(process.stdout, "write").mockImplementation(((
        text: string,
        callback?: (error?: Error | null) => void,
      ) => {
        callback?.();
        return true;
      }) as typeof process.stdout.write);
      const out = join(f.dataDir, "verdicts.json");
      if (multiple)
        await expect(main(["--data-dir", f.dataDir, "--out", out])).rejects.toThrow(
          "--self-node is required",
        );
      await main(["--data-dir", f.dataDir, "--out", out, ...selection]);
      expect(readVerdictFile(out).verdicts).toHaveLength(3);
      expect(readVerdictFile(out).self_node).toEqual({
        id: f.canonical.id,
        label: f.canonical.label,
      });
      if (multiple)
        await expect(main(["plan", "--data-dir", f.dataDir, "--verdicts", out])).rejects.toThrow(
          "--self-node is required",
        );
      await main(["plan", "--data-dir", f.dataDir, "--verdicts", out, ...selection]);
      const dry = JSON.parse(String(stdout.mock.calls.at(-1)![0]));
      expect(dry.mode).toBe("dry-run");
      expect(dry.plan.additionalSql).toHaveLength(4);
      expect(f.audit.list()).toHaveLength(0);
      await expect(
        main(["--data-dir", f.dataDir, "--verdicts", out, "--apply", ...selection]),
      ).rejects.toThrow("--backup-verified");
      const path = backup(f);
      await main([
        "apply",
        "--data-dir",
        f.dataDir,
        "--verdicts",
        out,
        "--backup-verified",
        path,
        ...selection,
      ]);
      await main([
        "apply",
        "--data-dir",
        f.dataDir,
        "--verdicts",
        out,
        "--backup-verified",
        path,
        ...selection,
      ]);
      expect(JSON.parse(String(stdout.mock.calls.at(-1)![0]))).toMatchObject({
        complete: true,
        status: "no changes",
        changed: false,
        run_id: null,
      });
      await main([
        "apply",
        "--data-dir",
        f.dataDir,
        "--resume",
        "--backup-verified",
        path,
        ...selection,
      ]);
      expect(f.audit.list().filter((row) => row.action === "judge-self-label:other")).toHaveLength(
        3,
      );
    },
  );
});
