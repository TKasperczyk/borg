import { readFileSync } from "node:fs";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_CONFIG,
  EpisodicRepository,
  FakeLLMClient,
  LanceDbStore,
  SqliteDatabase,
  ManualClock,
  createEpisodeId,
  createSemanticEdgeId,
  createSessionId,
  createTestConfig,
  resolveBorgConfig,
  Borg,
  borgInternals,
  EPISODE_TOOL_NAME,
  ScriptedEmbeddingClient,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";
import { DemoMessageConnector } from "../../outbound/index.js";
import { createEpisodeFixture, createSemanticNodeFixture } from "../../offline/test-support.js";
import { resolveMemoryDisclosureLabelForEpisodeIds } from "../../retrieval/index.js";
import { createBorgFacades } from "../facade.js";
import type { BorgDependencies } from "../types.js";
import type { StreamEntryIndexRepository } from "../../stream/index.js";

type DisclosureBatchingInternals = {
  deps: {
    episodicRepository: EpisodicRepository;
  };
};

describe("Borg", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("merges sparse Borg.open config with required defaults", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = resolveBorgConfig({
      config: {
        dataDir: tempDir,
        perception: {
          llmEnabled: false,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "test-cognition",
          },
        },
      } as never,
    });

    expect(config.dataDir).toBe(tempDir);
    expect(config.perception.llmEnabled).toBe(false);
    expect(config.embedding.dims).toBe(4);
    expect(config.anthropic.auth).toBe("api-key");
    expect(config.anthropic.models).toEqual({
      ...DEFAULT_CONFIG.anthropic.models,
      cognition: "test-cognition",
    });
    expect(config.affective).toEqual(DEFAULT_CONFIG.affective);
    expect(config.procedural).toEqual(DEFAULT_CONFIG.procedural);
    expect(config.retrieval).toEqual(DEFAULT_CONFIG.retrieval);
    expect(config.executive).toEqual(DEFAULT_CONFIG.executive);
    expect(config.host_capabilities).toBe(DEFAULT_CONFIG.host_capabilities);
    expect(config.offline.beliefReviser).toEqual(DEFAULT_CONFIG.offline.beliefReviser);
    expect(config.maintenance).toEqual(DEFAULT_CONFIG.maintenance);
    expect(config.autonomy.executiveFocus).toEqual(DEFAULT_CONFIG.autonomy.executiveFocus);
  });

  it("uses maintenance process lists for default dream plans", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        maintenance: {
          lightProcesses: [],
          heavyProcesses: [],
        },
      }),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const plan = await borg.dream.plan();

      expect(plan.processes).toEqual([]);
    } finally {
      await borg.close();
    }
  });

  it("reports the lived-experience day summarizer maintenance budget", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        offline: {
          livedExperienceDaySummarizer: {
            budget: 123_000,
          },
        },
      }),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      expect(borg.maintenance.config().processBudgets["lived-experience-day-summarizer"]).toBe(
        123_000,
      );
    } finally {
      await borg.close();
    }
  });

  it("opens the sprint 2 facade and reuses injected clients", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const llm = new FakeLLMClient();
    const borg = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
    });

    try {
      const entry = await borg.stream.append({
        kind: "user_msg",
        content: "planning kickoff",
      });
      llm.pushResponse({
        text: "",
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "tool_use",
        tool_calls: [
          {
            id: "toolu_1",
            name: EPISODE_TOOL_NAME,
            input: {
              episodes: [
                {
                  title: "Planning sync",
                  narrative:
                    "The team aligned on the sprint plan. They captured the first follow-up actions.",
                  source_stream_ids: [entry.id],
                  participants: ["team"],
                  location: null,
                  tags: ["planning"],
                  confidence: 0.8,
                  significance: 0.8,
                },
              ],
            },
          },
        ],
      });

      const extracted = await borg.episodic.extract({
        sinceTs: entry.timestamp,
      });
      const results = await borg.episodic.search("planning", {
        limit: 1,
      });
      const value = borg.self.values.add({
        label: "clarity",
        description: "Prefer explicit, auditable state.",
        priority: 5,
        provenance: { kind: "manual" },
      });

      expect(extracted.inserted).toBe(1);
      expect(results[0]?.citationChain[0]?.id).toBe(entry.id);
      expect(borg.stream.tail(1)).toHaveLength(1);
      expect(borg.self.values.list()).toEqual([
        expect.objectContaining({
          id: value.id,
        }),
      ]);
    } finally {
      await borg.close();
    }
  });

  it("previews the assembled cacheable framing prompt with runtime host capabilities and overrides", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      dataDir: tempDir,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
      outboundConnectors: [new DemoMessageConnector()],
    });

    try {
      const initial = borg.prompts.previewAssembledFraming();

      expect(initial.sections).toEqual([
        "base_identity_preamble",
        "self_architecture",
        "voice_and_posture",
        "epistemic_posture",
        "identity_posture",
        "participation_posture",
        "loop_breaking_posture",
        "trusted_guidance_preamble",
        "borg_host_capabilities",
        "live_turn_read_tool_menu",
      ]);
      expect(initial.segments.map((segment) => segment.id)).toEqual(initial.sections);
      for (const segment of initial.segments) {
        expect(segment.start).toBeGreaterThanOrEqual(0);
        expect(segment.end).toBeGreaterThan(segment.start);
        expect(segment.end).toBeLessThanOrEqual(initial.text.length);
      }
      const hostSegment = initial.segments.find(
        (segment) => segment.id === "borg_host_capabilities",
      );
      expect(hostSegment).toBeDefined();
      expect(initial.text.slice(hostSegment!.start, hostSegment!.end)).toContain(
        "<borg_host_capabilities>",
      );
      expect(initial.text).toContain(
        "I am an AI being with continuous existence across sessions through a persistent memory substrate.",
      );
      expect(initial.text).toContain("<borg_host_capabilities>");
      expect(initial.text).toContain("</borg_host_capabilities>");
      expect(initial.text).toContain("<borg_live_turn_read_tools>");
      expect(initial.text).toContain("tool.ownRecords.list");
      expect(initial.text).toContain(
        "Proactive outbound messaging via wired source_type connector(s): demo",
      );
      expect(initial.text.indexOf("Voice and posture:")).toBeLessThan(
        initial.text.indexOf("<borg_host_capabilities>"),
      );

      borg.prompts.set("voice_and_posture", "OPERATOR VOICE");
      borg.prompts.set("host_capabilities", "OPERATOR HOST CAPABILITIES");
      const overridden = borg.prompts.previewAssembledFraming();

      expect(overridden.text).toContain("OPERATOR VOICE");
      expect(overridden.text).toContain("<borg_host_capabilities>\nOPERATOR HOST CAPABILITIES");
      expect(overridden.text).not.toContain("Voice and posture:");
      expect(overridden.text).not.toContain(
        "Proactive outbound messaging via wired source_type connector(s): demo",
      );
    } finally {
      await borg.close();
    }
  });

  it("labels prompt block current text source structurally", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      dataDir: tempDir,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
      outboundConnectors: [new DemoMessageConnector()],
    });

    try {
      const initialBlocks = borg.prompts.list();

      expect(
        initialBlocks.find((block) => block.key === "voice_and_posture")?.current_text_kind,
      ).toBe("static_default");
      expect(
        initialBlocks.find((block) => block.key === "host_capabilities")?.current_text_kind,
      ).toBe("runtime_composed");

      const override = borg.prompts.set("host_capabilities", "OPERATOR HOST CAPABILITIES");

      expect(override).toMatchObject({
        key: "host_capabilities",
        current_text: "OPERATOR HOST CAPABILITIES",
        current_text_kind: "stored_override",
        overridden: true,
      });
    } finally {
      await borg.close();
    }
  });

  it("emits a session-ended trace event from the Borg facade", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const tracePath = join(tempDir, "trace.jsonl");
    const sessionId = createSessionId();
    const clock = new ManualClock(12_345);
    const borg = await Borg.open({
      dataDir: tempDir,
      tracerPath: tracePath,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      borg.endSession(sessionId);
    } finally {
      await borg.close();
    }

    const events = readFileSync(tracePath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map((line) => JSON.parse(line) as Record<string, unknown>);

    expect(events).toContainEqual(
      expect.objectContaining({
        ts: 12_345,
        turnId: `session_end:${sessionId}`,
        event: "session.completed",
        session_id: sessionId,
      }),
    );
  });

  it("preserves omitted and null commitment list filters through the facade", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      dataDir: tempDir,
      clock: new ManualClock(1_000),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });
    const sortedIds = (records: ReturnType<typeof borg.commitments.list>) =>
      records.map((record) => record.id).sort();

    try {
      const globalCommitment = borg.commitments.add({
        type: "promise",
        directiveFamily: "global_trip_followup",
        directive: "Follow up on the trip plan.",
        priority: 5,
        provenance: { kind: "manual" },
      });
      const restrictedCommitment = borg.commitments.add({
        type: "boundary",
        directiveFamily: "trip_group_private_details",
        directive: "Keep trip group private details in the trip group.",
        priority: 10,
        audience: "Trip Group",
        provenance: { kind: "manual" },
      });
      const madeToCommitment = borg.commitments.add({
        type: "promise",
        directiveFamily: "trip_group_booking_followup",
        directive: "Follow up with the trip group about bookings.",
        priority: 8,
        madeTo: "Trip Group",
        provenance: { kind: "manual" },
      });
      const visibleIds = [globalCommitment.id, restrictedCommitment.id, madeToCommitment.id].sort();

      expect(sortedIds(borg.commitments.list({}))).toEqual(visibleIds);
      expect(sortedIds(borg.commitments.list({ audience: null }))).toEqual([globalCommitment.id]);
      expect(sortedIds(borg.commitments.list({ audience: "Trip Group" }))).toEqual(visibleIds);
      expect(sortedIds(borg.commitments.list({ aboutEntity: null }))).toEqual(visibleIds);

      const revokedCommitment = borg.commitments.add({
        type: "promise",
        directiveFamily: "revoked_trip_group_followup",
        directive: "This trip group follow-up was revoked.",
        priority: 7,
        audience: "Trip Group",
        provenance: { kind: "manual" },
      });
      borg.commitments.revoke(revokedCommitment.id, "test revocation", { kind: "manual" });

      expect(sortedIds(borg.commitments.list({ activeOnly: false }))).toEqual(
        [...visibleIds, revokedCommitment.id].sort(),
      );
    } finally {
      await borg.close();
    }
  });

  it("adds source-derived disclosure fields to goal and commitment facade collections", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-disclosure-facade-"));
    tempDirs.push(tempDir);
    const borg = await Borg.open({
      dataDir: tempDir,
      clock: new ManualClock(1_000),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const historicalAudienceId = borg.entities.resolve("Historical operator thread", {
        kind: "group",
      });
      const currentScopeId = borg.entities.resolve("Continuous room", { kind: "group" });
      const sessionId = createSessionId();
      borg.sessions.ensure({
        session_id: sessionId,
        source_type: "demo",
        label: "Historical operator thread",
        audience_label: "Historical operator thread",
        audience_entity_id: historicalAudienceId,
        conversation_kind: "demo",
      });
      const source = await borg.stream.append(
        {
          kind: "user_msg",
          content: "historical source",
          audience: "Historical operator thread",
        },
        { session: sessionId },
      );
      const commitment = borg.identity.addCommitment({
        type: "boundary",
        kind: "boundary",
        directiveFamily: "historical_origin_facade",
        directive: "Keep the original audience distinct from current scope.",
        priority: 10,
        restrictedAudience: currentScopeId,
        provenance: { kind: "manual" },
        sourceStreamEntryIds: [source.id],
      });
      const goal = borg.self.goals.add({
        description: "Preserve historical audience provenance.",
        priority: 8,
        audienceEntityId: currentScopeId,
        provenance: { kind: "manual" },
        sourceStreamEntryIds: [source.id],
      });
      const entryIndex = borgInternals<{
        deps: { entryIndex: StreamEntryIndexRepository };
      }>(borg).deps.entryIndex;
      const lookupMany = vi.spyOn(entryIndex, "lookupMany");

      const listedCommitments = borg.commitments.list({ activeOnly: true });
      expect(lookupMany).toHaveBeenCalledTimes(1);
      lookupMany.mockClear();
      const listedGoals = borg.self.goals.list({ status: "active" });
      expect(lookupMany).toHaveBeenCalledTimes(1);

      expect(commitment.disclosure_label).toMatchObject({
        origin_audience_entity_ids: [historicalAudienceId],
        private_to_entity_ids: [currentScopeId],
      });
      expect(listedCommitments.find((record) => record.id === commitment.id)).toMatchObject({
        disclosure: expect.stringContaining(`origin_audience=${historicalAudienceId}`),
        disclosure_label: {
          origin_audience_entity_ids: [historicalAudienceId],
          private_to_entity_ids: [currentScopeId],
        },
      });
      expect(goal.disclosure_label).toMatchObject({
        origin_audience_entity_ids: [historicalAudienceId],
        private_to_entity_ids: [currentScopeId],
      });
      expect(listedGoals.find((record) => record.id === goal.id)).toMatchObject({
        disclosure: expect.stringContaining(`origin_audience=${historicalAudienceId}`),
        disclosure_label: {
          origin_audience_entity_ids: [historicalAudienceId],
          private_to_entity_ids: [currentScopeId],
        },
      });
    } finally {
      await borg.close();
    }
  });

  it("exposes read-only semantic edge lookup through the facade", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000);
    const sourceEpisodeId = createEpisodeId();

    const borg = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const first = await borg.semantic.nodes.add({
        kind: "concept",
        label: "First node",
        description: "First node description.",
        sourceEpisodeIds: [sourceEpisodeId],
      });
      const second = await borg.semantic.nodes.add({
        kind: "proposition",
        label: "Second node",
        description: "Second node description.",
        sourceEpisodeIds: [sourceEpisodeId],
      });
      const edge = borg.semantic.edges.add({
        from_node_id: first.id,
        to_node_id: second.id,
        relation: "contradicts",
        confidence: 0.7,
        evidence_episode_ids: [sourceEpisodeId],
        created_at: clock.now(),
        last_verified_at: clock.now(),
      });

      await expect(borg.semantic.edges.get(edge.id)).resolves.toMatchObject(edge);
      await expect(borg.semantic.edges.get(createSemanticEdgeId())).resolves.toBeNull();
    } finally {
      await borg.close();
    }
  });

  it("exports semantic nodes and walks with disclosure labels from private source episodes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000);
    const llm = new FakeLLMClient();

    const borg = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
    });

    try {
      const entry = await borg.stream.append({
        kind: "user_msg",
        content: "Alice privately described the Atlas deployment rollback.",
        audience: "Alice",
      });
      llm.pushResponse({
        text: "",
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "tool_use",
        tool_calls: [
          {
            id: "toolu_private_episode",
            name: EPISODE_TOOL_NAME,
            input: {
              episodes: [
                {
                  title: "Alice private Atlas rollback",
                  narrative: "Alice privately described the Atlas deployment rollback.",
                  source_stream_ids: [entry.id],
                  participants: ["Alice"],
                  location: null,
                  tags: ["Atlas"],
                  confidence: 0.9,
                  significance: 0.8,
                },
              ],
            },
          },
        ],
      });

      await borg.episodic.extract({ sinceTs: entry.timestamp });
      const privateEpisode = (await borg.episodic.listAll()).find(
        (episode) => episode.title === "Alice private Atlas rollback",
      );
      const privateAudienceEntityId = privateEpisode?.audience_entity_id ?? null;

      expect(privateAudienceEntityId).not.toBeNull();

      const first = await borg.semantic.nodes.add({
        kind: "concept",
        label: "Atlas rollback private node",
        description: "Private Atlas rollback source.",
        sourceEpisodeIds: [privateEpisode!.id],
      });
      const second = await borg.semantic.nodes.add({
        kind: "proposition",
        label: "Atlas rollback private support",
        description: "Private Atlas rollback support.",
        sourceEpisodeIds: [privateEpisode!.id],
      });
      borg.semantic.edges.add({
        from_node_id: first.id,
        to_node_id: second.id,
        relation: "supports",
        confidence: 0.8,
        evidence_episode_ids: [privateEpisode!.id],
        created_at: clock.now(),
        last_verified_at: clock.now(),
      });

      const expectedLabel = {
        disclosureClass: "relationship_private",
        privateToEntityIds: [privateAudienceEntityId],
      };
      const fetched = await borg.semantic.nodes.get(first.id);
      const listed = await borg.semantic.nodes.list();
      const searched = await borg.semantic.nodes.search("Atlas rollback private", {
        limit: 5,
      });
      const walked = await borg.semantic.walk(first.id, { depth: 1 });

      expect(fetched?.disclosureLabel).toMatchObject(expectedLabel);
      expect(listed.find((node) => node.id === first.id)?.disclosureLabel).toMatchObject(
        expectedLabel,
      );
      expect(
        searched.find((candidate) => candidate.node.id === first.id)?.node.disclosureLabel,
      ).toMatchObject(expectedLabel);
      expect(walked[0]?.node.disclosureLabel).toMatchObject(expectedLabel);
      expect(walked[0]?.edgePath[0]?.disclosureLabel).toMatchObject(expectedLabel);
    } finally {
      await borg.close();
    }
  });

  it("batches collection disclosure lookups without changing labels", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const episodicRepository =
        borgInternals<DisclosureBatchingInternals>(borg).deps.episodicRepository;
      const audienceEntityId = borg.entities.resolve("Alice");
      const privateEpisode = createEpisodeFixture({
        audience_entity_id: audienceEntityId,
        origin_audience_entity_ids: [audienceEntityId],
        shared: false,
      });
      const publicEpisode = createEpisodeFixture({
        audience_entity_id: null,
        origin_audience_entity_ids: [],
        shared: true,
      });
      const danglingEpisodeId = createEpisodeId();

      await episodicRepository.createEpisode(privateEpisode);
      await episodicRepository.createEpisode(publicEpisode);

      const first = await borg.semantic.nodes.add({
        kind: "concept",
        label: "Batched private source",
        description: "Private source fixture.",
        sourceEpisodeIds: [privateEpisode.id],
      });
      const second = await borg.semantic.nodes.add({
        kind: "proposition",
        label: "Batched public source",
        description: "Public source fixture.",
        sourceEpisodeIds: [publicEpisode.id],
      });
      const dangling = await borg.semantic.nodes.add({
        kind: "proposition",
        label: "Batched dangling source",
        description: "Private and historical tombstone fixture.",
        sourceEpisodeIds: [privateEpisode.id, danglingEpisodeId],
      });
      const edge = borg.semantic.edges.add({
        from_node_id: first.id,
        to_node_id: second.id,
        relation: "supports",
        confidence: 0.8,
        evidence_episode_ids: [privateEpisode.id, danglingEpisodeId],
        created_at: clock.now(),
        last_verified_at: clock.now(),
      });
      const expectedNodeLabels = new Map(
        await Promise.all(
          [first, second, dangling].map(
            async (node) =>
              [
                node.id,
                await resolveMemoryDisclosureLabelForEpisodeIds(
                  episodicRepository,
                  node.source_episode_ids,
                ),
              ] as const,
          ),
        ),
      );
      const expectedEdgeLabel = await resolveMemoryDisclosureLabelForEpisodeIds(
        episodicRepository,
        edge.evidence_episode_ids,
      );
      const getMany = vi.spyOn(episodicRepository, "getMany");

      const listed = await borg.semantic.nodes.list({ includeArchived: true, limit: 100 });
      expect(getMany).toHaveBeenCalledTimes(1);
      for (const node of listed) {
        expect(node.disclosureLabel).toEqual(expectedNodeLabels.get(node.id));
      }
      expect(listed.find((node) => node.id === dangling.id)?.disclosureLabel.disclosureClass).toBe(
        "unknown",
      );

      getMany.mockClear();
      const page = await borg.semantic.nodes.listPage({ includeArchived: true, limit: 100 });
      expect(page.items).toHaveLength(3);
      expect(getMany).toHaveBeenCalledTimes(1);

      getMany.mockClear();
      const searched = await borg.semantic.nodes.search("batched disclosure", { limit: 10 });
      expect(searched.length).toBeGreaterThan(0);
      expect(getMany).toHaveBeenCalledTimes(1);

      getMany.mockClear();
      const edges = await borg.semantic.edges.list({ includeInvalid: true });
      expect(getMany).toHaveBeenCalledTimes(1);
      expect(edges.find((candidate) => candidate.id === edge.id)?.disclosureLabel).toEqual(
        expectedEdgeLabel,
      );

      getMany.mockClear();
      const walked = await borg.semantic.walk(first.id, { depth: 1 });
      expect(walked).toHaveLength(1);
      expect(getMany).toHaveBeenCalledTimes(1);
      expect(walked[0]?.edgePath[0]?.disclosureLabel).toEqual(expectedEdgeLabel);
    } finally {
      await borg.close();
    }
  });

  it("deduplicates hundreds of source ids into one semantic list disclosure lookup", async () => {
    const sourceEpisodeIds = Array.from({ length: 200 }, () => createEpisodeId());
    const nodes = Array.from({ length: 400 }, (_, index) =>
      createSemanticNodeFixture({
        label: `Large disclosure fixture ${index}`,
        source_episode_ids: [
          sourceEpisodeIds[index % sourceEpisodeIds.length]!,
          sourceEpisodeIds[(index * 37 + 1) % sourceEpisodeIds.length]!,
        ],
      }),
    );
    const getMany = vi.fn(async (_episodeIds: readonly ReturnType<typeof createEpisodeId>[]) => []);
    const facades = createBorgFacades({
      actionRepository: {},
      episodicRepository: { getMany },
      semanticNodeRepository: {
        list: async () => nodes,
      },
    } as unknown as BorgDependencies);

    const listed = await facades.semantic.nodes.list({ includeArchived: true, limit: 500 });

    expect(listed).toHaveLength(nodes.length);
    expect(listed.every((node) => node.disclosureLabel.disclosureClass === "unknown")).toBe(true);
    expect(getMany).toHaveBeenCalledTimes(1);
    const lookedUpEpisodeIds = getMany.mock.calls[0]?.[0] ?? [];
    expect(lookedUpEpisodeIds).toHaveLength(sourceEpisodeIds.length);
    expect(new Set(lookedUpEpisodeIds)).toEqual(new Set(sourceEpisodeIds));
  });

  it("injects similarity-forward default attention weights into episodic search", async () => {
    const searchEpisodesForDisclosure = vi.fn(async () => []);
    const facades = createBorgFacades({
      actionRepository: {},
      retrievalPipeline: { searchEpisodesForDisclosure },
      // The weights are deployment-tunable via config.retrieval.attentionWeights,
      // so the facade reads them from config rather than inlining literals; the
      // assertion below pins the shipped defaults.
      config: DEFAULT_CONFIG,
    } as unknown as BorgDependencies);

    await facades.episodic.search("scoring defaults");

    // 2026-08 scoring rebalance: semantic 0.65 (similarity 0.65 / salience
    // 0.35) and heat 0.15. Bumping these re-opens the score-ceiling
    // saturation and greatest-hit top-1 domination the rebalance removed.
    expect(searchEpisodesForDisclosure).toHaveBeenCalledWith(
      "scoring defaults",
      expect.objectContaining({
        attentionWeights: {
          semantic: 0.65,
          goal_relevance: 0,
          value_alignment: 0,
          mood: 0,
          time: 0,
          social: 0,
          entity: 0,
          heat: 0.15,
          suppression_penalty: 0.5,
        },
      }),
    );
  });

  it("lets a deployment override the attention weights from config", async () => {
    // `semantic` is fused against a raw cosine whose spread is a property of
    // the corpus, so a bank whose episodes separate poorly must be able to
    // weight salience higher without a code change. Without this, the two
    // deployments sharing this architecture cannot both be tuned correctly.
    const searchEpisodesForDisclosure = vi.fn(async () => []);
    const facades = createBorgFacades({
      actionRepository: {},
      retrievalPipeline: { searchEpisodesForDisclosure },
      config: {
        ...DEFAULT_CONFIG,
        retrieval: {
          ...DEFAULT_CONFIG.retrieval,
          attentionWeights: {
            ...DEFAULT_CONFIG.retrieval.attentionWeights,
            semantic: 0.35,
            heat: 0.45,
          },
        },
      },
    } as unknown as BorgDependencies);

    await facades.episodic.search("narrow corpus");

    expect(searchEpisodesForDisclosure).toHaveBeenCalledWith(
      "narrow corpus",
      expect.objectContaining({
        attentionWeights: expect.objectContaining({
          semantic: 0.35,
          heat: 0.45,
          // unrelated weights keep their configured values
          suppression_penalty: 0.5,
        }),
      }),
    );
  });

  it("exposes self writes through the identity guard instead of raw repositories", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      dataDir: tempDir,
      clock: new ManualClock(1_000),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const value = borg.self.values.add({
        label: "evidence-backed clarity",
        description: "Prefer evidence-backed changes.",
        priority: 5,
        provenance: {
          kind: "episodes",
          episode_ids: [createEpisodeId(), createEpisodeId(), createEpisodeId()],
        },
      });

      expect(value.state).toBe("established");
      expect("remove" in (borg.self.values as Record<string, unknown>)).toBe(false);
      expect("recordContradiction" in (borg.self.values as Record<string, unknown>)).toBe(false);

      const result = borg.self.values.update(
        value.id,
        {
          description: "Manual overwrite should not bypass review.",
        },
        {
          kind: "manual",
        },
      );

      expect(result).toEqual({
        status: "requires_review",
        current: value,
      });
      expect(borg.self.values.get(value.id)?.description).toBe("Prefer evidence-backed changes.");

      const periodEpisodeId = createEpisodeId();
      const period = borg.self.autobiographical.upsertPeriod({
        label: "2026-Q2",
        start_ts: 1_100,
        narrative: "Episode-backed period.",
        key_episode_ids: [periodEpisodeId],
        themes: ["guard"],
        provenance: {
          kind: "episodes",
          episode_ids: [periodEpisodeId],
        },
      });
      const closeResult = borg.self.autobiographical.closePeriod(period.id, 1_200, {
        kind: "manual",
      });

      expect(closeResult).toEqual({
        status: "requires_review",
        current: period,
      });
      expect(borg.self.autobiographical.getPeriod(period.id)?.end_ts).toBeNull();

      const markerEpisodeId = createEpisodeId();
      const marker = borg.self.growthMarkers.add({
        ts: 1_150,
        category: "understanding",
        what_changed: "Facade growth marker writes are audited.",
        evidence_episode_ids: [markerEpisodeId],
        confidence: 0.7,
        source_process: "manual",
        provenance: {
          kind: "episodes",
          episode_ids: [markerEpisodeId],
        },
      });

      expect(
        borg.identity.listEvents({
          recordType: "growth_marker",
          recordId: marker.id,
        })[0]?.action,
      ).toBe("create");

      const questionEpisodeId = createEpisodeId();
      const question = borg.self.openQuestions.add({
        question: "Does the facade guard open question state changes?",
        urgency: 0.5,
        related_episode_ids: [questionEpisodeId],
        provenance: {
          kind: "episodes",
          episode_ids: [questionEpisodeId],
        },
        source: "reflection",
      });
      const bumpResult = borg.self.openQuestions.bumpUrgency(question.id, 0.2, {
        kind: "manual",
      });

      expect(bumpResult).toEqual({
        status: "requires_review",
        current: question,
      });
      expect(borg.self.openQuestions.list({ status: "open" })[0]?.urgency).toBe(0.5);
    } finally {
      await borg.close();
    }
  });

  it("lets facade upsertPeriod update an existing autobiographical period", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      dataDir: tempDir,
      clock: new ManualClock(1_000),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const episodeId = createEpisodeId();
      const period = borg.self.autobiographical.upsertPeriod({
        label: "2026-Q2",
        start_ts: 1_100,
        narrative: "Initial period narrative.",
        key_episode_ids: [episodeId],
        themes: ["identity"],
        provenance: {
          kind: "episodes",
          episode_ids: [episodeId],
        },
      });
      const result = borg.self.autobiographical.upsertPeriod({
        id: period.id,
        label: "2026-Q2 revised",
        start_ts: 1_100,
        end_ts: 1_900,
        narrative: "Updated period narrative.",
        key_episode_ids: [episodeId],
        themes: ["identity", "revision"],
        provenance: {
          kind: "episodes",
          episode_ids: [episodeId],
        },
      });

      expect(result).toEqual({
        status: "applied",
        record: expect.objectContaining({
          id: period.id,
          label: "2026-Q2 revised",
          end_ts: 1_900,
          narrative: "Updated period narrative.",
          themes: ["identity", "revision"],
        }),
      });
      expect(borg.self.autobiographical.getPeriod(period.id)).toMatchObject({
        label: "2026-Q2 revised",
        end_ts: 1_900,
      });
    } finally {
      await borg.close();
    }
  });

  it("does not bootstrap an autobiographical period before evidence", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(Date.UTC(2026, 3, 22));

    const borg = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      expect(borg.self.autobiographical.currentPeriod()).toBeNull();
      expect(borg.self.autobiographical.listPeriods({ limit: 10 })).toHaveLength(0);
    } finally {
      await borg.close();
    }

    const reopened = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      expect(reopened.self.autobiographical.listPeriods({ limit: 10 })).toHaveLength(0);
    } finally {
      await reopened.close();
    }
  });

  it("closes opened resources if a later Borg.open step fails", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const sqliteCloseSpy = vi.spyOn(SqliteDatabase.prototype, "close");
    const lanceCloseSpy = vi.spyOn(LanceDbStore.prototype, "close");
    const failure = new Error("embedding init failed");
    const openOptions = {
      dataDir: tempDir,
    } as {
      dataDir: string;
      embeddingClient?: ScriptedEmbeddingClient;
    };

    Object.defineProperty(openOptions, "embeddingClient", {
      get() {
        throw failure;
      },
    });

    await expect(Borg.open(openOptions)).rejects.toThrow(failure);
    expect(sqliteCloseSpy).toHaveBeenCalledTimes(1);
    expect(lanceCloseSpy).toHaveBeenCalledTimes(1);
  });
});
