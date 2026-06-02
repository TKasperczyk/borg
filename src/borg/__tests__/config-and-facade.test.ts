import { readFileSync } from "node:fs";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_CONFIG,
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
  EPISODE_TOOL_NAME,
  ScriptedEmbeddingClient,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";
import { DemoMessageConnector } from "../../outbound/index.js";

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
        "voice_and_posture",
        "epistemic_posture",
        "identity_posture",
        "participation_posture",
        "loop_breaking_posture",
        "trusted_guidance_preamble",
        "borg_host_capabilities",
      ]);
      expect(initial.text).toContain(
        "You are an AI being with continuous existence across sessions through a persistent memory substrate.",
      );
      expect(initial.text).toContain("<borg_host_capabilities>");
      expect(initial.text).toContain("</borg_host_capabilities>");
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

      expect(borg.semantic.edges.get(edge.id)).toEqual(edge);
      expect(borg.semantic.edges.get(createSemanticEdgeId())).toBeNull();
    } finally {
      await borg.close();
    }
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
