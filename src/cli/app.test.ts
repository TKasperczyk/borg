import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { Borg } from "../borg.js";
import { DEFAULT_CONFIG } from "../config/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import { FakeLLMClient } from "../llm/index.js";
import { LanceDbStore } from "../storage/lancedb/index.js";
import { openDatabase } from "../storage/sqlite/index.js";
import { episodicMigrations } from "../memory/episodic/migrations.js";
import { EpisodicRepository, createEpisodesTableSchema } from "../memory/episodic/repository.js";
import { selfMigrations } from "../memory/self/migrations.js";
import { retrievalMigrations } from "../retrieval/migrations.js";
import { FixedClock } from "../util/clock.js";
import { readJsonFile, writeJsonFileAtomic } from "../util/atomic-write.js";
import { runCli } from "./app.js";

const CONSOLIDATION_TOOL_NAME = "EmitConsolidation";
const EPISODE_TOOL_NAME = "EmitEpisodeCandidates";

class ScriptedEmbeddingClient implements EmbeddingClient {
  async embed(text: string): Promise<Float32Array> {
    return this.vector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.vector(text));
  }

  private vector(text: string): Float32Array {
    if (/atlas|pnpm|deploy/i.test(text)) {
      return Float32Array.from([1, 0, 0, 0]);
    }

    return Float32Array.from([0, 1, 0, 0]);
  }
}

function createOutputBuffer() {
  let value = "";

  return {
    stream: {
      write(chunk: string) {
        value += chunk;
        return true;
      },
    },
    read() {
      return value;
    },
  };
}

function createCliTempDir(tempDirs: string[]): string {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
  tempDirs.push(tempDir);
  writeJsonFileAtomic(join(tempDir, "config.json"), {
    embedding: {
      dims: 4,
    },
  });
  return tempDir;
}

function openTestBorg(tempDir: string, llm = new FakeLLMClient()) {
  return Borg.open({
    config: {
      dataDir: tempDir,
      perception: {
        useLlmFallback: false,
          modeWhenLlmAbsent: "problem_solving",
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
          cognition: "sonnet",
          background: "haiku",
          extraction: "haiku",
        },
      },
      offline: DEFAULT_CONFIG.offline,
    },
    clock: new FixedClock(1_000),
    embeddingDimensions: 4,
    embeddingClient: new ScriptedEmbeddingClient(),
    llmClient: llm,
  });
}

describe("cli", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("prints the version", async () => {
    const stdout = createOutputBuffer();
    const stderr = createOutputBuffer();

    const exitCode = await runCli(["node", "borg", "version"], {
      stdout: stdout.stream,
      stderr: stderr.stream,
    });

    expect(exitCode).toBe(0);
    expect(stdout.read()).toContain("borg 0.1.0");
    expect(stderr.read()).toBe("");
  });

  it("shows redacted config", async () => {
    const tempDir = createCliTempDir(tempDirs);

    const stdout = createOutputBuffer();
    const stderr = createOutputBuffer();

    const exitCode = await runCli(["node", "borg", "config", "show"], {
      stdout: stdout.stream,
      stderr: stderr.stream,
      dataDir: tempDir,
      env: {
        ANTHROPIC_API_KEY: "secret",
      },
    });

    expect(exitCode).toBe(0);
    expect(JSON.parse(stdout.read())).toMatchObject({
      dataDir: tempDir,
      anthropic: {
        apiKey: "[REDACTED]",
      },
    });
    expect(stderr.read()).toBe("");
  });

  it("reports oauth auth status from a mocked shared credentials file", async () => {
    const tempDir = createCliTempDir(tempDirs);
    const credentialsPath = join(tempDir, "claude-credentials.json");

    writeJsonFileAtomic(credentialsPath, {
      claudeAiOauth: {
        accessToken: "oauth-access-token",
        refreshToken: "oauth-refresh-token",
        expiresAt: Date.now() + 4 * 60 * 60_000,
      },
    });

    const stdout = createOutputBuffer();
    const stderr = createOutputBuffer();

    const exitCode = await runCli(["node", "borg", "auth", "status"], {
      stdout: stdout.stream,
      stderr: stderr.stream,
      dataDir: tempDir,
      env: {
        BORG_CLAUDE_CREDENTIALS_PATH: credentialsPath,
      },
    });

    expect(exitCode).toBe(0);
    expect(stdout.read()).toContain("oauth via");
    expect(stdout.read()).toContain("expires in");
    expect(stderr.read()).toBe("");
  });

  it("appends to and tails the stream", async () => {
    const tempDir = createCliTempDir(tempDirs);

    const appendOut = createOutputBuffer();
    const appendErr = createOutputBuffer();

    expect(
      await runCli(
        ["node", "borg", "stream", "append", "--kind", "user_msg", "--content", "hello"],
        {
          stdout: appendOut.stream,
          stderr: appendErr.stream,
          dataDir: tempDir,
        },
      ),
    ).toBe(0);

    const tailOut = createOutputBuffer();
    const tailErr = createOutputBuffer();

    expect(
      await runCli(["node", "borg", "stream", "tail", "--n", "1"], {
        stdout: tailOut.stream,
        stderr: tailErr.stream,
        dataDir: tempDir,
      }),
    ).toBe(0);

    expect(JSON.parse(tailOut.read())).toMatchObject({
      kind: "user_msg",
      content: "hello",
    });
    expect(appendErr.read()).toBe("");
    expect(tailErr.read()).toBe("");
  });

  it("surfaces invalid session ids as clean cli errors", async () => {
    const tempDir = createCliTempDir(tempDirs);

    const stdout = createOutputBuffer();
    const stderr = createOutputBuffer();

    const exitCode = await runCli(
      ["node", "borg", "stream", "tail", "--session", "not-a-session"],
      {
        stdout: stdout.stream,
        stderr: stderr.stream,
        dataDir: tempDir,
      },
    );

    expect(exitCode).toBe(1);
    expect(stderr.read()).toContain("Invalid session id");
  });

  it("shows an episode and manages self-band commands", async () => {
    const tempDir = createCliTempDir(tempDirs);

    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
    });

    await repo.insert({
      id: "ep_aaaaaaaaaaaaaaaa" as never,
      title: "Planning sync",
      narrative: "A short planning episode.",
      participants: ["team"],
      location: null,
      start_time: 1,
      end_time: 2,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.8,
      tags: ["planning"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 1,
      updated_at: 1,
    });
    db.close();
    await store.close();

    const showOut = createOutputBuffer();
    const showErr = createOutputBuffer();
    const showExit = await runCli(["node", "borg", "episode", "show", "ep_aaaaaaaaaaaaaaaa"], {
      stdout: showOut.stream,
      stderr: showErr.stream,
      dataDir: tempDir,
    });

    expect(showExit).toBe(0);
    expect(JSON.parse(showOut.read())).toMatchObject({
      episode: {
        id: "ep_aaaaaaaaaaaaaaaa",
      },
    });

    const goalAddOut = createOutputBuffer();
    const goalExit = await runCli(
      ["node", "borg", "goal", "add", "--description", "Ship Sprint 2", "--priority", "9"],
      {
        stdout: goalAddOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
      },
    );
    const goal = JSON.parse(goalAddOut.read()) as {
      id: string;
    };

    expect(goalExit).toBe(0);
    expect(goal.id).toMatch(/^goal_/);

    const goalProgressOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "goal", "progress", goal.id, "--note", "Almost there"], {
        stdout: goalProgressOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
      }),
    ).toBe(0);
    expect(JSON.parse(goalProgressOut.read())).toMatchObject({
      id: goal.id,
      progress_notes: "Almost there",
    });

    const valueAddOut = createOutputBuffer();
    expect(
      await runCli(
        [
          "node",
          "borg",
          "value",
          "add",
          "--label",
          "clarity",
          "--description",
          "Prefer explicit state.",
          "--priority",
          "5",
        ],
        {
          stdout: valueAddOut.stream,
          stderr: createOutputBuffer().stream,
          dataDir: tempDir,
        },
      ),
    ).toBe(0);
    const value = JSON.parse(valueAddOut.read()) as {
      id: string;
    };

    const traitOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "trait", "show"], {
        stdout: traitOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
      }),
    ).toBe(0);
    expect(JSON.parse(traitOut.read())).toEqual([]);

    const affirmOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "value", "affirm", value.id], {
        stdout: affirmOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
      }),
    ).toBe(0);
    expect(JSON.parse(affirmOut.read())).toMatchObject({
      id: value.id,
      affirmed: true,
    });
  });

  it("manages autobiographical periods, growth markers, and open questions", async () => {
    const tempDir = createCliTempDir(tempDirs);

    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
    });
    await repo.insert({
      id: "ep_cccccccccccccccc" as never,
      title: "Atlas note",
      narrative: "A short Atlas episode.",
      participants: ["team"],
      location: null,
      start_time: 1,
      end_time: 2,
      source_stream_ids: ["strm_cccccccccccccccc" as never],
      significance: 0.8,
      tags: ["atlas"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 1,
      updated_at: 1,
    });
    db.close();
    await store.close();

    const periodOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "period", "open", "2026-Q2"], {
        stdout: periodOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
      }),
    ).toBe(0);
    const period = JSON.parse(periodOut.read()) as { id: string };

    const growthOut = createOutputBuffer();
    expect(
      await runCli(
        [
          "node",
          "borg",
          "growth",
          "add",
          "understanding",
          "Learned Atlas rollback order",
          "--episode",
          "ep_cccccccccccccccc",
        ],
        {
          stdout: growthOut.stream,
          stderr: createOutputBuffer().stream,
          dataDir: tempDir,
        },
      ),
    ).toBe(0);
    expect(JSON.parse(growthOut.read())).toMatchObject({
      category: "understanding",
    });

    const questionOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "question", "add", "Why does Atlas fail?"], {
        stdout: questionOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
      }),
    ).toBe(0);
    const question = JSON.parse(questionOut.read()) as { id: string };

    const bumpOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "question", "bump", question.id, "0.2"], {
        stdout: bumpOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
      }),
    ).toBe(0);
    expect(JSON.parse(bumpOut.read())).toMatchObject({
      id: question.id,
    });

    const showPeriodOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "period", "show", period.id], {
        stdout: showPeriodOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
      }),
    ).toBe(0);
    expect(JSON.parse(showPeriodOut.read())).toMatchObject({
      id: period.id,
    });
  });

  it("supports ruminate and narrate dream commands", async () => {
    const tempDir = createCliTempDir(tempDirs);

    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
    });
    await repo.insert({
      id: "ep_aaaaaaaaaaaaaaaa" as never,
      title: "Atlas rehearsal",
      narrative: "Atlas stabilized after rehearsal.",
      participants: ["team"],
      location: null,
      start_time: 1,
      end_time: 2,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.8,
      tags: ["atlas"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 1,
      updated_at: 1,
    });
    await repo.insert({
      id: "ep_bbbbbbbbbbbbbbbb" as never,
      title: "Atlas follow-up",
      narrative: "Atlas debugging became clearer.",
      participants: ["team"],
      location: null,
      start_time: 2,
      end_time: 3,
      source_stream_ids: ["strm_bbbbbbbbbbbbbbbb" as never],
      significance: 0.8,
      tags: ["atlas"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 2,
      updated_at: 2,
    });
    db.close();
    await store.close();

    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 30,
          output_tokens: 20,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: "EmitRuminatorDecisions",
              input: {
                resolution_note: "Atlas recovered after rehearsal.",
                growth_marker: null,
              },
            },
          ],
        },
        {
          text: "",
          input_tokens: 30,
          output_tokens: 20,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_2",
              name: "EmitSelfNarratorObservations",
              input: {
                observation: {
                  category: "understanding",
                  what_changed: "Atlas debugging became clearer.",
                  before_description: "The failure path was fuzzy.",
                  after_description: "The failure path is clearer now.",
                  confidence: 0.8,
                  evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa", "ep_bbbbbbbbbbbbbbbb"],
                },
              },
            },
          ],
        },
      ],
    });
    const seedBorg = await openTestBorg(tempDir, llm);
    seedBorg.self.openQuestions.add({
      question: "Why does Atlas fail?",
      urgency: 0.8,
      source: "user",
    });
    await seedBorg.close();

    const ruminateOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "dream", "ruminate", "--dry-run", "--max-questions", "1"], {
        stdout: ruminateOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
        openBorg: () => openTestBorg(tempDir, llm),
      }),
    ).toBe(0);
    expect(JSON.parse(ruminateOut.read())).toMatchObject({
      dryRun: true,
    });

    const narrateOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "dream", "narrate", "--dry-run"], {
        stdout: narrateOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
        openBorg: () => openTestBorg(tempDir, llm),
      }),
    ).toBe(0);
    expect(JSON.parse(narrateOut.read())).toMatchObject({
      dryRun: true,
    });
  });

  it("shows and clears working memory", async () => {
    const tempDir = createCliTempDir(tempDirs);

    const showOut = createOutputBuffer();
    const showErr = createOutputBuffer();

    expect(
      await runCli(["node", "borg", "workmem", "show"], {
        stdout: showOut.stream,
        stderr: showErr.stream,
        dataDir: tempDir,
      }),
    ).toBe(0);
    expect(JSON.parse(showOut.read())).toMatchObject({
      session_id: "default",
      turn_counter: 0,
    });
    expect(showErr.read()).toBe("");

    const clearOut = createOutputBuffer();
    const clearErr = createOutputBuffer();

    expect(
      await runCli(["node", "borg", "workmem", "clear"], {
        stdout: clearOut.stream,
        stderr: clearErr.stream,
        dataDir: tempDir,
      }),
    ).toBe(0);
    expect(JSON.parse(clearOut.read())).toEqual({
      session: "default",
      cleared: true,
    });
    expect(clearErr.read()).toBe("");
  });

  it("runs dream maintenance and exposes audit commands", async () => {
    const tempDir = createCliTempDir(tempDirs);

    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
      clock: new FixedClock(100 * 24 * 60 * 60 * 1_000),
    });

    await repo.insert({
      id: "ep_dddddddddddddddd" as never,
      title: "Archive candidate",
      narrative: "A stale memory with low heat.",
      participants: ["team"],
      location: null,
      start_time: 1,
      end_time: 2,
      source_stream_ids: ["strm_dddddddddddddddd" as never],
      significance: 0.2,
      tags: ["quiet"],
      confidence: 0.8,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([0, 1, 0, 0]),
      created_at: 1,
      updated_at: 1,
    });
    db.close();
    await store.close();

    const dreamOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "dream", "curate"], {
        stdout: dreamOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
      }),
    ).toBe(0);
    const dreamResult = JSON.parse(dreamOut.read()) as {
      run_id: string;
    };
    expect(dreamResult.run_id).toMatch(/^run_/);

    const auditListOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "audit", "list", "--process", "curator"], {
        stdout: auditListOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
      }),
    ).toBe(0);
    const audits = JSON.parse(auditListOut.read()) as Array<{ id: number }>;
    expect(audits.length).toBeGreaterThan(0);

    const auditRevertOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "audit", "revert", String(audits[0]!.id)], {
        stdout: auditRevertOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
      }),
    ).toBe(0);
    expect(JSON.parse(auditRevertOut.read())).toMatchObject({
      id: audits[0]!.id,
    });
  });

  it("writes a dream plan file and applies it without another llm call", async () => {
    const tempDir = createCliTempDir(tempDirs);
    const planPath = join(tempDir, "maintenance-plan.json");

    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
      clock: new FixedClock(10_000),
    });

    await repo.insert({
      id: "ep_aaaaaaaaaaaaaaaa" as never,
      title: "Deploy prep one",
      narrative: "Atlas deploy prep documented the rollback steps.",
      participants: ["team"],
      location: null,
      start_time: 1,
      end_time: 2,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.8,
      tags: ["atlas", "deploy"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 1,
      updated_at: 1,
    });
    await repo.insert({
      id: "ep_bbbbbbbbbbbbbbbb" as never,
      title: "Deploy prep two",
      narrative: "The Atlas deploy checklist repeated the rollback prep.",
      participants: ["team"],
      location: null,
      start_time: 3,
      end_time: 4,
      source_stream_ids: ["strm_bbbbbbbbbbbbbbbb" as never],
      significance: 0.8,
      tags: ["atlas", "deploy"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([0.99, 0, 0, 0]),
      created_at: 3,
      updated_at: 3,
    });
    db.close();
    await store.close();

    const planningLlm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 15,
          output_tokens: 10,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: CONSOLIDATION_TOOL_NAME,
              input: {
                title: "Merged deploy prep",
                narrative: "The deploy prep notes were merged into one grounded summary.",
              },
            },
          ],
        },
      ],
    });
    const dryRunOut = createOutputBuffer();

    expect(
      await runCli(["node", "borg", "dream", "consolidate", "--dry-run", "--output", planPath], {
        stdout: dryRunOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
        openBorg: async () => openTestBorg(tempDir, planningLlm),
      }),
    ).toBe(0);
    expect(planningLlm.requests).toHaveLength(1);
    expect(readJsonFile(planPath)).toMatchObject({
      kind: "borg_maintenance_plan",
      processes: [expect.objectContaining({ process: "consolidator" })],
    });

    const applyOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "dream", "apply", "--plan", planPath], {
        stdout: applyOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
        openBorg: async () => openTestBorg(tempDir, new FakeLLMClient()),
      }),
    ).toBe(0);
    expect(JSON.parse(applyOut.read())).toMatchObject({
      dryRun: false,
    });

    const borg = await openTestBorg(tempDir, new FakeLLMClient());

    try {
      const episodes = await borg.episodic.list({
        limit: 10,
      });
      expect(episodes.items.some((episode) => episode.title === "Merged deploy prep")).toBe(true);
    } finally {
      await borg.close();
    }
  });

  it("rejects invalid audit ids on revert", async () => {
    const tempDir = createCliTempDir(tempDirs);

    const stdout = createOutputBuffer();
    const stderr = createOutputBuffer();
    const exitCode = await runCli(["node", "borg", "audit", "revert", "12junk"], {
      stdout: stdout.stream,
      stderr: stderr.stream,
      dataDir: tempDir,
    });

    expect(exitCode).toBe(1);
    expect(stderr.read()).toContain("Invalid audit id: 12junk");
  });

  it("runs a cognitive turn through the cli", async () => {
    const tempDir = createCliTempDir(tempDirs);

    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
      clock: new FixedClock(1_000),
    });

    await repo.insert({
      id: "ep_aaaaaaaaaaaaaaaa" as never,
      title: "Atlas deploy fix",
      narrative: "Atlas deploys recovered after rerunning pnpm install.",
      participants: ["team"],
      location: null,
      start_time: 1,
      end_time: 2,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.8,
      tags: ["atlas", "deploy"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 1,
      updated_at: 1,
    });
    db.close();
    await store.close();

    const stdout = createOutputBuffer();
    const stderr = createOutputBuffer();
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Check the Atlas deploy assumptions before answering.",
          input_tokens: 10,
          output_tokens: 5,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "Rerun pnpm install, then redeploy Atlas. Next step: redeploy Atlas.",
          input_tokens: 20,
          output_tokens: 10,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });

    expect(
      await runCli(
        ["node", "borg", "turn", "Atlas deploy has a pnpm failure", "--stakes", "high"],
        {
          stdout: stdout.stream,
          stderr: stderr.stream,
          dataDir: tempDir,
          openBorg: async () => openTestBorg(tempDir, llm),
        },
      ),
    ).toBe(0);

    expect(stdout.read()).toContain("Rerun pnpm install");
    expect(stdout.read()).toContain("[mode=problem_solving]");
    expect(stdout.read()).toContain("[path=system_2]");
    expect(stdout.read()).toContain("ep_aaaaaaaaaaaaaaaa");
    expect(stderr.read()).toBe("");
  });

  it("manages semantic, commitment, and review commands", async () => {
    const tempDir = createCliTempDir(tempDirs);
    const cliOptions = {
      dataDir: tempDir,
      openBorg: async () => openTestBorg(tempDir),
    };

    const addAtlasOut = createOutputBuffer();
    expect(
      await runCli(
        [
          "node",
          "borg",
          "semantic",
          "node",
          "add",
          "--kind",
          "entity",
          "--label",
          "Atlas",
          "--description",
          "Atlas service",
          "--source-episodes",
          "ep_aaaaaaaaaaaaaaaa",
        ],
        {
          stdout: addAtlasOut.stream,
          stderr: createOutputBuffer().stream,
          ...cliOptions,
        },
      ),
    ).toBe(0);
    const atlasNode = JSON.parse(addAtlasOut.read()) as {
      id: string;
    };

    const addRollbackOut = createOutputBuffer();
    expect(
      await runCli(
        [
          "node",
          "borg",
          "semantic",
          "node",
          "add",
          "--kind",
          "concept",
          "--label",
          "Rollback",
          "--description",
          "Rollback plan",
          "--source-episodes",
          "ep_aaaaaaaaaaaaaaaa",
        ],
        {
          stdout: addRollbackOut.stream,
          stderr: createOutputBuffer().stream,
          ...cliOptions,
        },
      ),
    ).toBe(0);
    const rollbackNode = JSON.parse(addRollbackOut.read()) as {
      id: string;
    };

    const showNodeOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "semantic", "node", "show", atlasNode.id], {
        stdout: showNodeOut.stream,
        stderr: createOutputBuffer().stream,
        ...cliOptions,
      }),
    ).toBe(0);
    expect(JSON.parse(showNodeOut.read())).toMatchObject({
      id: atlasNode.id,
      label: "Atlas",
    });

    const searchNodeOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "semantic", "node", "search", "Atlas"], {
        stdout: searchNodeOut.stream,
        stderr: createOutputBuffer().stream,
        ...cliOptions,
      }),
    ).toBe(0);
    expect(JSON.parse(searchNodeOut.read())).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          node: expect.objectContaining({
            id: atlasNode.id,
          }),
        }),
      ]),
    );

    const listNodesOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "semantic", "node", "list"], {
        stdout: listNodesOut.stream,
        stderr: createOutputBuffer().stream,
        ...cliOptions,
      }),
    ).toBe(0);
    expect(JSON.parse(listNodesOut.read())).toHaveLength(2);

    const addSupportOut = createOutputBuffer();
    expect(
      await runCli(
        [
          "node",
          "borg",
          "semantic",
          "edge",
          "add",
          "--from",
          atlasNode.id,
          "--to",
          rollbackNode.id,
          "--relation",
          "supports",
          "--evidence-episodes",
          "ep_aaaaaaaaaaaaaaaa",
        ],
        {
          stdout: addSupportOut.stream,
          stderr: createOutputBuffer().stream,
          ...cliOptions,
        },
      ),
    ).toBe(0);
    expect(JSON.parse(addSupportOut.read())).toMatchObject({
      relation: "supports",
    });

    const addContradictionOut = createOutputBuffer();
    expect(
      await runCli(
        [
          "node",
          "borg",
          "semantic",
          "edge",
          "add",
          "--from",
          atlasNode.id,
          "--to",
          rollbackNode.id,
          "--relation",
          "contradicts",
          "--evidence-episodes",
          "ep_aaaaaaaaaaaaaaaa",
        ],
        {
          stdout: addContradictionOut.stream,
          stderr: createOutputBuffer().stream,
          ...cliOptions,
        },
      ),
    ).toBe(0);
    expect(JSON.parse(addContradictionOut.read())).toMatchObject({
      relation: "contradicts",
    });

    const listEdgesOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "semantic", "edge", "list", "--from", atlasNode.id], {
        stdout: listEdgesOut.stream,
        stderr: createOutputBuffer().stream,
        ...cliOptions,
      }),
    ).toBe(0);
    expect(JSON.parse(listEdgesOut.read())).toHaveLength(2);

    const walkOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "semantic", "walk", atlasNode.id, "--depth", "2"], {
        stdout: walkOut.stream,
        stderr: createOutputBuffer().stream,
        ...cliOptions,
      }),
    ).toBe(0);
    expect(JSON.parse(walkOut.read())).toEqual([
      expect.objectContaining({
        node: expect.objectContaining({
          id: rollbackNode.id,
        }),
      }),
    ]);

    const addCommitmentOut = createOutputBuffer();
    expect(
      await runCli(
        [
          "node",
          "borg",
          "commitment",
          "add",
          "--type",
          "boundary",
          "--directive",
          "Do not discuss Atlas with Sam",
          "--priority",
          "10",
          "--audience",
          "Sam",
          "--about",
          "Atlas",
        ],
        {
          stdout: addCommitmentOut.stream,
          stderr: createOutputBuffer().stream,
          ...cliOptions,
        },
      ),
    ).toBe(0);
    const commitment = JSON.parse(addCommitmentOut.read()) as {
      id: string;
    };

    const listCommitmentsOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "commitment", "list", "--audience", "Sam"], {
        stdout: listCommitmentsOut.stream,
        stderr: createOutputBuffer().stream,
        ...cliOptions,
      }),
    ).toBe(0);
    expect(JSON.parse(listCommitmentsOut.read())).toEqual([
      expect.objectContaining({
        id: commitment.id,
      }),
    ]);

    const listReviewsOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "review", "list", "--kind", "contradiction"], {
        stdout: listReviewsOut.stream,
        stderr: createOutputBuffer().stream,
        ...cliOptions,
      }),
    ).toBe(0);
    const reviews = JSON.parse(listReviewsOut.read()) as Array<{ id: number }>;
    expect(reviews).toHaveLength(1);

    const resolveReviewOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "review", "resolve", String(reviews[0]!.id), "invalidate"], {
        stdout: resolveReviewOut.stream,
        stderr: createOutputBuffer().stream,
        ...cliOptions,
      }),
    ).toBe(0);
    expect(JSON.parse(resolveReviewOut.read())).toMatchObject({
      id: reviews[0]!.id,
      resolution: "invalidate",
    });

    const revokeCommitmentOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "commitment", "revoke", commitment.id], {
        stdout: revokeCommitmentOut.stream,
        stderr: createOutputBuffer().stream,
        ...cliOptions,
      }),
    ).toBe(0);
    expect(JSON.parse(revokeCommitmentOut.read())).toMatchObject({
      id: commitment.id,
    });

    const listCommitmentsAfterOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "commitment", "list", "--audience", "Sam"], {
        stdout: listCommitmentsAfterOut.stream,
        stderr: createOutputBuffer().stream,
        ...cliOptions,
      }),
    ).toBe(0);
    expect(JSON.parse(listCommitmentsAfterOut.read())).toEqual([]);
  });

  it("manages skill, mood, and social commands", async () => {
    const tempDir = createCliTempDir(tempDirs);
    const llm = new FakeLLMClient();
    const borg = await openTestBorg(tempDir, llm);
    const entry = await borg.stream.append({
      kind: "user_msg",
      content: "Rust lifetimes are frustrating.",
    });
    llm.pushResponse({
      text: "",
      input_tokens: 10,
      output_tokens: 10,
      stop_reason: "tool_use",
      tool_calls: [
        {
          id: "toolu_1",
          name: EPISODE_TOOL_NAME,
          input: {
            episodes: [
              {
                title: "Rust lifetime frustration",
                narrative: "The user was frustrated by Rust lifetimes.",
                source_stream_ids: [entry.id],
                participants: ["user"],
                location: null,
                tags: ["rust", "debugging"],
                confidence: 0.8,
                significance: 0.7,
              },
            ],
          },
        },
      ],
    });
    await borg.episodic.extract();
    const [storedEpisode] = (await borg.episodic.list()).items;
    borg.mood.update("default", {
      valence: -0.4,
      arousal: 0.5,
      reason: "seeded",
    });
    await borg.close();

    const addSkillOut = createOutputBuffer();
    expect(
      await runCli(
        [
          "node",
          "borg",
          "skill",
          "add",
          "--applies-when",
          "Rust lifetime debugging",
          "--approach",
          "Shrink borrow scopes.",
          "--episode",
          storedEpisode!.id,
        ],
        {
          stdout: addSkillOut.stream,
          stderr: createOutputBuffer().stream,
          dataDir: tempDir,
          openBorg: async () => openTestBorg(tempDir),
        },
      ),
    ).toBe(0);
    const skill = JSON.parse(addSkillOut.read()) as { id: string };

    const showSkillOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "skill", "show", skill.id], {
        stdout: showSkillOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
        openBorg: async () => openTestBorg(tempDir),
      }),
    ).toBe(0);
    expect(JSON.parse(showSkillOut.read())).toMatchObject({
      id: skill.id,
    });

    const moodOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "mood", "current", "--session", "default"], {
        stdout: moodOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
        openBorg: async () => openTestBorg(tempDir),
      }),
    ).toBe(0);
    expect(JSON.parse(moodOut.read())).toMatchObject({
      session_id: "default",
    });

    const socialUpsertOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "social", "upsert", "Sam"], {
        stdout: socialUpsertOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
        openBorg: async () => openTestBorg(tempDir),
      }),
    ).toBe(0);

    const socialTrustOut = createOutputBuffer();
    expect(
      await runCli(["node", "borg", "social", "adjust-trust", "Sam", "0.2"], {
        stdout: socialTrustOut.stream,
        stderr: createOutputBuffer().stream,
        dataDir: tempDir,
        openBorg: async () => openTestBorg(tempDir),
      }),
    ).toBe(0);
    expect(JSON.parse(socialTrustOut.read()).trust).toBeGreaterThan(0.5);
  });
});
