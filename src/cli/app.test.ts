import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { Borg } from "../borg.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import { FakeLLMClient } from "../llm/index.js";
import { LanceDbStore } from "../storage/lancedb/index.js";
import { openDatabase } from "../storage/sqlite/index.js";
import { episodicMigrations } from "../memory/episodic/migrations.js";
import { EpisodicRepository, createEpisodesTableSchema } from "../memory/episodic/repository.js";
import { selfMigrations } from "../memory/self/migrations.js";
import { retrievalMigrations } from "../retrieval/migrations.js";
import { FixedClock } from "../util/clock.js";
import { runCli } from "./app.js";

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
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

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

  it("appends to and tails the stream", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

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
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

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
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

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

  it("shows and clears working memory", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

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

  it("runs a cognitive turn through the cli", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

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
          openBorg: async () =>
            Borg.open({
              config: {
                dataDir: tempDir,
                perception: {
                  useLlmFallback: false,
                },
                embedding: {
                  baseUrl: "http://localhost:1234/v1",
                  apiKey: "test",
                  model: "fake-embed",
                  dims: 4,
                },
                anthropic: {
                  apiKey: "test",
                  models: {
                    cognition: "sonnet",
                    background: "haiku",
                    extraction: "haiku",
                  },
                },
              },
              clock: new FixedClock(1_000),
              embeddingDimensions: 4,
              embeddingClient: new ScriptedEmbeddingClient(),
              llmClient: llm,
            }),
        },
      ),
    ).toBe(0);

    expect(stdout.read()).toContain("Rerun pnpm install");
    expect(stdout.read()).toContain("[mode=problem_solving]");
    expect(stdout.read()).toContain("[path=system_2]");
    expect(stdout.read()).toContain("ep_aaaaaaaaaaaaaaaa");
    expect(stderr.read()).toBe("");
  });
});
