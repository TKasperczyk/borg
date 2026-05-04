import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  FakeLLMClient,
  type Borg,
  type GenerationSuppressionReason,
  type SessionId,
} from "../src/index.js";
import { MaintenanceScheduler, type MaintenanceTickResult } from "../src/offline/scheduler.js";
import { BorgTransport, type ChatWithBorgResult } from "../assessor/borg-transport.js";
import { runSimulation } from "./runner.js";
import type { PersonaSession } from "./persona.js";
import { tomPersona } from "./personas/tom.js";

const tempDirs: string[] = [];

function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "borg-simulator-runner-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(() => {
  vi.restoreAllMocks();

  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop() as string, { recursive: true, force: true });
  }
});

function spyMaintenanceTick() {
  return vi
    .spyOn(MaintenanceScheduler.prototype, "tick")
    .mockImplementation(async (cadence): Promise<MaintenanceTickResult> => {
      return {
        status: "ok",
        cadence,
        ts: Date.now(),
        processes: [],
        result: null,
      };
    });
}

function fakeSimulatorBorg(): Borg {
  return {
    mood: {
      current: () => ({ valence: 0, arousal: 0 }),
    },
    episodic: {
      list: async () => ({ items: [] }),
    },
    semantic: {
      nodes: {
        list: async () => [],
      },
      edges: {
        list: () => [],
      },
    },
    self: {
      openQuestions: {
        list: () => [],
      },
      goals: {
        list: () => [],
      },
    },
    stream: {
      tail: () => [],
    },
    maintenance: {
      scheduler: {
        tick: async (cadence: string) => ({
          status: "ok",
          cadence,
          ts: Date.now(),
          processes: [],
          result: null,
        }),
      },
    },
    review: {
      list: () => [],
    },
  } as unknown as Borg;
}

function fakePersonaSession(messages: readonly string[]): {
  session: PersonaSession;
  nextTurn: ReturnType<typeof vi.fn>;
  startNewSession: ReturnType<typeof vi.fn>;
} {
  let index = 0;
  const nextTurn = vi.fn(async () => {
    const message = messages[index] ?? messages.at(-1) ?? "persona turn";
    index += 1;
    return message;
  });
  const startNewSession = vi.fn();

  return {
    session: {
      nextTurn,
      startNewSession,
    } as unknown as PersonaSession,
    nextTurn,
    startNewSession,
  };
}

function mockTransportLifecycle(): void {
  vi.spyOn(BorgTransport.prototype, "open").mockResolvedValue(undefined);
  vi.spyOn(BorgTransport.prototype, "close").mockResolvedValue(undefined);
  vi.spyOn(BorgTransport.prototype, "getBorg").mockReturnValue(fakeSimulatorBorg());
}

function chatResult(input: {
  response: string;
  emitted: boolean;
  turnId: string;
  sessionId: SessionId;
  suppressionReason?: GenerationSuppressionReason;
}): ChatWithBorgResult {
  return {
    response: input.response,
    emitted: input.emitted,
    emission: input.emitted
      ? ({
          kind: "message",
          content: input.response,
          agentMessageId: `strm_${input.turnId}`,
        } as ChatWithBorgResult["emission"])
      : {
          kind: "suppressed",
          reason: input.suppressionReason ?? "no_output_tool",
        },
    turnId: input.turnId,
    sessionId: input.sessionId,
    usage: {
      input_tokens: 0,
      output_tokens: 0,
    },
    moodAfter: {
      valence: 0,
      arousal: 0,
    },
    toolCalls: [],
  };
}

describe("SimulatorRunner", () => {
  it("runs a 20-turn mock simulation with overseer checkpoints and metrics", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    spyMaintenanceTick();
    const report = await runSimulation({
      runId: "sim-runner-test",
      persona: tomPersona,
      totalTurns: 20,
      checkEvery: 10,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
      overseerRunner: async ({ turnCounter }) => ({
        ts: Date.now(),
        turn_counter: turnCounter,
        status: "healthy",
        observations: ["Mock overseer saw no degradation."],
        recommendation: "Continue.",
      }),
    });
    const metricsRows = readFileSync(metricsPath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map((line) => JSON.parse(line) as { turn_counter: number });

    expect(report.totalTurns).toBe(20);
    expect(Object.hasOwn(report, "probes")).toBe(false);
    expect(report.overseerCheckpoints).toHaveLength(2);
    expect(metricsRows).toHaveLength(20);
    expect(metricsRows.at(-1)?.turn_counter).toBe(20);
  });

  it("runs periodic maintenance ticks on cadence in mock mode", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const tickSpy = spyMaintenanceTick();

    await runSimulation({
      runId: "sim-runner-maintenance-test",
      persona: tomPersona,
      totalTurns: 20,
      checkEvery: 999,
      maintenanceEvery: 10,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(tickSpy).toHaveBeenCalledTimes(2);
    expect(tickSpy.mock.calls.map(([cadence]) => cadence)).toEqual(["light", "light"]);
  });

  it("passes the persona display name as the stable Borg audience", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const chatSpy = vi.spyOn(BorgTransport.prototype, "chat");
    spyMaintenanceTick();

    await runSimulation({
      runId: "sim-runner-audience-test",
      persona: tomPersona,
      totalTurns: 2,
      checkEvery: 999,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(chatSpy.mock.calls.map(([, options]) => options?.audience)).toEqual(["Tom", "Tom"]);
  });

  it("passes distinct session IDs after no_output_tool rotation and records them", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const chatSessionIds: SessionId[] = [];
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) => {
      const sessionId = options.sessionId as SessionId;
      chatSessionIds.push(sessionId);
      const emitted = chatSessionIds.length > 1;

      return chatResult({
        response: emitted ? "Second session response" : "",
        emitted,
        turnId: `turn-${chatSessionIds.length}`,
        sessionId,
        suppressionReason: "no_output_tool",
      });
    });

    const report = await runSimulation({
      runId: "sim-runner-session-id-test",
      persona: tomPersona,
      totalTurns: 2,
      checkEvery: 999,
      maxSessions: 3,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(chatSessionIds).toHaveLength(2);
    expect(chatSessionIds[0]).toMatch(/^sess_[a-z0-9]{16}$/);
    expect(chatSessionIds[1]).toMatch(/^sess_[a-z0-9]{16}$/);
    expect(chatSessionIds[0]).not.toBe(chatSessionIds[1]);
    expect(report.sessions).toEqual([
      {
        sessionIndex: 0,
        sessionId: chatSessionIds[0],
        startedAtTurn: 1,
        endedAtTurn: 1,
        endReason: "suppression",
        suppressionReason: "no_output_tool",
      },
      {
        sessionIndex: 1,
        sessionId: chatSessionIds[1],
        startedAtTurn: 2,
        endedAtTurn: 2,
        endReason: "run_complete",
      },
    ]);
  });

  it("continues the same session after a guard-driven suppression", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const persona = fakePersonaSession(["first persona turn", "second persona turn"]);
    const chatSessionIds: SessionId[] = [];
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) => {
      const sessionId = options.sessionId as SessionId;
      chatSessionIds.push(sessionId);
      const emitted = chatSessionIds.length > 1;

      return chatResult({
        response: emitted ? "Borg replied." : "",
        emitted,
        turnId: `turn-${chatSessionIds.length}`,
        sessionId,
        suppressionReason: "relational_guard_self_correction",
      });
    });

    const report = await runSimulation({
      runId: "sim-runner-guard-suppression-test",
      persona: tomPersona,
      personaSession: persona.session,
      totalTurns: 2,
      checkEvery: 999,
      maxSessions: 3,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(chatSessionIds).toHaveLength(2);
    expect(chatSessionIds[0]).toBe(chatSessionIds[1]);
    expect(persona.nextTurn).toHaveBeenCalledTimes(2);
    expect(persona.nextTurn.mock.calls.map(([previous]) => previous)).toEqual([null, null]);
    expect(persona.startNewSession).not.toHaveBeenCalled();
    expect(report.resultState).toBe("completed");
    expect(report.sessions).toHaveLength(1);
    expect(report.sessions[0]).toMatchObject({
      sessionIndex: 0,
      sessionId: chatSessionIds[0],
      startedAtTurn: 1,
      endedAtTurn: 2,
      endReason: "run_complete",
    });
    expect(report.suppressionEvents).toEqual([
      {
        sessionIndex: 0,
        sessionId: chatSessionIds[0],
        turn: 1,
        reason: "relational_guard_self_correction",
      },
    ]);
  });

  it("starts a new session after no_output_tool suppression", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const persona = fakePersonaSession(["first persona turn", "second persona turn"]);
    const chatSessionIds: SessionId[] = [];
    mockTransportLifecycle();
    vi.spyOn(BorgTransport.prototype, "chat").mockImplementation(async (_message, options = {}) => {
      const sessionId = options.sessionId as SessionId;
      chatSessionIds.push(sessionId);
      const emitted = chatSessionIds.length > 1;

      return chatResult({
        response: emitted ? "Borg replied." : "",
        emitted,
        turnId: `turn-${chatSessionIds.length}`,
        sessionId,
        suppressionReason: "no_output_tool",
      });
    });

    const report = await runSimulation({
      runId: "sim-runner-no-output-suppression-test",
      persona: tomPersona,
      personaSession: persona.session,
      totalTurns: 2,
      checkEvery: 999,
      maxSessions: 3,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
    });

    expect(chatSessionIds).toHaveLength(2);
    expect(chatSessionIds[0]).not.toBe(chatSessionIds[1]);
    expect(persona.nextTurn).toHaveBeenCalledTimes(2);
    expect(persona.startNewSession).toHaveBeenCalledTimes(1);
    expect(persona.startNewSession.mock.calls[0]?.[0]).toBe(
      "It's the next evening. You're back on the couch after dinner.",
    );
    expect(report.sessions).toHaveLength(2);
    expect(report.sessions[0]).toMatchObject({
      sessionIndex: 0,
      sessionId: chatSessionIds[0],
      startedAtTurn: 1,
      endedAtTurn: 1,
      endReason: "suppression",
      suppressionReason: "no_output_tool",
    });
    expect(report.sessions[1]).toMatchObject({
      sessionIndex: 1,
      sessionId: chatSessionIds[1],
      startedAtTurn: 2,
      endedAtTurn: 2,
      endReason: "run_complete",
    });
    expect(report.suppressionEvents).toEqual([]);
  });

  it("rotates sessions when Borg suppresses a turn and stops at maxSessions", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    spyMaintenanceTick();
    const report = await runSimulation({
      runId: "sim-runner-suppression-test",
      persona: tomPersona,
      totalTurns: 5,
      checkEvery: 999,
      maxSessions: 1,
      metricsPath,
      dataDir: join(dir, "data"),
      tracePath: join(dir, "trace.jsonl"),
      mock: true,
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "tool_use",
            tool_calls: [
              {
                id: "toolu_plan",
                name: "EmitTurnPlan",
                input: {
                  uncertainty: "",
                  verification_steps: [],
                  tensions: [],
                  voice_note: "",
                  referenced_episode_ids: [],
                  intents: [],
                },
              },
            ],
          },
          {
            text: "",
            input_tokens: 8,
            output_tokens: 4,
            stop_reason: "tool_use",
            tool_calls: [
              {
                id: "toolu_no_output",
                name: "no_output",
                input: {},
              },
            ],
          },
        ],
      }),
    });
    const metricsRows = readFileSync(metricsPath, "utf8")
      .trim()
      .split(/\r?\n/)
      .map((line) => JSON.parse(line) as { turn_counter: number });

    expect(report.resultState).toBe("max_sessions_reached");
    expect(report.sessions).toHaveLength(1);
    expect(report.sessions[0]).toMatchObject({
      sessionIndex: 0,
      startedAtTurn: 1,
      endedAtTurn: 1,
      endReason: "suppression",
    });
    expect(metricsRows).toHaveLength(1);
    expect(metricsRows[0]?.turn_counter).toBe(1);
  });
});
