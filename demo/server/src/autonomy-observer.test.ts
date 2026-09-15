import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  AutonomyError,
  compositeTracer,
  createTurnTracer,
  type AutonomyTickEventResult,
  type Borg,
  type TickResult,
  type TurnTracer,
} from "borg";
import { afterEach, describe, expect, it, vi } from "vitest";

import { wireAutonomySchedulerObserver } from "./autonomy-observer.js";
import { AutonomySourcePreparationError } from "../../../src/autonomy/scheduler.js";

function tick(overrides: Partial<TickResult> = {}): TickResult {
  return {
    status: "ok",
    ts: 1000,
    scannedSources: ["scheduled_wake"],
    dueEvents: 0,
    firedEvents: 0,
    budgetSkipped: 0,
    fleetCooldownSkipped: 0,
    errorCircuitSkipped: 0,
    busySkipped: 0,
    errorCount: 0,
    sourceErrorCount: 0,
    bookkeepingErrorCount: 0,
    events: [],
    ...overrides,
  };
}

function event(overrides: Partial<AutonomyTickEventResult> = {}): AutonomyTickEventResult {
  return {
    id: "private-event-id",
    sourceName: "scheduled_wake",
    sourceType: "trigger",
    sourceCategory: "operational",
    status: "fired",
    payload: { body: "private-wake-body", stream: { content: "private-stream-content" } },
    outcomeSummary: "private-outcome",
    turnResultId: "private-turn-id",
    ...overrides,
  };
}

function harness(trace?: TurnTracer) {
  const tracer = { enabled: true, includePayloads: true, emit: vi.fn<TurnTracer["emit"]>() };
  const logger = { warn: vi.fn(), error: vi.fn() };
  const scheduler = { setObserver: vi.fn<Borg["autonomy"]["scheduler"]["setObserver"]>() };
  wireAutonomySchedulerObserver(scheduler, trace ?? tracer, logger);
  const observer = scheduler.setObserver.mock.calls[0]![0]!;
  return { observer, tracer, logger };
}

const tempDirs: string[] = [];
afterEach(() => {
  for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("demo autonomy observer", () => {
  it.each(["ok", "disabled"] as const)("does not trace or log empty %s ticks", async (status) => {
    const { observer, tracer, logger } = harness();
    await observer.onTick?.(tick({ status }));
    expect(tracer.emit).not.toHaveBeenCalled();
    expect(logger.warn).not.toHaveBeenCalled();
    expect(logger.error).not.toHaveBeenCalled();
  });

  it.each([
    { field: "dueEvents", journal: false },
    { field: "budgetSkipped", journal: false },
    { field: "errorCount", journal: true },
    { field: "sourceErrorCount", journal: true },
    { field: "bookkeepingErrorCount", journal: true },
  ] as const)("traces ticks with only $field > 0", async ({ field, journal }) => {
    const { observer, tracer, logger } = harness();
    await observer.onTick?.(tick({ [field]: 1 }));
    expect(tracer.emit).toHaveBeenCalledExactlyOnceWith(
      "autonomy.tick",
      expect.objectContaining({ [field]: 1 }),
    );
    expect(logger.warn).toHaveBeenCalledTimes(journal ? 1 : 0);
    expect(logger.error).not.toHaveBeenCalled();
  });

  it("traces repeated budget-refusal-only ticks without journaling", async () => {
    const { observer, tracer, logger } = harness();
    for (const ts of [1000, 61_000]) {
      await observer.onTick?.(
        tick({
          ts,
          dueEvents: 1,
          budgetSkipped: 1,
          events: [event({ status: "budget_skipped" })],
        }),
      );
      expect(tracer.emit).toHaveBeenLastCalledWith("autonomy.tick", {
        turnId: `autonomy_tick_${process.pid}_${ts}`,
        ts,
        dueEvents: 1,
        admittedEvents: 0,
        firedEvents: 0,
        budgetSkipped: 1,
        fleetCooldownSkipped: 0,
        errorCircuitSkipped: 0,
        busySkipped: 0,
        errorCount: 0,
        sourceErrorCount: 0,
        bookkeepingErrorCount: 0,
        triggerNames: ["scheduled_wake"],
        errors: [],
        sourceErrors: [],
      });
    }
    expect(tracer.emit).toHaveBeenCalledTimes(2);
    expect(logger.warn).not.toHaveBeenCalled();
    expect(logger.error).not.toHaveBeenCalled();
  });

  it("counts admission by event status and projects only safe fields", async () => {
    const { observer, tracer, logger } = harness();
    const statuses: AutonomyTickEventResult["status"][] = [
      "fired",
      "error",
      "bookkeeping_error",
      "busy_skipped",
      "budget_skipped",
      "fleet_cooldown_skipped",
      "error_circuit_skipped",
    ];
    await observer.onTick?.(
      tick({
        dueEvents: 7,
        firedEvents: 2,
        budgetSkipped: 1,
        fleetCooldownSkipped: 1,
        errorCircuitSkipped: 1,
        busySkipped: 1,
        errorCount: 1,
        bookkeepingErrorCount: 1,
        events: statuses.map((status) => event({ status })),
      }),
    );
    const summary = {
      dueEvents: 7,
      admittedEvents: 4,
      firedEvents: 2,
      budgetSkipped: 1,
      fleetCooldownSkipped: 1,
      errorCircuitSkipped: 1,
      busySkipped: 1,
      errorCount: 1,
      sourceErrorCount: 0,
      bookkeepingErrorCount: 1,
    };
    expect(tracer.emit).toHaveBeenCalledExactlyOnceWith("autonomy.tick", {
      turnId: `autonomy_tick_${process.pid}_1000`,
      ts: 1000,
      ...summary,
      triggerNames: ["scheduled_wake"],
      errors: [],
      sourceErrors: [],
    });
    expect(logger.warn).toHaveBeenCalledExactlyOnceWith(`autonomy.tick ${JSON.stringify(summary)}`);
    expect(JSON.stringify(tracer.emit.mock.calls)).not.toContain("private-");
    expect(JSON.stringify(logger.warn.mock.calls)).not.toContain("private-");
  });

  it("bounds error messages and source names, including scan errors with no due event", async () => {
    const { observer, tracer, logger } = harness();
    const sourceName = "s".repeat(200) as AutonomyTickEventResult["sourceName"];
    const error = "e".repeat(1000);
    await observer.onTick?.(
      tick({
        errorCount: 1,
        sourceErrorCount: 1,
        sourceErrors: [{ sourceName, error }],
      }),
    );
    expect(tracer.emit).toHaveBeenLastCalledWith(
      "autonomy.tick",
      expect.objectContaining({
        dueEvents: 0,
        triggerNames: [],
        sourceErrors: [
          {
            sourceName: "s".repeat(128),
            errorName: "AutonomySourcePreparationError",
            error: "e".repeat(512),
          },
        ],
      }),
    );

    await observer.onTick?.(
      tick({
        dueEvents: 2,
        errorCount: 2,
        events: [
          event({ sourceName, status: "error", error }),
          event({ status: "error", error: "short" }),
        ],
      }),
    );
    expect(tracer.emit).toHaveBeenLastCalledWith(
      "autonomy.tick",
      expect.objectContaining({
        triggerNames: ["s".repeat(128), "scheduled_wake"],
        errors: [
          { sourceName: "s".repeat(128), errorName: "AutonomyError", error: "e".repeat(512) },
          { sourceName: "scheduled_wake", errorName: "AutonomyError", error: "short" },
        ],
        sourceErrors: [],
      }),
    );
    expect(logger.warn).toHaveBeenCalledTimes(2);
    expect(JSON.stringify(logger.warn.mock.calls)).not.toContain(error);
  });

  it.each([false, true])(
    "gates both error fields with payload tracing %s",
    async (includePayloads) => {
      const { observer, tracer, logger } = harness();
      tracer.includePayloads = includePayloads;
      const secret = `private exception/cause content ${"x".repeat(1000)}`;
      await observer.onTick?.(
        tick({
          dueEvents: 1,
          errorCount: 2,
          sourceErrorCount: 1,
          events: [event({ status: "error", errorName: "TypeError", error: secret })],
          sourceErrors: [
            {
              sourceName: "scheduled_wake",
              errorName: "AutonomySourcePreparationError",
              error: secret,
            },
          ],
        }),
      );
      expect(tracer.emit).toHaveBeenCalledExactlyOnceWith(
        "autonomy.tick",
        expect.objectContaining({
          errors: [
            {
              sourceName: "scheduled_wake",
              errorName: "TypeError",
              error: includePayloads ? secret.slice(0, 512) : "TypeError",
            },
          ],
          sourceErrors: [
            {
              sourceName: "scheduled_wake",
              errorName: "AutonomySourcePreparationError",
              error: includePayloads ? secret.slice(0, 512) : "AutonomySourcePreparationError",
            },
          ],
        }),
      );
      if (!includePayloads) expect(JSON.stringify(tracer.emit.mock.calls)).not.toContain("private");
      expect(JSON.stringify(logger.warn.mock.calls)).not.toContain("private");
    },
  );

  it("uses structural fallbacks instead of parsing untyped exception strings", async () => {
    const { observer, tracer } = harness();
    tracer.includePayloads = false;
    await observer.onTick?.(
      tick({
        dueEvents: 1,
        errorCount: 2,
        sourceErrorCount: 1,
        events: [event({ status: "error", error: "private content: arbitrary thrown string" })],
        sourceErrors: [{ sourceName: "scheduled_wake", error: "private source content" }],
      }),
    );
    expect(tracer.emit).toHaveBeenCalledWith(
      "autonomy.tick",
      expect.objectContaining({
        errors: [
          { sourceName: "scheduled_wake", errorName: "AutonomyError", error: "AutonomyError" },
        ],
        sourceErrors: [
          {
            sourceName: "scheduled_wake",
            errorName: "AutonomySourcePreparationError",
            error: "AutonomySourcePreparationError",
          },
        ],
      }),
    );
    expect(JSON.stringify(tracer.emit.mock.calls)).not.toContain("private");
  });

  it.each([false, true])(
    "gates JSONL errors per sink with file payload tracing %s",
    async (includePayloads) => {
      const dir = mkdtempSync(join(tmpdir(), "borg-autonomy-observer-"));
      tempDirs.push(dir);
      const tracePath = join(dir, "traces", "turns.jsonl");
      const liveTracer = { enabled: true, includePayloads: true, emit: vi.fn() };
      const { observer, logger } = harness(
        compositeTracer([
          createTurnTracer({
            env: { BORG_TRACE: tracePath, BORG_TRACE_PROMPTS: includePayloads ? "1" : "0" },
          }),
          liveTracer,
        ]),
      );
      await observer.onTick?.(tick());
      expect(existsSync(tracePath)).toBe(false);
      const secret = "exception-body-secret";
      await observer.onTick?.(
        tick({
          dueEvents: 1,
          errorCount: 2,
          sourceErrorCount: 1,
          events: [event({ status: "error", errorName: "TypeError", error: secret })],
          sourceErrors: [
            {
              sourceName: "scheduled_wake",
              errorName: "AutonomySourcePreparationError",
              error: secret,
            },
          ],
        }),
      );
      await observer.onTick?.(tick({ ts: 2000 }));
      const traceText = readFileSync(tracePath, "utf8");
      expect(JSON.parse(traceText)).toMatchObject({
        event: "autonomy.tick",
        ts: 1000,
        dueEvents: 1,
        admittedEvents: 1,
        firedEvents: 0,
        triggerNames: ["scheduled_wake"],
        errors: [
          {
            sourceName: "scheduled_wake",
            errorName: "TypeError",
            error: includePayloads ? secret : "TypeError",
          },
        ],
        sourceErrors: [
          {
            sourceName: "scheduled_wake",
            errorName: "AutonomySourcePreparationError",
            error: includePayloads ? secret : "AutonomySourcePreparationError",
          },
        ],
      });
      expect(traceText).not.toContain("private-");
      expect(liveTracer.emit).toHaveBeenCalledOnce();
      expect(liveTracer.emit).toHaveBeenCalledWith(
        "autonomy.tick",
        expect.objectContaining({
          errors: [expect.objectContaining({ error: secret })],
          sourceErrors: [expect.objectContaining({ error: secret })],
        }),
      );
      if (!includePayloads) expect(traceText).not.toContain(secret);
      expect(logger.warn).toHaveBeenCalledOnce();
      expect(JSON.stringify(logger.warn.mock.calls)).not.toContain(secret);
    },
  );

  it.each([false, true])(
    "reports aborted tick errors with payload tracing %s",
    async (includePayloads) => {
      const { observer, tracer, logger } = harness();
      tracer.includePayloads = includePayloads;
      const message = `private error text\n${"x".repeat(1000)}`;
      await observer.onError?.(new TypeError(message));
      expect(tracer.emit).toHaveBeenCalledExactlyOnceWith("autonomy.tick_error", {
        turnId: expect.stringMatching(/^autonomy_tick_error_/),
        ts: expect.any(Number),
        errorCount: 1,
        errors: [
          { errorName: "TypeError", error: includePayloads ? message.slice(0, 512) : "TypeError" },
        ],
      });
      expect(logger.warn).toHaveBeenCalledExactlyOnceWith(
        'autonomy.tick_error {"errorCount":1,"errorName":"TypeError","traceErrorCount":0}',
      );
      if (!includePayloads) expect(JSON.stringify(tracer.emit.mock.calls)).not.toContain("private");
    },
  );

  it("does not duplicate source and bookkeeping errors reported in completed ticks", async () => {
    const { observer, tracer, logger } = harness();
    await observer.onError?.(
      new AutonomySourcePreparationError("scheduled_wake", new Error("private")),
    );
    await observer.onError?.(new AutonomyError("private", { code: "AUTONOMY_BOOKKEEPING_FAILED" }));
    expect(tracer.emit).not.toHaveBeenCalled();
    expect(logger.warn).not.toHaveBeenCalled();
    await observer.onTick?.(
      tick({
        errorCount: 1,
        sourceErrorCount: 1,
        bookkeepingErrorCount: 1,
        sourceErrors: [
          {
            sourceName: "scheduled_wake",
            errorName: "AutonomySourcePreparationError",
            error: "source failed",
          },
        ],
      }),
    );
    expect(tracer.emit).toHaveBeenCalledExactlyOnceWith(
      "autonomy.tick",
      expect.objectContaining({ sourceErrorCount: 1 }),
    );
    expect(logger.warn).toHaveBeenCalledOnce();
  });

  it("contains error-observer sink failures and handles non-Error throws", () => {
    const { observer, tracer, logger } = harness();
    tracer.includePayloads = false;
    tracer.emit.mockImplementationOnce(() => {
      throw new Error("private trace failure");
    });
    expect(() => observer.onError?.("private thrown string")).not.toThrow();
    expect(logger.warn).toHaveBeenCalledExactlyOnceWith(
      'autonomy.tick_error {"errorCount":1,"errorName":"UnknownError","traceErrorCount":1}',
    );
    logger.warn.mockImplementation(() => {
      throw new Error("journal failed");
    });
    expect(() => observer.onError?.(new Error("private message"))).not.toThrow();
    expect(tracer.emit).toHaveBeenCalledTimes(2);
    expect(JSON.stringify(tracer.emit.mock.calls)).not.toContain("private");
  });

  it("reports trace write failures in one bounded journal line", async () => {
    const { observer, tracer, logger } = harness();
    const message = `trace failed\n${"e".repeat(1000)}`;
    tracer.emit.mockImplementation(() => {
      throw new Error(message);
    });
    await observer.onTick?.(tick({ dueEvents: 1 }));
    expect(logger.error).toHaveBeenCalledExactlyOnceWith(
      `Autonomy tick trace failed: ${JSON.stringify(message.slice(0, 512))}`,
    );
  });
});
