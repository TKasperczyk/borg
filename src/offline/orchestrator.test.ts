import { publicMemoryDisclosureLabel } from "../memory/common/disclosure-label.js";
import { createSemanticNodeId } from "../util/ids.js";
import { afterEach, describe, expect, it, vi } from "vitest";
import { refreshSerializedEmbeddings } from "../embeddings/serialized.js";
import type { MaintenancePlan } from "./plan-file.js";

import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../tracing/tracer.js";
import { StreamReader, StreamWriter } from "../stream/index.js";
import { createMaintenanceRunId, DEFAULT_SESSION_ID } from "../util/ids.js";

import { CuratorProcess } from "./curator/index.js";
import { MaintenanceOrchestrator } from "./orchestrator.js";
import { createEpisodeFixture, createOfflineTestHarness } from "./test-support.js";
import type {
  OfflineContext,
  OfflineProcess,
  OfflineProcessName,
  OfflineProcessPlan,
  OfflineResult,
} from "./types.js";
import { OFFLINE_PROCESS_NAMES } from "./types.js";

class CaptureTracer implements TurnTracer {
  readonly enabled = true;
  readonly includePayloads = false;
  readonly events: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];

  emit(event: TurnTraceEventName, data: TurnTraceData): void {
    this.events.push({ event, data });
  }
}

function baseContextFrom(ctx: OfflineContext) {
  const { runId: _runId, auditLog: _auditLog, streamWriter: _streamWriter, ...baseContext } = ctx;

  return baseContext;
}

function emptyResult(name: OfflineProcessName): OfflineResult {
  return {
    process: name,
    dryRun: false,
    changes: [],
    tokens_used: 0,
    errors: [],
    budget_exhausted: false,
  };
}

function emptyPlan(name: OfflineProcessName): OfflineProcessPlan {
  return {
    process: name,
    items: [],
    tokens_used: 0,
    errors: [],
    budget_exhausted: false,
  } as unknown as OfflineProcessPlan;
}

function fakeProcess(name: OfflineProcessName, result: OfflineResult = emptyResult(name)) {
  return {
    name,
    plan: async () => emptyPlan(name),
    preview: () => result,
    apply: async () => result,
    run: async () => result,
  } satisfies OfflineProcess;
}

function createProcessRegistry(
  overrides: Partial<Record<OfflineProcessName, OfflineProcess>>,
): Record<OfflineProcessName, OfflineProcess> {
  return Object.fromEntries(
    OFFLINE_PROCESS_NAMES.map((name) => [name, overrides[name] ?? fakeProcess(name)]),
  ) as Record<OfflineProcessName, OfflineProcess>;
}

describe("maintenance orchestrator", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("rejects saved vectors without text before creating a stream writer or applying a plan", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);
    const createStreamWriter = vi.fn(() => harness.createContext().streamWriter);
    const orchestrator = new MaintenanceOrchestrator({
      baseContext: baseContextFrom(harness.createContext()),
      auditLog: harness.auditLog,
      processRegistry: createProcessRegistry({}),
      createStreamWriter,
      prepareSerializedEmbeddings: (payload) =>
        refreshSerializedEmbeddings(payload, harness.createContext().embeddingClient),
    });
    const episodeId = createEpisodeFixture().id;
    const plan: MaintenancePlan = {
      kind: "borg_maintenance_plan",
      version: 2,
      run_id: createMaintenanceRunId(),
      created_at: 1,
      processes: [
        {
          process: "reflector",
          tokens_used: 0,
          errors: [],
          budget_exhausted: false,
          items: [
            {
              cluster_key: "test",
              episode_ids: [episodeId],
              source_disclosure_label: publicMemoryDisclosureLabel(),
              candidate_support_edges: [],
              target: {
                mode: "update",
                node_id: createSemanticNodeId(),
                patch: {
                  description: "replacement",
                  confidence: 0.7,
                  source_episode_ids: [episodeId],
                  last_verified_at: 1,
                  embedding: [1, 2, 3, 4],
                  archived: false,
                },
              },
              review: { kind: "new_insight", reason: "queued" },
            },
          ],
        },
      ],
    };
    await expect(orchestrator.apply(plan)).rejects.toMatchObject({
      code: "SERIALIZED_EMBEDDING_INCOMPATIBLE",
    });
    expect(createStreamWriter).not.toHaveBeenCalled();
  });

  it("emits a dream_report and links audit rows to the run id", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const episode = createEpisodeFixture({
      title: "Archive candidate",
      created_at: 1_000_000 - 50 * 24 * 60 * 60 * 1_000,
      updated_at: 1_000_000 - 50 * 24 * 60 * 60 * 1_000,
    });
    await harness.episodicRepository.createEpisode(episode);
    const process = new CuratorProcess({
      episodicRepository: harness.episodicRepository,
      traitsRepository: harness.traitsRepository,
      moodRepository: harness.moodRepository,
      socialRepository: harness.socialRepository,
      registry: harness.registry,
    });

    const {
      runId: _runId,
      auditLog: _auditLog,
      streamWriter: _streamWriter,
      ...baseContext
    } = harness.createContext();
    const orchestrator = new MaintenanceOrchestrator({
      baseContext,
      auditLog: harness.auditLog,
      createStreamWriter: () =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId: DEFAULT_SESSION_ID,
          clock: harness.clock,
        }),
      processRegistry: createProcessRegistry({ curator: process }),
    });

    const runId = createMaintenanceRunId();
    const result = await orchestrator.run({
      runId,
      processes: [process],
      opts: {
        dryRun: false,
      },
    });

    expect(result.run_id).toBe(runId);
    expect(harness.auditLog.list()[0]?.run_id).toBe(result.run_id);

    const reader = new StreamReader({
      dataDir: harness.tempDir,
      sessionId: DEFAULT_SESSION_ID,
    });
    const dreamReport = reader.tail(1)[0];

    expect(dreamReport).toMatchObject({
      kind: "dream_report",
      content: expect.objectContaining({
        run_id: result.run_id,
      }),
    });
  });

  it("preserves a caller run id through plan, preview, apply, and tracing context", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);
    const seenRunIds: string[] = [];
    const process = {
      ...fakeProcess("curator"),
      plan: async (ctx: OfflineContext) => {
        seenRunIds.push(ctx.runId);
        return emptyPlan("curator");
      },
      apply: async (ctx: OfflineContext) => {
        seenRunIds.push(ctx.runId);
        return emptyResult("curator");
      },
    } satisfies OfflineProcess;
    const orchestrator = new MaintenanceOrchestrator({
      baseContext: baseContextFrom(harness.createContext()),
      auditLog: harness.auditLog,
      createStreamWriter: () =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId: DEFAULT_SESSION_ID,
          clock: harness.clock,
        }),
      processRegistry: createProcessRegistry({ curator: process }),
    });
    const runId = createMaintenanceRunId();

    const plan = await orchestrator.plan({ runId, processes: [process] });
    const preview = orchestrator.preview(plan);
    const applied = await orchestrator.apply(plan);

    expect(plan.run_id).toBe(runId);
    expect(preview.run_id).toBe(runId);
    expect(applied.run_id).toBe(runId);
    expect(seenRunIds).toEqual([runId, runId]);
  });

  it("traces process error details on offline_process.completed", async () => {
    const tracer = new CaptureTracer();
    const harness = await createOfflineTestHarness({ tracer });
    cleanup.push(harness.cleanup);
    const longMessage = `overseer target failed ${"x".repeat(350)}`;

    const result = {
      process: "overseer",
      dryRun: false,
      changes: [],
      tokens_used: 42,
      errors: [
        {
          process: "overseer",
          message: longMessage,
          code: "OVERSEER_TARGET_FAILED",
          target_type: "episode",
          target_id: "ep_trace_target",
        },
        { process: "overseer", message: "second failure" },
        { process: "overseer", message: "third failure" },
        { process: "overseer", message: "fourth failure" },
      ],
      budget_exhausted: true,
      candidate_stats: {
        proposed: 1,
        accepted: 0,
        rejected: 1,
      },
    } satisfies OfflineResult;
    const process = fakeProcess("overseer", result);
    const orchestrator = new MaintenanceOrchestrator({
      baseContext: baseContextFrom(harness.createContext()),
      auditLog: harness.auditLog,
      createStreamWriter: () =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId: DEFAULT_SESSION_ID,
          clock: harness.clock,
        }),
      processRegistry: createProcessRegistry({
        overseer: process,
      }),
    });

    await orchestrator.run({
      processes: [process],
      opts: {
        dryRun: false,
      },
    });

    expect(tracer.events).toContainEqual({
      event: "offline_process.completed",
      data: expect.objectContaining({
        process_name: "overseer",
        errors: 4,
        error_details: [
          {
            message: longMessage.slice(0, 300),
            code: "OVERSEER_TARGET_FAILED",
            target_type: "episode",
            target_id: "ep_trace_target",
          },
          { message: "second failure" },
          { message: "third failure" },
        ],
        tokens_used: 42,
        budget_exhausted: true,
      }),
    });
  });

  it("uses explicit candidate stats and keeps the legacy fallback for unstatted results", async () => {
    const tracer = new CaptureTracer();
    const harness = await createOfflineTestHarness({ tracer });
    cleanup.push(harness.cleanup);

    const explicitResult = {
      process: "overseer",
      dryRun: false,
      changes: [],
      tokens_used: 0,
      errors: [
        {
          process: "overseer",
          message: "target failed",
        },
      ],
      budget_exhausted: false,
      notes: ["tension scaffolding dropped: 0"],
      candidate_stats: {
        proposed: 3,
        accepted: 2,
        rejected: 1,
        truncated: 1,
      },
    } satisfies OfflineResult;
    const fallbackResult = {
      process: "curator",
      dryRun: false,
      changes: [
        {
          process: "curator",
          action: "archive",
          targets: {
            episode_id: "ep_curator_target",
          },
        },
      ],
      tokens_used: 0,
      errors: [
        {
          process: "curator",
          message: "first failure",
        },
        {
          process: "curator",
          message: "second failure",
        },
      ],
      budget_exhausted: false,
    } satisfies OfflineResult;
    const overseer = fakeProcess("overseer", explicitResult);
    const curator = fakeProcess("curator", fallbackResult);
    const orchestrator = new MaintenanceOrchestrator({
      baseContext: baseContextFrom(harness.createContext()),
      auditLog: harness.auditLog,
      createStreamWriter: () =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId: DEFAULT_SESSION_ID,
          clock: harness.clock,
        }),
      processRegistry: createProcessRegistry({
        overseer,
        curator,
      }),
    });

    await orchestrator.run({
      processes: [overseer, curator],
      opts: {
        dryRun: false,
      },
    });

    expect(tracer.events).toContainEqual({
      event: "offline_process.completed",
      data: expect.objectContaining({
        process_name: "overseer",
        candidates_proposed: 3,
        candidates_accepted: 2,
        candidates_rejected: 1,
        candidates_truncated: 1,
        notes: ["tension scaffolding dropped: 0", "candidate_cap_truncated:1"],
        errors: 1,
      }),
    });
    expect(tracer.events).toContainEqual({
      event: "offline_process.completed",
      data: expect.objectContaining({
        process_name: "curator",
        candidates_proposed: 1,
        candidates_accepted: 1,
        candidates_rejected: 2,
        errors: 2,
      }),
    });
    const dreamReport = new StreamReader({
      dataDir: harness.tempDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(1)[0];

    expect(dreamReport?.content).toMatchObject({
      notes: ["overseer: tension scaffolding dropped: 0"],
    });
  });

  it("carries ruminator due, visited, and budget-cut counts into the dream report", async () => {
    const harness = await createOfflineTestHarness({});
    cleanup.push(harness.cleanup);
    const schedulerNote =
      "rumination scheduler: due=9; selected=8; visited=3; budget_cut=5; budget_cut_question_ids=oq_cut";
    const ruminator = fakeProcess("ruminator", {
      process: "ruminator",
      dryRun: false,
      changes: [],
      tokens_used: 150_010,
      errors: [],
      budget_exhausted: true,
      notes: [schedulerNote],
    });
    const orchestrator = new MaintenanceOrchestrator({
      baseContext: baseContextFrom(harness.createContext()),
      auditLog: harness.auditLog,
      createStreamWriter: () =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId: DEFAULT_SESSION_ID,
          clock: harness.clock,
        }),
      processRegistry: createProcessRegistry({ ruminator }),
    });

    await orchestrator.run({ processes: [ruminator], opts: { dryRun: false } });

    const dreamReport = new StreamReader({
      dataDir: harness.tempDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(1)[0];
    expect(dreamReport?.content).toMatchObject({
      notes: [`ruminator: ${schedulerNote}`, "Budget exhausted: ruminator"],
    });
  });
});
