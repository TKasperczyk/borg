import { afterEach, describe, expect, it } from "vitest";

import { StreamReader, StreamWriter } from "../stream/index.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";

import { CuratorProcess } from "./curator/index.js";
import { MaintenanceOrchestrator } from "./orchestrator.js";
import { createEpisodeFixture, createOfflineTestHarness } from "./test-support.js";

describe("maintenance orchestrator", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("emits a dream_report and links audit rows to the run id", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const episode = createEpisodeFixture({
      title: "Archive candidate",
      created_at: 1_000_000 - 50 * 24 * 60 * 60 * 1_000,
      updated_at: 1_000_000 - 50 * 24 * 60 * 60 * 1_000,
    });
    await harness.episodicRepository.insert(episode);
    const process = new CuratorProcess({
      episodicRepository: harness.episodicRepository,
      registry: harness.registry,
    });

    const orchestrator = new MaintenanceOrchestrator({
      baseContext: {
        config: harness.config,
        clock: harness.clock,
        embeddingClient: harness.embeddingClient,
        llm: {
          cognition: harness.llmClient,
          background: harness.llmClient,
          extraction: harness.llmClient,
        },
        episodicRepository: harness.episodicRepository,
        semanticNodeRepository: harness.semanticNodeRepository,
        semanticEdgeRepository: harness.semanticEdgeRepository,
        reviewQueueRepository: harness.reviewQueueRepository,
        valuesRepository: harness.valuesRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        entityRepository: harness.entityRepository,
        commitmentRepository: harness.commitmentRepository,
      },
      auditLog: harness.auditLog,
      createStreamWriter: () =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId: DEFAULT_SESSION_ID,
          clock: harness.clock,
        }),
      processRegistry: {
        consolidator: process,
        reflector: process,
        curator: process,
        overseer: process,
      },
    });

    const result = await orchestrator.run({
      processes: [process],
      opts: {
        dryRun: false,
      },
    });

    expect(result.run_id).toBeDefined();
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
});
