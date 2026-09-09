import { mkdtempSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Borg } from "../borg.js";
import { TurnOrchestrator } from "../cognition/turn-orchestrator.js";
import { AgentDeliveryRepository } from "../cognition/ingestion/agent-deliveries.js";
import { FakeEmbeddingClient } from "../embeddings/index.js";
import { FakeLLMClient, createFakeEmitAnswerResponse } from "../llm/test-support/fake-client.js";
import { TurnReflectionCoordinator } from "../cognition/reflection/turn-reflection-coordinator.js";
import { StreamWriter } from "../stream/index.js";
import { DeliveryWaiterRegistry } from "../sidecar/delivery-waiter-registry.js";
import { ResponseWaiterRegistry } from "../sidecar/response-waiter-registry.js";
import { createMemoryHandler } from "../sidecar/memory-handler.js";
import { createSessionId } from "../util/ids.js";
import type { BorgDependencies, BorgOpenOptions } from "./types.js";
import * as turnSetup from "./turn-setup.js";
type ClaimResponse = { deliveries: { terminal_entry_id: string; content: string }[] };

const cleanups: Array<() => void | Promise<void>> = [];
afterEach(async () => {
  while (cleanups.length) await cleanups.pop()!();
  vi.restoreAllMocks();
});

async function harness() {
  const dataDir = mkdtempSync(join(tmpdir(), "borg-native-inbox-"));
  cleanups.push(() => rmSync(dataDir, { recursive: true, force: true }));
  const onWake = vi.fn();
  const deliveryWaiters = new DeliveryWaiterRegistry({ onWake });
  const inboxWaiters = new ResponseWaiterRegistry();
  let borg: Borg;
  const open = async (inbox: BorgOpenOptions["inbox"]) => {
    borg = await Borg.open({
      dataDir,
      env: {},
      liveExtraction: false,
      embeddingClient: new FakeEmbeddingClient(4),
      embeddingDimensions: 4,
      llmClient: new FakeLLMClient({
        responses: Array.from({ length: 100 }, () =>
          createFakeEmitAnswerResponse("Committed native answer"),
        ),
      }),
      inbox: {
        ...inbox,
        onDeliveryAvailable: (sessionId) => deliveryWaiters.notify("probe", sessionId),
        onTerminalCommitted: (entry) => inboxWaiters.resolveTerminal("probe", entry),
        sessionPredicate: (session) => session?.source_type === "teams_inbox",
      },
    });
    return borg;
  };
  await open({ native: { tools: "none", agentDeliveries: true } });
  cleanups.push(async () => {
    deliveryWaiters.shutdown();
    inboxWaiters.shutdown();
    await borg.close();
  });
  const sessionId = createSessionId();
  const sender = borg!.entities.resolve("Sender", {
    kind: "person",
    provenance: "transport_sender",
  });
  const audience = borg!.entities.resolve("Room", {
    kind: "group",
    provenance: "transport_audience_label",
  });
  const session = {
    session_id: sessionId,
    source_type: "teams_inbox" as const,
    source_external_id: "room-external",
    label: "Room",
    audience_label: "Room",
    audience_entity_id: audience,
    conversation_kind: "thread" as const,
  };
  borg!.sessions.ensure(session);
  const enqueue = (externalMessageId = "question-1") =>
    borg!.enqueueMessage({
      session,
      senderEntityId: sender,
      audience: "Room",
      audienceEntityId: audience,
      userMessage: "A memory question",
      sourceMessageKey: {
        source_type: "teams_inbox",
        source_external_id: "room-external",
        external_message_id: externalMessageId,
      },
    });
  const server: Server = createServer(
    createMemoryHandler({
      token: "test-token",
      deliveryWaiters,
      inboxWaiters,
      pool: { listTenantIds: async () => ["probe"], withTenant: async (_tenant, fn) => fn(borg!) },
    }),
  );
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  cleanups.push(() => new Promise<void>((resolve) => server.close(() => resolve())));
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const post = async (path: string, body: object) => {
    const response = await fetch(`${base}${path}`, {
      method: "POST",
      headers: {
        "x-borg-token": "test-token",
        "content-type": "application/json",
      },
      body: JSON.stringify({ tenant: "probe", ...body }),
    });
    expect(response.status).toBe(200);
    return response.json();
  };
  return { borg: borg!, dataDir, open, sessionId, enqueue, deliveryWaiters, onWake, post };
}

describe("Borg native inbox wiring", () => {
  it.each([false, true])(
    "uses the real run lifecycle before publishing (post-response failure=%s)",
    async (failAfterResponse) => {
      const h = await harness();
      await h.enqueue();
      const internal = h.borg as unknown as { deps: BorgDependencies };
      let persistedAgentEntry:
        | Parameters<TurnReflectionCoordinator["run"]>[0]["persistedAgentEntry"]
        | undefined;
      let finish!: () => void;
      const held = new Promise<void>((resolve) => {
        finish = resolve;
      });
      const originalReflection = TurnReflectionCoordinator.prototype.run;
      vi.spyOn(TurnReflectionCoordinator.prototype, "run").mockImplementation(async function (
        this: TurnReflectionCoordinator,
        input,
      ) {
        persistedAgentEntry = input.persistedAgentEntry;
        await held;
        if (failAfterResponse) throw new Error("Injected post-response failure");
        return originalReflection.call(this, input);
      });
      // Call through: perception, generation, persistence, post-response processing,
      // TurnOrchestrator.run's catch/rollback, and the real stream index all execute.
      const run = vi.spyOn(TurnOrchestrator.prototype, "run");
      const drain = h.borg.inbox.catchUp.tick(h.sessionId);
      try {
        await vi.waitFor(() => expect(persistedAgentEntry).toBeDefined());
        expect(internal.deps.entryIndex.lookup(persistedAgentEntry!.id)?.active).toBe(true);
        expect(
          h.borg.inbox.deliveries.claim({ sessionIds: [h.sessionId], leaseMs: 1000 }).deliveries,
        ).toEqual([]);
        expect(h.onWake).not.toHaveBeenCalled();
      } finally {
        finish();
      }
      expect((await drain).status).toBe(failAfterResponse ? "error" : "drained");
      expect(run).toHaveBeenCalledTimes(1);
      expect(internal.deps.entryIndex.lookup(persistedAgentEntry!.id)?.active).toBe(
        !failAfterResponse,
      );
      if (failAfterResponse) {
        expect(
          h.borg.inbox.deliveries.claim({ sessionIds: [h.sessionId], leaseMs: 1000 }).deliveries,
        ).toEqual([]);
        await h.borg.close();
        const reopened = await h.open({ native: { tools: "none", agentDeliveries: true } });
        await reopened.inbox.catchUp.tick(h.sessionId);
        expect(
          reopened.inbox.deliveries.claim({ sessionIds: [h.sessionId], leaseMs: 1000 }).deliveries,
        ).toEqual([]);
      } else {
        expect(
          h.borg.inbox.deliveries.claim({ sessionIds: [h.sessionId], leaseMs: 1000 }).deliveries,
        ).toEqual([
          expect.objectContaining({
            terminal_entry_id: persistedAgentEntry!.id,
            content: "Committed native answer",
          }),
        ]);
      }
    },
  );
  it("retries a failed delivery projection before skipping an answered prefix", async () => {
    const h = await harness();
    await h.enqueue();
    const run = vi.spyOn(TurnOrchestrator.prototype, "run").mockImplementation(async (input) => {
      await h.borg.inbox.appendBacklogTerminal({
        sessionId: h.sessionId,
        sourceEntryIds: input.inboundBatch!.entryIds,
        turnId: "native-delivery-retry",
        terminal: { kind: "agent_msg", content: "Committed before queue recovered" },
      });
      return { turn_id: "native-delivery-retry" } as Awaited<ReturnType<TurnOrchestrator["run"]>>;
    });
    const create = vi.spyOn(AgentDeliveryRepository.prototype, "create").mockImplementation(() => {
      throw new Error("test delivery storage unavailable");
    });
    expect((await h.borg.inbox.catchUp.tick(h.sessionId)).status).toBe("error");
    create.mockRestore();
    await h.borg.inbox.catchUp.tick(h.sessionId);
    expect(run).toHaveBeenCalledTimes(1);
    expect(
      h.borg.inbox.deliveries.claim({ sessionIds: [h.sessionId], leaseMs: 1000 }).deliveries,
    ).toEqual([expect.objectContaining({ content: "Committed before queue recovered" })]);
  });
  it("has no executable tools and defers the delivery HTTP wake until successful run completion", async () => {
    const build = vi.spyOn(turnSetup, "buildTurnOrchestrator");
    const h = await harness();
    expect(build.mock.calls[0]![0].toolDispatcher.listTools("deliberator")).toEqual([]);
    expect(build.mock.calls[0]![0].toolDispatcher.listTools("autonomous")).toEqual([]);
    const queued = await h.enqueue();
    const waiting = h.post("/memory/agent-deliveries/claim", {
      sidecar_session_ids: [h.sessionId],
      wait_ms: 5000,
    });
    await vi.waitFor(() => expect(h.deliveryWaiters.size()).toBe(1));
    let finish!: () => void;
    const afterCommit = new Promise<void>((resolve) => {
      finish = resolve;
    });
    let appended = false;
    const run = vi.spyOn(TurnOrchestrator.prototype, "run").mockImplementation(async (input) => {
      expect(input.inboundBatch?.entryIds).toEqual([queued.streamEntryId]);
      await h.borg.inbox.appendBacklogTerminal({
        sessionId: h.sessionId,
        sourceEntryIds: input.inboundBatch!.entryIds,
        turnId: "native-test-turn",
        terminal: { kind: "agent_msg", content: "Committed native answer" },
      });
      appended = true;
      await afterCommit;
      return { turn_id: "native-test-turn" } as Awaited<ReturnType<TurnOrchestrator["run"]>>;
    });
    const drain = h.borg.inbox.catchUp.tick(h.sessionId);
    try {
      await vi.waitFor(() => expect(appended).toBe(true));
      expect(
        h.borg.inbox.deliveries.claim({ sessionIds: [h.sessionId], leaseMs: 1000 }).deliveries,
      ).toEqual([]);
      expect(h.onWake).not.toHaveBeenCalled();
      finish();
      expect((await drain).status).toBe("drained");
      const result = (await waiting) as ClaimResponse;
      expect(result.deliveries).toHaveLength(1);
      expect(result.deliveries[0]).toMatchObject({
        content: "Committed native answer",
        sidecar_session_id: h.sessionId,
      });
      expect(h.onWake).toHaveBeenCalledWith({
        tenant: "probe",
        sessionIds: [h.sessionId],
        wake: "available",
      });
      expect(run).toHaveBeenCalledWith(
        expect.objectContaining({ origin: "user", lockMode: "try" }),
      );
      const answer = await h.post("/memory/await-response", {
        sidecar_session_id: h.sessionId,
        entry_id: queued.streamEntryId,
        timeout_ms: 0,
      });
      expect(answer).toMatchObject({
        status: "answered",
        terminal_id: result.deliveries[0]!.terminal_entry_id,
        reply: "Committed native answer",
      });
    } finally {
      finish();
    }
    expect((await drain).status).toBe("drained");
    expect(
      (
        (await h.post("/memory/agent-deliveries/claim", {
          sidecar_session_ids: [h.sessionId],
        })) as ClaimResponse
      ).deliveries,
    ).toEqual([]);
    await h.borg.inbox.catchUp.tick(h.sessionId);
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("recovers a committed native answer after reopen without regenerating or delivering HTTP-runner terminals", async () => {
    const h = await harness();
    await h.borg.close();
    const borg = await h.open({ runner: { async run() {} } });
    const queued = await h.enqueue();
    const terminal = await borg.inbox.appendBacklogTerminal({
      sessionId: h.sessionId,
      sourceEntryIds: [queued.streamEntryId],
      turnId: "native-crash-window",
      terminal: { kind: "agent_msg", content: "Already committed" },
    });
    const internal = borg as unknown as { deps: BorgDependencies };
    const writer = new StreamWriter({
      dataDir: h.dataDir,
      sessionId: h.sessionId,
      entryIndex: internal.deps.entryIndex,
    });
    await writer.append({
      kind: "internal_event",
      turn_id: "native-crash-window",
      content: {
        event: "native_inbox_turn_committed",
        terminal_entry_id: terminal.terminalEntry.id,
      },
    });
    writer.close();
    const interrupted = await h.enqueue("interrupted-question");
    await borg.inbox.appendBacklogTerminal({
      sessionId: h.sessionId,
      sourceEntryIds: [interrupted.streamEntryId],
      turnId: "unfinished-native-turn",
      terminal: { kind: "agent_msg", content: "No successful lifecycle marker" },
    });
    const legacyQueued = await h.enqueue("legacy-question");
    await borg.inbox.appendBacklogTerminal({
      sessionId: h.sessionId,
      sourceEntryIds: [legacyQueued.streamEntryId],
      terminal: { kind: "agent_msg", content: "HTTP runner reply, already handled" },
    });
    const observedQueued = await h.enqueue("observed-question");
    await borg.inbox.appendBacklogTerminal({
      sessionId: h.sessionId,
      sourceEntryIds: [observedQueued.streamEntryId],
      turnId: "native-observed",
      terminal: { kind: "agent_observed", reason: "No answer" },
    });
    expect(
      borg.inbox.deliveries.claim({ sessionIds: [h.sessionId], leaseMs: 1000 }).deliveries,
    ).toEqual([]);
    await borg.close();
    const reopened = await h.open({ native: { tools: "none", agentDeliveries: true } });
    const run = vi.spyOn(TurnOrchestrator.prototype, "run");
    await reopened.inbox.catchUp.tick(h.sessionId);
    expect(run).not.toHaveBeenCalled();
    const deliveries = reopened.inbox.deliveries.claim({
      sessionIds: [h.sessionId],
      leaseMs: 1000,
    }).deliveries;
    expect(deliveries).toHaveLength(1);
    expect(deliveries[0]).toMatchObject({
      terminal_entry_id: terminal.terminalEntry.id,
      content: "Already committed",
    });
  });

  it("rejects simultaneous native and override runner selection before opening storage", async () => {
    await expect(
      Borg.open({
        inbox: { native: { tools: "none", agentDeliveries: true }, runner: { async run() {} } },
      }),
    ).rejects.toThrow("mutually exclusive");
  });
});
