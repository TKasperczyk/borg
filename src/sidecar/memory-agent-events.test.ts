import { mkdtempSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Borg } from "../borg.js";
import type { BorgOpenOptions } from "../borg/types.js";
import type {
  AgentDelivery,
  AgentDeliveryRepository,
} from "../cognition/ingestion/agent-deliveries.js";
import type { BacklogTerminalService } from "../cognition/ingestion/backlog-terminal.js";
import { FakeEmbeddingClient } from "../embeddings/index.js";
import type { LLMCompleteOptions } from "../llm/index.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import { ManualClock } from "../util/clock.js";
import { createSessionId, createStreamEntryId } from "../util/ids.js";
import { createMemoryHandler, type MemoryPool } from "./memory-handler.js";
import { DeliveryWaiterRegistry } from "./delivery-waiter-registry.js";
import { ResponseWaiterRegistry } from "./response-waiter-registry.js";
import { TeamAgentTaskEventRunner } from "./team-agent-task-event-runner.js";
import { TeamAgentTurnRunner } from "./team-agent-turn-runner.js";

const cleanups: Array<() => void | Promise<void>> = [];
afterEach(async () => {
  while (cleanups.length > 0) await cleanups.pop()!();
  vi.restoreAllMocks();
});

async function harness(
  options: {
    fetchFn?: typeof fetch;
    tenantCount?: number;
    liveExtraction?: boolean;
    taskEventsEnabled?: boolean;
  } = {},
) {
  const dataDir = mkdtempSync(join(tmpdir(), "borg-task-events-"));
  cleanups.push(() => rmSync(dataDir, { recursive: true, force: true }));
  const clock = new ManualClock(Date.now());
  const deliveryWaiters = new DeliveryWaiterRegistry();
  const inboxWaiters = new ResponseWaiterRegistry();
  const fetchFn = vi.fn(
    options.fetchFn ??
      (async () => Response.json({ action: "reply", content: "The report is ready." })),
  );
  const borgs = new Map<string, Borg>();
  const repositories = new Map<string, AgentDeliveryRepository>();
  const terminals = new Map<string, BacklogTerminalService>();
  const activityProjections: Parameters<Borg["activity"]["projectRepliedTurn"]>[0][] = [];
  const llmClient = new FakeLLMClient();
  const open = async (tenant: string, taskEventsEnabled = options.taskEventsEnabled !== false) => {
    const borg = await Borg.open({
      dataDir: join(dataDir, tenant),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new FakeEmbeddingClient(4),
      llmClient,
      liveExtraction: options.liveExtraction ?? false,
      inbox: {
        ...(taskEventsEnabled ? { taskEventsEnabled: true } : {}),
        runner: (context) =>
          new TeamAgentTurnRunner({
            ...context,
            tenant,
            baseUrl: "http://team-agent:8080",
            apiToken: "agent-token",
            timeoutMs: 1000,
            staleMs: 600_000,
            clock,
            fetchFn,
          }),
        taskEventRunner: (context) => {
          repositories.set(tenant, context.deliveries);
          terminals.set(tenant, context.terminal);
          return new TeamAgentTaskEventRunner({
            ...context,
            tenant,
            baseUrl: "http://team-agent:8080",
            apiToken: "agent-token",
            timeoutMs: 1000,
            fetchFn,
            activity: {
              projectRepliedTurn(input) {
                activityProjections.push(input);
                return context.activity.projectRepliedTurn(input);
              },
            },
          });
        },
        onDeliveryAvailable: (sessionId) => deliveryWaiters.notify(tenant, sessionId),
        sessionPredicate: (session) => session?.source_type === "teams_inbox",
        settleMs: 20,
        maxSettleMs: 100,
      } satisfies NonNullable<BorgOpenOptions["inbox"]>,
    });
    borgs.set(tenant, borg);
    borg.entities.ensureSelf("team-agent", { provenance: "config_default_user" });
    return borg;
  };
  const first = await open("alpha");
  if (options.tenantCount === 2) await open("beta");
  const sessionId = createSessionId();
  for (const borg of borgs.values()) {
    const audience = borg.entities.resolveExternal({
      source: "team-agent.conversation",
      externalId: "origin-conversation",
      canonicalName: "Origin room",
      kind: "group",
      provenance: "transport_audience_label",
    });
    borg.sessions.ensure({
      session_id: sessionId,
      source_type: "teams_inbox",
      source_external_id: "origin-conversation",
      label: "Origin room",
      audience_label: "Origin room",
      audience_entity_id: audience,
      conversation_kind: "channel",
      status: "active",
    });
  }
  // Exercise the production contract: only short mutations hold the tenant's chain.
  const chains = new Map<string, Promise<unknown>>();
  const exclusives: boolean[] = [];
  const pool: MemoryPool = {
    listTenantIds: async () => [...borgs.keys()],
    async withTenant(tenant, fn, opts) {
      const borg = borgs.get(tenant);
      if (borg === undefined) throw new Error("tenant unavailable");
      exclusives.push(opts?.exclusive === true);
      if (!opts?.exclusive) return fn(borg);
      const previous = chains.get(tenant) ?? Promise.resolve();
      const work = (async () => {
        await previous;
        return fn(borg);
      })();
      chains.set(
        tenant,
        work.catch(() => undefined),
      );
      return work;
    },
  };
  const server: Server = createServer(
    createMemoryHandler({
      pool,
      token: "secret",
      inboxWaiters,
      deliveryWaiters,
    }),
  );
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  cleanups.push(async () => {
    deliveryWaiters.shutdown();
    inboxWaiters.shutdown();
    await new Promise<void>((resolve) => server.close(() => resolve()));
    for (const borg of borgs.values()) await borg.close();
  });
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const post = (path: string, body: unknown, token = "secret", signal?: AbortSignal) =>
    fetch(`${base}${path}`, {
      method: "POST",
      headers: { "x-borg-token": token, "content-type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  const eventBody = (eventId = "event-1") => ({
    tenant: "alpha",
    sidecar_session_id: sessionId,
    event_id: eventId,
    task_id: "task-1",
    task_version: 3,
    kind: "task_completed",
    occurred_at: "2026-01-01T12:00:00+02:00", // Old task events must never be sealed as stale.
    outcome: {
      status: "succeeded",
      summary: "Created the report",
      artifacts: [{ label: "Report", url: "https://example.com/report" }],
    },
    origin: { source_entry_ids: [] as string[] },
  });
  const claimBody = { tenant: "alpha", sidecar_session_ids: [sessionId], wait_ms: 0 };
  return {
    borg: first,
    borgs,
    sessionId,
    fetchFn,
    llmClient,
    clock,
    deliveryWaiters,
    terminals,
    repositories,
    activityProjections,
    eventBody,
    claimBody,
    post,
    exclusives,
    enqueue: (eventId?: string) => post("/memory/agent-events", eventBody(eventId)),
    async restart(taskEventsEnabled = options.taskEventsEnabled !== false) {
      await borgs.get("alpha")!.close();
      return open("alpha", taskEventsEnabled);
    },
  };
}

describe("agent event and delivery routes", () => {
  it("keeps task enqueue and terminal writes disabled by default despite a configured runner factory", async () => {
    const h = await harness({ taskEventsEnabled: false });
    expect(h.terminals.has("alpha")).toBe(false);
    expect((await h.enqueue()).status).toBe(503);
    expect(h.borg.inbox.listUnansweredTaskEvents(h.sessionId)).toEqual([]);
    await expect(
      h.borg.stream.append(
        {
          kind: "agent_msg",
          content: "Disabled task",
          response_to: {
            kind: "task_event",
            event_id: "event",
            event_entry_id: createStreamEntryId(),
            task_id: "task",
            task_version: 1,
          },
        },
        { session: h.sessionId },
      ),
    ).rejects.toMatchObject({ code: "TASK_EVENT_LANE_DISABLED" });
    expect(h.fetchFn).not.toHaveBeenCalled();
  });

  it("leaves persisted tasks undrained on a restart with the lane disabled", async () => {
    const h = await harness();
    await h.enqueue();
    const disabled = await h.restart(false);
    disabled.inbox.catchUp.start();
    expect(await disabled.inbox.catchUp.tick(h.sessionId)).toMatchObject({ status: "empty" });
    await disabled.inbox.catchUp.stop();
    expect(disabled.inbox.listUnansweredTaskEvents(h.sessionId)).toHaveLength(1);
    expect(h.fetchFn).not.toHaveBeenCalled();
    const enabled = await h.restart(true);
    expect(await enabled.inbox.catchUp.tick(h.sessionId)).toMatchObject({ status: "drained" });
  });

  it("extracts the metadata outcome with audience and event provenance when live extraction is enabled", async () => {
    let pendingUserId: ReturnType<typeof createStreamEntryId>;
    const h = await harness({
      liveExtraction: true,
      fetchFn: async () => {
        const user = await h.borg.stream.append(
          { kind: "user_msg", content: "Unanswered user outside the task window" },
          { session: h.sessionId },
        );
        pendingUserId = user.id;
        return Response.json({ action: "reply", content: "Done." });
      },
    });
    await h.enqueue();
    const taskEvent = h.borg.inbox.listUnansweredTaskEvents(h.sessionId)[0]!;
    h.llmClient.pushResponse((request: LLMCompleteOptions) => {
      const prompt = String(request.messages[0]!.content);
      const reply = prompt
        .split("\n")
        .filter((line) => line.startsWith("{"))
        .map((line) => JSON.parse(line))
        .find((item) => item.kind === "agent_msg");
      expect(reply.task_event_context).toMatchObject({
        source_entry_id: taskEvent.entry.id,
        audience: "(audience routing label) Origin room",
        event: { event_id: "event-1", outcome: { summary: "Created the report" } },
      });
      expect(prompt).not.toContain("Unanswered user outside the task window");
      return {
        text: "",
        input_tokens: 10,
        output_tokens: 20,
        stop_reason: "tool_use",
        tool_calls: [
          {
            id: "episode",
            name: "EmitEpisodeCandidates",
            input: {
              episodes: [
                {
                  title: "Report created",
                  narrative: "I completed the report for Origin room.",
                  source_stream_ids: [reply.id],
                  participants: ["team-agent"],
                  location: null,
                  tags: ["report"],
                  confidence: 0.9,
                  significance: 0.7,
                },
              ],
              relational_slot_updates: [],
            },
          },
        ],
      };
    });
    expect(await h.borg.inbox.catchUp.tick(h.sessionId)).toMatchObject({ status: "drained" });
    const terminal = h.borg.inbox.findTaskEventTerminal(h.sessionId, taskEvent)!;
    const [episode] = await h.borg.episodic.listAll();
    const audienceId = h.borg.sessions.get(h.sessionId)!.audience_entity_id!;
    expect(episode).toMatchObject({
      audience_entity_id: audienceId,
      origin_audience_entity_ids: [audienceId],
      shared: false,
      narrative: "I completed the report for Origin room.",
    });
    expect(episode!.source_stream_ids).toEqual(
      expect.arrayContaining([taskEvent.entry.id, terminal.id]),
    );
    expect(episode!.source_stream_ids).toHaveLength(2);
    expect(
      h.borg.inbox.findTerminalCoveringEntry({ sessionId: h.sessionId, entryId: pendingUserId! }),
    ).toEqual({ status: "pending" });
  });

  it("answers a pending user while an earlier task terminal's ingestion repair keeps failing", async () => {
    const h = await harness();
    await h.enqueue();
    const ingest = vi
      .spyOn(h.terminals.get("alpha")!, "ingestTaskEventTerminal")
      .mockRejectedValue(new Error("task ingestion unavailable"));
    expect(await h.borg.inbox.catchUp.tick(h.sessionId)).toMatchObject({ status: "error" });
    const user = await h.borg.stream.append(
      {
        kind: "user_msg",
        content: "A new request",
        conversation: { type: "channel", name: "Origin room" },
        source_message_key: {
          source_type: "teams_inbox",
          source_external_id: "origin-conversation",
          external_message_id: "new-request",
        },
        metadata: {
          teams_inbox: {
            thread_id: "thread",
            sender: { external_id: "requester", display_name: "Requester", bot: false },
            mentioned: true,
            quotes_bot: false,
          },
        },
      },
      { session: h.sessionId },
    );
    expect(await h.borg.inbox.catchUp.tick(h.sessionId)).toMatchObject({ status: "drained" });
    expect(ingest).toHaveBeenCalledTimes(1);
    expect(
      h.borg.inbox.findTerminalCoveringEntry({ sessionId: h.sessionId, entryId: user.id }).status,
    ).toBe("found");
    expect(h.fetchFn.mock.calls.map(([url]) => String(url))).toEqual([
      "http://team-agent:8080/v1/chat/task-result",
      "http://team-agent:8080/v1/chat/observe",
    ]);
  });

  it("binds ack receipts to claim generations so A's retried ack cannot give B's lease to C", async () => {
    const h = await harness();
    await h.enqueue();
    await h.borg.inbox.catchUp.tick(h.sessionId);
    const claim = async () =>
      (await (await h.post("/memory/agent-deliveries/claim", h.claimBody)).json()) as {
        deliveries: AgentDelivery[];
      };
    const [a] = (await claim()).deliveries;
    const ack = {
      tenant: "alpha",
      delivery_id: a!.delivery_id,
      claim_generation: a!.claim_generation,
      outcome: "failed_retryable",
    };
    expect(await (await h.post("/memory/agent-deliveries/ack", ack)).json()).toEqual({
      status: "acknowledged",
    });
    const [b] = (await claim()).deliveries;
    expect(b!.claim_generation).toBe(a!.claim_generation + 1);
    const replay = await h.post("/memory/agent-deliveries/ack", ack);
    expect(replay.status).toBe(200);
    expect(await replay.json()).toEqual({ status: "acknowledged" });
    expect(await claim()).toEqual({ deliveries: [] });
    const missingGeneration = { tenant: "alpha", delivery_id: a!.delivery_id, outcome: "sent" };
    expect((await h.post("/memory/agent-deliveries/ack", missingGeneration)).status).toBe(400);
  });

  it("authenticates, validates, rejects unknown/non-inbox sessions, and deduplicates per tenant/session", async () => {
    const h = await harness({ tenantCount: 2 });
    expect((await h.post("/memory/agent-events", h.eventBody(), "wrong")).status).toBe(401);
    expect(
      (await h.post("/memory/agent-events", { ...h.eventBody(), task_version: 1.1 })).status,
    ).toBe(400);
    expect(
      (
        await h.post("/memory/agent-events", {
          ...h.eventBody(),
          sidecar_session_id: createSessionId(),
        })
      ).status,
    ).toBe(404);
    const legacy = createSessionId();
    h.borg.sessions.ensure({
      session_id: legacy,
      source_type: "console",
      label: "Console",
      audience_label: "Console",
      conversation_kind: "dm",
    });
    expect(
      (await h.post("/memory/agent-events", { ...h.eventBody(), sidecar_session_id: legacy }))
        .status,
    ).toBe(404);
    const responses = await Promise.all([h.enqueue(), h.enqueue()]);
    const bodies = await Promise.all(
      responses.map((response) => response.json() as Promise<{ status: string; entry_id: string }>),
    );
    expect(responses.map((response) => response.status)).toEqual([200, 200]);
    expect(bodies.map((body) => body.status).sort()).toEqual(["duplicate", "enqueued"]);
    expect(bodies[0]!.entry_id).toBe(bodies[1]!.entry_id);
    const event = h.borg.inbox.listUnansweredTaskEvents(h.sessionId);
    expect(event).toHaveLength(1);
    expect(event[0]!.entry).toMatchObject({
      kind: "internal_event",
      metadata: { task_event: { schema_version: 1, task_version: 3 } },
    });
    expect(
      (await h.post("/memory/agent-events", { ...h.eventBody(), tenant: "beta" })).status,
    ).toBe(200);
    expect(h.borgs.get("beta")!.inbox.listUnansweredTaskEvents(h.sessionId)).toHaveLength(1);
    expect(h.exclusives.every(Boolean)).toBe(true);
  });

  it("calls the originating conversation, stamps one event, projects self + audience, and creates one delivery", async () => {
    const h = await harness();
    const origin = await h.borg.stream.append(
      {
        kind: "user_msg",
        content: "Build it",
        turn_id: "origin-turn",
        metadata: {
          teams_inbox: {
            thread_id: "thread",
            sender: { external_id: "requester", display_name: "Requester", bot: false },
            mentioned: true,
            quotes_bot: false,
          },
        },
      },
      { session: h.sessionId },
    );
    const body = h.eventBody();
    body.origin.source_entry_ids = [origin.id];
    await h.post("/memory/agent-events", body);
    const taskEvent = h.borg.inbox.listUnansweredTaskEvents(h.sessionId)[0]!;
    const ingest = vi.spyOn(h.terminals.get("alpha")!, "ingestTaskEventTerminal");
    expect((await h.borg.inbox.catchUp.tick(h.sessionId)).status).toBe("drained");
    const [url, request] = h.fetchFn.mock.calls[0]!;
    expect(String(url)).toBe("http://team-agent:8080/v1/chat/task-result");
    expect(request).toMatchObject({
      redirect: "error",
      headers: { authorization: "Bearer agent-token" },
    });
    expect(JSON.parse(request!.body as string)).toEqual({
      model: "alpha",
      sidecar_session_id: h.sessionId,
      conversation: { external_id: "origin-conversation", type: "channel", name: "Origin room" },
      event: {
        event_id: "event-1",
        event_entry_id: taskEvent.entry.id,
        task_id: "task-1",
        task_version: 3,
        kind: body.kind,
        occurred_at: body.occurred_at,
        outcome: body.outcome,
      },
      requester: { external_id: "requester", display_name: "Requester" },
    });
    const terminal = h.borg.inbox.findTaskEventTerminal(h.sessionId, taskEvent)!;
    expect(terminal.response_to).toEqual({
      kind: "task_event",
      event_id: "event-1",
      event_entry_id: taskEvent.entry.id,
      task_id: "task-1",
      task_version: 3,
    });
    expect(ingest).toHaveBeenCalledWith(terminal);
    const claim = (await (await h.post("/memory/agent-deliveries/claim", h.claimBody)).json()) as {
      deliveries: AgentDelivery[];
    };
    expect(claim.deliveries).toEqual([
      expect.objectContaining({
        sidecar_session_id: h.sessionId,
        terminal_entry_id: terminal.id,
        task_id: "task-1",
        content: "The report is ready.",
        created_at: new Date(terminal.timestamp).toISOString(),
      }),
    ]);
    const audienceId = h.borg.sessions.get(h.sessionId)!.audience_entity_id!;
    const selfId = h.borg.entities.getSelf()!.id;
    expect(h.activityProjections).toHaveLength(1);
    expect(h.activityProjections[0]!.borgReplied).toMatchObject({
      speakerEntityId: selfId,
      actorEntityId: selfId,
      audienceEntityId: audienceId,
      participantEntityIds: [selfId, audienceId],
      sourceStreamEntryIds: [terminal.id],
    });
    const activities = h.borg.activity.listRecentVisibleOtherSessionEvents({
      currentSessionId: createSessionId(),
      audienceEntityIds: [audienceId],
      sinceMs: 0,
      limit: 10,
      kinds: ["borg_replied"],
    });
    expect(activities).toHaveLength(1);
    expect(h.borg.inbox.listUnansweredTaskEvents(h.sessionId)).toEqual([]);
    await h.enqueue();
    await h.borg.inbox.catchUp.tick(h.sessionId);
    expect(h.fetchFn).toHaveBeenCalledTimes(1);
  });

  it.each([400, 401, 404, 422, 429])(
    "records a deterministic reply for HTTP %s",
    async (status) => {
      const h = await harness({ fetchFn: async () => new Response("rejected", { status }) });
      await h.enqueue();
      await h.borg.inbox.catchUp.tick(h.sessionId);
      const result = (await (
        await h.post("/memory/agent-deliveries/claim", h.claimBody)
      ).json()) as { deliveries: AgentDelivery[] };
      expect(result.deliveries[0]!.content).toBe("Task task-1 finished: Created the report");
      expect(h.borg.inbox.listUnansweredTaskEvents(h.sessionId)).toEqual([]);
    },
  );

  it.each(["empty", "whitespace", "malformed", "silent", "server", "network"])(
    "retries %s responses without a terminal or delivery",
    async (kind) => {
      const h = await harness({
        fetchFn: async () => {
          if (kind === "network") throw new Error("network error");
          if (kind === "server") return new Response("unavailable", { status: 503 });
          if (kind === "malformed") return new Response("{");
          if (kind === "silent") return Response.json({ action: "silent" });
          return Response.json({ action: "reply", content: kind === "empty" ? "" : " \n " });
        },
      });
      await h.enqueue();
      expect((await h.borg.inbox.catchUp.tick(h.sessionId)).status).toBe("error");
      expect(h.borg.inbox.listUnansweredTaskEvents(h.sessionId)).toHaveLength(1);
      expect(await (await h.post("/memory/agent-deliveries/claim", h.claimBody)).json()).toEqual({
        deliveries: [],
      });
      h.fetchFn.mockResolvedValue(Response.json({ action: "reply", content: "Recovered" }));
      expect((await h.borg.inbox.catchUp.tick(h.sessionId)).status).toBe("drained");
    },
  );

  it("wakes long polls, atomically leases, retries failures, and keeps sent acknowledgements idempotent", async () => {
    const h = await harness({ tenantCount: 2 });
    const pending = h.post("/memory/agent-deliveries/claim", { ...h.claimBody, wait_ms: 2000 });
    await vi.waitFor(() => expect(h.deliveryWaiters.size()).toBe(1));
    await h.enqueue();
    await h.borg.inbox.catchUp.tick(h.sessionId);
    const { deliveries } = (await (await pending).json()) as { deliveries: AgentDelivery[] };
    expect(deliveries).toHaveLength(1);
    const id = deliveries[0]!.delivery_id;
    expect(await (await h.post("/memory/agent-deliveries/claim", h.claimBody)).json()).toEqual({
      deliveries: [],
    });
    expect(
      (
        await h.post("/memory/agent-deliveries/ack", {
          tenant: "beta",
          delivery_id: id,
          claim_generation: 1,
          outcome: "sent",
        })
      ).status,
    ).toBe(404);
    await h.post("/memory/agent-deliveries/ack", {
      tenant: "alpha",
      delivery_id: id,
      claim_generation: 1,
      outcome: "failed_retryable",
      error: "connector",
    });
    const concurrent = await Promise.all([
      h.post("/memory/agent-deliveries/claim", h.claimBody),
      h.post("/memory/agent-deliveries/claim", h.claimBody),
    ]);
    const claimed = await Promise.all(
      concurrent.map((res) => res.json() as Promise<{ deliveries: AgentDelivery[] }>),
    );
    expect(claimed.flatMap((body) => body.deliveries)).toHaveLength(1);
    h.clock.advance(120_001);
    expect(
      await (await h.post("/memory/agent-deliveries/claim", h.claimBody)).json(),
    ).toMatchObject({ deliveries: [{ delivery_id: id }] });
    const ack = {
      tenant: "alpha",
      delivery_id: id,
      claim_generation: 3,
      outcome: "sent",
      teams_message_id: "teams-id",
    };
    const ackResponse = await h.post("/memory/agent-deliveries/ack", ack);
    expect(ackResponse.status).toBe(200);
    expect(await ackResponse.json()).toEqual({
      status: "acknowledged",
    });
    expect(
      await (
        await h.post("/memory/agent-deliveries/ack", { ...ack, outcome: "failed_retryable" })
      ).json(),
    ).toEqual({ status: "acknowledged" });
    expect(await (await h.post("/memory/agent-deliveries/claim", h.claimBody)).json()).toEqual({
      deliveries: [],
    });
    expect(h.deliveryWaiters.size()).toBe(0);
  });

  it("times out an empty claim and removes waiters on disconnect and shutdown", async () => {
    const h = await harness();
    expect(
      await (await h.post("/memory/agent-deliveries/claim", { ...h.claimBody, wait_ms: 5 })).json(),
    ).toEqual({ deliveries: [] });
    const controller = new AbortController();
    const pending = h.post(
      "/memory/agent-deliveries/claim",
      { ...h.claimBody, wait_ms: 2000 },
      "secret",
      controller.signal,
    );
    const rejected = expect(pending).rejects.toThrow();
    await vi.waitFor(() => expect(h.deliveryWaiters.size()).toBe(1));
    controller.abort();
    await rejected;
    await vi.waitFor(() => expect(h.deliveryWaiters.size()).toBe(0));
    const shutdown = h.post("/memory/agent-deliveries/claim", { ...h.claimBody, wait_ms: 2000 });
    await vi.waitFor(() => expect(h.deliveryWaiters.size()).toBe(1));
    h.deliveryWaiters.shutdown();
    expect(await (await shutdown).json()).toEqual({ deliveries: [] });
    expect(h.deliveryWaiters.size()).toBe(0);
  });

  it("long-polls until an existing lease expires without a new delivery notification", async () => {
    const h = await harness();
    vi.spyOn(h.clock, "now").mockImplementation(() => Date.now());
    await h.enqueue();
    await h.borg.inbox.catchUp.tick(h.sessionId);
    const first = (await (
      await h.post("/memory/agent-deliveries/claim", { ...h.claimBody, lease_ms: 250 })
    ).json()) as { deliveries: AgentDelivery[] };
    const pending = h.post("/memory/agent-deliveries/claim", { ...h.claimBody, wait_ms: 2000 });
    await vi.waitFor(() => expect(h.deliveryWaiters.size()).toBe(1));
    expect(await (await pending).json()).toEqual({
      deliveries: [{ ...first.deliveries[0], claim_generation: 2 }],
    });
    expect(h.deliveryWaiters.size()).toBe(0);
  });

  it.each([
    ["dm", "personal"],
    ["thread", "groupChat"],
    ["channel", "channel"],
  ] as const)(
    "maps %s sessions to %s and preserves null requester when provenance is unavailable",
    async (kind, type) => {
      const h = await harness();
      h.borg.sessions.ensure({ ...h.borg.sessions.get(h.sessionId)!, conversation_kind: kind });
      const event = h.eventBody();
      event.origin.source_entry_ids = [createStreamEntryId()];
      await h.post("/memory/agent-events", event);
      await h.borg.inbox.catchUp.tick(h.sessionId);
      expect(JSON.parse(h.fetchFn.mock.calls[0]![1]!.body as string)).toMatchObject({
        requester: null,
        conversation: { external_id: "origin-conversation", type },
      });
    },
  );

  it("drains task-only sessions on startup, oldest first, and repairs task activity through maintenance", async () => {
    const h = await harness();
    await h.enqueue("first");
    await h.enqueue("second");
    const reopened = await h.restart();
    reopened.inbox.catchUp.start();
    await vi.waitFor(() => expect(h.fetchFn).toHaveBeenCalledTimes(2));
    await reopened.inbox.catchUp.stop();
    expect(
      h.fetchFn.mock.calls.map(([, init]) => JSON.parse(init!.body as string).event.event_id),
    ).toEqual(["first", "second"]);
    const event = reopened.inbox.listUnansweredTaskEvents(h.sessionId);
    expect(event).toEqual([]);
    // Model a terminal that committed before its activity projection or delivery.
    await h.enqueue("unprojected");
    const pending = reopened.inbox.listUnansweredTaskEvents(h.sessionId)[0]!;
    await h.terminals.get("alpha")!.appendTaskEventTerminal({
      sessionId: h.sessionId,
      responseTo: {
        kind: "task_event",
        event_id: pending.event.event_id,
        event_entry_id: pending.entry.id,
        task_id: pending.event.task_id,
        task_version: pending.event.task_version,
      },
      content: "Recovered result",
    });
    const dry = reopened.inbox.reconcileReplyActivity({ dryRun: true });
    expect(dry).toMatchObject({ already_recorded: 2, inserted: 1 });
    expect(reopened.inbox.reconcileReplyActivity({ dryRun: false })).toMatchObject({ inserted: 1 });
    expect(reopened.inbox.reconcileReplyActivity({ dryRun: false })).toMatchObject({
      already_recorded: 3,
      inserted: 0,
    });
  });

  it("repairs a committed terminal without a delivery after restart, without another HTTP call", async () => {
    const h = await harness();
    await h.enqueue();
    const taskEvent = h.borg.inbox.listUnansweredTaskEvents(h.sessionId)[0]!;
    vi.spyOn(h.repositories.get("alpha")!, "create").mockImplementationOnce(() => {
      throw new Error("simulated crash");
    });
    expect((await h.borg.inbox.catchUp.tick(h.sessionId)).status).toBe("error");
    const terminal = h.borg.inbox.findTaskEventTerminal(h.sessionId, taskEvent)!;
    expect(terminal).not.toBeNull();
    const reopened = await h.restart();
    reopened.inbox.catchUp.start();
    await vi.waitFor(() =>
      expect(h.repositories.get("alpha")!.hasTerminal(h.sessionId, terminal.id)).toBe(true),
    );
    await reopened.inbox.catchUp.stop();
    expect(h.fetchFn).toHaveBeenCalledTimes(1);
    const claim = (await (await h.post("/memory/agent-deliveries/claim", h.claimBody)).json()) as {
      deliveries: AgentDelivery[];
    };
    expect(claim.deliveries).toHaveLength(1);
    expect(claim.deliveries[0]!.terminal_entry_id).toBe(terminal.id);
    expect(await (await h.enqueue()).json()).toMatchObject({ status: "duplicate" });
  });
});
