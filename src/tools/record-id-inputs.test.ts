import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { buildToolDispatcher, type BuildToolDispatcherOptions } from "../borg/tools-setup.js";
import { openDatabase, type SqliteDatabase } from "../storage/sqlite/index.js";
import { StreamWriter } from "../stream/index.js";
import { ManualClock } from "../util/clock.js";
import { ToolError } from "../util/errors.js";
import { DEFAULT_SESSION_ID, type EntityId, type SessionId } from "../util/ids.js";
import { toToolInputSchema } from "../llm/index.js";
import { createOutboundPostTool } from "./internal/outbound-post.js";
import { ToolDispatcher, type ToolDefinition } from "./dispatcher.js";
import { toAnthropicToolDefinitions, SHORT_HANDLE_MENU_GUIDANCE } from "./anthropic.js";
import { RECORD_ID_STORES } from "./record-id-resolver.js";
import { resolveFinalizerNonTerminalTools } from "../cognition/deliberation/autonomous-finalizer-tools.js";

const ids = Object.fromEntries(
  Object.entries(RECORD_ID_STORES).map(([kind, store]) => [
    kind,
    store.helper.prefix + "test000000000001",
  ]),
) as Record<keyof typeof RECORD_ID_STORES, string>;
const range = { since: "2026-01-01T00:00:00Z", until: "2026-12-01T00:00:00Z" };
const cases = [
  ["episodic.get", { episode_id: ids.episode }, 1],
  ["stream.get", { stream_entry_id: ids.stream_entry }, 1],
  ["semantic.walk", { node_id: ids.semantic_node, relation: "supports" }, 1],
  [
    "openQuestions.create",
    {
      question: "合成の問い",
      related_episode_ids: [ids.episode],
      related_semantic_node_ids: [ids.semantic_node],
    },
    2,
  ],
  [
    "openQuestions.resolve",
    {
      open_question_id: ids.open_question,
      resolution_note: "合成の証拠",
      resolution_evidence_episode_ids: [ids.episode],
      resolution_evidence_stream_entry_ids: [ids.stream_entry],
    },
    3,
  ],
  [
    "openQuestions.correctResolution",
    {
      open_question_id: ids.open_question,
      correction_note: "合成の訂正",
      evidence: [
        { id: ids.episode, basis: "own_read" },
        { id: ids.stream_entry, basis: "reported", reported_by: "合成の話者" },
      ],
    },
    3,
  ],
  ["openQuestions.ruminations", { ...range, open_question_id: ids.open_question }, 1],
  ["goals.retire", { goal_id: ids.goal, disposition: "superseded", reason: "合成の理由" }, 1],
  [
    "goals.block",
    {
      goal_id: ids.goal,
      blocker: { kind: "goal", goal_id: ids.goal },
      attempt_status: "attempted_unavailable",
      reason: "合成の理由",
      attempt_evidence: { kind: "stream_entry", id: ids.stream_entry },
    },
    3,
  ],
  [
    "goals.block",
    {
      goal_id: ids.goal,
      blocker: { kind: "entity", entity_id: ids.entity },
      attempt_status: "attempted_unavailable",
      reason: "合成の理由",
      attempt_evidence: { kind: "scheduled_wake", id: ids.scheduled_wake },
    },
    3,
  ],
  [
    "goals.block",
    {
      goal_id: ids.goal,
      blocker: { kind: "until", until: 123456 },
      attempt_status: "attempted_unavailable",
      reason: "合成の理由",
      attempt_evidence: { kind: "executive_step_outcome", id: ids.executive_step },
    },
    2,
  ],
  [
    "goals.block",
    {
      goal_id: ids.goal,
      blocker: { kind: "until", until: 123456 },
      attempt_status: "attempted_unavailable",
      reason: "合成の理由",
      attempt_evidence: { kind: "resolved_open_question", id: ids.open_question },
    },
    2,
  ],
  ["goals.unblock", { goal_id: ids.goal, reason: "合成の理由" }, 1],
  ["scheduledWakes.cancel", { scheduled_wake_id: ids.scheduled_wake }, 1],
  [
    "identityEvents.listForCognition",
    { recordType: "semantic_edge", recordId: ids.semantic_edge },
    1,
  ],
  ["ownRecords.list", { ...range, session_id: ids.session }, 1],
  ["outbound.post", { target_session_id: ids.session, instruction: "合成の指示" }, 1],
] as const;

function shorten(value: unknown, length = 4): unknown {
  if (Array.isArray(value)) return value.map((item) => shorten(item, length));
  if (value !== null && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, shorten(item, length)]),
    );
  if (typeof value === "string" && Object.values(ids).includes(value))
    return value.slice(0, value.lastIndexOf("_") + 1 + length);
  return value;
}

describe("internal tool record ID boundary", () => {
  let db: SqliteDatabase;
  let dir: string;
  let dispatcher: ToolDispatcher;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "borg-handle-tools-"));
    db = openDatabase(":memory:");
    for (const [kind, store] of Object.entries(RECORD_ID_STORES)) {
      const timestamp =
        "timestamp" in store ? `, "${store.timestamp}" INTEGER DEFAULT 1700000000000` : "";
      db.exec(`CREATE TABLE "${store.table}" ("${store.column}" TEXT PRIMARY KEY${timestamp})`);
      db.prepare(`INSERT INTO "${store.table}" ("${store.column}") VALUES (?)`).run(
        ids[kind as keyof typeof ids],
      );
    }
    dispatcher = buildToolDispatcher({
      db,
      dataDir: dir,
      clock: new ManualClock(1000),
      // This inventory test replaces invocation; real repository behavior is
      // covered by the internal-tools and correction integration tests.
      createStreamWriter: (sessionId: SessionId = DEFAULT_SESSION_ID) =>
        new StreamWriter({ dataDir: dir, sessionId }),
    } as unknown as BuildToolDispatcherOptions);
    dispatcher.register(createOutboundPostTool({} as never));
  });
  afterEach(() => {
    db.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it.each(cases)(
    "%s preserves full-ID inputs and echoes every resolved field (%#)",
    async (name, fullInput, count) => {
      const tool = dispatcher.getDefinition(`tool.${name}`)!;
      const invoke = vi.fn(async (_input: unknown, _context: unknown) => ({ accepted: true }));
      tool.invoke = invoke;
      tool.outputSchema = z.object({ accepted: z.boolean() });
      const full = await dispatcher.dispatch({
        toolName: tool.name,
        input: fullInput,
        origin: "autonomous",
      });
      expect(full).toMatchObject({ ok: true, output: { accepted: true } });
      expect((full as { output: unknown }).output).toEqual({ accepted: true });
      const parsed = tool.inputSchema.parse(fullInput);
      expect(invoke.mock.calls[0]![0]).toEqual(parsed);
      for (const length of [4, 12]) {
        const input = shorten(fullInput, length);
        const before = structuredClone(input);
        const result = await dispatcher.dispatch({
          toolName: tool.name,
          input,
          origin: "autonomous",
          sessionId: DEFAULT_SESSION_ID,
          audienceEntityId: "ent_else000000000000" as EntityId,
          currentSenderBorgRole: null,
        });
        expect(result.ok, JSON.stringify(result)).toBe(true);
        const resolutions = (
          result as { output: { id_resolutions: { input: string; resolved_id: string }[] } }
        ).output.id_resolutions;
        expect(resolutions).toHaveLength(count);
        expect(
          resolutions.every(
            (item) => item.resolved_id.startsWith(item.input) && item.resolved_id !== item.input,
          ),
        ).toBe(true);
        expect(invoke.mock.calls.at(-1)![0]).toEqual(parsed);
        expect(input).toEqual(before);
      }
    },
  );

  it("pins the inventory and adds scoped guidance exactly once to every composed menu", () => {
    const tools = dispatcher.listTools("autonomous");
    const changed = tools.filter((tool) => tool.recordIdInputs !== undefined);
    expect(changed.map((tool) => tool.name).sort()).toEqual(
      [...new Set(cases.map(([name]) => `tool.${name}`))].sort(),
    );
    expect(changed.flatMap((tool) => tool.recordIdInputs!).length).toBe(21);
    const structuredOutput: ToolDefinition = {
      name: "EmitSyntheticReflection",
      description: "Synthetic structured reflection output.",
      inputSchema: z.object({ goal_id: z.string() }),
      outputSchema: z.object({}),
      allowedOrigins: ["autonomous", "deliberator"],
      writeScope: "write",
      invoke: async () => ({}),
    };
    const menus = [
      tools,
      dispatcher.listTools("deliberator"),
      ...(["autonomous", "user", "directed_outbound"] as const).map((turnOrigin) => [
        structuredOutput,
        ...resolveFinalizerNonTerminalTools({ dispatcher, turnOrigin }),
      ]),
    ];
    for (const menu of menus) {
      const baseline = menu.map((tool) => ({
        name: tool.name,
        description: tool.description,
        inputSchema: toToolInputSchema(tool.inputSchema),
      }));
      const actual = toAnthropicToolDefinitions(menu);
      const after = JSON.stringify(actual);
      const before = JSON.stringify(baseline);
      expect(after.split(SHORT_HANDLE_MENU_GUIDANCE)).toHaveLength(2);
      expect(after.length - before.length).toBe(SHORT_HANDLE_MENU_GUIDANCE.length + 1);
      expect(actual.map((tool) => tool.inputSchema)).toEqual(
        baseline.map((tool) => tool.inputSchema),
      );
      expect(actual.find((tool) => tool.name === structuredOutput.name)?.description).toBe(
        menu.includes(structuredOutput) ? structuredOutput.description : undefined,
      );
    }
    expect(toAnthropicToolDefinitions([structuredOutput])[0]!.description).toBe(
      structuredOutput.description,
    );
  });

  it.each(cases.filter(([name]) => name !== "openQuestions.create"))(
    "%s passes unknown handles through its schema and native checks (%#)",
    async (name, fullInput) => {
      const tool = dispatcher.getDefinition(`tool.${name}`)!;
      const invoke = vi.fn(async () => ({ accepted: true }));
      tool.invoke = invoke;
      tool.outputSchema = z.object({ accepted: z.boolean() });
      const input = JSON.parse(JSON.stringify(shorten(fullInput)).replaceAll("_test", "_none"));
      const absent = JSON.parse(
        JSON.stringify(fullInput).replaceAll("test000000000001", "none000000000000"),
      );
      const result = await dispatcher.dispatch({
        toolName: tool.name,
        input,
        origin: "autonomous",
      });
      expect(result).toMatchObject({ ok: true, output: { accepted: true } });
      expect(invoke).toHaveBeenCalledWith(tool.inputSchema.parse(absent), expect.any(Object));
      // A known-absent address is not a successful resolution to a stored record.
      expect(result.ok && result.output).not.toHaveProperty("id_resolutions");
    },
  );

  it("does not invoke a write on ambiguous or too-short handles and retains resolution echoes on errors", async () => {
    db.prepare("INSERT INTO open_questions (id) VALUES (?)").run("oq_test000000000002");
    const tool = dispatcher.getDefinition("tool.openQuestions.correctResolution")!;
    const invoke = vi.fn();
    tool.invoke = invoke;
    const input = {
      open_question_id: "oq_test",
      correction_note: "synthetic",
      evidence: [{ id: ids.episode, basis: "own_read" }],
    };
    const ambiguous = await dispatcher.dispatch({
      toolName: tool.name,
      input,
      origin: "autonomous",
    });
    expect(ambiguous).toMatchObject({
      ok: false,
      errorCode: "ambiguous_id_prefix",
      error: expect.stringContaining('"omitted_count":0'),
    });
    expect(
      await dispatcher.dispatch({
        toolName: tool.name,
        input: { ...input, open_question_id: "oq_tes" },
        origin: "autonomous",
      }),
    ).toMatchObject({ ok: false, errorCode: "invalid_id_prefix" });
    expect(invoke).not.toHaveBeenCalled();
    invoke.mockRejectedValueOnce(
      new ToolError("native citation check", {
        code: "OPEN_QUESTION_CITATION_NOT_FOUND",
      }),
    );
    const missingCitation = await dispatcher.dispatch({
      toolName: tool.name,
      input: {
        ...input,
        open_question_id: ids.open_question,
        evidence: [{ id: "ep_none", basis: "own_read" }],
      },
      origin: "autonomous",
    });
    expect(missingCitation).toMatchObject({
      ok: false,
      errorCode: "OPEN_QUESTION_CITATION_NOT_FOUND",
    });
    expect(invoke).toHaveBeenCalledWith(
      expect.objectContaining({ evidence: [{ id: "ep_none000000000000", basis: "own_read" }] }),
      expect.any(Object),
    );
    const resolvedThenFailed = await dispatcher.dispatch({
      toolName: tool.name,
      input: {
        ...input,
        open_question_id: ids.open_question,
        evidence: [{ id: "ep_test", basis: "own_read" }],
      },
      origin: "deliberator",
    });
    expect(resolvedThenFailed).toMatchObject({ ok: false }); // origin guard remains first
    expect(invoke).toHaveBeenCalledOnce();
    const ep = dispatcher.getDefinition("tool.episodic.get")!;
    ep.invoke = async () => {
      throw new Error("synthetic failure after resolution");
    };
    const failed = await dispatcher.dispatch({
      toolName: ep.name,
      input: { episode_id: "ep_test" },
      origin: "autonomous",
    });
    expect(failed).toMatchObject({
      ok: false,
      error: expect.stringContaining(`"resolved_id":"${ids.episode}"`),
    });
  });

  it("rejects malformed non-ID fields even when the handle is missing", async () => {
    const tool = dispatcher.getDefinition("tool.goals.retire")!;
    expect(
      await dispatcher.dispatch({
        toolName: tool.name,
        input: { goal_id: "goal_none", disposition: "done", reason: "" },
        origin: "autonomous",
      }),
    ).toMatchObject({ ok: false, errorCode: "TOOL_INPUT_INVALID" });
  });
});
