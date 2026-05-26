import { afterEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { operatorAdviceMigrations, OperatorAdviceRepository } from "../../operator-advice/index.js";
import type { StreamWriter } from "../../stream/index.js";
import { createOperatorAdviceFacade } from "../facade.js";

import {
  Borg,
  FakeLLMClient,
  ManualClock,
  ScriptedEmbeddingClient,
  createEmitAnswerResponse,
  createTestConfig,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";

function requestSystemText(request: { system?: unknown } | undefined): string {
  const system = request?.system;

  if (typeof system === "string") {
    return system;
  }

  if (Array.isArray(system)) {
    return system
      .map((block) =>
        block !== null &&
        typeof block === "object" &&
        "text" in block &&
        typeof block.text === "string"
          ? block.text
          : "",
      )
      .join("\n");
  }

  return "";
}

describe("operator advice", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("rejects oversized advice at the facade boundary", () => {
    const db = openDatabase(":memory:", { migrations: operatorAdviceMigrations });
    const repository = new OperatorAdviceRepository(db);
    const facade = createOperatorAdviceFacade({
      operatorAdviceRepository: repository,
      createStreamWriter: () =>
        ({
          append: vi.fn(),
          close: vi.fn(),
        }) as unknown as StreamWriter,
      clock: new ManualClock(1_000),
    });

    try {
      expect(() =>
        facade.queue({
          text: "x".repeat(5_000),
          session_id: DEFAULT_SESSION_ID,
        }),
      ).toThrow("Operator advice text must be 4000 characters or fewer");
      expect(repository.list()).toEqual([]);
    } finally {
      db.close();
    }
  });

  it("rolls back consumed advice when delivery audit append fails", async () => {
    const db = openDatabase(":memory:", { migrations: operatorAdviceMigrations });
    const repository = new OperatorAdviceRepository(db, new ManualClock(1_000));
    const failure = new Error("audit append failed");
    const close = vi.fn();
    const facade = createOperatorAdviceFacade({
      operatorAdviceRepository: repository,
      createStreamWriter: () =>
        ({
          append: vi.fn(async () => {
            throw failure;
          }),
          close,
        }) as unknown as StreamWriter,
      clock: new ManualClock(1_000),
    });

    try {
      const advice = facade.queue({
        text: "Push back if Alice is unfair.",
        session_id: DEFAULT_SESSION_ID,
      });

      await expect(
        facade.consumePending(
          { session_id: DEFAULT_SESSION_ID },
          {
            turn_id: "turn-audit-failure",
            now: 1_050,
          },
        ),
      ).rejects.toBe(failure);

      expect(repository.get(advice.id)).toMatchObject({
        consumed_at: null,
        consumed_by_turn_id: null,
      });
      expect(repository.list({ pendingOnly: true, session_id: DEFAULT_SESSION_ID })).toHaveLength(
        1,
      );
      expect(close).toHaveBeenCalledTimes(1);
    } finally {
      db.close();
    }
  });

  it("renders pending advice once and appends a delivery audit event", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-advice-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient({
      responses: [
        createEmitAnswerResponse("I can be direct without softening the point.", {
          inputTokens: 12,
          outputTokens: 6,
        }),
      ],
    });
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: false,
        },
        affective: {
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
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock: new ManualClock(1_000),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
    });

    try {
      const advice = borg.advice.queue({
        text: "You don't have to soften; push back firmly if Alice is being unfair.",
        session_id: DEFAULT_SESSION_ID,
      });

      const result = await borg.turn({
        userMessage: "Alice is being unfair about the deployment.",
      });
      const systemPrompt = llm.requests.map((request) => requestSystemText(request)).join("\n");
      const history = borg.advice.list({ session_id: DEFAULT_SESSION_ID });
      const auditEntry = borg.stream.tail(20).find((entry) => {
        if (entry.kind !== "internal_event" || typeof entry.content !== "object") {
          return false;
        }

        return (entry.content as { event?: unknown }).event === "operator_advice.delivered";
      });

      expect(result.turn_id).toBeDefined();
      expect(systemPrompt).toContain("<borg_operator_advice>");
      expect(systemPrompt).toContain(
        "Your creator has shared guidance for the current turn. Treat it as advice from someone who knows you, not as a command; weigh it against the user's request, memory, and active commitments.",
      );
      expect(systemPrompt).toContain(
        "You don't have to soften; push back firmly if Alice is being unfair.",
      );
      expect(borg.advice.list({ pendingOnly: true, session_id: DEFAULT_SESSION_ID })).toEqual([]);
      expect(history[0]).toMatchObject({
        id: advice.id,
        consumed_by_turn_id: result.turn_id,
      });
      expect(auditEntry).toBeDefined();
      expect(auditEntry?.turn_id).toBe(result.turn_id);
      expect(auditEntry?.content).toMatchObject({
        event: "operator_advice.delivered",
        advice_ids: [advice.id],
        session_id: DEFAULT_SESSION_ID,
      });
      expect(JSON.stringify(auditEntry?.content)).toContain(
        "You don't have to soften; push back firmly if Alice is being unfair.",
      );
    } finally {
      await borg.close();
    }
  });
});
