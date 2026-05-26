import { afterEach, describe, expect, it, vi } from "vitest";

import type { SessionsRepository } from "../../sessions/index.js";
import type { StreamWriter } from "../../stream/index.js";
import { StreamError } from "../../util/errors.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";

import {
  Borg,
  FakeLLMClient,
  ScriptedEmbeddingClient,
  borgInternals,
  createTestConfig,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";

describe("sessions facade", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  async function openTestBorg(): Promise<Borg> {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-sessions-"));
    tempDirs.push(tempDir);

    return Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        offline: {
          consolidator: { enabled: false },
          reflector: { enabled: false },
          curator: { enabled: false },
          overseer: { enabled: false },
          reviewResolver: { enabled: false },
          ruminator: { enabled: false },
          selfNarrator: { enabled: false },
          proceduralSynthesizer: { enabled: false },
          beliefReviser: { enabled: false },
          semanticExtractor: { enabled: false },
        },
      }),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });
  }

  function ensureDefaultSession(borg: Borg): void {
    borg.sessions.ensure({
      session_id: DEFAULT_SESSION_ID,
      source_type: "demo",
      label: "demo",
      audience_label: "alice",
      conversation_kind: "demo",
    });
  }

  it("sets participation policy and appends an audit event", async () => {
    const borg = await openTestBorg();

    try {
      ensureDefaultSession(borg);

      const updated = await borg.sessions.setParticipationPolicy(DEFAULT_SESSION_ID, "observing", {
        reason: "borg is dominating the thread",
      });
      const auditEntry = borg.stream.tail(5).find((entry) => {
        return (
          entry.kind === "internal_event" &&
          typeof entry.content === "object" &&
          entry.content !== null &&
          (entry.content as { event?: unknown }).event === "participation_policy.changed"
        );
      });

      expect(updated.participation_policy).toBe("observing");
      expect(borg.sessions.get(DEFAULT_SESSION_ID)?.participation_policy).toBe("observing");
      expect(auditEntry?.turn_id).toBeUndefined();
      expect(auditEntry?.content).toEqual({
        event: "participation_policy.changed",
        session_id: DEFAULT_SESSION_ID,
        previous: "active",
        next: "observing",
        reason: "borg is dominating the thread",
        operator: true,
      });
    } finally {
      await borg.close();
    }
  });

  it("rolls participation policy back when audit append fails", async () => {
    const borg = await openTestBorg();
    const failure = new Error("audit append failed");
    const close = vi.fn();

    try {
      ensureDefaultSession(borg);
      await borg.sessions.setParticipationPolicy(DEFAULT_SESSION_ID, "muted");

      const internal = borgInternals<{
        deps: {
          createStreamWriter: (sessionId: typeof DEFAULT_SESSION_ID) => StreamWriter;
        };
      }>(borg);
      internal.deps.createStreamWriter = () =>
        ({
          append: vi.fn(async () => {
            throw failure;
          }),
          close,
        }) as unknown as StreamWriter;

      await expect(
        borg.sessions.setParticipationPolicy(DEFAULT_SESSION_ID, "paused", {
          reason: "pause during incident review",
        }),
      ).rejects.toBe(failure);
      expect(borg.sessions.get(DEFAULT_SESSION_ID)?.participation_policy).toBe("muted");
      expect(close).toHaveBeenCalledTimes(1);
    } finally {
      await borg.close();
    }
  });

  it("keeps participation policy when audit append committed but index update failed", async () => {
    const borg = await openTestBorg();
    const failure = new StreamError("stream index update failed after committed append", {
      code: "STREAM_INDEX_UPDATE_FAILED",
    });
    const append = vi.fn(async () => {
      throw failure;
    });
    const close = vi.fn();

    try {
      ensureDefaultSession(borg);

      const internal = borgInternals<{
        deps: {
          createStreamWriter: (sessionId: typeof DEFAULT_SESSION_ID) => StreamWriter;
        };
      }>(borg);
      internal.deps.createStreamWriter = () =>
        ({
          append,
          close,
        }) as unknown as StreamWriter;

      await expect(
        borg.sessions.setParticipationPolicy(DEFAULT_SESSION_ID, "paused", {
          reason: "pause during index incident",
        }),
      ).rejects.toBe(failure);
      expect(borg.sessions.get(DEFAULT_SESSION_ID)?.participation_policy).toBe("paused");
      expect(append).toHaveBeenCalledWith({
        kind: "internal_event",
        content: {
          event: "participation_policy.changed",
          session_id: DEFAULT_SESSION_ID,
          previous: "active",
          next: "paused",
          reason: "pause during index incident",
          operator: true,
        },
      });
      expect(close).toHaveBeenCalledTimes(1);
    } finally {
      await borg.close();
    }
  });

  it("chains audit append and DB rollback failures", async () => {
    const borg = await openTestBorg();
    const appendFailure = new Error("audit append failed before commit");
    const rollbackFailure = new Error("rollback DB update failed");
    const close = vi.fn();

    try {
      ensureDefaultSession(borg);
      await borg.sessions.setParticipationPolicy(DEFAULT_SESSION_ID, "muted");

      const internal = borgInternals<{
        deps: {
          createStreamWriter: (sessionId: typeof DEFAULT_SESSION_ID) => StreamWriter;
          sessionsRepository: SessionsRepository;
        };
      }>(borg);
      internal.deps.createStreamWriter = () =>
        ({
          append: vi.fn(async () => {
            throw appendFailure;
          }),
          close,
        }) as unknown as StreamWriter;

      const originalSetParticipationPolicy =
        internal.deps.sessionsRepository.setParticipationPolicy.bind(
          internal.deps.sessionsRepository,
        );
      vi.spyOn(internal.deps.sessionsRepository, "setParticipationPolicy").mockImplementation(
        (sessionId, policy, options) => {
          if (policy === "muted") {
            throw rollbackFailure;
          }

          return originalSetParticipationPolicy(sessionId, policy, options);
        },
      );

      let thrown: unknown;
      try {
        await borg.sessions.setParticipationPolicy(DEFAULT_SESSION_ID, "paused", {
          reason: "pause during incident review",
        });
      } catch (error) {
        thrown = error;
      }

      expect(thrown).toBeInstanceOf(Error);
      expect((thrown as Error).message).toContain("Failed to roll back participation policy");
      expect((thrown as Error).message).toContain("audit append failed before commit");
      expect((thrown as Error).message).toContain("rollback DB update failed");
      expect((thrown as Error).cause).toBe(appendFailure);
      expect(borg.sessions.get(DEFAULT_SESSION_ID)?.participation_policy).toBe("paused");
      expect(close).toHaveBeenCalledTimes(1);
    } finally {
      await borg.close();
    }
  });
});
