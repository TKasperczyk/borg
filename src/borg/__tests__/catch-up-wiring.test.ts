import { afterEach, describe, expect, it, vi } from "vitest";

import {
  Borg,
  FakeLLMClient,
  ScriptedEmbeddingClient,
  createTestConfig,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";

describe("Borg chat response catch-up wiring", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("composes onStreamAppend with the catch-up wake hook", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-catch-up-wiring-"));
    tempDirs.push(tempDir);
    const observerError = new Error("observer failed");
    const callerObserver = vi.fn(() => {
      throw observerError;
    });
    vi.spyOn(console, "error").mockImplementation(() => {});

    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
      }),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
      onStreamAppend: callerObserver,
    });

    try {
      const catchUpOnAppend = vi.spyOn(borg.inbox.catchUp, "onAppend");

      await borg.stream.append({
        kind: "user_msg",
        content: "queued",
      });

      expect(callerObserver).toHaveBeenCalledTimes(1);
      expect(catchUpOnAppend).toHaveBeenCalledTimes(1);
    } finally {
      await borg.close();
    }
  });
});
