import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StreamWriter } from "../../stream/index.js";
import {
  appendInternalFailureEvent,
  appendOpenQuestionHookFailureEvent,
} from "./review-open-question-hook.js";

describe("failure event reporting", () => {
  const dirs: string[] = [];
  afterEach(() => {
    vi.restoreAllMocks();
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  it.each(["synchronous", "asynchronous"])(
    "logs and retains a %s append failure without rejecting the live caller",
    async (mode) => {
      const dataDir = mkdtempSync(join(tmpdir(), "borg-failure-report-"));
      dirs.push(dataDir);
      const writer = new StreamWriter({ dataDir });
      const failure = new Error("diagnostic append failed");
      vi.spyOn(writer, "append").mockImplementation(() => {
        if (mode === "synchronous") throw failure;
        return Promise.reject(failure);
      });
      const log = vi.spyOn(console, "error").mockImplementation(() => {});
      await expect(
        appendInternalFailureEvent(writer, "test", new Error("original failure")),
      ).resolves.toBeUndefined();
      await expect(
        appendOpenQuestionHookFailureEvent(writer, "test", new Error("original failure")),
      ).resolves.toBeUndefined();
      expect(log).toHaveBeenCalledTimes(2);
      await expect(writer.failureReports.flush()).rejects.toMatchObject({
        errors: [failure, failure],
      });
      await expect(writer.failureReports.flush()).rejects.toMatchObject({
        errors: [failure, failure],
      });
      writer.close();
    },
  );
});
