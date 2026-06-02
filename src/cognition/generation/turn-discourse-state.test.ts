import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { StreamReader, StreamWriter } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, createStreamEntryId } from "../../util/ids.js";
import { TurnDiscourseStateService } from "./turn-discourse-state.js";

describe("TurnDiscourseStateService", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("persists no-output primary reason and structural flags to suppression markers", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-turn-discourse-state-"));
    tempDirs.push(tempDir);

    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(1_000),
    });
    const service = new TurnDiscourseStateService({
      tracer: {
        enabled: false,
        includePayloads: false,
        emit() {},
      },
      clock: new FixedClock(1_000),
    });

    await service.appendSuppressionMarker({
      streamWriter: writer,
      reason: "finalizer_no_output",
      userEntryId: createStreamEntryId(),
      turnId: "turn-llm-primary",
      audience: "service-test",
      noOutputCategories: ["closure"],
      primaryNoOutputReason: "low_value_echo",
      structuralNoOutputFlags: ["with_open_question", "open_question_rendered"],
    });

    await service.appendSuppressionMarker({
      streamWriter: writer,
      reason: "finalizer_no_output",
      userEntryId: createStreamEntryId(),
      turnId: "turn-derived-primary",
      audience: "service-test",
      noOutputCategories: ["when_borg_addressed", "with_state_delta"],
      primaryNoOutputReason: "when_borg_addressed",
      structuralNoOutputFlags: [
        "borg_directly_addressed",
        "with_state_delta",
        "current_turn_state_delta",
      ],
    });

    writer.close();

    const entries = new StreamReader({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(10);
    const suppressedEntries = entries.filter((entry) => entry.kind === "agent_suppressed");
    const contents = suppressedEntries.map((entry) => entry.content as Record<string, unknown>);

    expect(contents).toHaveLength(2);
    expect(contents[0]).toMatchObject({
      reason: "finalizer_no_output",
      turn_id: "turn-llm-primary",
      no_output_categories: ["closure"],
      primary_no_output_reason: "low_value_echo",
      structural_no_output_flags: ["with_open_question", "open_question_rendered"],
    });
    expect(contents[1]).toMatchObject({
      reason: "finalizer_no_output",
      turn_id: "turn-derived-primary",
      no_output_categories: ["when_borg_addressed", "with_state_delta"],
      primary_no_output_reason: "when_borg_addressed",
      structural_no_output_flags: [
        "borg_directly_addressed",
        "with_state_delta",
        "current_turn_state_delta",
      ],
    });
  });

  it("persists invalid finalizer tool diagnostics to suppression markers", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-turn-discourse-state-"));
    tempDirs.push(tempDir);

    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(1_000),
    });
    const service = new TurnDiscourseStateService({
      tracer: {
        enabled: false,
        includePayloads: false,
        emit() {},
      },
      clock: new FixedClock(1_000),
    });

    await service.appendSuppressionMarker({
      streamWriter: writer,
      reason: "invalid_tool_after_regenerate",
      userEntryId: createStreamEntryId(),
      turnId: "turn-invalid-tool-after-regenerate",
      audience: "service-test",
      finalizerInvalidTool: {
        tool_name: "EmitAnswer",
        reason: "schema payload was invalid",
        attempt: "regenerate",
      },
    });

    writer.close();

    const suppressedEntry = new StreamReader({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
    })
      .tail(10)
      .find((entry) => entry.kind === "agent_suppressed");

    expect(suppressedEntry?.content).toMatchObject({
      reason: "invalid_tool_after_regenerate",
      turn_id: "turn-invalid-tool-after-regenerate",
      finalizer_invalid_tool: {
        tool_name: "EmitAnswer",
        reason: "schema payload was invalid",
        attempt: "regenerate",
      },
    });
  });
});
