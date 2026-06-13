import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { createWorkingMemory } from "../memory/working/index.js";
import {
  StreamEntryIndexRepository,
  streamEntryIndexMigrations,
  StreamReader,
  StreamWriter,
} from "../stream/index.js";
import { openDatabase } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";
import { appendRecentSuppression } from "./generation/discourse-state.js";
import { hydrateTurnMechanismEvidence } from "./mechanism-evidence.js";

describe("turn mechanism evidence", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("hydrates recent suppression diagnostics from the source stream entry", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-mechanism-evidence-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...streamEntryIndexMigrations],
    });
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new ManualClock(100),
      entryIndex,
    });

    try {
      const suppressionEntry = await writer.append({
        kind: "agent_suppressed",
        turn_id: "turn-suppressed",
        content: {
          reason: "finalizer_no_output",
          turn_id: "turn-suppressed",
          no_output_categories: ["with_open_question"],
          primary_no_output_reason: "other",
          structural_no_output_flags: ["open_question_rendered"],
          finalizer_invalid_tool: {
            tool_name: "EmitNoOutput",
            reason: "schema_error",
            attempt: "initial",
          },
        },
      });
      const workingMemory = appendRecentSuppression(createWorkingMemory(DEFAULT_SESSION_ID, 100), {
        turnId: "turn-suppressed",
        reason: "finalizer_no_output",
        ts: 100,
        sourceStreamEntryId: suppressionEntry.id,
      });

      const evidence = await hydrateTurnMechanismEvidence({
        dataDir: tempDir,
        sessionId: DEFAULT_SESSION_ID,
        workingMemory,
        entryIndex,
        createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      });

      expect(evidence.recentSuppressions).toHaveLength(1);
      expect(evidence.recentSuppressions[0]?.diagnostic).toEqual({
        noOutputCategories: ["with_open_question"],
        primaryNoOutputReason: "other",
        structuralNoOutputFlags: ["open_question_rendered"],
        finalizerInvalidTool: {
          tool_name: "EmitNoOutput",
          reason: "schema_error",
          attempt: "initial",
        },
      });
    } finally {
      writer.close();
      db.close();
    }
  });
});
