import { appendFileSync, mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { BorgTransport } from "./borg-transport.js";
import { recallScenario } from "./scenarios/recall.js";
import {
  DEFAULT_SESSION_ID,
  getSessionStreamPath,
  getStreamDirectory,
  StreamWriter,
} from "../src/stream/index.js";
import { ManualClock } from "../src/util/clock.js";
import { createSessionId } from "../src/util/ids.js";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("BorgTransport", () => {
  it("opens a mock Borg, runs a traced turn, and reads the trace summary", async () => {
    const transport = new BorgTransport({
      runId: "transport-test",
      scenario: recallScenario,
      mock: true,
    });

    try {
      await transport.open();
      const result = await transport.chat("What's my dog's name?");
      const summary = transport.readTrace(result.turnId);

      expect(result.response).toContain("Otto");
      expect(result.turnId).toHaveLength(36);
      expect(summary).toContain("tool.episodic.search");
      expect(transport.readTraceEvents().some((record) => record.turnId === result.turnId)).toBe(
        true,
      );
    } finally {
      await transport.close();
    }
  });

  it("keeps maintenance enabled when explicitly opted in", async () => {
    const disabledTransport = new BorgTransport({
      runId: "transport-maintenance-default-test",
      scenario: recallScenario,
      mock: true,
    });
    const enabledTransport = new BorgTransport({
      runId: "transport-maintenance-enabled-test",
      scenario: recallScenario,
      mock: true,
      maintenance: true,
    });

    try {
      await disabledTransport.open();
      await enabledTransport.open();

      expect(disabledTransport.getBorg().maintenance.scheduler.isEnabled()).toBe(false);
      expect(enabledTransport.getBorg().maintenance.scheduler.isEnabled()).toBe(true);
    } finally {
      await disabledTransport.close();
      await enabledTransport.close();
    }
  });

  it("passes an explicit audience through to Borg turns", async () => {
    const transport = new BorgTransport({
      runId: "transport-audience-test",
      scenario: recallScenario,
      mock: true,
    });

    try {
      await transport.open();
      const turnSpy = vi.spyOn(transport.getBorg(), "turn");

      await transport.chat("What's my dog's name?", { audience: "Tom" });

      expect(turnSpy.mock.calls[0]?.[0].audience).toBe("Tom");
    } finally {
      await transport.close();
    }
  });

  it("reads a timestamp-sorted transcript across all stream sessions with append-order ties", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-transport-transcript-"));
    const clock = new ManualClock(0);
    const sessionId = createSessionId();
    const defaultWriter = new StreamWriter({
      dataDir,
      clock,
    });
    const sessionWriter = new StreamWriter({
      dataDir,
      sessionId,
      clock,
    });

    try {
      clock.set(200);
      const late = await defaultWriter.append({
        kind: "user_msg",
        content: "late default user message",
      });

      clock.set(100);
      const earlySession = await sessionWriter.append({
        kind: "agent_msg",
        content: "early session agent message",
      });
      await sessionWriter.append({
        kind: "thought",
        content: "not part of transcript",
      });
      clock.set(150);
      const tiedFromDefault = await defaultWriter.append({
        kind: "user_msg",
        content: "same timestamp default user message",
      });
      const tiedSecondFromDefault = await defaultWriter.append({
        kind: "agent_msg",
        content: "same timestamp default agent message",
      });

      const transport = new BorgTransport({
        runId: "transport-transcript-test",
        scenario: recallScenario,
        mock: true,
        dataDir,
        keep: true,
      });
      const transcript = await transport.readTranscript();

      expect(transcript.map((entry) => entry.id)).toEqual([
        earlySession.id,
        tiedFromDefault.id,
        tiedSecondFromDefault.id,
        late.id,
      ]);
      expect(transcript.map((entry) => entry.content)).toEqual([
        earlySession.content,
        tiedFromDefault.content,
        tiedSecondFromDefault.content,
        late.content,
      ]);
      expect(transcript.map((entry) => entry.kind)).toEqual([
        earlySession.kind,
        tiedFromDefault.kind,
        tiedSecondFromDefault.kind,
        late.kind,
      ]);
    } finally {
      defaultWriter.close();
      sessionWriter.close();
      rmSync(dataDir, { recursive: true, force: true });
    }
  });

  it("keeps append order for same-timestamp entries when stream ids would reverse them", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-transport-transcript-tie-"));
    const streamDir = getStreamDirectory(dataDir);
    const streamPath = getSessionStreamPath(dataDir, DEFAULT_SESSION_ID);

    try {
      mkdirSync(streamDir, { recursive: true });
      appendFileSync(
        streamPath,
        [
          {
            id: "strm_zzzzzzzzzzzzzzzz",
            timestamp: 500,
            kind: "user_msg",
            content: "first in append order",
            session_id: DEFAULT_SESSION_ID,
          },
          {
            id: "strm_aaaaaaaaaaaaaaaa",
            timestamp: 500,
            kind: "agent_msg",
            content: "second in append order",
            session_id: DEFAULT_SESSION_ID,
          },
        ]
          .map((entry) => `${JSON.stringify(entry)}\n`)
          .join(""),
      );

      const transport = new BorgTransport({
        runId: "transport-transcript-tie-test",
        scenario: recallScenario,
        mock: true,
        dataDir,
        keep: true,
      });
      const transcript = await transport.readTranscript();

      expect(transcript.map((entry) => entry.id)).toEqual([
        "strm_zzzzzzzzzzzzzzzz",
        "strm_aaaaaaaaaaaaaaaa",
      ]);
    } finally {
      rmSync(dataDir, { recursive: true, force: true });
    }
  });

  it("keeps the Banks user turn before its same-timestamp agent response", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-transport-transcript-banks-"));
    const streamDir = getStreamDirectory(dataDir);
    const streamPath = getSessionStreamPath(dataDir, DEFAULT_SESSION_ID);

    try {
      mkdirSync(streamDir, { recursive: true });
      appendFileSync(
        streamPath,
        [
          {
            id: "strm_zzzzzzzzzzzzzzzz",
            timestamp: 700,
            kind: "user_msg",
            content: "Have you read Use of Weapons?",
            session_id: DEFAULT_SESSION_ID,
          },
          {
            id: "strm_aaaaaaaaaaaaaaaa",
            timestamp: 700,
            kind: "agent_msg",
            content: "I haven't read Banks the way you've read Banks...",
            session_id: DEFAULT_SESSION_ID,
          },
        ]
          .map((entry) => `${JSON.stringify(entry)}\n`)
          .join(""),
      );

      const transport = new BorgTransport({
        runId: "transport-transcript-banks-test",
        scenario: recallScenario,
        mock: true,
        dataDir,
        keep: true,
      });
      const transcript = await transport.readTranscript();

      expect(transcript.map((entry) => entry.content)).toEqual([
        "Have you read Use of Weapons?",
        "I haven't read Banks the way you've read Banks...",
      ]);
    } finally {
      rmSync(dataDir, { recursive: true, force: true });
    }
  });
});
