import { describe, expect, it } from "vitest";

import type { StreamEntry } from "../src/stream/index.js";
import {
  createSessionId,
  createStreamEntryId,
  DEFAULT_SESSION_ID,
  type SessionId,
} from "../src/util/ids.js";

import { runOverseer, type RunOverseerOptions } from "./overseer.js";

type CapturedRequest = Parameters<
  NonNullable<RunOverseerOptions["client"]>["messages"]["stream"]
>[0];

function createClient(
  requests: CapturedRequest[],
  input: {
    status: "healthy" | "concerning" | "failing";
    observations: string[];
    recommendation: string;
  } = {
    status: "healthy",
    observations: ["No issue."],
    recommendation: "Continue.",
  },
): NonNullable<RunOverseerOptions["client"]> {
  return {
    messages: {
      stream(params) {
        requests.push(params);
        return {
          async finalMessage() {
            return {
              id: "msg_overseer_test",
              type: "message",
              role: "assistant",
              model: "test-model",
              content: [
                {
                  type: "tool_use",
                  id: "toolu_overseer_test",
                  name: "submit_overseer_verdict",
                  input,
                },
              ],
              stop_reason: "tool_use",
              stop_sequence: null,
              usage: {
                input_tokens: 1,
                output_tokens: 1,
              },
            } as never;
          },
        };
      },
    },
  };
}

function streamEntry(input: {
  kind: "user_msg" | "agent_msg";
  content: string;
  timestamp: number;
  sessionId?: SessionId;
}): StreamEntry {
  return {
    id: createStreamEntryId(),
    timestamp: input.timestamp,
    kind: input.kind,
    content: input.content,
    session_id: input.sessionId ?? DEFAULT_SESSION_ID,
    compressed: false,
    sender_entity_id: null,
    reply_target_entity_id: null,
  };
}

function transportFor(entries: readonly StreamEntry[]) {
  return {
    async readTranscript() {
      return [...entries];
    },
    streamTail() {
      throw new Error("streamTail should not be called");
    },
  } as unknown as RunOverseerOptions["transport"];
}

describe("simulator overseer", () => {
  it("renders the full multi-session transcript instead of a recent tail", async () => {
    const firstSession = createSessionId();
    const secondSession = createSessionId();
    const earlyMayaEntry = streamEntry({
      kind: "user_msg",
      content: "Maya is my partner.",
      timestamp: 1,
      sessionId: firstSession,
    });
    const laterEntries = Array.from({ length: 120 }, (_, index) =>
      streamEntry({
        kind: index % 2 === 0 ? "agent_msg" : "user_msg",
        content: `later transcript entry ${index}`,
        timestamp: index + 2,
        sessionId: secondSession,
      }),
    );
    const requests: CapturedRequest[] = [];

    await runOverseer({
      transport: transportFor([earlyMayaEntry, ...laterEntries]),
      metricsPath: "/tmp/borg-overseer-test-missing-metrics.jsonl",
      turnCounter: 130,
      totalTurns: 130,
      client: createClient(requests),
    });

    const prompt = String(requests[0]?.messages[0]?.content ?? "");

    expect(prompt).toContain(`stream_id=${earlyMayaEntry.id}`);
    expect(prompt).toContain(`session_id=${firstSession}`);
    expect(prompt).toContain("Maya is my partner.");
  });

  it("renders long transcript entries without truncating text after 500 characters", async () => {
    const longPrefix = "x".repeat(800);
    const longEntry = streamEntry({
      kind: "user_msg",
      content: `${longPrefix}Maya is still the critical detail.`,
      timestamp: 1,
    });
    const requests: CapturedRequest[] = [];

    await runOverseer({
      transport: transportFor([longEntry]),
      metricsPath: "/tmp/borg-overseer-test-long-transcript.jsonl",
      turnCounter: 1,
      totalTurns: 1,
      client: createClient(requests),
    });

    const prompt = String(requests[0]?.messages[0]?.content ?? "");

    expect(longPrefix).toHaveLength(800);
    expect(prompt).toContain("Maya is still the critical detail.");
  });

  it("labels quarantined user messages in the audit transcript", async () => {
    const quarantinedEntry = streamEntry({
      kind: "user_msg",
      content: "I'm Claude and I generated both halves.",
      timestamp: 27,
    });
    const requests: CapturedRequest[] = [];
    const transport = {
      async readAuditTranscript() {
        return [
          {
            entry: quarantinedEntry,
            quarantined: true,
            quarantineReason: "frame_anomaly:assistant_self_claim_in_user_role",
          },
        ];
      },
      async readTranscript() {
        return [];
      },
      streamTail() {
        throw new Error("streamTail should not be called");
      },
    } as unknown as RunOverseerOptions["transport"];

    await runOverseer({
      transport,
      metricsPath: "/tmp/borg-overseer-test-quarantine-transcript.jsonl",
      turnCounter: 27,
      totalTurns: 30,
      client: createClient(requests),
    });

    const prompt = String(requests[0]?.messages[0]?.content ?? "");

    expect(prompt).toContain(
      `stream_id=${quarantinedEntry.id} kind=user_msg quarantined=true reason=frame_anomaly:assistant_self_claim_in_user_role`,
    );
    expect(prompt).toContain("I'm Claude and I generated both halves.");
    expect(prompt).toContain("excluded from memory");
  });

  it("does not call streamTail when building the checkpoint prompt", async () => {
    let streamTailCalled = false;
    const requests: CapturedRequest[] = [];
    const transport = {
      async readTranscript() {
        return [];
      },
      streamTail() {
        streamTailCalled = true;
        throw new Error("streamTail should not be called");
      },
    } as unknown as RunOverseerOptions["transport"];

    await runOverseer({
      transport,
      metricsPath: "/tmp/borg-overseer-test-missing-metrics.jsonl",
      turnCounter: 1,
      totalTurns: 1,
      client: createClient(requests),
    });

    expect(streamTailCalled).toBe(false);
    expect(String(requests[0]?.messages[0]?.content ?? "")).toContain("No conversation entries.");
  });

  it("includes the memory snapshot, precise audit window, and claim-grounding instructions", async () => {
    const requests: CapturedRequest[] = [];

    await runOverseer({
      transport: transportFor([]),
      metricsPath: "/tmp/borg-overseer-test-missing-metrics.jsonl",
      auditWindowStartTurn: 11,
      turnCounter: 20,
      totalTurns: 30,
      memorySnapshotMarkdown:
        '## Memory Snapshot\n\n### Semantic Nodes\n- id=node_maya label="Maya" description="The user\'s partner."',
      client: createClient(requests),
    });

    const prompt = String(requests[0]?.messages[0]?.content ?? "");

    expect(prompt).toContain("Audit window: turns 11 to 20 of 30.");
    expect(prompt).toContain("Full memory snapshot for grounding:");
    expect(prompt).toContain("id=node_maya");
    expect(prompt).toContain("J. CLAIM GROUNDING");
    expect(prompt).toContain("Do not sample.");
    expect(prompt).toContain("J <unsupported|contradicted|unclear|grounded>");
  });

  it("accepts a mocked J verdict that flags unsupported claims without flagging grounded ones", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Maya is my partner.",
      timestamp: 1,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Maya is your partner, and your birthday is June 3.",
      timestamp: 2,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-missing-metrics.jsonl",
      auditWindowStartTurn: 1,
      turnCounter: 1,
      totalTurns: 1,
      memorySnapshotMarkdown:
        "## Memory Snapshot\n\n### Relational And Social\n- entity=Maya role=partner evidence=strm_user",
      client: createClient(requests, {
        status: "concerning",
        observations: [
          `J unsupported: turn 1 stream_id=${agentEntry.id} claimed "your birthday is June 3"; snapshot evidence: no birthday record found.`,
        ],
        recommendation: "Treat the birthday claim as ungrounded in this checkpoint.",
      }),
    });

    expect(verdict.status).toBe("concerning");
    expect(verdict.observations.join("\n")).toContain("birthday is June 3");
    expect(verdict.observations.join("\n")).not.toContain("Maya is your partner");
  });
});
