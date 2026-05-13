import { describe, expect, it } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { StreamEntry } from "../src/stream/index.js";
import {
  createSessionId,
  createStreamEntryId,
  DEFAULT_SESSION_ID,
  type SessionId,
} from "../src/util/ids.js";

import { runOverseer, type RunOverseerOptions } from "./overseer.js";
import type { MetricsRow } from "./types.js";

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
  turnId?: string;
}): StreamEntry {
  return {
    id: createStreamEntryId(),
    timestamp: input.timestamp,
    kind: input.kind,
    content: input.content,
    ...(input.turnId === undefined ? {} : { turn_id: input.turnId }),
    session_id: input.sessionId ?? DEFAULT_SESSION_ID,
    compressed: false,
    sender_entity_id: null,
    reply_target_entity_id: null,
  };
}

function metricsRow(turn: number): MetricsRow {
  return {
    event: "turn_metrics",
    ts: turn,
    turn_counter: turn,
    turnId: `turn-${turn}`,
    transport_chat_attempts: 1,
    episode_count: 0,
    semantic_node_count: 0,
    semantic_edge_count: 0,
    semantic_nodes_added_since_last_check: 0,
    semantic_edges_added_since_last_check: 0,
    open_question_count: 0,
    active_goal_count: 0,
    generation_suppression_count: 0,
    mood_valence: 0,
    mood_arousal: 0,
    retrieval_latency_ms: null,
    deliberation_latency_ms: null,
    borg_input_tokens: 0,
    borg_output_tokens: 0,
    open_question_resolved_count: 0,
    action_record_count_total: 0,
    action_record_count_by_state: {
      considering: 0,
      committed_to_do: 0,
      scheduled: 0,
      completed: 0,
      not_done: 0,
      unknown: 0,
    },
    recent_completed_action_count: 0,
    commitment_count_active: 0,
    commitment_count_superseded: 0,
    pending_action_count: 0,
    pending_action_merge_count: 0,
    relational_slot_count_by_state: {
      established: 0,
      contested: 0,
      quarantined: 0,
      revoked: 0,
    },
    review_queue_open_count_by_type: {
      contradiction: 0,
      duplicate: 0,
      new_insight: 0,
      misattribution: 0,
      temporal_drift: 0,
      identity_inconsistency: 0,
      correction: 0,
      belief_revision: 0,
      skill_split: 0,
    },
    frame_anomaly_classifier_calls: 0,
    frame_anomaly_classified_normal_count: 0,
    frame_anomaly_actual_anomaly_count: 0,
    frame_anomaly_degraded_count: 0,
    frame_anomaly_degraded_fallback_match_count: 0,
    quarantined_user_entry_count: 0,
    early_extractors_skipped_frame_anomaly_count: 0,
    overseer_due_on_suppressed_turn: false,
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

  it("renders transcript turn ids and a full audit-window turn map", async () => {
    const dir = mkdtempSync(join(tmpdir(), "borg-overseer-turn-map-"));
    const metricsPath = join(dir, "metrics.jsonl");
    try {
      writeFileSync(
        metricsPath,
        Array.from({ length: 7 }, (_, index) => JSON.stringify(metricsRow(index + 11))).join("\n"),
      );
      const agentEntry = streamEntry({
        kind: "agent_msg",
        content: "Maya is your partner.",
        timestamp: 12,
        turnId: "turn-12",
      });
      const requests: CapturedRequest[] = [];

      await runOverseer({
        transport: transportFor([agentEntry]),
        metricsPath,
        auditWindowStartTurn: 11,
        turnCounter: 17,
        totalTurns: 20,
        client: createClient(requests),
      });

      const prompt = String(requests[0]?.messages[0]?.content ?? "");

      expect(prompt).toContain(`turn_counter=12 turn_id=turn-12`);
      expect(prompt).toContain(`stream_id=${agentEntry.id}`);
      expect(prompt).toContain("Audit window turn map:");
      expect(prompt).toContain("turn=11 turn_id=turn-11 event=turn_metrics");
      expect(prompt).toContain("turn=17 turn_id=turn-17 event=turn_metrics");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
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
