import { describe, expect, it } from "vitest";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { StreamEntry } from "../src/stream/index.js";
import {
  createSessionId,
  createStreamEntryId,
  DEFAULT_SESSION_ID,
  type SessionId,
} from "../src/util/ids.js";

import {
  runOverseer,
  validateOverseerVerdict,
  type FindingCarryoverCache,
  type OverseerAuditContext,
  type RunOverseerOptions,
} from "./overseer.js";
import type { MetricsRow, RawOverseerVerdict } from "./types.js";

type CapturedRequest = Parameters<
  NonNullable<RunOverseerOptions["client"]>["messages"]["stream"]
>[0];

function createClient(
  requests: CapturedRequest[],
  input: RawOverseerVerdict = {
    status: "healthy",
    observations: ["No issue."],
    recommendation: "Continue.",
    findings: [],
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
    semantic_node_count_by_status: {
      active: 0,
      superseded: 0,
      contradicted: 0,
      quarantined: 0,
    },
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
    action_record_count_committed_to_do: 0,
    action_record_count_canonicalized: 0,
    action_record_count_active: 0,
    action_record_creation_source_per_turn: {
      extractor: 0,
      reflector: 0,
      api: 0,
      unknown: 0,
    },
    action_record_creation_count_this_turn: 0,
    recent_completed_action_count: 0,
    commitment_count_active: 0,
    commitment_count_superseded: 0,
    commitment_count_revoked: 0,
    commitment_count_expired: 0,
    commitment_count_canonicalized: 0,
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
    goal_promotion_salvaged_promotions: 0,
    goal_promotion_skipped_promotions: 0,
    goal_promotion_initial_step_downgraded: 0,
    goal_promotion_dedup_skipped_extractor_signal: 0,
    goal_promotion_dedup_skipped_embedding: 0,
    goal_promotion_dedup_degraded: 0,
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

function auditContextFor(
  entries: readonly StreamEntry[],
  window: OverseerAuditContext["window"],
): OverseerAuditContext {
  return {
    window,
    chronology_rule: "Stream ts is authoritative for tests.",
    assistant_emitted: entries
      .filter((entry) => entry.kind === "agent_msg")
      .map((entry) => ({
        stream_entry_id: entry.id,
        ts: entry.timestamp,
        turn_counter: null,
        turn_id: entry.turn_id ?? null,
        session_id: entry.session_id,
        text: entry.content as string,
      })),
    user_messages: entries
      .filter((entry) => entry.kind === "user_msg")
      .map((entry) => ({
        stream_entry_id: entry.id,
        ts: entry.timestamp,
        turn_counter: null,
        turn_id: entry.turn_id ?? null,
        session_id: entry.session_id,
        text: entry.content as string,
        sender_entity_id: entry.sender_entity_id,
        quarantined: false,
        quarantine_reason: null,
      })),
    prompt_visible_memory: {
      summary: "Test memory.",
      note: "Test prompt-visible memory.",
    },
    snapshot_state: {
      markdown: "Test memory.",
      note: "Test snapshot state.",
    },
    metrics_window: [],
  };
}

describe("simulator overseer", () => {
  it("demotes same-impact carryover findings without changing the cached incident", () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Your birthday is June 3.",
      timestamp: 10,
    });
    const cache: FindingCarryoverCache = new Map([
      [
        agentEntry.id,
        {
          status_impact: "concerning",
          cached_at_turn: 40,
          category: "J",
          claim_status: "unsupported",
        },
      ],
    ]);
    const validated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Birthday claim still lacks support."],
        recommendation: "Do not double count.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            quoted_emitted_span: "Your birthday is June 3",
            evidence_summary: "Birthday claim lacks support.",
          },
        ],
      },
      auditContextFor([agentEntry], { from_turn: 41, to_turn: 50 }),
      cache,
    );

    expect(validated.status).toBe("healthy");
    expect(validated.findings[0]).toMatchObject({
      status_impact: "none",
      carryover_demoted: true,
      carryover_original_status_impact: "concerning",
      carryover_cached_status_impact: "concerning",
      carryover_cached_stream_entry_id: agentEntry.id,
      carryover_cached_at_turn: 40,
    });
    expect(cache.get(agentEntry.id)).toMatchObject({
      status_impact: "concerning",
      cached_at_turn: 40,
    });
  });

  it("passes through higher-impact carryover findings as escalations", () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Your birthday is June 3.",
      timestamp: 11,
    });
    const cache: FindingCarryoverCache = new Map([
      [
        agentEntry.id,
        {
          status_impact: "concerning",
          cached_at_turn: 40,
          category: "J",
          claim_status: "unsupported",
        },
      ],
    ]);
    const validated = validateOverseerVerdict(
      {
        status: "failing",
        observations: ["Birthday claim escalated."],
        recommendation: "Treat as serious.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "failing",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            quoted_emitted_span: "Your birthday is June 3",
            evidence_summary: "The same unsupported claim became a failing pattern.",
          },
        ],
      },
      auditContextFor([agentEntry], { from_turn: 41, to_turn: 50 }),
      cache,
    );

    expect(validated.status).toBe("failing");
    expect(validated.findings[0]?.carryover_demoted).toBeUndefined();
    expect(cache.get(agentEntry.id)).toMatchObject({
      status_impact: "failing",
      cached_at_turn: 50,
    });
  });

  it("does not dedup findings without assistant stream IDs", () => {
    const cache: FindingCarryoverCache = new Map([
      [
        "strm_cached",
        {
          status_impact: "concerning",
          cached_at_turn: 40,
          category: "I",
          claim_status: "grounded",
        },
      ],
    ]);
    const validated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Instrumentation concern in this metrics window."],
        recommendation: "Inspect metrics.",
        findings: [
          {
            category: "I",
            claim_status: "grounded",
            source_kind: "snapshot_memory",
            status_impact: "concerning",
            metrics_turn_counter: 50,
            evidence_summary: "Retrieval latency grew in the current metrics window.",
          },
        ],
      },
      auditContextFor([], { from_turn: 41, to_turn: 50 }),
      cache,
    );

    expect(validated.status).toBe("concerning");
    expect(validated.findings[0]).toMatchObject({
      status_impact: "concerning",
    });
    expect(validated.findings[0]?.carryover_demoted).toBeUndefined();
    expect(cache.size).toBe(1);
  });

  it("dedups same-verdict duplicate stream IDs only against the pre-verdict cache snapshot", () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "I fabricated one detail and then another.",
      timestamp: 12,
    });
    const cache: FindingCarryoverCache = new Map();
    const validated = validateOverseerVerdict(
      {
        status: "failing",
        observations: ["Two findings cite the same emitted entry."],
        recommendation: "Cache the max impact after the checkpoint.",
        findings: [
          {
            category: "H",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "The emitted entry contained a soft epistemic issue.",
          },
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "failing",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            quoted_emitted_span: "I fabricated one detail",
            evidence_summary: "The emitted entry contained a failing unsupported claim.",
          },
        ],
      },
      auditContextFor([agentEntry], { from_turn: 1, to_turn: 10 }),
      cache,
    );

    expect(validated.status).toBe("failing");
    expect(validated.findings.map((finding) => finding.carryover_demoted)).toEqual([
      undefined,
      undefined,
    ]);
    expect(cache.get(agentEntry.id)).toMatchObject({
      status_impact: "failing",
      cached_at_turn: 10,
    });
  });

  it("recomputes status as healthy when all status-driving findings are carryover", () => {
    const firstEntry = streamEntry({
      kind: "agent_msg",
      content: "Your birthday is June 3.",
      timestamp: 13,
    });
    const secondEntry = streamEntry({
      kind: "agent_msg",
      content: "Your birthday is June 3 again.",
      timestamp: 14,
    });
    const cache: FindingCarryoverCache = new Map([
      [
        firstEntry.id,
        {
          status_impact: "concerning",
          cached_at_turn: 40,
          category: "J",
          claim_status: "unsupported",
        },
      ],
      [
        secondEntry.id,
        {
          status_impact: "concerning",
          cached_at_turn: 40,
          category: "J",
          claim_status: "unsupported",
        },
      ],
    ]);
    const validated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Both unsupported findings are prior incidents."],
        recommendation: "Do not downgrade this checkpoint.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: firstEntry.id,
            assistant_ts: firstEntry.timestamp,
            quoted_emitted_span: "Your birthday is June 3",
            evidence_summary: "Prior unsupported birthday claim.",
          },
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: secondEntry.id,
            assistant_ts: secondEntry.timestamp,
            quoted_emitted_span: "Your birthday is June 3 again",
            evidence_summary: "Prior unsupported birthday claim repeated.",
          },
        ],
      },
      auditContextFor([firstEntry, secondEntry], { from_turn: 41, to_turn: 50 }),
      cache,
    );

    expect(validated.status).toBe("healthy");
    expect(validated.findings.every((finding) => finding.carryover_demoted === true)).toBe(true);
  });

  it("backfills legacy status-driving findings from cited assistant source handles", () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "The night 14 plan is too dense.",
      timestamp: 20,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "I should have fixed that density instead of flagging it.",
      timestamp: 21,
    });
    const cache: FindingCarryoverCache = new Map();
    const initial = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Legacy B finding cited source handles only in prose."],
        recommendation: "Seed carryover from the assistant handle.",
        findings: [
          {
            category: "B",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            evidence_summary: `Ben (${userEntry.id}) caught the issue; Borg acknowledged it in ${agentEntry.id}.`,
          },
        ],
      },
      auditContextFor([userEntry, agentEntry], { from_turn: 31, to_turn: 40 }),
      cache,
    );

    expect(initial.findings[0]).toMatchObject({
      assistant_stream_entry_id: agentEntry.id,
      status_impact: "concerning",
    });
    expect(cache.get(agentEntry.id)).toMatchObject({
      status_impact: "concerning",
      cached_at_turn: 40,
      category: "B",
      claim_status: "grounded",
    });

    const repeated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Same incident surfaced in the next window."],
        recommendation: "Demote as carryover.",
        findings: [
          {
            category: "B",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Borg repeated the same density-fix incident.",
          },
        ],
      },
      auditContextFor([userEntry, agentEntry], { from_turn: 41, to_turn: 50 }),
      cache,
    );

    expect(repeated.status).toBe("healthy");
    expect(repeated.findings[0]).toMatchObject({
      status_impact: "none",
      carryover_demoted: true,
      carryover_cached_stream_entry_id: agentEntry.id,
      carryover_cached_at_turn: 40,
    });
  });

  it("does not backfill legacy findings from user-only source handles", () => {
    const firstUserEntry = streamEntry({
      kind: "user_msg",
      content: "I caught the issue.",
      timestamp: 30,
    });
    const secondUserEntry = streamEntry({
      kind: "user_msg",
      content: "I confirmed the issue.",
      timestamp: 31,
    });
    const cache: FindingCarryoverCache = new Map();
    const initial = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Finding cites only user stream handles."],
        recommendation: "Do not seed carryover from user messages.",
        findings: [
          {
            category: "B",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            evidence_summary: `Ben ${firstUserEntry.id} and Alice ${secondUserEntry.id} caught the issue.`,
          },
        ],
      },
      auditContextFor([firstUserEntry, secondUserEntry], { from_turn: 31, to_turn: 40 }),
      cache,
    );

    expect(initial.findings[0]?.assistant_stream_entry_id).toBeUndefined();
    expect(cache.size).toBe(0);

    const repeated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["A later malformed finding cites the user handle directly."],
        recommendation: "It should not be demoted.",
        findings: [
          {
            category: "B",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: firstUserEntry.id,
            evidence_summary: "The user handle was never a cached Borg output incident.",
          },
        ],
      },
      auditContextFor([firstUserEntry, secondUserEntry], { from_turn: 41, to_turn: 50 }),
      cache,
    );

    expect(repeated.status).toBe("concerning");
    expect(repeated.findings[0]?.carryover_demoted).toBeUndefined();
  });

  it("does not backfill legacy findings from unknown source handles", () => {
    const cache: FindingCarryoverCache = new Map();
    const validated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Finding cites a stream handle outside the audit context."],
        recommendation: "Do not seed unknown handles.",
        findings: [
          {
            category: "B",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            evidence_summary: "Borg allegedly acknowledged the issue in strm_unknownlegacy123.",
          },
        ],
      },
      auditContextFor([], { from_turn: 31, to_turn: 40 }),
      cache,
    );

    expect(validated.status).toBe("concerning");
    expect(validated.findings[0]?.assistant_stream_entry_id).toBeUndefined();
    expect(cache.size).toBe(0);
  });

  it("dedups same-stream findings across different categories", () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "I should have fixed that density instead of flagging it.",
      timestamp: 40,
    });
    const cache: FindingCarryoverCache = new Map();
    const initial = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Category B finding seeded the incident."],
        recommendation: "Cache by stream ID.",
        findings: [
          {
            category: "B",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Borg acknowledged a density issue instead of preventing it.",
          },
        ],
      },
      auditContextFor([agentEntry], { from_turn: 31, to_turn: 40 }),
      cache,
    );

    expect(initial.status).toBe("concerning");
    expect(cache.get(agentEntry.id)).toMatchObject({
      status_impact: "concerning",
      category: "B",
    });

    const repeated = validateOverseerVerdict(
      {
        status: "concerning",
        observations: ["Category J finding cites the same emitted entry."],
        recommendation: "Dedup by stream ID alone.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            quoted_emitted_span: "fixed that density",
            evidence_summary: "Same emitted entry, different category.",
          },
        ],
      },
      auditContextFor([agentEntry], { from_turn: 41, to_turn: 50 }),
      cache,
    );

    expect(repeated.status).toBe("healthy");
    expect(repeated.findings[0]).toMatchObject({
      category: "J",
      status_impact: "none",
      carryover_demoted: true,
      carryover_cached_stream_entry_id: agentEntry.id,
      carryover_cached_at_turn: 40,
    });
  });

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

    expect(prompt).toContain(`"stream_entry_id": "${earlyMayaEntry.id}"`);
    expect(prompt).toContain(`"session_id": "${firstSession}"`);
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

    expect(prompt).toContain(`"stream_entry_id": "${quarantinedEntry.id}"`);
    expect(prompt).toContain('"quarantined": true');
    expect(prompt).toContain(
      '"quarantine_reason": "frame_anomaly:assistant_self_claim_in_user_role"',
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
    expect(String(requests[0]?.messages[0]?.content ?? "")).toContain("no conversation entries.");
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
    expect(prompt).toContain("Structured audit context (JSON):");
    expect(prompt).toContain('"prompt_visible_memory"');
    expect(prompt).toContain("id=node_maya");
    expect(prompt).toContain("J. CLAIM GROUNDING");
    expect(prompt).toContain("Do not sample.");
    expect(prompt).toContain("quoted_emitted_span");
    expect(prompt).toContain("Stream `ts` is authoritative");
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

      expect(prompt).toContain('"turn_counter": 12');
      expect(prompt).toContain('"turn_id": "turn-12"');
      expect(prompt).toContain(`"stream_entry_id": "${agentEntry.id}"`);
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
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            metrics_turn_counter: 1,
            quoted_emitted_span: "your birthday is June 3",
            evidence_summary: "No birthday record found in snapshot state.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("concerning");
    expect(verdict.observations.join("\n")).toContain("birthday is June 3");
    expect(verdict.observations.join("\n")).not.toContain("Maya is your partner");
  });

  it("rejects a J contradicted finding without a quoted emitted span and downgrades all-rejected verdicts", async () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Seville was deferred to a future trip.",
      timestamp: 20,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-j-missing-quote.jsonl",
      turnCounter: 7,
      totalTurns: 10,
      client: createClient(requests, {
        status: "concerning",
        observations: ["J contradicted: Borg claimed a Seville-inclusive itinerary."],
        recommendation: "Inspect the itinerary recall.",
        findings: [
          {
            category: "J",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Borg allegedly included Seville in the itinerary.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.raw_verdict.status).toBe("concerning");
    expect(verdict.findings).toEqual([]);
    expect(verdict.rejected_findings).toHaveLength(1);
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("quoted_emitted_span");
  });

  it("rejects a J contradicted finding whose quoted span is not in the cited assistant entry", async () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Madrid, Granada, and San Sebastian remain the three anchors.",
      timestamp: 30,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-j-bad-quote.jsonl",
      turnCounter: 8,
      totalTurns: 10,
      client: createClient(requests, {
        status: "concerning",
        observations: ["J contradicted: Borg supposedly included Seville."],
        recommendation: "Inspect the emitted turn.",
        findings: [
          {
            category: "J",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            quoted_emitted_span: "Madrid, Granada, Seville, and San Sebastian",
            evidence_summary: "The quote does not match the emitted turn.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("not a verbatim substring");
  });

  it("rejects a temporal C claim when timestamps contradict the claimed ordering", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Fair trade.",
      timestamp: 100,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "You called that fair trade.",
      timestamp: 115,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-c-temporal.jsonl",
      turnCounter: 40,
      totalTurns: 50,
      client: createClient(requests, {
        status: "concerning",
        observations: ["C: Borg recalled fair trade before Alice had said it."],
        recommendation: "Check turn chronology.",
        findings: [
          {
            category: "C",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            cited_evidence_stream_ids: [userEntry.id],
            cited_evidence_ts: [userEntry.timestamp],
            temporal_direction: "claim_before_evidence",
            evidence_summary: "Borg recalled fair trade before Alice had said it.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("assistant before evidence");
  });

  it("keeps a failing A-I status impact when a separate J finding is rejected", async () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "I will now narrate the user's interior thoughts.",
      timestamp: 120,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-status-impact.jsonl",
      turnCounter: 41,
      totalTurns: 50,
      client: createClient(requests, {
        status: "failing",
        observations: ["A: operational identity collapse plus a malformed J claim."],
        recommendation: "Stop and inspect identity drift.",
        findings: [
          {
            category: "A",
            claim_status: "grounded",
            source_kind: "emitted_output",
            status_impact: "failing",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Borg narrated user interior thoughts in its emitted output.",
          },
          {
            category: "J",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Malformed J finding without quote.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("failing");
    expect(verdict.findings).toHaveLength(1);
    expect(verdict.findings[0]?.status_impact).toBe("failing");
    expect(verdict.rejected_findings).toHaveLength(1);
  });

  it("rejects A-I findings missing status_impact without downgrading raw failing to healthy", async () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "I will now narrate the user's interior thoughts.",
      timestamp: 121,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-missing-ai-impact.jsonl",
      turnCounter: 41,
      totalTurns: 50,
      client: createClient(requests, {
        status: "failing",
        observations: ["A: operational identity collapse."],
        recommendation: "Stop and inspect identity drift.",
        findings: [
          {
            category: "A",
            claim_status: "grounded",
            source_kind: "emitted_output",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Borg narrated user interior thoughts in emitted output.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("failing");
    expect(verdict.findings).toEqual([]);
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("status_impact");
  });

  it("rejects temporal C findings that supply turn counters as timestamps", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Fair trade.",
      timestamp: 100,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "You called that fair trade.",
      timestamp: 115,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-c-turn-counter-ts.jsonl",
      turnCounter: 40,
      totalTurns: 50,
      client: createClient(requests, {
        status: "concerning",
        observations: ["C: Borg recalled fair trade before Alice said it."],
        recommendation: "Check timestamp citations.",
        findings: [
          {
            category: "C",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: 36,
            cited_evidence_stream_ids: [userEntry.id],
            cited_evidence_ts: [37],
            temporal_direction: "claim_before_evidence",
            evidence_summary: "Borg recalled fair trade before Alice said it.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("assistant_ts=36");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("resolved stream ts=115");
  });

  it("rejects C temporal claims with prose cues but no temporal_direction", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Fair trade.",
      timestamp: 100,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "You called that fair trade.",
      timestamp: 115,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-c-missing-direction.jsonl",
      turnCounter: 40,
      totalTurns: 50,
      client: createClient(requests, {
        status: "concerning",
        observations: ["C: Borg recalled fair trade before Alice said it."],
        recommendation: "Check timestamp citations.",
        findings: [
          {
            category: "C",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            cited_evidence_stream_ids: [userEntry.id],
            cited_evidence_ts: [userEntry.timestamp],
            evidence_summary: "Borg recalled fair trade before Alice said it.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("temporal_direction");
  });

  it("rejects temporal C findings whose structured direction conflicts with their prose claim", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Fair trade.",
      timestamp: 100,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "You called that fair trade.",
      timestamp: 115,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-c-direction-conflict.jsonl",
      turnCounter: 40,
      totalTurns: 50,
      client: createClient(requests, {
        status: "concerning",
        observations: ["C: Borg recalled fair trade before Alice said it."],
        recommendation: "Check temporal direction.",
        findings: [
          {
            category: "C",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            cited_evidence_stream_ids: [userEntry.id],
            cited_evidence_ts: [userEntry.timestamp],
            temporal_direction: "claim_after_evidence",
            evidence_summary: "Borg recalled fair trade before Alice said it.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain(
      "temporal_direction=claim_after_evidence conflicts",
    );
  });

  it("allows simultaneous C claims within the timestamp tolerance", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Fair trade.",
      timestamp: 1_000,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "You called that fair trade at the same time.",
      timestamp: 1_050,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-c-simultaneous-valid.jsonl",
      turnCounter: 40,
      totalTurns: 50,
      client: createClient(requests, {
        status: "concerning",
        observations: ["C: The attribution was simultaneous with the user message."],
        recommendation: "Check batch ordering.",
        findings: [
          {
            category: "C",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            cited_evidence_stream_ids: [userEntry.id],
            cited_evidence_ts: [userEntry.timestamp],
            temporal_direction: "claim_simultaneous",
            evidence_summary: "The attribution was simultaneous with the user message.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("concerning");
    expect(verdict.findings).toHaveLength(1);
    expect(verdict.rejected_findings).toEqual([]);
  });

  it("rejects simultaneous C claims outside the timestamp tolerance", async () => {
    const userEntry = streamEntry({
      kind: "user_msg",
      content: "Fair trade.",
      timestamp: 1_000,
    });
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "You called that fair trade much later.",
      timestamp: 1_200,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([userEntry, agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-c-simultaneous-rejected.jsonl",
      turnCounter: 40,
      totalTurns: 50,
      client: createClient(requests, {
        status: "concerning",
        observations: ["C: The attribution was simultaneous with the user message."],
        recommendation: "Check batch ordering.",
        findings: [
          {
            category: "C",
            claim_status: "contradicted",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            cited_evidence_stream_ids: [userEntry.id],
            cited_evidence_ts: [userEntry.timestamp],
            temporal_direction: "claim_simultaneous",
            evidence_summary: "The attribution was simultaneous with the user message.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("more than 100ms");
  });

  it("rejects a J unsupported finding without a quoted emitted span", async () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Your birthday is June 3.",
      timestamp: 130,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-j-unsupported-missing-quote.jsonl",
      turnCounter: 42,
      totalTurns: 50,
      client: createClient(requests, {
        status: "concerning",
        observations: ["J unsupported: birthday claim lacks support."],
        recommendation: "Drop the birthday claim.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            evidence_summary: "Birthday claim lacks support.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("healthy");
    expect(verdict.rejected_findings[0]?.claim_status).toBe("unsupported");
    expect(verdict.rejected_findings[0]?.validation_warning).toContain("quoted_emitted_span");
  });

  it("persists raw and validated verdicts in audit JSONL for exact replay", async () => {
    const dir = mkdtempSync(join(tmpdir(), "borg-overseer-audit-replay-"));
    const auditContextPath = join(dir, "overseer-audit.jsonl");
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Madrid and Granada remain the plan.",
      timestamp: 140,
    });
    const requests: CapturedRequest[] = [];

    try {
      const verdict = await runOverseer({
        transport: transportFor([agentEntry]),
        metricsPath: join(dir, "metrics.jsonl"),
        auditContextPath,
        turnCounter: 43,
        totalTurns: 50,
        client: createClient(requests, {
          status: "concerning",
          observations: ["J contradicted with missing quote."],
          recommendation: "Inspect.",
          findings: [
            {
              category: "J",
              claim_status: "contradicted",
              source_kind: "emitted_output",
              status_impact: "concerning",
              assistant_stream_entry_id: agentEntry.id,
              assistant_ts: agentEntry.timestamp,
              evidence_summary: "Missing quoted emitted span.",
            },
          ],
        }),
      });
      const [line] = readFileSync(auditContextPath, "utf8").trim().split(/\r?\n/);
      const record = JSON.parse(line ?? "{}") as {
        audit_context: OverseerAuditContext;
        raw_verdict: RawOverseerVerdict;
        validated_verdict: {
          status: string;
          findings: unknown[];
          rejected_findings: unknown[];
        };
      };
      const replayed = validateOverseerVerdict(record.raw_verdict, record.audit_context);

      expect(record.raw_verdict).toEqual(verdict.raw_verdict);
      expect(replayed).toEqual({
        status: record.validated_verdict.status,
        findings: record.validated_verdict.findings,
        rejected_findings: record.validated_verdict.rejected_findings,
      });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("downgrades failing to concerning when only some non-grounded findings are rejected", async () => {
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Your birthday is June 3.",
      timestamp: 50,
    });
    const requests: CapturedRequest[] = [];
    const verdict = await runOverseer({
      transport: transportFor([agentEntry]),
      metricsPath: "/tmp/borg-overseer-test-partial-rejection.jsonl",
      turnCounter: 9,
      totalTurns: 10,
      client: createClient(requests, {
        status: "failing",
        observations: ["One unsupported birthday claim and one malformed claim."],
        recommendation: "Inspect manually.",
        findings: [
          {
            category: "J",
            claim_status: "unsupported",
            source_kind: "emitted_output",
            status_impact: "concerning",
            assistant_stream_entry_id: agentEntry.id,
            assistant_ts: agentEntry.timestamp,
            quoted_emitted_span: "Your birthday is June 3",
            evidence_summary: "Birthday lacks support.",
          },
          {
            category: "J",
            claim_status: "contradicted",
            source_kind: "snapshot_memory",
            status_impact: "concerning",
            evidence_summary: "Malformed emitted-output attribution.",
          },
        ],
      }),
    });

    expect(verdict.status).toBe("concerning");
    expect(verdict.findings).toHaveLength(1);
    expect(verdict.rejected_findings).toHaveLength(1);
  });

  it("emits a trace event when validation rejects a finding", async () => {
    const dir = mkdtempSync(join(tmpdir(), "borg-overseer-rejected-trace-"));
    const tracePath = join(dir, "trace.jsonl");
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Madrid and Granada remain the plan.",
      timestamp: 60,
    });
    const transport = Object.assign(transportFor([agentEntry]), { tracePath });
    const requests: CapturedRequest[] = [];

    try {
      await runOverseer({
        transport,
        metricsPath: join(dir, "metrics.jsonl"),
        turnCounter: 10,
        totalTurns: 10,
        client: createClient(requests, {
          status: "concerning",
          observations: ["J contradicted with missing quote."],
          recommendation: "Inspect.",
          findings: [
            {
              category: "J",
              claim_status: "contradicted",
              source_kind: "emitted_output",
              status_impact: "concerning",
              assistant_stream_entry_id: agentEntry.id,
              assistant_ts: agentEntry.timestamp,
              evidence_summary: "Missing quoted emitted span.",
            },
          ],
        }),
      });

      const trace = readFileSync(tracePath, "utf8");

      expect(trace).toContain("overseer_finding_rejected");
      expect(trace).toContain("quoted_emitted_span");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("emits a trace event when carryover dedup demotes a finding", async () => {
    const dir = mkdtempSync(join(tmpdir(), "borg-overseer-carryover-trace-"));
    const tracePath = join(dir, "trace.jsonl");
    const agentEntry = streamEntry({
      kind: "agent_msg",
      content: "Your birthday is June 3.",
      timestamp: 70,
    });
    const transport = Object.assign(transportFor([agentEntry]), { tracePath });
    const requests: CapturedRequest[] = [];
    const carryoverCache: FindingCarryoverCache = new Map([
      [
        agentEntry.id,
        {
          status_impact: "concerning",
          cached_at_turn: 40,
          category: "J",
          claim_status: "unsupported",
        },
      ],
    ]);

    try {
      await runOverseer({
        transport,
        metricsPath: join(dir, "metrics.jsonl"),
        turnCounter: 50,
        totalTurns: 50,
        carryoverCache,
        client: createClient(requests, {
          status: "concerning",
          observations: ["J unsupported: birthday claim lacks support."],
          recommendation: "Do not double count.",
          findings: [
            {
              category: "J",
              claim_status: "unsupported",
              source_kind: "emitted_output",
              status_impact: "concerning",
              assistant_stream_entry_id: agentEntry.id,
              assistant_ts: agentEntry.timestamp,
              quoted_emitted_span: "Your birthday is June 3",
              evidence_summary: "Birthday claim lacks support.",
            },
          ],
        }),
      });

      const trace = readFileSync(tracePath, "utf8");

      expect(trace).toContain("overseer_finding_carryover_demoted");
      expect(trace).toContain('"cached_at_turn":40');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
