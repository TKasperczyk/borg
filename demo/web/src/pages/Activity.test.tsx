import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { installMockWebSocket } from "../__tests__/mock-websocket";
import type { ActivityResponse, AutonomyStateResponse, JournalResponse, StreamResponse } from "../api/types";
import { LiveProvider } from "../live/useLive";
import { ActivityPage } from "./Activity";

const day = "2026-06-11";
const previousDay = "2026-06-10";
const baseTs = Date.UTC(2026, 5, 11, 10, 0, 0);
const previousJournalTs = Date.UTC(2026, 5, 10, 7, 55, 0);

const activityFixture: ActivityResponse = {
  day,
  days: [day, previousDay],
  truncated: false,
  digest: {
    turns: 2,
    autonomous_wakes: 1,
    emissions: 1,
    silences: 1,
    observations: 0,
    suppressions: 0,
    dream_changes: 3,
    journal_notes: 1,
  },
  rows: [
    {
      id: "turn:s_user:turn_user",
      kind: "turn",
      started_at: baseTs,
      session_id: "s_user",
      session_label: "operator",
      origin: "user",
      trigger: null,
      outcome: "emitted",
      suppression_reason: null,
      duration_ms: null,
      excerpt: "answer from user turn",
      turn_id: "turn_user",
    },
    {
      id: "turn:s_auto:turn_auto",
      kind: "turn",
      started_at: baseTs + 1_000,
      session_id: "s_auto",
      session_label: "self",
      origin: "autonomous",
      trigger: "scheduled_reflection",
      outcome: "deliberate-silence",
      suppression_reason: "finalizer_no_output",
      duration_ms: null,
      excerpt: "finalizer_no_output",
      turn_id: "turn_auto",
    },
    {
      id: "dream:entry_1",
      kind: "dream",
      started_at: baseTs + 2_000,
      session_id: "default",
      session_label: "operator",
      origin: "dream",
      trigger: null,
      outcome: "dream",
      suppression_reason: null,
      duration_ms: null,
      excerpt: "summarized maintenance",
      turn_id: null,
      dream: {
        run_id: "run_1",
        process_count: 2,
        changes: 3,
        errors: 0,
      },
    },
  ],
};

const autonomyFixture: AutonomyStateResponse = {
  scheduler: { enabled: true },
  wake_sources: [
    {
      name: "scheduled_reflection",
      enabled: null,
      wake_source_type: "trigger",
      source_category: "contemplative",
      last_fired: baseTs,
      wake_count: 1,
    },
  ],
  wake_budget: null,
  self_scheduled_wakes: [
    {
      id: "sw_1",
      due_at: baseTs + 60_000,
      note: "follow up later",
      created_at: baseTs - 60_000,
      status: "pending",
    },
  ],
  can_cancel_wakes: false,
  recent_wakes: [],
};

const journalFixture: JournalResponse = {
  entries: [
    {
      id: 1,
      self_entity_id: "ent_self",
      self_label: "self",
      text: "private note",
      disclosure_class: "self_private",
      created_at: baseTs,
      updated_at: baseTs,
      source_turn_id: "turn_user",
      marker_stream_entry_id: null,
    },
    {
      id: 2,
      self_entity_id: "ent_self",
      self_label: "self",
      text: "long id note",
      disclosure_class: "self_private",
      created_at: previousJournalTs,
      updated_at: previousJournalTs,
      source_turn_id: "12345678-90ab-cdef-1234-567890abcdef",
      marker_stream_entry_id: null,
    },
  ],
};

const streamFixture: StreamResponse = {
  next_cursor: null,
  entries: [
    {
      id: "entry_answer",
      timestamp: baseTs + 2,
      kind: "agent_msg",
      content: "full answer body",
      turn_id: "turn_user",
      sender_entity_id: null,
      reply_target_entity_id: null,
      session_id: "s_user",
      sender_label: null,
      session_label: "operator",
      audience_label: null,
    },
  ],
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function renderActivity() {
  const ws = installMockWebSocket();
  const requests: Array<{ url: string; method: string; body: unknown }> = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    requests.push({
      url,
      method,
      body: init?.body === undefined ? null : JSON.parse(String(init.body)),
    });

    if (url === "/api/activity") {
      return json(activityFixture);
    }
    if (url === `/api/activity?day=${day}`) {
      return json(activityFixture);
    }
    if (url === `/api/activity?day=${previousDay}`) {
      return json({ ...activityFixture, day: previousDay, rows: [] });
    }
    if (url === "/api/autonomy") {
      return json(autonomyFixture);
    }
    if (url === "/api/journal?limit=10") {
      return json(journalFixture);
    }
    if (url === "/api/stream?session=s_user&limit=200") {
      return json(streamFixture);
    }
    return json({}, 404);
  });

  render(
    <LiveProvider>
      <ActivityPage />
    </LiveProvider>,
  );

  return { requests, ws };
}

describe("ActivityPage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("renders feed rows, digest stats, trigger chips, and origin filters", async () => {
    renderActivity();

    expect(await screen.findByText("answer from user turn")).toBeTruthy();
    expect(screen.getAllByText("scheduled_reflection").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("3 changes")).toBeTruthy();
    expect(screen.getByText("journal notes")).toBeTruthy();
    expect(screen.getByText("private note")).toBeTruthy();
    expect(screen.getByText("JUN 10 · 09:55")).toBeTruthy();
    expect(screen.queryByText(/JUN 11 ·/)).toBeNull();
    expect(screen.getByText("12345678").getAttribute("title")).toBe(
      "12345678-90ab-cdef-1234-567890abcdef",
    );
    expect(screen.getByLabelText("scheduled_reflection state unknown")).toBeTruthy();
    expect(screen.getByText(/state unknown/)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "AUTONOMOUS" }));

    expect(screen.queryByText("answer from user turn")).toBeNull();
    expect(screen.getByText("finalizer_no_output")).toBeTruthy();
  });

  it("fetches row details lazily on expand and renders journal notes for the turn", async () => {
    const { requests } = renderActivity();

    await screen.findByText("answer from user turn");
    expect(requests.some((request) => request.url.startsWith("/api/stream"))).toBe(false);

    fireEvent.click(screen.getByText("answer from user turn"));

    expect(await screen.findByText("answered: full answer body")).toBeTruthy();
    expect(screen.getAllByText("private note").length).toBeGreaterThanOrEqual(2);
    expect(requests.some((request) => request.url === "/api/stream?session=s_user&limit=200")).toBe(
      true,
    );
  });

  it("renders cached live phase durations only after observed frames", async () => {
    const { ws } = renderActivity();

    await screen.findByText("answer from user turn");
    fireEvent.click(screen.getByText("answer from user turn"));
    expect(screen.queryByText(/DELIB 20ms/)).toBeNull();

    act(() => {
      ws.instances[0]!.receive({
        type: "turn:phase:completed",
        ts: 1,
        event: "turn_phase.completed",
        data: {
          turnId: "turn_user",
          turn_id: "turn_user",
          session_id: "s_user",
          phase: "retrieval",
          ts: 1,
          duration_ms: 10,
        },
      });
      ws.instances[0]!.receive({
        type: "turn:phase:completed",
        ts: 2,
        event: "turn_phase.completed",
        data: {
          turnId: "turn_user",
          turn_id: "turn_user",
          session_id: "s_user",
          phase: "delib",
          ts: 2,
          duration_ms: 20,
        },
      });
    });

    expect(await screen.findByText(/DELIB 20ms/)).toBeTruthy();
  });

  it("refetches with a day query when a day tab is selected", async () => {
    const { requests } = renderActivity();

    await screen.findByText("answer from user turn");
    expect(requests.some((request) => request.url === "/api/journal?limit=10")).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "JUN 10" }));

    await waitFor(() =>
      expect(requests.some((request) => request.url === `/api/activity?day=${previousDay}`)).toBe(
        true,
      ),
    );
    expect(requests.filter((request) => request.url.startsWith("/api/journal")).length).toBe(1);
    expect(requests.some((request) => request.url.startsWith("/api/journal?limit=10&day="))).toBe(
      false,
    );
  });
});
