import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { EvidenceLedger, LiveFrame, TurnPhaseName, WsState } from "../api/types";
import type { LiveEventHandler, LiveEvents } from "./use-live-events";
import { TURN_SNAPSHOT_CACHE_LIMIT, useTurnStream } from "./use-turn-stream";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function makeLiveSource(): {
  emit: (frame: LiveFrame) => void;
  live: (connectionCount?: number, wsState?: WsState) => LiveEvents;
} {
  const handlers = new Set<LiveEventHandler>();
  return {
    live: (connectionCount = 1, wsState = "live") => ({
      wsState,
      connectionCount,
      subscribe: (handler) => {
        handlers.add(handler);
        return () => {
          handlers.delete(handler);
        };
      },
    }),
    emit: (frame) => {
      for (const handler of handlers) {
        handler(frame);
      }
    },
  };
}

function phaseFrame(
  type: "turn:phase:started" | "turn:phase:completed" | "turn:phase:failed",
  phase: TurnPhaseName,
  turnId: string,
  sessionId?: string,
): LiveFrame {
  const event =
    type === "turn:phase:started"
      ? "turn_phase.started"
      : type === "turn:phase:completed"
        ? "turn_phase.completed"
        : "turn_phase.failed";

  return {
    type,
    event,
    ts: Date.now(),
    data: {
      turnId,
      turn_id: turnId,
      session_id: sessionId,
      phase,
      duration_ms: type === "turn:phase:started" ? undefined : 12,
      sub: type === "turn:phase:started" ? "running" : "done",
    },
  };
}

function terminalFrame(turnId: string, sessionId?: string): LiveFrame {
  return {
    type: "turn:terminal",
    event: "turn.terminal",
    ts: Date.now(),
    data: {
      turnId,
      turn_id: turnId,
      session_id: sessionId,
      outcome: "reflected",
      duration_ms: 42,
    },
  };
}

function DetailProbe({ live }: { live: LiveEvents }) {
  const turnStream = useTurnStream(live, { sessionId: "default" });
  const detailLines = [...turnStream.detailByPhase.values()].flat();

  return (
    <>
      <button
        type="button"
        onClick={() => {
          void turnStream.runTurn({
            message: "hello borg",
            external_message_id: "detail-test-message",
            audience: "alice",
            session: "default",
          });
        }}
      >
        start
      </button>
      <pre data-testid="details">{detailLines.join("\n")}</pre>
    </>
  );
}

function StateProbe({ live, sessionId = "default" }: { live: LiveEvents; sessionId?: string }) {
  const turnStream = useTurnStream(live, { sessionId });
  const phaseStatus = (phase: TurnPhaseName) =>
    turnStream.phases.find((candidate) => candidate.id === phase)?.status ?? "missing";

  return (
    <>
      <button
        type="button"
        onClick={() => {
          void turnStream.runTurn({
            message: "hello borg",
            external_message_id: "state-test-message",
            audience: "alice",
            session: sessionId,
          });
        }}
      >
        start
      </button>
      <output data-testid="active-turn">{turnStream.activeTurnId ?? "idle"}</output>
      <output data-testid="running">{String(turnStream.running)}</output>
      <output data-testid="ingest-status">{phaseStatus("ingest")}</output>
      <output data-testid="retrieval-status">{phaseStatus("retrieval")}</output>
      <output data-testid="token-text">{[...turnStream.tokenTextByPhase.values()].join("")}</output>
    </>
  );
}

function SnapshotProbe({ live }: { live: LiveEvents }) {
  const turnStream = useTurnStream(live, { sessionId: "default" });
  const cached = turnStream.flowSnapshotByTurn.get("turn_cached");

  return (
    <>
      <output data-testid="snapshot-size">{turnStream.flowSnapshotByTurn.size}</output>
      <output data-testid="snapshot-text">
        {cached?.tokenTextByPhase.get("turn_cached:final") ?? ""}
      </output>
      <output data-testid="snapshot-has-oldest">
        {String(turnStream.flowSnapshotByTurn.has("turn_0"))}
      </output>
      <output data-testid="snapshot-has-latest">
        {String(turnStream.flowSnapshotByTurn.has(`turn_${TURN_SNAPSHOT_CACHE_LIMIT}`))}
      </output>
    </>
  );
}

function LedgerProbe({ live }: { live: LiveEvents }) {
  const turnStream = useTurnStream(live, { sessionId: "default" });
  const entry = turnStream.ledgerByTurn.get("turn_private")?.sections[0]?.entries[0];

  return (
    <>
      <output data-testid="ledger-class">
        {entry?.disclosure_label?.disclosure_class ?? "missing"}
      </output>
      <output data-testid="ledger-note">{entry?.disclosure_note ?? "missing"}</output>
      <output data-testid="ledger-audience">
        {entry?.current_audience_entity_id ?? "missing"}
      </output>
    </>
  );
}

function emptyLedger(): EvidenceLedger {
  return {
    sections: [],
    sharedState: null,
    transcriptIncluded: false,
    transcriptCompacted: false,
    originalTranscriptTokenEstimate: 0,
    compactedTranscriptEntryCount: 0,
    rawPreservedUserTranscriptEntryCount: 0,
    estimatedTokens: 0,
  };
}

function CacheResetProbe({ live }: { live: LiveEvents }) {
  const turnStream = useTurnStream(live, { sessionId: "default" });
  const [result, setResult] = useState("none");

  return (
    <>
      <button type="button" onClick={() => setResult(String(turnStream.resetCaches()))}>
        reset caches
      </button>
      <output data-testid="reset-result">{result}</output>
      <output data-testid="cache-running">{String(turnStream.running)}</output>
      <output data-testid="cache-snapshots">{turnStream.flowSnapshotByTurn.size}</output>
      <output data-testid="cache-tail">{turnStream.eventTail.length}</output>
      <output data-testid="cache-ledgers">{turnStream.ledgerByTurn.size}</output>
    </>
  );
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("useTurnStream", () => {
  it("observes selected-session external turns and switches to the next external turn", () => {
    const source = makeLiveSource();

    render(<StateProbe live={source.live()} />);

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "ingest", "turn_without_session"));
    });

    expect(screen.getByTestId("active-turn")).toHaveTextContent("idle");
    expect(screen.getByTestId("running")).toHaveTextContent("false");

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "ingest", "turn_external_a", "default"));
      source.emit({
        type: "turn:token",
        ts: Date.now(),
        session_id: "default",
        turn_id: "turn_external_a",
        phase: "ingest",
        chunk_text: "external A",
        sequence: 1,
      });
    });

    expect(screen.getByTestId("active-turn")).toHaveTextContent("turn_external_a");
    expect(screen.getByTestId("running")).toHaveTextContent("true");
    expect(screen.getByTestId("ingest-status")).toHaveTextContent("running");
    expect(screen.getByTestId("token-text")).toHaveTextContent("external A");

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "retrieval", "turn_external_b", "default"));
    });

    expect(screen.getByTestId("active-turn")).toHaveTextContent("turn_external_b");
    expect(screen.getByTestId("ingest-status")).toHaveTextContent("queue");
    expect(screen.getByTestId("retrieval-status")).toHaveTextContent("running");
    expect(screen.getByTestId("token-text")).toHaveTextContent("");

    act(() => {
      source.emit(terminalFrame("turn_external_a", "default"));
    });

    expect(screen.getByTestId("active-turn")).toHaveTextContent("turn_external_b");
    expect(screen.getByTestId("retrieval-status")).toHaveTextContent("running");
  });

  it("lets an operator turn preempt an observed turn and keep control until completion", async () => {
    const source = makeLiveSource();
    const fetchMock = vi.fn(() =>
      Promise.resolve(
        jsonResponse({ ok: true, status: "enqueued", stream_entry_id: "strm_user_operator" }),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<StateProbe live={source.live()} />);

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "ingest", "turn_observed", "default"));
    });
    expect(screen.getByTestId("active-turn")).toHaveTextContent("turn_observed");

    fireEvent.click(screen.getByRole("button", { name: "start" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(screen.getByTestId("active-turn")).toHaveTextContent("idle");

    act(() => {
      source.emit(phaseFrame("turn:phase:completed", "ingest", "turn_observed", "default"));
      source.emit(phaseFrame("turn:phase:started", "ingest", "turn_operator", "default"));
      source.emit(phaseFrame("turn:phase:started", "retrieval", "turn_external", "default"));
    });

    expect(screen.getByTestId("active-turn")).toHaveTextContent("turn_operator");
    expect(screen.getByTestId("ingest-status")).toHaveTextContent("running");
    expect(screen.getByTestId("retrieval-status")).toHaveTextContent("queue");

    act(() => {
      source.emit({
        type: "stream:append",
        ts: Date.now(),
        entries: [
          {
            id: "strm_agent_operator",
            timestamp: 1,
            kind: "agent_msg",
            content: "operator answer",
            turn_id: "turn_operator",
            audience: "alice",
            sender_entity_id: null,
            reply_target_entity_id: null,
            response_to: {
              source_entry_ids: ["strm_user_operator"],
            },
            session_id: "default",
            compressed: false,
          },
        ],
      });
      source.emit(terminalFrame("turn_operator", "default"));
      source.emit(phaseFrame("turn:phase:started", "retrieval", "turn_external", "default"));
    });

    expect(screen.getByTestId("active-turn")).toHaveTextContent("turn_external");
    expect(screen.getByTestId("retrieval-status")).toHaveTextContent("running");
  });

  it("accumulates active turn phase detail frames by phase", async () => {
    const source = makeLiveSource();
    const fetchMock = vi.fn(() =>
      Promise.resolve(
        jsonResponse({ ok: true, status: "enqueued", stream_entry_id: "strm_user_abc" }),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<DetailProbe live={source.live()} />);

    fireEvent.click(screen.getByRole("button", { name: "start" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    act(() => {
      source.emit({
        type: "turn:phase:detail",
        ts: Date.now(),
        turn_id: "turn_hook",
        session_id: "default",
        phase: "retrieval",
        event: "retrieval.completed",
        summary: "episodeCount=2 semanticHits=4 confidence=0.82",
      });
    });

    await waitFor(() =>
      expect(screen.getByTestId("details")).toHaveTextContent(
        "retrieval.completed · episodeCount=2 semanticHits=4 confidence=0.82",
      ),
    );
  });

  it("retains bounded per-turn flow snapshots from live frames", () => {
    const source = makeLiveSource();

    render(<SnapshotProbe live={source.live()} />);

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "final", "turn_cached", "default"));
      source.emit({
        type: "turn:token",
        ts: Date.now(),
        turn_id: "turn_cached",
        session_id: "default",
        phase: "final",
        chunk_text: "cached text",
        sequence: 1,
      });
      source.emit(terminalFrame("turn_cached", "default"));
    });

    expect(screen.getByTestId("snapshot-text")).toHaveTextContent("cached text");

    act(() => {
      for (let index = 0; index <= TURN_SNAPSHOT_CACHE_LIMIT; index += 1) {
        source.emit(terminalFrame(`turn_${index}`, "default"));
      }
    });

    expect(screen.getByTestId("snapshot-size")).toHaveTextContent(
      String(TURN_SNAPSHOT_CACHE_LIMIT),
    );
    expect(screen.getByTestId("snapshot-has-oldest")).toHaveTextContent("false");
    expect(screen.getByTestId("snapshot-has-latest")).toHaveTextContent("true");
  });

  it("does not reset retained caches during an active turn", () => {
    const source = makeLiveSource();

    render(<CacheResetProbe live={source.live()} />);

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "ingest", "turn_active", "default"));
    });

    expect(screen.getByTestId("cache-running")).toHaveTextContent("true");
    expect(screen.getByTestId("cache-snapshots")).toHaveTextContent("1");
    expect(screen.getByTestId("cache-tail")).toHaveTextContent("1");

    fireEvent.click(screen.getByRole("button", { name: "reset caches" }));

    expect(screen.getByTestId("reset-result")).toHaveTextContent("false");
    expect(screen.getByTestId("cache-snapshots")).toHaveTextContent("1");
    expect(screen.getByTestId("cache-tail")).toHaveTextContent("1");
  });

  it("resets retained turn caches when idle", () => {
    const source = makeLiveSource();

    render(<CacheResetProbe live={source.live()} />);

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "ingest", "turn_idle", "default"));
      source.emit({
        type: "evidence_ledger:built",
        ts: Date.now(),
        session_id: "default",
        turn_id: "turn_idle",
        ledger: emptyLedger(),
      });
      source.emit(terminalFrame("turn_idle", "default"));
    });

    expect(screen.getByTestId("cache-running")).toHaveTextContent("false");
    expect(screen.getByTestId("cache-snapshots")).toHaveTextContent("1");
    expect(screen.getByTestId("cache-tail")).toHaveTextContent("3");
    expect(screen.getByTestId("cache-ledgers")).toHaveTextContent("1");

    fireEvent.click(screen.getByRole("button", { name: "reset caches" }));

    expect(screen.getByTestId("reset-result")).toHaveTextContent("true");
    expect(screen.getByTestId("cache-snapshots")).toHaveTextContent("0");
    expect(screen.getByTestId("cache-tail")).toHaveTextContent("0");
    expect(screen.getByTestId("cache-ledgers")).toHaveTextContent("0");
  });

  it("normalizes disclosure metadata on live cached ledgers", () => {
    const source = makeLiveSource();

    render(<LedgerProbe live={source.live()} />);

    act(() => {
      source.emit({
        type: "evidence_ledger:built",
        ts: Date.now(),
        session_id: "default",
        turn_id: "turn_private",
        ledger: {
          ...emptyLedger(),
          sections: [
            {
              id: "shared_state",
              label: "shared state",
              entries: [
                {
                  id: "entry_private",
                  source_type: "shared_state",
                  session_scope: "global",
                  actor: "memory",
                  trust_rank: 1,
                  text: "private state",
                  state_metadata: {
                    disclosure_label: {
                      disclosure_class: "relationship_private",
                      origin_audience_entity_ids: ["ent_aaaaaaaaaaaaaaaa"],
                      private_to_entity_ids: ["ent_aaaaaaaaaaaaaaaa"],
                      public_to_entity_ids: [],
                    },
                    disclosure_note: "live private",
                    current_audience_entity_id: "ent_aaaaaaaaaaaaaaaa",
                  },
                },
              ],
            },
          ],
        },
      });
    });

    expect(screen.getByTestId("ledger-class")).toHaveTextContent("relationship_private");
    expect(screen.getByTestId("ledger-note")).toHaveTextContent("live private");
    expect(screen.getByTestId("ledger-audience")).toHaveTextContent("ent_aaaaaaaaaaaaaaaa");
  });
});
