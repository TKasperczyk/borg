import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { postTurn } from "../api/client";
import type {
  EvidenceLedger,
  LiveFrame,
  PhaseEventData,
  StreamEntry,
  TurnPhaseFrame,
  TurnPhaseName,
  TurnRequest
} from "../api/types";
import type { LiveEvents } from "./use-live-events";

export type PhaseStatus = "queue" | "running" | "done" | "fail";

export type PhaseState = {
  id: TurnPhaseName;
  name: string;
  sub: string;
  status: PhaseStatus;
  durationMs?: number;
  startedAt?: number;
};

export type TailEvent = {
  id: string;
  ts: string;
  kind: string;
  body: string;
  isNew: boolean;
};

const TURN_REFLECT_TIMEOUT_MS = 60_000;

const PHASES: ReadonlyArray<Pick<PhaseState, "id" | "name">> = [
  { id: "perception", name: "perception" },
  { id: "frame", name: "frame gate" },
  { id: "extract", name: "extraction" },
  { id: "retrieval", name: "retrieval" },
  { id: "ledger", name: "evidence ledger" },
  { id: "shared", name: "shared state" },
  { id: "delib", name: "deliberation" },
  { id: "reflect", name: "reflection" }
];

export function initialPhases(): PhaseState[] {
  return PHASES.map((phase) => ({
    ...phase,
    sub: "waiting",
    status: "queue"
  }));
}

function turnIdFromPhase(data: PhaseEventData): string {
  return data.turn_id ?? data.turnId;
}

function hms(ts: number): string {
  const date = new Date(ts);
  return date.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

function contentText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  try {
    return JSON.stringify(content ?? null);
  } catch {
    return String(content);
  }
}

function tailKindForEntry(entry: StreamEntry): string {
  if (entry.kind === "user_msg") {
    return "user";
  }
  if (entry.kind === "agent_msg") {
    return "assistant";
  }
  if (entry.kind === "user_image_attachment") {
    return "perception";
  }
  if (entry.kind === "tool_call" || entry.kind === "tool_result") {
    return "tool";
  }
  if (entry.kind === "dream_report") {
    return "dream";
  }
  return "internal";
}

function phaseTailKind(frame: TurnPhaseFrame): string {
  const phase = frame.data.phase;
  if (phase === "perception" || phase === "frame" || phase === "extract") {
    return "perception";
  }
  if (phase === "delib") {
    return "thought";
  }
  if (phase === "ledger" || phase === "shared") {
    return "tool";
  }
  return "internal";
}

function tailRowsFromFrame(frame: LiveFrame): TailEvent[] {
  if (frame.type === "stream:append") {
    return frame.entries.map((entry) => ({
      id: `${frame.type}:${entry.id}`,
      ts: hms(frame.ts),
      kind: tailKindForEntry(entry),
      body: `${entry.kind} · ${contentText(entry.content)}`,
      isNew: true
    }));
  }

  if (frame.type === "evidence_ledger:built") {
    const count = frame.ledger?.sections.reduce((sum, section) => sum + section.entries.length, 0) ?? 0;
    return [
      {
        id: `${frame.type}:${frame.turn_id}:${frame.ts}`,
        ts: hms(frame.ts),
        kind: "tool",
        body: `evidence ledger built · turn ${frame.turn_id} · ${count} entries`,
        isNew: true
      }
    ];
  }

  return [
    {
      id: `${frame.type}:${turnIdFromPhase(frame.data)}:${frame.data.phase ?? "unknown"}:${frame.ts}`,
      ts: hms(frame.ts),
      kind: phaseTailKind(frame),
      body: `${frame.type.replace("turn:phase:", "")} · ${frame.data.phase ?? "unknown"}${frame.data.sub === undefined ? "" : ` · ${frame.data.sub}`}`,
      isNew: true
    }
  ];
}

function tailRowsFromEntries(entries: readonly StreamEntry[]): TailEvent[] {
  return [...entries]
    .sort((left, right) => {
      if (left.timestamp !== right.timestamp) {
        return left.timestamp - right.timestamp;
      }
      return left.id.localeCompare(right.id);
    })
    .slice(-60)
    .reverse()
    .map((entry) => ({
      id: `stream:append:${entry.id}`,
      ts: hms(entry.timestamp),
      kind: tailKindForEntry(entry),
      body: `${entry.kind} · ${contentText(entry.content)}`,
      isNew: false
    }));
}

function updatePhase(phases: PhaseState[], frame: TurnPhaseFrame): PhaseState[] {
  const phaseName = frame.data.phase;
  if (phaseName === undefined) {
    return phases;
  }

  const status: PhaseStatus =
    frame.type === "turn:phase:started" ? "running" : frame.type === "turn:phase:completed" ? "done" : "fail";

  return phases.map((phase) => {
    if (phase.id !== phaseName) {
      return phase;
    }

    return {
      ...phase,
      status,
      sub: frame.data.sub ?? phase.sub,
      durationMs: frame.data.duration_ms ?? phase.durationMs,
      startedAt: frame.type === "turn:phase:started" ? frame.ts : phase.startedAt
    };
  });
}

export type TurnStreamState = {
  activeTurnId: string | null;
  running: boolean;
  phases: PhaseState[];
  eventTail: TailEvent[];
  ledgerByTurn: Map<string, EvidenceLedger>;
  lastPhase: string;
  runTurn: (input: TurnRequest) => Promise<void>;
  resetForReconnect: () => void;
  replaceTailFromEntries: (entries: readonly StreamEntry[]) => void;
};

export function useTurnStream(live: LiveEvents): TurnStreamState {
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [phases, setPhases] = useState<PhaseState[]>(initialPhases);
  const [eventTail, setEventTail] = useState<TailEvent[]>([]);
  const [ledgerByTurn, setLedgerByTurn] = useState(() => new Map<string, EvidenceLedger>());
  const [lastPhase, setLastPhase] = useState("idle");
  const clearTimersRef = useRef<number[]>([]);
  const reflectTimeoutRef = useRef<number | null>(null);

  const clearReflectTimeout = useCallback(() => {
    if (reflectTimeoutRef.current !== null) {
      window.clearTimeout(reflectTimeoutRef.current);
      reflectTimeoutRef.current = null;
    }
  }, []);

  const markTailSettled = useCallback((ids: readonly string[]) => {
    const timer = window.setTimeout(() => {
      setEventTail((current) =>
        current.map((event) => (ids.includes(event.id) ? { ...event, isNew: false } : event))
      );
    }, 800);
    clearTimersRef.current.push(timer);
  }, []);

  const pushTail = useCallback(
    (rows: TailEvent[]) => {
      if (rows.length === 0) {
        return;
      }
      setEventTail((current) => [...rows, ...current].slice(0, 60));
      markTailSettled(rows.map((row) => row.id));
    },
    [markTailSettled]
  );

  useEffect(
    () => () => {
      for (const timer of clearTimersRef.current) {
        window.clearTimeout(timer);
      }
      clearReflectTimeout();
    },
    [clearReflectTimeout]
  );

  useEffect(() => {
    return live.subscribe((frame) => {
      pushTail(tailRowsFromFrame(frame));

      if (frame.type === "evidence_ledger:built" && frame.ledger !== null) {
        const ledger = frame.ledger;
        setLedgerByTurn((current) => {
          const next = new Map(current);
          next.set(frame.turn_id, ledger);
          return next;
        });
        setActiveTurnId((current) => current ?? frame.turn_id);
        return;
      }

      if (!frame.type.startsWith("turn:phase:")) {
        return;
      }

      const phaseFrame = frame as TurnPhaseFrame;
      const frameTurnId = turnIdFromPhase(phaseFrame.data);
      setActiveTurnId((current) => current ?? frameTurnId);
      setPhases((current) => updatePhase(current, phaseFrame));

      if (phaseFrame.data.phase !== undefined) {
        const timing =
          phaseFrame.data.duration_ms === undefined ? "" : ` ${Math.round(phaseFrame.data.duration_ms)}ms`;
        const suffix =
          phaseFrame.type === "turn:phase:failed" ? "failed" : phaseFrame.type === "turn:phase:completed" ? "ok" : "run";
        setLastPhase(`${phaseFrame.data.phase} ${suffix}${timing}`);
      }

      if (
        phaseFrame.type === "turn:phase:completed" &&
        phaseFrame.data.phase === "reflect"
      ) {
        clearReflectTimeout();
        setRunning(false);
      }

      if (phaseFrame.type === "turn:phase:failed") {
        clearReflectTimeout();
        setRunning(false);
      }
    });
  }, [clearReflectTimeout, live, pushTail]);

  const runTurn = useCallback(async (input: TurnRequest) => {
    if (running) {
      return;
    }

    setRunning(true);
    setActiveTurnId(null);
    setPhases(initialPhases());
    setLastPhase("turn queued");
    clearReflectTimeout();
    reflectTimeoutRef.current = window.setTimeout(() => {
      reflectTimeoutRef.current = null;
      setRunning(false);
      setLastPhase("reflect timeout");
    }, TURN_REFLECT_TIMEOUT_MS);

    try {
      const result = await postTurn(input);
      setActiveTurnId(result.turn_id);
    } catch {
      clearReflectTimeout();
      setRunning(false);
      setLastPhase("turn failed");
    }
  }, [clearReflectTimeout, running]);

  const resetForReconnect = useCallback(() => {
    setLastPhase("ws reconnected");
  }, []);

  const replaceTailFromEntries = useCallback((entries: readonly StreamEntry[]) => {
    setEventTail(tailRowsFromEntries(entries));
  }, []);

  return useMemo(
    () => ({
      activeTurnId,
      running,
      phases,
      eventTail,
      ledgerByTurn,
      lastPhase,
      runTurn,
      resetForReconnect,
      replaceTailFromEntries
    }),
    [
      activeTurnId,
      eventTail,
      lastPhase,
      ledgerByTurn,
      phases,
      replaceTailFromEntries,
      resetForReconnect,
      runTurn,
      running
    ]
  );
}
