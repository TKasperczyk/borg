import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { postTurn } from "../api/client";
import type {
  EvidenceLedger,
  LiveFrame,
  LiveTokenFlushFrame,
  LiveTokenFrame,
  PhaseEventData,
  StreamEntry,
  TurnTerminalFrame,
  TurnTerminalOutcome,
  TurnPhaseFrame,
  TurnPhaseName,
  TurnRequest,
} from "../api/types";
import { formatTime, sortStreamEntries, streamContentText } from "../lib/stream-utils";
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
  { id: "ingest", name: "ingest" },
  { id: "audience", name: "audience" },
  { id: "perception", name: "perception" },
  { id: "frame", name: "frame gate" },
  { id: "extract", name: "extraction" },
  { id: "closure_loop", name: "closure-loop check" },
  { id: "generation_gate", name: "generation gate" },
  { id: "retrieval", name: "retrieval" },
  { id: "ledger", name: "evidence ledger" },
  { id: "shared", name: "shared state" },
  { id: "delib", name: "deliberation" },
  { id: "final", name: "finalizer" },
  { id: "guards", name: "guards" },
  { id: "persist", name: "persistence" },
  { id: "reflect", name: "reflection" },
];

export function initialPhases(): PhaseState[] {
  return PHASES.map((phase) => ({
    ...phase,
    sub: "waiting",
    status: "queue",
  }));
}

function turnIdFromPhase(data: PhaseEventData): string {
  return data.turn_id ?? data.turnId;
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

function turnIdFromTerminal(frame: TurnTerminalFrame): string {
  return frame.data.turn_id ?? frame.data.turnId;
}

function turnIdFromLiveFrame(frame: LiveFrame): string | null {
  if (
    frame.type === "turn:token" ||
    frame.type === "turn:token:flush" ||
    frame.type === "evidence_ledger:built" ||
    frame.type === "turn:delib_path" ||
    frame.type === "turn:final_attempt"
  ) {
    return frame.turn_id;
  }

  if (frame.type === "turn:terminal") {
    return turnIdFromTerminal(frame);
  }

  if (
    frame.type === "turn:phase:started" ||
    frame.type === "turn:phase:completed" ||
    frame.type === "turn:phase:failed"
  ) {
    return turnIdFromPhase(frame.data);
  }

  return null;
}

function tokenKey(turnId: string, phase: TurnPhaseName): string {
  return `${turnId}:${phase}`;
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
  if (frame.type === "turn:token" || frame.type === "turn:token:flush") {
    return [];
  }

  if (frame.type === "stream:append") {
    return frame.entries.map((entry) => ({
      id: `${frame.type}:${entry.id}`,
      ts: formatTime(frame.ts),
      kind: tailKindForEntry(entry),
      body: `${entry.kind} · ${streamContentText(entry.content)}`,
      isNew: true,
    }));
  }

  if (frame.type === "evidence_ledger:built") {
    const count =
      frame.ledger?.sections.reduce((sum, section) => sum + section.entries.length, 0) ?? 0;
    return [
      {
        id: `${frame.type}:${frame.turn_id}:${frame.ts}`,
        ts: formatTime(frame.ts),
        kind: "tool",
        body: `evidence ledger built · turn ${frame.turn_id} · ${count} entries`,
        isNew: true,
      },
    ];
  }

  if (frame.type === "turn:terminal") {
    const turnId = turnIdFromTerminal(frame);
    return [
      {
        id: `${frame.type}:${turnId}:${frame.ts}`,
        ts: formatTime(frame.ts),
        kind: "internal",
        body: `terminal · turn ${turnId} · ${frame.data.outcome}`,
        isNew: true,
      },
    ];
  }

  if (frame.type === "turn:delib_path") {
    return [
      {
        id: `${frame.type}:${frame.turn_id}:${frame.ts}`,
        ts: formatTime(frame.ts),
        kind: "thought",
        body: `deliberation path · ${frame.path}`,
        isNew: true,
      },
    ];
  }

  if (frame.type === "turn:final_attempt") {
    return [
      {
        id: `${frame.type}:${frame.turn_id}:${frame.attempt}:${frame.ts}`,
        ts: formatTime(frame.ts),
        kind: "internal",
        body: `finalizer re-attempt · #${frame.attempt}`,
        isNew: true,
      },
    ];
  }

  if (frame.type === "dream:process:started") {
    return [
      {
        id: `${frame.type}:${frame.process}:${frame.ts}`,
        ts: formatTime(frame.ts),
        kind: "dream",
        body: `dream process started · ${frame.process}`,
        isNew: true,
      },
    ];
  }

  if (frame.type === "dream:process:completed") {
    return [
      {
        id: `${frame.type}:${frame.process}:${frame.ts}`,
        ts: formatTime(frame.ts),
        kind: "dream",
        body: `dream process ${frame.errors > 0 ? "failed" : "completed"} · ${frame.process}${
          frame.duration_ms === undefined ? "" : ` · ${Math.round(frame.duration_ms)}ms`
        }`,
        isNew: true,
      },
    ];
  }

  return [
    {
      id: `${frame.type}:${turnIdFromPhase(frame.data)}:${frame.data.phase ?? "unknown"}:${frame.ts}`,
      ts: formatTime(frame.ts),
      kind: phaseTailKind(frame),
      body: `${frame.type.replace("turn:phase:", "")} · ${frame.data.phase ?? "unknown"}${frame.data.sub === undefined ? "" : ` · ${frame.data.sub}`}`,
      isNew: true,
    },
  ];
}

function tailRowsFromEntries(entries: readonly StreamEntry[]): TailEvent[] {
  return sortStreamEntries(entries)
    .slice(-60)
    .reverse()
    .map((entry) => ({
      id: `stream:append:${entry.id}`,
      ts: formatTime(entry.timestamp),
      kind: tailKindForEntry(entry),
      body: `${entry.kind} · ${streamContentText(entry.content)}`,
      isNew: false,
    }));
}

function updatePhase(phases: PhaseState[], frame: TurnPhaseFrame): PhaseState[] {
  const phaseName = frame.data.phase;
  if (phaseName === undefined) {
    return phases;
  }

  const status: PhaseStatus =
    frame.type === "turn:phase:started"
      ? "running"
      : frame.type === "turn:phase:completed"
        ? "done"
        : "fail";

  return phases.map((phase) => {
    if (phase.id !== phaseName) {
      return phase;
    }

    return {
      ...phase,
      status,
      sub: frame.data.sub ?? phase.sub,
      durationMs: frame.data.duration_ms ?? phase.durationMs,
      startedAt: frame.type === "turn:phase:started" ? frame.ts : phase.startedAt,
    };
  });
}

function appendTokenText(
  current: Map<string, string>,
  frame: LiveTokenFrame | LiveTokenFlushFrame,
): Map<string, string> {
  const next = new Map(current);
  const key = tokenKey(frame.turn_id, frame.phase);

  if (frame.type === "turn:token:flush") {
    next.set(key, frame.full_text);
    return next;
  }

  next.set(key, `${next.get(key) ?? ""}${frame.chunk_text}`);
  return next;
}

export type TurnStreamState = {
  activeTurnId: string | null;
  running: boolean;
  phases: PhaseState[];
  tokenTextByPhase: Map<string, string>;
  terminalOutcome: TurnTerminalOutcome | null;
  delibPath: "system_1" | "system_2" | null;
  finalAttempt: number;
  eventTail: TailEvent[];
  ledgerByTurn: Map<string, EvidenceLedger>;
  lastPhase: string;
  runTurn: (input: TurnRequest) => Promise<boolean>;
  resetForReconnect: () => void;
  replaceTailFromEntries: (entries: readonly StreamEntry[]) => void;
};

export function useTurnStream(live: LiveEvents): TurnStreamState {
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [phases, setPhases] = useState<PhaseState[]>(initialPhases);
  const [tokenTextByPhase, setTokenTextByPhase] = useState(() => new Map<string, string>());
  const [terminalOutcome, setTerminalOutcome] = useState<TurnTerminalOutcome | null>(null);
  const [delibPath, setDelibPath] = useState<"system_1" | "system_2" | null>(null);
  const [finalAttempt, setFinalAttempt] = useState(1);
  const [eventTail, setEventTail] = useState<TailEvent[]>([]);
  const [ledgerByTurn, setLedgerByTurn] = useState(() => new Map<string, EvidenceLedger>());
  const [lastPhase, setLastPhase] = useState("idle");
  const clearTimersRef = useRef<number[]>([]);
  const reflectTimeoutRef = useRef<number | null>(null);
  const activeTurnIdRef = useRef<string | null>(null);
  const lastDrivenTurnIdRef = useRef<string | null>(null);
  const pendingTurnRef = useRef<{
    requestSeq: number;
    turnId: string | null;
    ignoredTurnIds: ReadonlySet<string>;
  } | null>(null);
  const runSeqRef = useRef(0);

  useEffect(() => {
    activeTurnIdRef.current = activeTurnId;
    if (activeTurnId !== null) {
      lastDrivenTurnIdRef.current = activeTurnId;
    }
  }, [activeTurnId]);

  const acceptsTurnFrame = useCallback((turnId: string): boolean => {
    const activeTurnId = activeTurnIdRef.current;

    if (activeTurnId !== null) {
      return turnId === activeTurnId;
    }

    const pendingTurn = pendingTurnRef.current;

    if (pendingTurn === null) {
      return false;
    }

    if (pendingTurn.turnId !== null) {
      return turnId === pendingTurn.turnId;
    }

    if (pendingTurn.ignoredTurnIds.has(turnId)) {
      return false;
    }

    pendingTurn.turnId = turnId;
    activeTurnIdRef.current = turnId;
    lastDrivenTurnIdRef.current = turnId;
    setActiveTurnId(turnId);
    return true;
  }, []);

  const clearReflectTimeout = useCallback(() => {
    if (reflectTimeoutRef.current !== null) {
      window.clearTimeout(reflectTimeoutRef.current);
      reflectTimeoutRef.current = null;
    }
  }, []);

  const markTailSettled = useCallback((ids: readonly string[]) => {
    const timer = window.setTimeout(() => {
      setEventTail((current) =>
        current.map((event) => (ids.includes(event.id) ? { ...event, isNew: false } : event)),
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
    [markTailSettled],
  );

  useEffect(
    () => () => {
      for (const timer of clearTimersRef.current) {
        window.clearTimeout(timer);
      }
      clearReflectTimeout();
    },
    [clearReflectTimeout],
  );

  useEffect(() => {
    return live.subscribe((frame) => {
      const frameTurnId = turnIdFromLiveFrame(frame);

      if (frameTurnId !== null && !acceptsTurnFrame(frameTurnId)) {
        return;
      }

      pushTail(tailRowsFromFrame(frame));

      if (frame.type === "evidence_ledger:built" && frame.ledger !== null) {
        const ledger = frame.ledger;
        setLedgerByTurn((current) => {
          const next = new Map(current);
          next.set(frame.turn_id, ledger);
          return next;
        });
        return;
      }

      if (frame.type === "turn:token" || frame.type === "turn:token:flush") {
        setTokenTextByPhase((current) => appendTokenText(current, frame));
        return;
      }

      if (frame.type === "turn:delib_path") {
        setDelibPath(frame.path);
        return;
      }

      if (frame.type === "turn:final_attempt") {
        setFinalAttempt(frame.attempt);
        return;
      }

      if (frame.type === "turn:terminal") {
        const frameTurnId = turnIdFromTerminal(frame);
        const currentTurnId = activeTurnIdRef.current;

        setLastPhase(`terminal ${frame.data.outcome}`);
        setTerminalOutcome(frame.data.outcome);
        setTokenTextByPhase(new Map());

        if (currentTurnId === null || currentTurnId === frameTurnId) {
          clearReflectTimeout();
          setRunning(false);
        }

        return;
      }

      if (!frame.type.startsWith("turn:phase:")) {
        return;
      }

      const phaseFrame = frame as TurnPhaseFrame;
      setPhases((current) => updatePhase(current, phaseFrame));

      if (phaseFrame.data.phase !== undefined) {
        const timing =
          phaseFrame.data.duration_ms === undefined
            ? ""
            : ` ${Math.round(phaseFrame.data.duration_ms)}ms`;
        const suffix =
          phaseFrame.type === "turn:phase:failed"
            ? "failed"
            : phaseFrame.type === "turn:phase:completed"
              ? "ok"
              : "run";
        setLastPhase(`${phaseFrame.data.phase} ${suffix}${timing}`);
      }

      if (phaseFrame.type === "turn:phase:completed" && phaseFrame.data.phase === "reflect") {
        clearReflectTimeout();
        setRunning(false);
      }

      if (phaseFrame.type === "turn:phase:failed") {
        clearReflectTimeout();
        setRunning(false);
      }
    });
  }, [acceptsTurnFrame, clearReflectTimeout, live, pushTail]);

  const runTurn = useCallback(
    async (input: TurnRequest) => {
      if (running) {
        return false;
      }

      const requestSeq = runSeqRef.current + 1;
      runSeqRef.current = requestSeq;
      pendingTurnRef.current = {
        requestSeq,
        turnId: null,
        ignoredTurnIds: new Set(
          [activeTurnIdRef.current, lastDrivenTurnIdRef.current].filter(
            (turnId): turnId is string => turnId !== null,
          ),
        ),
      };
      activeTurnIdRef.current = null;
      setRunning(true);
      setActiveTurnId(null);
      setPhases(initialPhases());
      setTokenTextByPhase(new Map());
      setTerminalOutcome(null);
      setDelibPath(null);
      setFinalAttempt(1);
      setLastPhase("turn queued");
      clearReflectTimeout();
      reflectTimeoutRef.current = window.setTimeout(() => {
        reflectTimeoutRef.current = null;
        setRunning(false);
        setLastPhase("reflect timeout");
      }, TURN_REFLECT_TIMEOUT_MS);

      try {
        const result = await postTurn(input);
        const pendingTurn = pendingTurnRef.current;

        if (runSeqRef.current === requestSeq && pendingTurn?.requestSeq === requestSeq) {
          if (pendingTurn.turnId === null) {
            pendingTurn.turnId = result.turn_id;
            activeTurnIdRef.current = result.turn_id;
            lastDrivenTurnIdRef.current = result.turn_id;
            setActiveTurnId(result.turn_id);
          }
        }
        return true;
      } catch {
        if (runSeqRef.current === requestSeq) {
          pendingTurnRef.current = null;
          clearReflectTimeout();
          setRunning(false);
          setLastPhase("turn failed");
        }
        return false;
      }
    },
    [clearReflectTimeout, running],
  );

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
      tokenTextByPhase,
      terminalOutcome,
      delibPath,
      finalAttempt,
      eventTail,
      ledgerByTurn,
      lastPhase,
      runTurn,
      resetForReconnect,
      replaceTailFromEntries,
    }),
    [
      activeTurnId,
      delibPath,
      eventTail,
      finalAttempt,
      lastPhase,
      ledgerByTurn,
      phases,
      terminalOutcome,
      tokenTextByPhase,
      replaceTailFromEntries,
      resetForReconnect,
      runTurn,
      running,
    ],
  );
}
