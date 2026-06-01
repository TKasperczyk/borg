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
  TurnPhaseDetailFrame,
  TurnPhaseFrame,
  TurnPhaseName,
  TurnRequest,
  TurnResponse,
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

type ActiveTurnDriver = "operator" | "observed";

const TURN_REFLECT_TIMEOUT_MS = 60_000;
const DETAIL_LINES_PER_PHASE = 10;

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
    frame.type === "turn:final_attempt" ||
    frame.type === "turn:phase:detail"
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

function responseToSourceEntryIds(entry: StreamEntry): string[] {
  const sourceEntryIds = entry.response_to?.source_entry_ids;
  if (!Array.isArray(sourceEntryIds)) {
    return [];
  }

  return sourceEntryIds.filter((entryId): entryId is string => typeof entryId === "string");
}

function sessionIdFromLiveFrame(frame: LiveFrame): string | null {
  if (frame.session_id !== undefined) {
    return frame.session_id;
  }

  if (frame.type === "stream:append") {
    return frame.entries[0]?.session_id ?? null;
  }

  if (
    frame.type === "turn:phase:started" ||
    frame.type === "turn:phase:completed" ||
    frame.type === "turn:phase:failed" ||
    frame.type === "turn:terminal"
  ) {
    return frame.data.session_id ?? null;
  }

  return null;
}

function tokenKey(turnId: string, phase: string): string {
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

  if (frame.type === "turn:phase:detail") {
    return [
      {
        id: `detail:${frame.turn_id}:${frame.event}:${frame.ts}`,
        ts: formatTime(frame.ts),
        kind: frame.event.split(".")[0] ?? frame.event,
        body: `${frame.event} · ${frame.summary}`,
        isNew: true,
      },
    ];
  }

  if (frame.type === "borg:reset") {
    return [];
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

  if (
    frame.type === "turn:phase:started" ||
    frame.type === "turn:phase:completed" ||
    frame.type === "turn:phase:failed"
  ) {
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

  return [];
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

function appendPhaseDetail(
  current: Map<string, string[]>,
  frame: TurnPhaseDetailFrame,
): Map<string, string[]> {
  if (frame.phase === undefined) {
    return current;
  }

  const next = new Map(current);
  const key = tokenKey(frame.turn_id, frame.phase);
  const line = `${frame.event} · ${frame.summary}`;
  next.set(key, [...(next.get(key) ?? []), line].slice(-DETAIL_LINES_PER_PHASE));
  return next;
}

export type TurnStreamState = {
  activeTurnId: string | null;
  running: boolean;
  phases: PhaseState[];
  tokenTextByPhase: Map<string, string>;
  detailByPhase: Map<string, string[]>;
  terminalOutcome: TurnTerminalOutcome | null;
  delibPath: "system_1" | "system_2" | null;
  finalAttempt: number;
  eventTail: TailEvent[];
  ledgerByTurn: Map<string, EvidenceLedger>;
  lastPhase: string;
  runTurn: (input: TurnRequest) => Promise<TurnResponse | null>;
  resetForReconnect: () => void;
  replaceTailFromEntries: (entries: readonly StreamEntry[]) => void;
};

export function useTurnStream(
  live: LiveEvents,
  input: { sessionId?: string } = {},
): TurnStreamState {
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [phases, setPhases] = useState<PhaseState[]>(initialPhases);
  const [tokenTextByPhase, setTokenTextByPhase] = useState(() => new Map<string, string>());
  const [detailByPhase, setDetailByPhase] = useState(() => new Map<string, string[]>());
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
  const activeTurnDriverRef = useRef<ActiveTurnDriver | null>(null);
  const supersededTurnIdsRef = useRef<Set<string>>(new Set());
  const runningRef = useRef(false);
  const outstandingStreamEntryIdsRef = useRef<Set<string>>(new Set());
  const pendingPostCountRef = useRef(0);
  const pendingTurnRef = useRef<{
    requestSeq: number;
    turnId: string | null;
    ignoredTurnIds: ReadonlySet<string>;
  } | null>(null);
  const runSeqRef = useRef(0);

  const setRunningState = useCallback((next: boolean) => {
    runningRef.current = next;
    setRunning(next);
  }, []);

  useEffect(() => {
    activeTurnIdRef.current = activeTurnId;
    if (activeTurnId !== null) {
      lastDrivenTurnIdRef.current = activeTurnId;
    }
  }, [activeTurnId]);

  const clearReflectTimeout = useCallback(() => {
    if (reflectTimeoutRef.current !== null) {
      window.clearTimeout(reflectTimeoutRef.current);
      reflectTimeoutRef.current = null;
    }
  }, []);

  const resetLiveTurnDisplay = useCallback(
    (lastPhase: string) => {
      setRunningState(true);
      setPhases(initialPhases());
      setTokenTextByPhase(new Map());
      setDetailByPhase(new Map());
      setTerminalOutcome(null);
      setDelibPath(null);
      setFinalAttempt(1);
      setLastPhase(lastPhase);
      clearReflectTimeout();
      reflectTimeoutRef.current = window.setTimeout(() => {
        reflectTimeoutRef.current = null;
        setRunningState(false);
        setLastPhase("reflect timeout");
      }, TURN_REFLECT_TIMEOUT_MS);
    },
    [clearReflectTimeout, setRunningState],
  );

  const rememberSupersededTurnId = useCallback((turnId: string) => {
    supersededTurnIdsRef.current.add(turnId);
    if (supersededTurnIdsRef.current.size <= 32) {
      return;
    }

    const oldestTurnId = supersededTurnIdsRef.current.values().next().value;
    if (oldestTurnId !== undefined) {
      supersededTurnIdsRef.current.delete(oldestTurnId);
    }
  }, []);

  const beginPendingLiveTurn = useCallback(
    (ignoredTurnIds: ReadonlySet<string>) => {
      const requestSeq = runSeqRef.current + 1;
      runSeqRef.current = requestSeq;
      pendingTurnRef.current = {
        requestSeq,
        turnId: null,
        ignoredTurnIds,
      };
      activeTurnDriverRef.current = "operator";
      activeTurnIdRef.current = null;
      for (const turnId of ignoredTurnIds) {
        rememberSupersededTurnId(turnId);
      }
      setActiveTurnId(null);
      resetLiveTurnDisplay("turn queued");
    },
    [rememberSupersededTurnId, resetLiveTurnDisplay],
  );

  const canAdoptObservedTurn = useCallback(
    (frameTurnId: string, frameSessionId: string | null): boolean => {
      return (
        input.sessionId !== undefined &&
        frameSessionId === input.sessionId &&
        !(activeTurnDriverRef.current === "operator" && runningRef.current) &&
        pendingPostCountRef.current === 0 &&
        frameTurnId.length > 0
      );
    },
    [input.sessionId],
  );

  const beginObservedLiveTurn = useCallback(
    (turnId: string) => {
      const previousTurnId = activeTurnIdRef.current;
      if (previousTurnId !== null && previousTurnId !== turnId) {
        rememberSupersededTurnId(previousTurnId);
      }

      pendingTurnRef.current = null;
      activeTurnDriverRef.current = "observed";
      activeTurnIdRef.current = turnId;
      lastDrivenTurnIdRef.current = turnId;
      setActiveTurnId(turnId);
      resetLiveTurnDisplay("observing turn");
    },
    [rememberSupersededTurnId, resetLiveTurnDisplay],
  );

  const acceptsTurnFrame = useCallback(
    (turnId: string, frameSessionId: string | null): boolean => {
      if (supersededTurnIdsRef.current.has(turnId)) {
        return false;
      }

      const activeTurnId = activeTurnIdRef.current;

      if (activeTurnId !== null) {
        if (turnId === activeTurnId) {
          return true;
        }

        if (canAdoptObservedTurn(turnId, frameSessionId)) {
          beginObservedLiveTurn(turnId);
          return true;
        }

        return false;
      }

      if (activeTurnDriverRef.current === "operator" && runningRef.current) {
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
      }

      if (canAdoptObservedTurn(turnId, frameSessionId)) {
        beginObservedLiveTurn(turnId);
        return true;
      }

      return false;
    },
    [beginObservedLiveTurn, canAdoptObservedTurn],
  );

  const ignoredCurrentTurnIds = useCallback((): ReadonlySet<string> => {
    return new Set(
      [activeTurnIdRef.current, lastDrivenTurnIdRef.current].filter(
        (turnId): turnId is string => turnId !== null,
      ),
    );
  }, []);

  const beginPendingLiveTurnFromCurrent = useCallback(() => {
    beginPendingLiveTurn(ignoredCurrentTurnIds());
  }, [beginPendingLiveTurn, ignoredCurrentTurnIds]);

  const completeActiveLiveTurn = useCallback(
    (input: { releaseForOutstanding: boolean }) => {
      clearReflectTimeout();

      if (input.releaseForOutstanding && outstandingStreamEntryIdsRef.current.size > 0) {
        beginPendingLiveTurnFromCurrent();
        return;
      }

      setRunningState(false);
    },
    [beginPendingLiveTurnFromCurrent, clearReflectTimeout, setRunningState],
  );

  const markResponseSourcesAnswered = useCallback((entries: readonly StreamEntry[]) => {
    for (const entry of entries) {
      for (const sourceEntryId of responseToSourceEntryIds(entry)) {
        outstandingStreamEntryIdsRef.current.delete(sourceEntryId);
      }
    }
  }, []);

  useEffect(() => {
    pendingTurnRef.current = null;
    activeTurnIdRef.current = null;
    lastDrivenTurnIdRef.current = null;
    activeTurnDriverRef.current = null;
    supersededTurnIdsRef.current.clear();
    outstandingStreamEntryIdsRef.current.clear();
    pendingPostCountRef.current = 0;
    clearReflectTimeout();
    setActiveTurnId(null);
    setRunningState(false);
    setPhases(initialPhases());
    setTokenTextByPhase(new Map());
    setDetailByPhase(new Map());
    setTerminalOutcome(null);
    setDelibPath(null);
    setFinalAttempt(1);
    setEventTail([]);
    setLedgerByTurn(new Map());
    setLastPhase("idle");
  }, [clearReflectTimeout, input.sessionId, setRunningState]);

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
      const frameSessionId = sessionIdFromLiveFrame(frame);
      if (
        input.sessionId !== undefined &&
        frameSessionId !== null &&
        frameSessionId !== input.sessionId
      ) {
        return;
      }

      const frameTurnId = turnIdFromLiveFrame(frame);

      if (frameTurnId !== null && !acceptsTurnFrame(frameTurnId, frameSessionId)) {
        return;
      }

      if (frame.type === "stream:append") {
        markResponseSourcesAnswered(frame.entries);
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

      if (frame.type === "borg:reset") {
        setDetailByPhase(new Map());
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
        setDetailByPhase(new Map());

        if (currentTurnId === null || currentTurnId === frameTurnId) {
          completeActiveLiveTurn({ releaseForOutstanding: true });
        }

        return;
      }

      if (frame.type === "turn:phase:detail") {
        setDetailByPhase((current) => appendPhaseDetail(current, frame));
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
        completeActiveLiveTurn({ releaseForOutstanding: false });
      }

      if (phaseFrame.type === "turn:phase:failed") {
        completeActiveLiveTurn({ releaseForOutstanding: false });
      }
    });
  }, [
    acceptsTurnFrame,
    completeActiveLiveTurn,
    input.sessionId,
    live,
    markResponseSourcesAnswered,
    pushTail,
  ]);

  const runTurn = useCallback(
    async (input: TurnRequest) => {
      const waitingForLiveTurn =
        pendingTurnRef.current !== null &&
        pendingTurnRef.current.turnId === null &&
        activeTurnIdRef.current === null;
      let startedPendingTurn = false;

      if (
        activeTurnDriverRef.current === "observed" ||
        !runningRef.current ||
        (!waitingForLiveTurn && activeTurnIdRef.current === null)
      ) {
        beginPendingLiveTurnFromCurrent();
        startedPendingTurn = true;
      }

      pendingPostCountRef.current += 1;
      try {
        const result = await postTurn(input);
        if (result.status === "enqueued") {
          outstandingStreamEntryIdsRef.current.add(result.stream_entry_id);
          if (!runningRef.current) {
            beginPendingLiveTurnFromCurrent();
          }
        }
        return result;
      } catch {
        const remainingPosts = Math.max(0, pendingPostCountRef.current - 1);
        if (
          startedPendingTurn &&
          remainingPosts === 0 &&
          activeTurnIdRef.current === null &&
          pendingTurnRef.current?.turnId === null
        ) {
          pendingTurnRef.current = null;
          clearReflectTimeout();
          setRunningState(false);
          setLastPhase("turn failed");
        }
        return null;
      } finally {
        pendingPostCountRef.current = Math.max(0, pendingPostCountRef.current - 1);
      }
    },
    [beginPendingLiveTurnFromCurrent, clearReflectTimeout, setRunningState],
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
      detailByPhase,
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
      detailByPhase,
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
