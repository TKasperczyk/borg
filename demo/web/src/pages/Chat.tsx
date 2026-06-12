import {
  type KeyboardEvent,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  ApiError,
  ensureOperatorSession,
  fetchInflight,
  fetchLedger,
  fetchSessions,
  fetchStream,
  fetchTurns,
  postTurn,
} from "../api/client";
import type {
  EvidenceLedger,
  LiveFrame,
  SessionRecord,
  StreamEntry,
  StreamResponse,
  TurnHistoryRow,
  TurnsResponse,
} from "../api/types";
import { TURN_PHASES, type TurnPhaseName } from "../api/types";
import { useQuery } from "../api/useQuery";
import { hms, hm } from "../format/time";
import { newId } from "../format/uid";
import { useLive } from "../live/useLive";
import { useAppState } from "../state/app-state";
import { moodLabel, useMood } from "../state/mood";
import {
  ThreadArtifactList,
  isDeliberateSilence,
  threadItemsFromEntries,
} from "./chat/artifacts";
import { applyDraftFrame, EMPTY_DRAFT_STATE, type DraftState } from "./chat/draft";
import { summarizeLedger } from "./chat/ledger";
import {
  outcomeFromTerminalAndEntry,
  terminalSummaryFromFrame,
  type TerminalSummary,
} from "./chat/outcome";
import { SwarmCanvas } from "./chat/SwarmCanvas";
import {
  PHASE_LABELS,
  applyPhaseFrame,
  initialPhaseGridState,
  seedPhaseGridFromInflight,
  type TurnPhaseGridState,
} from "./chat/turnPhase";

type InFlightTurn = {
  turnId: string;
};

type InFlightBySession = Record<string, InFlightTurn | undefined>;

function shortId(id: string): string {
  return id.length <= 8 ? id : id.slice(0, 8);
}

type TraceItem = {
  id: string;
  time: string;
  text: string;
};

type DelibPath = "system_1" | "system_2" | null;

type ChatPageProps = {
  onActiveSessionChange?: (sessionId: string | null) => void;
};

type SessionStreamData = {
  sessionId: string | null;
  response: StreamResponse;
};

type SessionTurnsData = {
  sessionId: string | null;
  response: TurnsResponse;
};

type LedgerQueryData = {
  turnId: string | null;
  ledger: EvidenceLedger | null;
};

function mergeEntries(entries: readonly StreamEntry[]): StreamEntry[] {
  const byId = new Map<string, StreamEntry>();
  for (const entry of entries) {
    byId.set(entry.id, entry);
  }

  return [...byId.values()].sort((left, right) => left.timestamp - right.timestamp);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function streamEntryFromLive(value: unknown): StreamEntry | null {
  if (!isRecord(value)) {
    return null;
  }
  if (typeof value.id !== "string" || typeof value.kind !== "string" || typeof value.session_id !== "string") {
    return null;
  }

  return value as StreamEntry;
}

function frameSessionId(frame: LiveFrame): string | null {
  if (
    frame.type === "turn:phase:started" ||
    frame.type === "turn:phase:completed" ||
    frame.type === "turn:phase:failed"
  ) {
    return frame.data.session_id ?? null;
  }
  if (
    frame.type === "turn:token" ||
    frame.type === "turn:token:flush" ||
    frame.type === "turn:delib_path" ||
    frame.type === "turn:final_attempt" ||
    frame.type === "evidence_ledger:built" ||
    frame.type === "turn:phase:detail"
  ) {
    return frame.session_id ?? null;
  }
  if (frame.type === "turn:terminal") {
    return frame.data.session_id;
  }
  if (frame.type === "stream:append") {
    return frame.session_id ?? streamEntryFromLive(frame.entries[0])?.session_id ?? null;
  }

  return null;
}

function applyInFlightFrame(state: InFlightBySession, frame: LiveFrame): InFlightBySession {
  if (frame.type === "turn:phase:started") {
    const sessionId = frame.data.session_id;
    if (sessionId === undefined) {
      return state;
    }

    return {
      ...state,
      [sessionId]: { turnId: frame.data.turn_id },
    };
  }

  if (frame.type === "turn:terminal") {
    const sessionId = frame.data.session_id;
    const current = state[sessionId];
    if (current?.turnId !== frame.data.turn_id) {
      return state;
    }

    return {
      ...state,
      [sessionId]: undefined,
    };
  }

  return state;
}

function sessionMeta(session: SessionRecord): string {
  return `${session.conversation_kind} · ${session.status} · ${hm(new Date(session.last_activity_at))}`;
}

function sessionHeaderMeta(session: SessionRecord | null): string {
  if (session === null) {
    return "";
  }

  return `${session.conversation_kind} · ${session.status} · ${session.message_count} messages`;
}

function formatError(error: unknown): string {
  if (error instanceof ApiError) {
    return `${error.status} ${error.message}`;
  }

  return error instanceof Error ? error.message : String(error);
}

function formatTraceFrame(frame: LiveFrame): string | null {
  if (
    frame.type === "turn:phase:started" ||
    frame.type === "turn:phase:completed" ||
    frame.type === "turn:phase:failed"
  ) {
    const suffix =
      frame.type === "turn:phase:completed" || frame.type === "turn:phase:failed"
        ? ` ${Math.round(frame.data.duration_ms ?? 0)}ms`
        : "";
    return `${frame.event} ${frame.data.phase}${suffix}`;
  }
  if (frame.type === "evidence_ledger:built") {
    const count =
      frame.ledger?.sections.reduce((sum, section) => sum + section.entries.length, 0) ?? 0;
    return `evidence_ledger.built entries=${count}`;
  }
  if (frame.type === "turn:delib_path") {
    return `deliberation.path.completed path=${frame.path}`;
  }
  if (frame.type === "turn:terminal") {
    return `turn.terminal outcome=${frame.data.outcome}`;
  }
  if (frame.type === "turn:final_attempt") {
    return `turn:final_attempt attempt=${frame.attempt}`;
  }
  if (frame.type === "maintenance:tick") {
    return `maintenance.tick cadence=${frame.cadence} status=${frame.status}`;
  }
  if (frame.type === "dream:process:started" || frame.type === "dream:process:completed") {
    return `${frame.type} ${frame.process} ${frame.phase}`;
  }
  if (frame.type === "turn:phase:detail") {
    return `${frame.event} ${frame.summary}`;
  }

  return null;
}

function shortTurnId(turnId: string): string {
  return turnId.length <= 12 ? turnId : `${turnId.slice(0, 10)}…`;
}

function latestTerminalEntry(
  entries: readonly StreamEntry[],
  terminal: TerminalSummary | null,
): StreamEntry | null {
  if (terminal === null) {
    return null;
  }

  return (
    entries.find(
      (entry) =>
        entry.turn_id === terminal.turnId &&
        (entry.kind === "agent_msg" ||
          entry.kind === "agent_suppressed" ||
          entry.kind === "agent_observed"),
    ) ?? null
  );
}

function swarmOutcome(
  terminal: TerminalSummary | null,
  terminalEntry: StreamEntry | null,
  turn: TurnHistoryRow | undefined,
): "idle" | "emitted" | "silence" | "observed" | "suppressed" | "error" {
  if (terminal === null) {
    return "idle";
  }
  if (
    terminal.outcome === "suppressed_action" ||
    terminal.outcome === "suppressed_closure" ||
    terminal.outcome === "suppressed_generation_gate"
  ) {
    return "suppressed";
  }
  if (terminal.outcome === "aborted" || terminal.outcome === "error") {
    return "error";
  }
  if (terminalEntry?.kind === "agent_msg") {
    return "emitted";
  }
  if (terminalEntry?.kind === "agent_observed") {
    return "observed";
  }
  if (terminalEntry?.kind === "agent_suppressed") {
    return isDeliberateSilence(terminalEntry, turn) ? "silence" : "suppressed";
  }

  return "idle";
}

function activePhase(state: TurnPhaseGridState): TurnPhaseName | null {
  return (
    TURN_PHASES.find((phase) => state.phases[phase].state === "active") ?? null
  );
}

export function ChatPage({ onActiveSessionChange }: ChatPageProps) {
  const appState = useAppState();
  const mood = useMood();
  const { onFrame, subscribeSession, unsubscribeSession } = useLive();
  const sessions = useQuery("sessions", fetchSessions);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [liveEntries, setLiveEntries] = useState<StreamEntry[]>([]);
  const [inFlightBySession, setInFlightBySession] = useState<InFlightBySession>({});
  const [phaseGrid, setPhaseGrid] = useState<TurnPhaseGridState>(() => initialPhaseGridState());
  const [draft, setDraft] = useState<DraftState>(EMPTY_DRAFT_STATE);
  const [trace, setTrace] = useState<TraceItem[]>([]);
  const [composer, setComposer] = useState("");
  const [composerError, setComposerError] = useState<string | null>(null);
  const [railError, setRailError] = useState<string | null>(null);
  const [delibPath, setDelibPath] = useState<DelibPath>(null);
  const [lastTerminal, setLastTerminal] = useState<TerminalSummary | null>(null);
  const [liveLedger, setLiveLedger] = useState<{ turnId: string; ledger: EvidenceLedger } | null>(
    null,
  );
  const [evidencePulse, setEvidencePulse] = useState(0);
  const [tokenPulse, setTokenPulse] = useState(0);
  const threadRef = useRef<HTMLDivElement | null>(null);
  const pinnedRef = useRef(true);
  const activeTurnIdRef = useRef<string | null>(null);
  const terminalSeenRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (activeSessionId !== null) {
      return;
    }

    const available = sessions.data?.sessions ?? [];
    const preferred =
      available.find((session) => session.session_id === appState.data?.active_session) ??
      available[0] ??
      null;
    if (preferred !== null) {
      setActiveSessionId(preferred.session_id);
    }
  }, [activeSessionId, appState.data?.active_session, sessions.data?.sessions]);

  useEffect(() => {
    setLiveEntries([]);
    setPhaseGrid(initialPhaseGridState());
    setDraft(EMPTY_DRAFT_STATE);
    setTrace([]);
    setDelibPath(null);
    setLastTerminal(null);
    setLiveLedger(null);
    activeTurnIdRef.current = null;
  }, [activeSessionId]);

  useEffect(() => {
    onActiveSessionChange?.(activeSessionId);
  }, [activeSessionId, onActiveSessionChange]);

  // A page mounted (or switched to this session) mid-turn has missed the
  // turn's frames -- and the live ring buffer cannot replay a long turn. Seed
  // running state and the phase grid from the server's in-flight snapshot;
  // live frames take over from there.
  useEffect(() => {
    if (activeSessionId === null) {
      return undefined;
    }

    const sessionId = activeSessionId;
    let cancelled = false;

    fetchInflight(sessionId)
      .then(({ inflight }) => {
        if (
          cancelled ||
          inflight === null ||
          inflight.session_id !== sessionId ||
          terminalSeenRef.current.has(inflight.turn_id)
        ) {
          return;
        }

        activeTurnIdRef.current = inflight.turn_id;
        setInFlightBySession((current) => ({
          ...current,
          [sessionId]: { turnId: inflight.turn_id },
        }));
        setPhaseGrid((current) =>
          // A live frame may have already advanced this turn's grid while the
          // fetch was in flight; never clobber fresher same-turn state.
          current.turnId === inflight.turn_id ? current : seedPhaseGridFromInflight(inflight),
        );
      })
      .catch(() => {
        // Best-effort seeding only; live frames remain the source of truth.
      });

    return () => {
      cancelled = true;
    };
  }, [activeSessionId]);

  const stream = useQuery<SessionStreamData>(
    `stream:${activeSessionId ?? ""}`,
    async () => {
      if (activeSessionId === null) {
        return { sessionId: null, response: { entries: [], next_cursor: null } };
      }

      return { sessionId: activeSessionId, response: await fetchStream(activeSessionId) };
    },
  );
  const turns = useQuery<SessionTurnsData>(
    `turns:${activeSessionId ?? ""}`,
    async () => {
      if (activeSessionId === null) {
        return { sessionId: null, response: { rows: [], next_cursor: null } };
      }

      return { sessionId: activeSessionId, response: await fetchTurns(activeSessionId) };
    },
  );
  const turnsMatchSession = turns.data?.sessionId === activeSessionId;
  const turnRows = turnsMatchSession ? (turns.data?.response.rows ?? []) : [];
  const latestTurnId = turnRows[0]?.turn_id ?? null;
  const ledgerQuery = useQuery<LedgerQueryData>(
    `ledger:${latestTurnId ?? ""}`,
    async () => {
      if (latestTurnId === null) {
        return { turnId: null, ledger: null };
      }

      try {
        const response = await fetchLedger(latestTurnId);
        return { turnId: response.turn_id, ledger: response.ledger };
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
          return { turnId: latestTurnId, ledger: null };
        }
        throw error;
      }
    },
  );

  const sessionIds = useMemo(
    () => (sessions.data?.sessions ?? []).map((session) => session.session_id),
    [sessions.data?.sessions],
  );
  const sessionIdsKey = sessionIds.join("\n");

  useEffect(() => {
    if (sessionIds.length === 0) {
      return undefined;
    }

    for (const sessionId of sessionIds) {
      subscribeSession(sessionId);
    }

    return () => {
      for (const sessionId of sessionIds) {
        unsubscribeSession(sessionId);
      }
    };
  }, [sessionIds, sessionIdsKey, subscribeSession, unsubscribeSession]);

  useEffect(
    () =>
      onFrame("*", (frame) => {
        if (frame.type === "turn:terminal") {
          // Guards the in-flight seed against resurrecting a turn that ended
          // while the snapshot fetch was in transit. Bounded ring of recents.
          terminalSeenRef.current.add(frame.data.turn_id);
          if (terminalSeenRef.current.size > 64) {
            const oldest = terminalSeenRef.current.values().next().value;
            if (oldest !== undefined) {
              terminalSeenRef.current.delete(oldest);
            }
          }
        }
        setInFlightBySession((current) => applyInFlightFrame(current, frame));

        const sessionId = frameSessionId(frame);
        if (sessionId !== activeSessionId && frame.type !== "maintenance:tick") {
          return;
        }

        const traceText = formatTraceFrame(frame);
        if (traceText !== null) {
          setTrace((current) =>
            [
              ...current,
              { id: `${frame.ts}:${traceText}:${current.length}`, time: hms(new Date(frame.ts)), text: traceText },
            ].slice(-40),
          );
        }

        if (
          frame.type === "turn:phase:started" ||
          frame.type === "turn:phase:completed" ||
          frame.type === "turn:phase:failed"
        ) {
          if (frame.type === "turn:phase:started" && activeTurnIdRef.current !== frame.data.turn_id) {
            activeTurnIdRef.current = frame.data.turn_id;
            setDelibPath(null);
            setLastTerminal(null);
            setLiveLedger(null);
          }
          setPhaseGrid((current) => applyPhaseFrame(current, frame));
        }

        if (frame.type === "turn:delib_path") {
          setDelibPath(frame.path);
        }

        if (frame.type === "turn:token" || frame.type === "turn:token:flush" || frame.type === "turn:terminal") {
          setDraft((current) => applyDraftFrame(current, frame));
        }

        if (frame.type === "turn:token" && frame.phase === "final") {
          setTokenPulse((current) => current + 1);
        }

        if (frame.type === "turn:terminal") {
          setLastTerminal(terminalSummaryFromFrame(frame));
        }

        if (frame.type === "evidence_ledger:built" && frame.ledger !== null) {
          setLiveLedger({ turnId: frame.turn_id, ledger: frame.ledger });
          setEvidencePulse((current) => current + 1);
        }

        if (frame.type === "stream:append") {
          const entries = frame.entries
            .map(streamEntryFromLive)
            .filter((entry): entry is StreamEntry => entry !== null)
            .filter((entry) => entry.session_id === activeSessionId);
          if (entries.length > 0) {
            setLiveEntries((current) => mergeEntries([...current, ...entries]));
          }
        }
      }),
    [activeSessionId, onFrame],
  );

  const currentSession =
    sessions.data?.sessions.find((session) => session.session_id === activeSessionId) ?? null;
  const streamMatchSession = stream.data?.sessionId === activeSessionId;
  const streamReady = streamMatchSession && !stream.loading;
  const allEntries = useMemo(
    () =>
      mergeEntries([
        ...(streamMatchSession ? (stream.data?.response.entries ?? []) : []),
        ...liveEntries,
      ]),
    [liveEntries, stream.data?.response.entries, streamMatchSession],
  );
  const threadItems = useMemo(
    () => threadItemsFromEntries(allEntries, turnRows, draft.withheldByTurn),
    [allEntries, draft.withheldByTurn, turnRows],
  );
  const terminalEntry = latestTerminalEntry(allEntries, lastTerminal);
  const terminalTurnRow =
    lastTerminal === null
      ? undefined
      : turnRows.find((turn) => turn.turn_id === lastTerminal.turnId);
  const currentInFlight = activeSessionId === null ? undefined : inFlightBySession[activeSessionId];
  const running = currentInFlight !== undefined;
  const outcome = outcomeFromTerminalAndEntry(lastTerminal, terminalEntry, running, terminalTurnRow);
  const currentPhase = activePhase(phaseGrid);
  const ledgerMatchesLatest = ledgerQuery.data?.turnId === latestTurnId;
  const ledgerTurnLabel = liveLedger?.turnId ?? (ledgerMatchesLatest ? ledgerQuery.data?.turnId : null) ?? "";
  const currentLedger =
    liveLedger?.ledger ?? (ledgerMatchesLatest ? (ledgerQuery.data?.ledger ?? null) : null);
  const ledgerSummary = summarizeLedger(currentLedger);
  const coreOutcome = running ? "idle" : swarmOutcome(lastTerminal, terminalEntry, terminalTurnRow);

  useLayoutEffect(() => {
    const element = threadRef.current;
    if (element === null || !pinnedRef.current) {
      return;
    }

    element.scrollTop = element.scrollHeight;
  }, [threadItems, draft.current?.text]);

  const updatePinned = useCallback(() => {
    const element = threadRef.current;
    if (element === null) {
      return;
    }

    pinnedRef.current = element.scrollHeight - element.scrollTop - element.clientHeight < 96;
  }, []);

  const send = useCallback(async () => {
    const message = composer.trim();
    if (message.length === 0 || activeSessionId === null) {
      return;
    }

    setComposer("");
    setComposerError(null);
    try {
      await postTurn({
        message,
        external_message_id: newId(),
        session: activeSessionId,
      });
      stream.refetch();
      turns.refetch();
      sessions.refetch();
    } catch (error) {
      setComposerError(formatError(error));
    }
  }, [activeSessionId, composer, sessions, stream, turns]);

  const onComposerKey = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === "Enter") {
        event.preventDefault();
        void send();
      }
    },
    [send],
  );

  const ensureOperator = useCallback(async () => {
    setRailError(null);
    try {
      const session = await ensureOperatorSession();
      setActiveSessionId(session.session_id);
      sessions.refetch();
    } catch (error) {
      setRailError(formatError(error));
    }
  }, [sessions]);

  return (
    <main className="chat-layout">
      <aside className="sessions-rail">
        <div className="sessions-head">
          <span>SESSIONS</span>
          <span>{sessions.data?.sessions.length ?? 0}</span>
        </div>
        <div className="session-list">
          {(sessions.data?.sessions ?? []).map((session) => {
            const active = session.session_id === activeSessionId;
            const inFlight = inFlightBySession[session.session_id] !== undefined;
            return (
              <button
                key={session.session_id}
                className={active ? "session-row session-row-active" : "session-row"}
                type="button"
                onClick={() => setActiveSessionId(session.session_id)}
              >
                <div className="session-row-main">
                  <span className={inFlight ? "session-dot session-dot-live pulse" : "session-dot"} />
                  <span className="session-name">{session.label}</span>
                </div>
                <div className="session-meta">{sessionMeta(session)}</div>
              </button>
            );
          })}
        </div>
        <div className="sessions-foot">
          <button className="ensure-session" type="button" onClick={() => void ensureOperator()}>
            + ENSURE OPERATOR SESSION
          </button>
          {railError === null ? null : <div className="rail-error">{railError}</div>}
        </div>
      </aside>

      <section className="thread-column">
        <header className="chat-thread-head">
          <span className="thread-title">{currentSession?.label ?? "CHAT"}</span>
          <span className="thread-subtitle">{sessionHeaderMeta(currentSession)}</span>
          <span className={running ? "turn-count turn-count-live" : "turn-count"}>
            {running
              ? "● turn in flight"
              : !turnsMatchSession || turns.loading
                ? "loading"
                : turnRows.length === 0
                  ? "idle"
                  : `${turnRows.length} turns`}
          </span>
        </header>

        <div ref={threadRef} className="thread-scroll" onScroll={updatePinned}>
          {!streamReady ? <div className="thread-empty">loading stream…</div> : null}
          {streamReady && threadItems.length === 0 ? (
            <div className="thread-empty">no stream entries yet</div>
          ) : streamReady ? (
            <ThreadArtifactList items={threadItems} />
          ) : null}
          {draft.current !== null && draft.current.sessionId === activeSessionId ? (
            <article className="thread-artifact thread-agent thread-draft">
              <div className="artifact-meta">
                <span className="entity-label">ENTITY</span>
                <span>streaming · final</span>
              </div>
              <div className="artifact-body pretty-wrap">
                {draft.current.text}
                <span className="draft-cursor blink" />
              </div>
            </article>
          ) : null}
          <div className="thread-tail" />
        </div>

        <footer className="composer-wrap">
          <div className="composer-row">
            <input
              className="composer-input"
              value={composer}
              onChange={(event) => setComposer(event.target.value)}
              onKeyDown={onComposerKey}
              placeholder="message the entity…"
            />
            <button className="send-button" type="button" onClick={() => void send()}>
              SEND ▸
            </button>
          </div>
          {composerError === null ? null : <div className="composer-error">{composerError}</div>}
          <div className="composer-hint">
            ENTER to send ·{" "}
            {running ? "● turn in flight" : "the entity decides whether to answer, observe, or stay silent"}
          </div>
        </footer>
      </section>

      <aside className="cognition-panel">
        <div className="cognition-head">
          <span>COGNITION</span>
        </div>
        <div className="mind-core">
          <SwarmCanvas
            phase={currentPhase}
            delibPath={delibPath}
            outcome={coreOutcome}
            arousal={mood.arousal}
            hue={mood.hue}
            inFlight={running}
            evidencePulse={evidencePulse}
            tokenPulse={tokenPulse}
          />
          <div className="core-caption">MIND CORE · particle swarm</div>
        </div>

        <div className={`outcome-line outcome-${outcome.tone}`}>
          <span className={outcome.pulse ? "outcome-dot pulse" : "outcome-dot"} />
          <span>{outcome.text}</span>
        </div>

        <section className="cog-block">
          <div className="turn-head">
            <span>TURN</span>
            <span>
              {running && currentInFlight !== undefined
                ? `${shortTurnId(currentInFlight.turnId)} · ${
                    currentPhase === null ? "…" : PHASE_LABELS[currentPhase]
                  }`
                : "idle — awaiting input"}
            </span>
            <span className={delibPath === null ? "path-badge" : "path-badge path-badge-active"}>
              {delibPath === "system_1"
                ? "SYS_1 FAST"
                : delibPath === "system_2"
                  ? "SYS_2 DELIB"
                  : running
                    ? "PATH …"
                    : "PATH —"}
            </span>
          </div>
          <div className="phase-grid">
            {TURN_PHASES.map((phase) => {
              const cell = phaseGrid.phases[phase];
              return (
                <div key={phase} className="phase-cell">
                  <span className={`phase-dot phase-${cell.state}`} />
                  <span className={cell.state === "active" ? "phase-name phase-active-name" : "phase-name"}>
                    {cell.label}
                  </span>
                  <span className="phase-duration">
                    {cell.state === "active"
                      ? "…"
                      : cell.durationMs === null
                        ? "—"
                        : `${Math.round(cell.durationMs)}ms`}
                  </span>
                </div>
              );
            })}
          </div>
        </section>

        <section className="cog-block mood-block">
          <div className="block-head">
            <span>MOOD</span>
            <strong>{moodLabel(mood.mood?.valence ?? 0, mood.mood?.arousal ?? 0)}</strong>
          </div>
          <MoodBar label="VALENCE" value={mood.mood?.valence ?? 0} centered />
          <MoodBar label="AROUSAL" value={mood.arousal} />
          <div className="mood-note">accent hue follows mood</div>
        </section>

        <section className="cog-block">
          <div className="block-head">
            <span>EVIDENCE RECALLED</span>
            {ledgerTurnLabel.length === 0 ? null : (
              <span title={ledgerTurnLabel}>{shortId(ledgerTurnLabel)}</span>
            )}
          </div>
          {ledgerSummary === null || ledgerSummary.totalEntries === 0 ? (
            <div className="ledger-empty">no ledger for current turn yet</div>
          ) : (
            <>
              <div className="ledger-chips">
                {ledgerSummary.chips.map((chip) => (
                  <div key={chip.key} className="ledger-chip">
                    <strong>{chip.key}</strong>
                    <span>{chip.value}</span>
                  </div>
                ))}
              </div>
              {ledgerSummary.disclosureCount === 0 ? null : (
                <div className="ledger-note">
                  {ledgerSummary.disclosureCount} items carry non-public disclosure labels — recalled,
                  not necessarily disclosable
                </div>
              )}
            </>
          )}
        </section>

        <section className="cog-block">
          <div className="trace-title">TRACE</div>
          <div className="trace-feed">
            {trace.map((item) => (
              <div key={item.id} className="trace-row">
                <span>{item.time}</span> {item.text}
              </div>
            ))}
          </div>
        </section>
      </aside>
    </main>
  );
}

function MoodBar({
  label,
  value,
  centered = false,
}: {
  label: string;
  value: number;
  centered?: boolean;
}) {
  const clamped = centered ? Math.max(-1, Math.min(1, value)) : Math.max(0, Math.min(1, value));
  const percentage = centered ? ((clamped + 1) / 2) * 100 : clamped * 100;
  const left = centered ? Math.min(50, percentage) : 0;
  const width = centered ? Math.abs(percentage - 50) : percentage;

  return (
    <div className="mood-row">
      <span>{label}</span>
      <div className="mood-track">
        {centered ? <div className="mood-center" /> : null}
        <div className="mood-fill" style={{ left: `${left}%`, width: `${width}%` }} />
      </div>
      <span>{centered ? `${clamped >= 0 ? "+" : ""}${clamped.toFixed(2)}` : clamped.toFixed(2)}</span>
    </div>
  );
}
