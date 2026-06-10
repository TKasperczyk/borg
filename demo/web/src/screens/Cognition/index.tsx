import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  getAssembledPrompt,
  getCommitments,
  getIdentity,
  getSharedState,
  getStream,
  getTurns,
} from "../../api/client";
import type {
  SharedStateResponse,
  SessionRecord,
  StreamEntry,
  StreamEntryKind,
  StreamResponse,
  TurnHistoryOutcomeClass,
  TurnHistoryRow,
} from "../../api/types";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi, type ApiHookState } from "../../hooks/use-api";
import type { TurnStreamState } from "../../hooks/use-turn-stream";
import type { AudienceDisplayIdentity } from "../../lib/audience-identity";
import {
  formatTimestamp,
  mergeEntries,
  sortStreamEntries,
  streamContentText,
} from "../../lib/stream-utils";
import { ParticipationPolicyControl } from "../../components/ParticipationPolicyControl";
import { Tag, type TagKind } from "../../components/Tag";
import { shortId } from "../screen-utils";
import { ChatInput } from "./ChatInput";
import { ChatStream } from "./ChatStream";
import type { ChatDeliveryStatus, ChatStreamEntry } from "./chat-utils";
import { Xray, type XrayTabId } from "./Xray";

const CHAT_KINDS: readonly StreamEntryKind[] = [
  "user_msg",
  "agent_msg",
  "user_image_attachment",
  "agent_suppressed",
  "agent_observed",
];
const TURN_HISTORY_LIMIT = 12;
const DEMO_SOURCE_TYPE = "demo";
const LAZY_XRAY_TABS = new Set<XrayTabId>(["shared", "commitments", "open_qs", "prompt"]);

type ApiDataState<T> = Pick<ApiHookState<T>, "data" | "loading" | "error">;

// crypto.randomUUID() exists only in secure contexts (HTTPS or localhost). The
// demo is reached over plain HTTP on the LAN, where randomUUID is undefined.
// crypto.getRandomValues IS available in non-secure contexts, so derive a
// UUIDv4 from it, with a last-resort fallback.
function makeClientMessageId(): string {
  const webCrypto = globalThis.crypto as Crypto | undefined;
  if (webCrypto !== undefined && typeof webCrypto.randomUUID === "function") {
    return webCrypto.randomUUID();
  }
  if (webCrypto !== undefined && typeof webCrypto.getRandomValues === "function") {
    const bytes = webCrypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6]! & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8]! & 0x3f) | 0x80; // variant 10
    const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0"));
    return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10, 16).join("")}`;
  }
  return `msg-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export type CognitionScreenProps = {
  sessionId: string;
  audienceValue: string | null;
  audienceDisplay: AudienceDisplayIdentity;
  audienceEntityId?: string | null;
  turnStream: TurnStreamState;
  session?: SessionRecord | null;
  onSessionPolicyChanged?: () => Promise<void>;
};

function isChatEntry(entry: StreamEntry, sessionId: string, audience: string): boolean {
  return (
    entry.session_id === sessionId && CHAT_KINDS.includes(entry.kind) && entry.audience === audience
  );
}

function entryExternalMessageId(entry: ChatStreamEntry): string | null {
  return entry.external_message_id ?? entry.source_message_key?.external_message_id ?? null;
}

function sameUserMessageForReconcile(optimistic: ChatStreamEntry, real: StreamEntry): boolean {
  if (optimistic.kind !== "user_msg" || real.kind !== "user_msg") {
    return false;
  }

  const optimisticExternalId = entryExternalMessageId(optimistic);
  const realExternalId = entryExternalMessageId(real);
  if (optimisticExternalId !== null && realExternalId !== null) {
    return optimisticExternalId === realExternalId;
  }

  return (
    optimistic.session_id === real.session_id &&
    optimistic.audience === real.audience &&
    optimistic.sender_entity_id === real.sender_entity_id &&
    streamContentText(optimistic.content) === streamContentText(real.content)
  );
}

function withoutReconciledOptimisticEntries(
  optimisticEntries: readonly ChatStreamEntry[],
  realEntries: readonly StreamEntry[],
): ChatStreamEntry[] {
  return optimisticEntries.filter(
    (optimistic) => !realEntries.some((entry) => sameUserMessageForReconcile(optimistic, entry)),
  );
}

function optimisticUserEntry(input: {
  externalMessageId: string;
  message: string;
  sessionId: string;
  audience: string;
  audienceEntityId?: string | null;
  senderLabel?: string | null;
  status: ChatDeliveryStatus;
}): ChatStreamEntry {
  return {
    id: `optimistic:${input.externalMessageId}`,
    timestamp: Date.now(),
    kind: "user_msg",
    content: input.message,
    audience: input.audience,
    sender_entity_id: input.audienceEntityId ?? null,
    sender_label: input.senderLabel ?? null,
    reply_target_entity_id: null,
    session_id: input.sessionId,
    compressed: false,
    external_message_id: input.externalMessageId,
    source_message_key: {
      source_type: DEMO_SOURCE_TYPE,
      source_external_id: input.sessionId,
      external_message_id: input.externalMessageId,
    },
    optimistic_status: input.status,
  };
}

function upsertOptimisticEntry(
  current: readonly ChatStreamEntry[],
  entry: ChatStreamEntry,
): ChatStreamEntry[] {
  if (current.some((item) => entryExternalMessageId(item) === entryExternalMessageId(entry))) {
    return [...current];
  }

  return [...current, entry];
}

function markOptimisticStatus(
  current: readonly ChatStreamEntry[],
  externalMessageId: string,
  status: ChatDeliveryStatus,
): ChatStreamEntry[] {
  return current.map((entry) =>
    entryExternalMessageId(entry) === externalMessageId
      ? { ...entry, optimistic_status: status }
      : entry,
  );
}

function useLazyScreenApi<T>(
  loader: () => Promise<T>,
  deps: readonly unknown[],
  enabled: boolean,
): ApiDataState<T> {
  const [state, setState] = useState<ApiDataState<T>>({
    data: null,
    loading: false,
    error: null,
  });
  const requestSeqRef = useRef(0);
  const startedRef = useRef(false);

  useEffect(() => {
    requestSeqRef.current += 1;
    startedRef.current = false;
    setState({ data: null, loading: false, error: null });
  }, deps);

  useEffect(() => {
    if (!enabled || startedRef.current) {
      return;
    }

    let cancelled = false;
    const requestSeq = requestSeqRef.current + 1;
    requestSeqRef.current = requestSeq;
    startedRef.current = true;
    setState({ data: null, loading: true, error: null });

    void loader()
      .then((data) => {
        if (!cancelled && requestSeqRef.current === requestSeq) {
          setState({ data, loading: false, error: null });
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled && requestSeqRef.current === requestSeq) {
          setState({
            data: null,
            loading: false,
            error: caught instanceof Error ? caught : new Error(String(caught)),
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [enabled, ...deps]);

  if (enabled && !startedRef.current && state.data === null && state.error === null) {
    return { data: null, loading: true, error: null };
  }

  return state;
}

function turnOutcomeKind(outcome: TurnHistoryOutcomeClass): TagKind {
  if (outcome === "emitted" || outcome === "deliberate-silence") {
    return "acc";
  }
  if (outcome === "failed" || outcome === "emission-failed") {
    return "bad";
  }
  if (outcome === "guard-blocked") {
    return "warn";
  }
  if (outcome === "observed") {
    return "info";
  }
  return "";
}

function StaticIdChip({ label }: { label: string }) {
  return (
    <span className="id-chip static-id-chip">
      <span className="id-chip-main id-chip-static">{label}</span>
    </span>
  );
}

function TurnHistoryStrip({
  rows,
  selectedTurnId,
  loading,
  error,
  onSelect,
  onLive,
}: {
  rows: readonly TurnHistoryRow[];
  selectedTurnId: string | null;
  loading: boolean;
  error: Error | null;
  onSelect: (turnId: string) => void;
  onLive: () => void;
}) {
  return (
    <section className="turn-history" aria-label="Turn history">
      <div className="turn-history-head">
        <span className="title">turns</span>
        <button
          className={`turn-history-live ${selectedTurnId === null ? "active" : ""}`.trim()}
          type="button"
          onClick={onLive}
          aria-pressed={selectedTurnId === null}
        >
          live
        </button>
      </div>
      <div className="turn-history-list">
        {loading && rows.length === 0 ? (
          <div className="turn-history-empty">loading turns</div>
        ) : null}
        {error !== null && rows.length === 0 ? (
          <div className="turn-history-empty">turn history unavailable</div>
        ) : null}
        {!loading && error === null && rows.length === 0 ? (
          <div className="turn-history-empty">no turns yet</div>
        ) : null}
        {rows.map((row) => (
          <button
            key={row.turn_id}
            className={`turn-history-row ${selectedTurnId === row.turn_id ? "active" : ""}`.trim()}
            type="button"
            onClick={() => onSelect(row.turn_id)}
            aria-pressed={selectedTurnId === row.turn_id}
          >
            <span className="turn-history-row-top">
              <span className="turn-history-turn">{formatTimestamp(row.started_at)}</span>
              <StaticIdChip label={shortId(row.turn_id)} />
              <Tag kind={turnOutcomeKind(row.outcome)}>{row.outcome}</Tag>
            </span>
            <span className="turn-history-meta">
              <span>{row.audience === null ? "global audience" : "session audience"}</span>
            </span>
            {row.suppression_reason === null ? null : (
              <span className="turn-history-reason">{row.suppression_reason}</span>
            )}
          </button>
        ))}
      </div>
    </section>
  );
}

export function CognitionScreen({
  sessionId,
  audienceValue,
  audienceDisplay,
  audienceEntityId,
  turnStream,
  session = null,
  onSessionPolicyChanged,
}: CognitionScreenProps) {
  const live = useLiveEventsContext();
  const [chatEntries, setChatEntries] = useState<StreamEntry[]>([]);
  const [optimisticEntries, setOptimisticEntries] = useState<ChatStreamEntry[]>([]);
  const [chatNextCursor, setChatNextCursor] = useState<string | null>(null);
  const [loadingOlderChat, setLoadingOlderChat] = useState(false);
  const [olderChatError, setOlderChatError] = useState<Error | null>(null);
  const [replayTurnId, setReplayTurnId] = useState<string | null>(null);
  const [xrayTab, setXrayTab] = useState<XrayTabId>("flow");
  const [activatedXrayTabs, setActivatedXrayTabs] = useState<ReadonlySet<XrayTabId>>(
    () => new Set(),
  );
  const previousConnectionCountRef = useRef(live.connectionCount);
  const olderChatRequestSeqRef = useRef(0);
  const chatFilterKey = `${sessionId}\n${audienceValue ?? ""}`;
  const chatFilterKeyRef = useRef(chatFilterKey);
  chatFilterKeyRef.current = chatFilterKey;
  const participationPolicy = session?.participation_policy ?? "active";
  const participationPolicyLocked = session?.audience_role === "operator";

  const streamApi = useApi(
    () =>
      audienceValue === null
        ? Promise.resolve({ entries: [], next_cursor: null } satisfies StreamResponse)
        : getStream({ session: sessionId, audience: audienceValue, kinds: CHAT_KINDS, limit: 50 }),
    [audienceValue, sessionId],
  );
  const turnsApi = useApi(
    () => getTurns({ session: sessionId, limit: TURN_HISTORY_LIMIT }),
    [sessionId],
  );
  const sharedStateApi = useLazyScreenApi(
    () =>
      audienceValue === null
        ? Promise.resolve({ audience: "", entries: [] } satisfies SharedStateResponse)
        : getSharedState(audienceValue),
    [audienceValue],
    audienceValue !== null && activatedXrayTabs.has("shared"),
  );
  const commitmentsApi = useLazyScreenApi(
    () => getCommitments(),
    [],
    activatedXrayTabs.has("commitments"),
  );
  const identityApi = useLazyScreenApi(getIdentity, [], activatedXrayTabs.has("open_qs"));
  const promptApi = useLazyScreenApi(getAssembledPrompt, [], activatedXrayTabs.has("prompt"));
  const resetForReconnect = turnStream.resetForReconnect;
  const replaceTailFromEntries = turnStream.replaceTailFromEntries;
  const refetchTurns = turnsApi.refetch;

  useEffect(() => {
    olderChatRequestSeqRef.current += 1;
    setChatEntries([]);
    setOptimisticEntries([]);
    setChatNextCursor(null);
    setLoadingOlderChat(false);
    setOlderChatError(null);
    setReplayTurnId(null);
  }, [audienceValue, sessionId]);

  useEffect(() => {
    if (turnStream.activeTurnId === null || turnStream.terminalOutcome === null) {
      return;
    }

    void refetchTurns();
  }, [refetchTurns, turnStream.activeTurnId, turnStream.terminalOutcome]);

  useEffect(() => {
    const streamData = streamApi.data;

    if (streamData !== null) {
      setOptimisticEntries((current) =>
        withoutReconciledOptimisticEntries(current, streamData.entries),
      );
      setChatEntries((current) =>
        mergeEntries(
          current.filter(
            (entry) => entry.session_id === sessionId && entry.audience === audienceValue,
          ),
          streamData.entries,
        ),
      );
      setChatNextCursor(streamData.next_cursor);
      setOlderChatError(null);
    }
  }, [audienceValue, sessionId, streamApi.data]);

  useEffect(() => {
    return live.subscribe((frame) => {
      if (frame.type !== "stream:append") {
        return;
      }

      if (audienceValue === null) {
        return;
      }

      const matching = frame.entries.filter((entry) =>
        isChatEntry(entry, sessionId, audienceValue),
      );
      if (matching.length > 0) {
        setOptimisticEntries((current) => withoutReconciledOptimisticEntries(current, matching));
        setChatEntries((current) => mergeEntries(current, matching));
      }
    });
  }, [audienceValue, live, sessionId]);

  useEffect(() => {
    const previousConnectionCount = previousConnectionCountRef.current;
    previousConnectionCountRef.current = live.connectionCount;

    if (live.connectionCount <= 1 || live.connectionCount === previousConnectionCount) {
      return;
    }

    let cancelled = false;
    void (async () => {
      if (audienceValue === null) {
        resetForReconnect();
        return;
      }

      try {
        const stream = await getStream({
          session: sessionId,
          audience: audienceValue,
          kinds: CHAT_KINDS,
          limit: 50,
        });

        if (cancelled) {
          return;
        }

        setChatEntries((current) =>
          mergeEntries(
            current.filter(
              (entry) => entry.session_id === sessionId && entry.audience === audienceValue,
            ),
            stream.entries,
          ),
        );
        setOptimisticEntries((current) =>
          withoutReconciledOptimisticEntries(current, stream.entries),
        );
        replaceTailFromEntries(stream.entries);
      } catch {
        // The standing useApi calls retain the previous visible error/data state.
      } finally {
        if (!cancelled) {
          resetForReconnect();
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [audienceValue, live.connectionCount, replaceTailFromEntries, resetForReconnect, sessionId]);

  const loadOlderChat = useCallback(async (): Promise<boolean> => {
    if (audienceValue === null || chatNextCursor === null || loadingOlderChat) {
      return false;
    }

    const requestSeq = olderChatRequestSeqRef.current + 1;
    const requestFilterKey = chatFilterKeyRef.current;
    olderChatRequestSeqRef.current = requestSeq;
    setLoadingOlderChat(true);
    setOlderChatError(null);

    try {
      const stream = await getStream({
        session: sessionId,
        audience: audienceValue,
        kinds: CHAT_KINDS,
        limit: 50,
        before: chatNextCursor,
      });

      if (
        olderChatRequestSeqRef.current !== requestSeq ||
        chatFilterKeyRef.current !== requestFilterKey
      ) {
        return false;
      }

      setOptimisticEntries((current) =>
        withoutReconciledOptimisticEntries(current, stream.entries),
      );
      setChatEntries((current) =>
        mergeEntries(
          current.filter(
            (entry) => entry.session_id === sessionId && entry.audience === audienceValue,
          ),
          stream.entries,
        ),
      );
      setChatNextCursor(stream.next_cursor);
      return stream.entries.length > 0;
    } catch (caught) {
      if (
        olderChatRequestSeqRef.current === requestSeq &&
        chatFilterKeyRef.current === requestFilterKey
      ) {
        setOlderChatError(caught instanceof Error ? caught : new Error(String(caught)));
      }
      return false;
    } finally {
      if (olderChatRequestSeqRef.current === requestSeq) {
        setLoadingOlderChat(false);
      }
    }
  }, [audienceValue, chatNextCursor, loadingOlderChat, sessionId]);

  const visibleChatEntries = useMemo(
    () => sortStreamEntries([...chatEntries, ...optimisticEntries]) as ChatStreamEntry[],
    [chatEntries, optimisticEntries],
  );
  const liveLedger =
    turnStream.activeTurnId === null
      ? undefined
      : turnStream.ledgerByTurn.get(turnStream.activeTurnId);
  const replaySnapshot =
    replayTurnId === null ? undefined : turnStream.flowSnapshotByTurn.get(replayTurnId);
  const replayLedger =
    replayTurnId === null ? undefined : turnStream.ledgerByTurn.get(replayTurnId);
  const replayTurn =
    replayTurnId === null
      ? null
      : (turnsApi.data?.rows.find((row) => row.turn_id === replayTurnId) ?? null);
  const xrayState =
    replayTurnId !== null && replaySnapshot !== undefined
      ? replaySnapshot
      : {
          phases: turnStream.phases,
          tokenTextByPhase: turnStream.tokenTextByPhase,
          detailByPhase: turnStream.detailByPhase,
          terminalOutcome: turnStream.terminalOutcome,
          delibPath: turnStream.delibPath,
          finalAttempt: turnStream.finalAttempt,
        };
  const xrayTurnId = replayTurnId ?? turnStream.activeTurnId;
  const xrayLedger = replayTurnId === null ? liveLedger : replayLedger;
  const tracePlaceholder =
    replayTurnId !== null && replaySnapshot === undefined
      ? "trace unavailable this browser session"
      : null;

  const send = async (input: { message: string; attachments?: readonly File[] }) => {
    if (audienceValue === null) {
      return false;
    }

    const externalMessageId = makeClientMessageId();
    const optimisticEntry = optimisticUserEntry({
      externalMessageId,
      message: input.message,
      sessionId,
      audience: audienceValue,
      audienceEntityId,
      senderLabel: audienceDisplay.label,
      status: "queued",
    });

    setOptimisticEntries((current) => upsertOptimisticEntry(current, optimisticEntry));

    const result = await turnStream.runTurn({
      ...input,
      external_message_id: externalMessageId,
      audience: audienceValue,
      audience_entity_id: audienceEntityId,
      session: sessionId,
    });

    if (result === null) {
      setOptimisticEntries((current) =>
        current.filter((entry) => entryExternalMessageId(entry) !== externalMessageId),
      );
      return false;
    }

    setOptimisticEntries((current) => {
      const sent = markOptimisticStatus(current, externalMessageId, "sent");
      const realEntry = chatEntries.find((entry) => entry.id === result.stream_entry_id);
      return realEntry === undefined ? sent : withoutReconciledOptimisticEntries(sent, [realEntry]);
    });

    return true;
  };

  const selectXrayTab = (tab: XrayTabId) => {
    setXrayTab(tab);
    if (LAZY_XRAY_TABS.has(tab)) {
      setActivatedXrayTabs((current) => new Set([...current, tab]));
    }
  };

  return (
    <div className={`cog ${turnStream.running ? "" : "cog-xray-idle"}`}>
      <div className="chat">
        <ChatStream
          entries={visibleChatEntries}
          sessionId={sessionId}
          session={session}
          audienceValue={audienceValue}
          audienceDisplay={audienceDisplay}
          running={turnStream.running}
          hasOlder={chatNextCursor !== null}
          loadingOlder={loadingOlderChat}
          olderError={olderChatError}
          onLoadOlder={loadOlderChat}
        />
        <ParticipationPolicyControl
          sessionId={sessionId}
          policy={participationPolicy}
          onChanged={onSessionPolicyChanged ?? (async () => undefined)}
          locked={participationPolicyLocked}
        />
        <ChatInput
          audience={audienceDisplay.label ?? "unknown"}
          onSend={send}
          disabled={audienceValue === null}
        />
        <TurnHistoryStrip
          rows={turnsApi.data?.rows ?? []}
          selectedTurnId={replayTurnId}
          loading={turnsApi.loading}
          error={turnsApi.error}
          onSelect={setReplayTurnId}
          onLive={() => setReplayTurnId(null)}
        />
      </div>
      <div className="cog-divider"></div>
      <Xray
        phases={xrayState.phases}
        activeTurnId={xrayTurnId}
        tokenTextByPhase={xrayState.tokenTextByPhase}
        detailByPhase={xrayState.detailByPhase}
        terminalOutcome={xrayState.terminalOutcome}
        delibPath={xrayState.delibPath}
        finalAttempt={xrayState.finalAttempt}
        cachedLedger={xrayLedger}
        audience={audienceValue}
        tracePlaceholder={tracePlaceholder}
        particleEnabled={replayTurnId === null}
        replayTurn={replayTurn}
        sharedStateApi={sharedStateApi}
        commitmentsApi={commitmentsApi}
        identityApi={identityApi}
        promptApi={promptApi}
        activeTab={xrayTab}
        onTabChange={selectXrayTab}
      />
    </div>
  );
}
