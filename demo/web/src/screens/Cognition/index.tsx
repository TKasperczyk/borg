import { useEffect, useMemo, useRef, useState } from "react";

import { getStream, setSessionPolicy } from "../../api/client";
import type {
  SessionParticipationPolicy,
  SessionRecord,
  StreamChatKind,
  StreamEntry,
} from "../../api/types";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import type { TurnStreamState } from "../../hooks/use-turn-stream";
import { mergeEntries, sortStreamEntries, streamContentText } from "../../lib/stream-utils";
import { ChatInput } from "./ChatInput";
import { ChatStream } from "./ChatStream";
import type { ChatDeliveryStatus, ChatStreamEntry } from "./chat-utils";
import { Xray } from "./Xray";

const CHAT_KINDS: readonly StreamChatKind[] = ["user_msg", "agent_msg", "user_image_attachment"];
const CHAT_PANEL_LIMIT = 16;
const DEMO_SOURCE_TYPE = "demo";

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
  audience: string;
  audienceEntityId?: string | null;
  turnStream: TurnStreamState;
  session?: SessionRecord | null;
  onSessionPolicyChanged?: () => Promise<void>;
};

function isChatEntry(entry: StreamEntry, sessionId: string, audience: string): boolean {
  return (
    entry.session_id === sessionId &&
    CHAT_KINDS.includes(entry.kind as StreamChatKind) &&
    entry.audience === audience
  );
}

function entryExternalMessageId(entry: ChatStreamEntry): string | null {
  return entry.external_message_id ?? entry.source_message_key?.external_message_id ?? null;
}

function sameUserMessageForReconcile(
  optimistic: ChatStreamEntry,
  real: StreamEntry,
): boolean {
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
    (optimistic) =>
      !realEntries.some((entry) => sameUserMessageForReconcile(optimistic, entry)),
  );
}

function optimisticUserEntry(input: {
  externalMessageId: string;
  message: string;
  sessionId: string;
  audience: string;
  status: ChatDeliveryStatus;
}): ChatStreamEntry {
  return {
    id: `optimistic:${input.externalMessageId}`,
    timestamp: Date.now(),
    kind: "user_msg",
    content: input.message,
    audience: input.audience,
    sender_entity_id: null,
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

const PARTICIPATION_POLICIES: readonly SessionParticipationPolicy[] = [
  "active",
  "paused",
  "observing",
  "muted",
];

function ParticipationPolicyControl({
  sessionId,
  policy,
  onChanged,
  locked = false,
}: {
  sessionId: string;
  policy: SessionParticipationPolicy;
  onChanged: () => Promise<void>;
  locked?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [selectedPolicy, setSelectedPolicy] = useState<SessionParticipationPolicy>(policy);
  const [reason, setReason] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setSelectedPolicy(policy);
    setReason("");
  }, [policy, sessionId]);

  const submit = () => {
    if (submitting) {
      return;
    }

    void (async () => {
      setSubmitting(true);
      setError(null);
      try {
        await setSessionPolicy(sessionId, selectedPolicy, reason);
        setReason("");
        setOpen(false);
        await onChanged();
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
      } finally {
        setSubmitting(false);
      }
    })();
  };

  return (
    <section className="participation-policy" aria-label="Participation policy">
      <div className="participation-policy-head">
        <span className="participation-policy-title">Participation</span>
        <button
          className={`participation-policy-badge ${policy === "active" ? "active" : "warn"}`}
          type="button"
          onClick={() => setOpen((current) => !current)}
          aria-label={`participation policy ${policy}`}
          disabled={locked}
        >
          {policy}
        </button>
      </div>
      {open ? (
        <div className="participation-policy-editor">
          <select
            aria-label="participation policy selection"
            value={selectedPolicy}
            onChange={(event) =>
              setSelectedPolicy(event.target.value as SessionParticipationPolicy)
            }
          >
            {PARTICIPATION_POLICIES.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
          <input
            aria-label="participation policy reason"
            value={reason}
            maxLength={500}
            onChange={(event) => setReason(event.target.value)}
            placeholder="reason"
          />
          <button className="btn sm primary" type="button" onClick={submit} disabled={submitting}>
            apply
          </button>
        </div>
      ) : null}
      {error === null ? null : <div className="participation-policy-error">{error}</div>}
    </section>
  );
}

export function CognitionScreen({
  sessionId,
  audience,
  audienceEntityId,
  turnStream,
  session = null,
  onSessionPolicyChanged,
}: CognitionScreenProps) {
  const live = useLiveEventsContext();
  const [chatEntries, setChatEntries] = useState<StreamEntry[]>([]);
  const [optimisticEntries, setOptimisticEntries] = useState<ChatStreamEntry[]>([]);
  const previousConnectionCountRef = useRef(live.connectionCount);
  const participationPolicy = session?.participation_policy ?? "active";
  const participationPolicyLocked = session?.audience_role === "operator";

  const streamApi = useApi(
    () => getStream({ session: sessionId, audience, kinds: CHAT_KINDS, limit: 50 }),
    [audience, sessionId],
  );
  const resetForReconnect = turnStream.resetForReconnect;
  const replaceTailFromEntries = turnStream.replaceTailFromEntries;

  useEffect(() => {
    setChatEntries([]);
    setOptimisticEntries([]);
  }, [audience, sessionId]);

  useEffect(() => {
    const streamData = streamApi.data;

    if (streamData !== null) {
      setOptimisticEntries((current) =>
        withoutReconciledOptimisticEntries(current, streamData.entries),
      );
      setChatEntries((current) =>
        mergeEntries(
          current.filter((entry) => entry.session_id === sessionId && entry.audience === audience),
          streamData.entries,
        ),
      );
    }
  }, [audience, sessionId, streamApi.data]);

  useEffect(() => {
    return live.subscribe((frame) => {
      if (frame.type !== "stream:append") {
        return;
      }

      const matching = frame.entries.filter((entry) => isChatEntry(entry, sessionId, audience));
      if (matching.length > 0) {
        setOptimisticEntries((current) => withoutReconciledOptimisticEntries(current, matching));
        setChatEntries((current) => mergeEntries(current, matching));
      }
    });
  }, [audience, live, sessionId]);

  useEffect(() => {
    const previousConnectionCount = previousConnectionCountRef.current;
    previousConnectionCountRef.current = live.connectionCount;

    if (live.connectionCount <= 1 || live.connectionCount === previousConnectionCount) {
      return;
    }

    let cancelled = false;
    void (async () => {
      try {
        const stream = await getStream({
          session: sessionId,
          audience,
          kinds: CHAT_KINDS,
          limit: 50,
        });

        if (cancelled) {
          return;
        }

        setChatEntries((current) =>
          mergeEntries(
            current.filter(
              (entry) => entry.session_id === sessionId && entry.audience === audience,
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
  }, [audience, live.connectionCount, replaceTailFromEntries, resetForReconnect, sessionId]);

  const visibleChatEntries = useMemo(
    () =>
      (sortStreamEntries([...chatEntries, ...optimisticEntries]) as ChatStreamEntry[]).slice(
        -CHAT_PANEL_LIMIT,
      ),
    [chatEntries, optimisticEntries],
  );

  const send = async (input: { message: string; attachments?: readonly File[] }) => {
    const externalMessageId = makeClientMessageId();
    const optimisticEntry = optimisticUserEntry({
      externalMessageId,
      message: input.message,
      sessionId,
      audience,
      status: "queued",
    });

    setOptimisticEntries((current) => upsertOptimisticEntry(current, optimisticEntry));

    const result = await turnStream.runTurn({
      ...input,
      external_message_id: externalMessageId,
      audience,
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
      return realEntry === undefined
        ? sent
        : withoutReconciledOptimisticEntries(sent, [realEntry]);
    });

    return true;
  };

  return (
    <div className="cog">
      <div className="chat">
        <ChatStream
          entries={visibleChatEntries}
          sessionId={sessionId}
          audience={audience}
          running={turnStream.running}
        />
        <ParticipationPolicyControl
          sessionId={sessionId}
          policy={participationPolicy}
          onChanged={onSessionPolicyChanged ?? (async () => undefined)}
          locked={participationPolicyLocked}
        />
        <ChatInput audience={audience} onSend={send} />
      </div>
      <div className="cog-divider"></div>
      <Xray
        phases={turnStream.phases}
        activeTurnId={turnStream.activeTurnId}
        tokenTextByPhase={turnStream.tokenTextByPhase}
        detailByPhase={turnStream.detailByPhase}
        terminalOutcome={turnStream.terminalOutcome}
        delibPath={turnStream.delibPath}
        finalAttempt={turnStream.finalAttempt}
      />
    </div>
  );
}
