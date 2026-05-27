import { useEffect, useRef, useState } from "react";

import { getStream, setSessionPolicy } from "../../api/client";
import type {
  SessionParticipationPolicy,
  SessionRecord,
  StreamChatKind,
  StreamEntry,
  TurnStakes,
} from "../../api/types";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import type { TurnStreamState } from "../../hooks/use-turn-stream";
import { mergeEntries } from "../../lib/stream-utils";
import { ChatInput } from "./ChatInput";
import { ChatStream } from "./ChatStream";
import { Xray } from "./Xray";

const CHAT_KINDS: readonly StreamChatKind[] = ["user_msg", "agent_msg", "user_image_attachment"];
const CHAT_PANEL_LIMIT = 16;

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
  }, [audience, sessionId]);

  useEffect(() => {
    const streamData = streamApi.data;

    if (streamData !== null) {
      setChatEntries((current) =>
        mergeEntries(
          current.filter((entry) => entry.session_id === sessionId && entry.audience === audience),
          streamData.entries,
        ),
      );
    }
  }, [audience, streamApi.data]);

  useEffect(() => {
    return live.subscribe((frame) => {
      if (frame.type !== "stream:append") {
        return;
      }

      const matching = frame.entries.filter((entry) => isChatEntry(entry, sessionId, audience));
      if (matching.length > 0) {
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

  const send = async (input: {
    message: string;
    stakes: TurnStakes;
    attachments?: readonly File[];
  }) => {
    return turnStream.runTurn({
      ...input,
      audience,
      audience_entity_id: audienceEntityId,
      session: sessionId,
    });
  };

  return (
    <div className="cog">
      <div className="chat">
        <ChatStream
          entries={chatEntries.slice(-CHAT_PANEL_LIMIT)}
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
        <ChatInput audience={audience} running={turnStream.running} onSend={send} />
      </div>
      <div className="cog-divider"></div>
      <Xray
        phases={turnStream.phases}
        activeTurnId={turnStream.activeTurnId}
        tokenTextByPhase={turnStream.tokenTextByPhase}
        terminalOutcome={turnStream.terminalOutcome}
        delibPath={turnStream.delibPath}
        finalAttempt={turnStream.finalAttempt}
      />
    </div>
  );
}
