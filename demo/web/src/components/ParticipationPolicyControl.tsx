import { useEffect, useState } from "react";

import { setSessionPolicy } from "../api/client";
import type { SessionParticipationPolicy } from "../api/types";

const PARTICIPATION_POLICIES: readonly SessionParticipationPolicy[] = [
  "active",
  "paused",
  "observing",
  "muted",
];

const PARTICIPATION_POLICY_LINES: Record<SessionParticipationPolicy, string> = {
  active: "normal participation",
  observing: "can observe but will not answer",
  paused: "will not process active participation",
  muted: "will stay silent",
};

export function ParticipationPolicyControl({
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
      <div className="participation-policy-line">
        <span className="current">{policy}</span> · {PARTICIPATION_POLICY_LINES[policy]}
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
