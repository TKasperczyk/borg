import { useState } from "react";

import { RESET_CONFIRM_TOKEN, postAdminReset } from "../api/client";
import { Modal } from "./Modal";

export type ResetButtonProps = {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
};

export function ResetButton({ open, onOpenChange }: ResetButtonProps = {}) {
  const [internalOpen, setInternalOpen] = useState(false);
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dialogOpen = open ?? internalOpen;

  const canConfirm = typed === RESET_CONFIRM_TOKEN && !busy;

  function setDialogOpen(nextOpen: boolean): void {
    if (open === undefined) {
      setInternalOpen(nextOpen);
    }
    onOpenChange?.(nextOpen);
  }

  function close(): void {
    if (busy) {
      return;
    }
    setDialogOpen(false);
    setTyped("");
    setError(null);
  }

  async function confirm(): Promise<void> {
    if (!canConfirm) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await postAdminReset();
      window.location.reload();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Reset failed");
      setBusy(false);
    }
  }

  return (
    <>
      <button
        type="button"
        className="topbar-reset"
        onClick={() => setDialogOpen(true)}
        title="Wipe all borg state and restart with an empty substrate"
      >
        reset
      </button>
      <Modal
        open={dialogOpen}
        title={busy ? "resetting borg..." : "reset borg to clean slate"}
        onClose={close}
        footer={
          busy ? null : (
            <>
              <button type="button" className="btn sm ghost" onClick={close}>
                cancel
              </button>
              <button
                type="button"
                className="btn sm primary"
                disabled={!canConfirm}
                onClick={() => void confirm()}
              >
                reset borg
              </button>
            </>
          )
        }
      >
        {busy ? (
          <div className="modal-form" aria-live="polite">
            <div className="dream-running">
              <span className="dream-running-spinner" aria-hidden="true" />
              <div style={{ color: "var(--text)", fontFamily: "var(--sans)", lineHeight: 1.5 }}>
                Wiping substrate and reopening borg. The page will reload when complete.
              </div>
            </div>
          </div>
        ) : (
          <div className="modal-form">
            <div style={{ color: "var(--text)", fontFamily: "var(--sans)", lineHeight: 1.5 }}>
              This deletes the entire borg substrate: the conversation stream, every memory band
              (episodic, semantic, identity, commitments, social, relational, procedural,
              affective), all attachments, the dream audit log, and the review queue.
            </div>
            <div className="dim">
              The operation is irreversible. No backup is taken. Re-open borg with an empty
              substrate, then the UI reloads.
            </div>
            <label className="modal-field">
              <span>
                type <code>{RESET_CONFIRM_TOKEN}</code> to confirm
              </span>
              <input
                autoFocus
                value={typed}
                onChange={(event) => setTyped(event.target.value)}
                placeholder={RESET_CONFIRM_TOKEN}
                spellCheck={false}
                autoComplete="off"
              />
            </label>
            {error === null ? null : (
              <div className="warn" role="alert">
                {error}
              </div>
            )}
          </div>
        )}
      </Modal>
    </>
  );
}
