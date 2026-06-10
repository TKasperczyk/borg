import { useEffect, useState } from "react";

import { deletePrompt, putPrompt } from "../../api/client";
import type { PromptBlockView, PromptKey } from "../../api/types";
import { Modal } from "../../components/Modal";
import { Tag } from "../../components/Tag";
import { copyText } from "../../lib/clipboard";
import { dateLabel } from "../screen-utils";

type PromptDiffLine = {
  id: string;
  kind: "same" | "added" | "removed";
  text: string;
};

function isRuntimeComposedHostBlock(block: PromptBlockView): boolean {
  return block.current_text_kind === "runtime_composed";
}

function promptLines(text: string): string[] {
  return text.length === 0 ? [""] : text.split("\n");
}

function diffPromptLines(before: string, after: string): PromptDiffLine[] {
  const left = promptLines(before);
  const right = promptLines(after);
  const dp = Array.from({ length: left.length + 1 }, () => Array<number>(right.length + 1).fill(0));

  for (let i = left.length - 1; i >= 0; i -= 1) {
    for (let j = right.length - 1; j >= 0; j -= 1) {
      dp[i]![j] =
        left[i] === right[j] ? dp[i + 1]![j + 1]! + 1 : Math.max(dp[i + 1]![j]!, dp[i]![j + 1]!);
    }
  }

  const lines: PromptDiffLine[] = [];
  let i = 0;
  let j = 0;
  let id = 0;

  while (i < left.length && j < right.length) {
    if (left[i] === right[j]) {
      lines.push({ id: `same-${id}`, kind: "same", text: left[i]! });
      i += 1;
      j += 1;
    } else if (dp[i + 1]![j]! >= dp[i]![j + 1]!) {
      lines.push({ id: `removed-${id}`, kind: "removed", text: left[i]! });
      i += 1;
    } else {
      lines.push({ id: `added-${id}`, kind: "added", text: right[j]! });
      j += 1;
    }
    id += 1;
  }

  while (i < left.length) {
    lines.push({ id: `removed-${id}`, kind: "removed", text: left[i]! });
    i += 1;
    id += 1;
  }

  while (j < right.length) {
    lines.push({ id: `added-${id}`, kind: "added", text: right[j]! });
    j += 1;
    id += 1;
  }

  return lines;
}

function PromptDiff({ before, after }: { before: string; after: string }) {
  return (
    <div className="prompt-freeze-diff" aria-label="static default versus saved override diff">
      <div className="prompt-freeze-diff-head">
        <span>static default</span>
        <span>saved static override</span>
      </div>
      <div className="prompt-freeze-diff-lines">
        {diffPromptLines(before, after).map((line) => (
          <div key={line.id} className={`prompt-freeze-diff-line ${line.kind}`}>
            <span className="prompt-freeze-diff-marker">
              {line.kind === "added" ? "+" : line.kind === "removed" ? "-" : " "}
            </span>
            <span>{line.text.length === 0 ? " " : line.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function PromptEditor({
  block,
  refetch,
}: {
  block: PromptBlockView;
  refetch: () => Promise<void>;
}) {
  const [drafts, setDrafts] = useState<Partial<Record<PromptKey, string>>>({});
  const [busy, setBusy] = useState<"save" | "reset" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmDraft, setConfirmDraft] = useState<string | null>(null);
  const [copyStatus, setCopyStatus] = useState<string | null>(null);

  useEffect(() => {
    setError(null);
    setCopyStatus(null);
    setConfirmDraft(null);
  }, [block.key]);

  const draft = drafts[block.key] ?? block.current_text;
  const runtimeComposed = isRuntimeComposedHostBlock(block);
  const dirty = draft !== block.current_text;
  const canSave = dirty && draft.trim().length > 0 && busy === null;
  const canReset = block.overridden && busy === null;

  function setBlockDraft(text: string): void {
    setDrafts((current) => ({ ...current, [block.key]: text }));
  }

  function clearBlockDraft(key: PromptKey): void {
    setDrafts((current) => {
      if (current[key] === undefined) {
        return current;
      }
      const next = { ...current };
      delete next[key];
      return next;
    });
  }

  async function save(text: string): Promise<boolean> {
    const key = block.key;
    setBusy("save");
    setError(null);
    try {
      await putPrompt(key, text);
      await refetch();
      clearBlockDraft(key);
      return true;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Save failed");
      return false;
    } finally {
      setBusy(null);
    }
  }

  function requestSave(): void {
    if (!canSave) {
      return;
    }

    if (runtimeComposed) {
      setError(null);
      setConfirmDraft(draft);
      return;
    }

    void save(draft);
  }

  function closeConfirm(): void {
    if (busy === null) {
      setConfirmDraft(null);
    }
  }

  async function confirmSave(): Promise<void> {
    if (confirmDraft === null) {
      return;
    }

    const saved = await save(confirmDraft);
    if (saved) {
      setConfirmDraft(null);
    }
  }

  async function reset(): Promise<void> {
    const key = block.key;
    setBusy("reset");
    setError(null);
    try {
      await deletePrompt(key);
      await refetch();
      clearBlockDraft(key);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Reset failed");
    } finally {
      setBusy(null);
    }
  }

  async function copy(label: string, text: string): Promise<void> {
    setCopyStatus(null);
    try {
      await copyText(text);
      setCopyStatus(`copied ${label}`);
    } catch (cause) {
      setCopyStatus(cause instanceof Error ? cause.message : "Copy failed");
    }
  }

  return (
    <>
      <section className="detail prompt-editor" aria-label="selected prompt block editor">
        <div className="prompt-editor-head">
          <div>
            <h2>{block.label}</h2>
            <div className="meta-line">
              <span>{block.key}</span>
              <Tag>{block.current_text_kind}</Tag>
              {block.overridden ? (
                <Tag kind="warn">overridden · {dateLabel(block.updated_at)}</Tag>
              ) : runtimeComposed ? (
                <Tag kind="info">runtime composed (connector-injected)</Tag>
              ) : (
                <Tag>default</Tag>
              )}
            </div>
          </div>
          <div className="prompt-copy-actions">
            <button
              type="button"
              className="btn sm ghost"
              onClick={() => void copy("default text", block.default_text)}
            >
              copy default text
            </button>
            <button
              type="button"
              className="btn sm ghost"
              onClick={() => void copy("current text", block.current_text)}
            >
              copy current text
            </button>
          </div>
        </div>
        <div className="prompt-card-desc">{block.description}</div>
        <div className="prompt-editor-reference">
          <div className="prompt-reference-panel">
            <div className="prompt-reference-label">static default</div>
            <pre>{block.default_text}</pre>
          </div>
          <div className="prompt-reference-panel">
            <div className="prompt-reference-label">
              {runtimeComposed ? "live connector-composed current" : "current text"}
            </div>
            <pre>{block.current_text}</pre>
          </div>
        </div>
        <label className="prompt-editor-label" htmlFor={`prompt-editor-${block.key}`}>
          edited override
        </label>
        <textarea
          id={`prompt-editor-${block.key}`}
          className="prompt-card-text"
          value={draft}
          onChange={(event) => setBlockDraft(event.target.value)}
          spellCheck={false}
          rows={Math.min(20, Math.max(6, draft.split("\n").length + 1))}
        />
        <div className="prompt-card-actions">
          <button
            type="button"
            className="btn sm primary"
            disabled={!canSave}
            onClick={requestSave}
          >
            {busy === "save" ? "saving..." : "save"}
          </button>
          {block.overridden ? (
            <button
              type="button"
              className="btn sm danger"
              disabled={!canReset}
              onClick={() => void reset()}
            >
              {busy === "reset" ? "resetting..." : "reset to default"}
            </button>
          ) : null}
          {error === null ? null : (
            <span className="warn" role="alert">
              {error}
            </span>
          )}
          {copyStatus === null ? null : <span className="prompt-copy-status">{copyStatus}</span>}
        </div>
        <div className="prompt-editor-diff">
          <div className="prompt-reference-label">default to edited diff</div>
          <PromptDiff before={block.default_text} after={draft} />
        </div>
      </section>
      <Modal
        open={confirmDraft !== null}
        title="freeze host capabilities override?"
        onClose={closeConfirm}
        footer={
          <>
            <button
              type="button"
              className="btn sm ghost"
              disabled={busy !== null}
              onClick={closeConfirm}
            >
              cancel
            </button>
            <button
              type="button"
              className="btn sm primary"
              disabled={busy !== null}
              onClick={() => void confirmSave()}
            >
              {busy === "save" ? "saving..." : "save static override"}
            </button>
          </>
        }
      >
        <div className="modal-form">
          <div className="prompt-freeze-warning">
            Saving freezes the current live connector-composed host capabilities into a STATIC
            override; future connector/outbound-capability injection will no longer update this
            block. Reset removes the override and restores live injection.
          </div>
          <PromptDiff before={block.default_text} after={confirmDraft ?? ""} />
        </div>
      </Modal>
    </>
  );
}
