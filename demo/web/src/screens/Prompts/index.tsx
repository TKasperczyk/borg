import { useEffect, useState } from "react";

import { deletePrompt, getAssembledPrompt, getPrompts, putPrompt } from "../../api/client";
import type { PromptAssembledResponse, PromptBlockView } from "../../api/types";
import { Modal } from "../../components/Modal";
import { Tag } from "../../components/Tag";
import { useApi } from "../../hooks/use-api";
import { dateLabel } from "../screen-utils";

export function PromptsScreen() {
  const api = useApi(getPrompts, []);
  const previewApi = useApi(getAssembledPrompt, []);
  const [previewOpen, setPreviewOpen] = useState(false);

  async function refetchAll(): Promise<void> {
    await Promise.all([api.refetch(), previewApi.refetch()]);
  }

  if (api.loading && api.data === null) {
    return <div className="notice">loading prompts</div>;
  }
  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
  }
  if (api.data === null) {
    return <div className="notice">no prompts available</div>;
  }

  return (
    <div className="prompts-screen">
      <div className="prompts-intro">
        <div className="prompts-title">prompt overrides</div>
        <div className="prompts-sub">
          Edit the 5 voice/posture/capabilities blocks that frame borg's system prompt. Defaults
          live in the code; overrides live in the substrate and take effect from the next turn
          onwards. Reset to default removes the override.
        </div>
      </div>
      <AssembledPromptPreview
        data={previewApi.data}
        error={previewApi.error}
        loading={previewApi.loading}
        open={previewOpen}
        onToggle={() => setPreviewOpen((current) => !current)}
      />
      <div className="prompts-list">
        {api.data.blocks.map((block) => (
          <PromptCard key={block.key} block={block} refetch={refetchAll} />
        ))}
      </div>
    </div>
  );
}

type PromptDiffLine = {
  id: string;
  kind: "same" | "added" | "removed";
  text: string;
};

function isRuntimeComposedHostBlock(block: PromptBlockView): boolean {
  return (
    block.key === "host_capabilities" &&
    !block.overridden &&
    block.current_text !== block.default_text
  );
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

function AssembledPromptPreview({
  data,
  error,
  loading,
  open,
  onToggle,
}: {
  data: PromptAssembledResponse | null;
  error: Error | null;
  loading: boolean;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="prompt-preview-panel">
      <div className="prompt-preview-head">
        <div>
          <div className="prompt-preview-title">assembled framing preview</div>
          <div className="prompt-preview-note">
            static framing prompt -- the cacheable prefix; per-turn dynamic context (retrieval,
            evidence ledger, commitments, current message) is added at runtime and not shown here.
          </div>
        </div>
        <button type="button" className="btn sm ghost" onClick={onToggle}>
          {open ? "hide assembled prompt" : "preview assembled prompt"}
        </button>
      </div>
      {open ? (
        <div className="prompt-preview-body">
          {loading && data === null ? <div className="notice">loading assembled prompt</div> : null}
          {error === null ? null : <div className="notice bad">{error.message}</div>}
          {data === null ? null : (
            <>
              <div className="prompt-preview-sections">
                {data.sections.map((section) => (
                  <Tag key={section}>{section}</Tag>
                ))}
              </div>
              <pre className="prompt-preview-text">{data.text}</pre>
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}

function PromptCard({ block, refetch }: { block: PromptBlockView; refetch: () => Promise<void> }) {
  const [draft, setDraft] = useState<string>(block.current_text);
  const [busy, setBusy] = useState<"save" | "reset" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmDraft, setConfirmDraft] = useState<string | null>(null);

  useEffect(() => {
    setDraft(block.current_text);
  }, [block.current_text, block.updated_at, block.overridden]);

  const runtimeComposed = isRuntimeComposedHostBlock(block);
  const dirty = draft !== block.current_text;
  const canSave = dirty && draft.trim().length > 0 && busy === null;
  const canReset = block.overridden && busy === null;

  async function save(text: string): Promise<boolean> {
    setBusy("save");
    setError(null);
    try {
      await putPrompt(block.key, text);
      await refetch();
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
    setBusy("reset");
    setError(null);
    try {
      await deletePrompt(block.key);
      await refetch();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Reset failed");
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <div className={`prompt-card ${block.overridden ? "overridden" : ""}`}>
        <div className="prompt-card-head">
          <div className="prompt-card-label">{block.label}</div>
          {block.overridden ? (
            <Tag kind="warn">overridden · {dateLabel(block.updated_at)}</Tag>
          ) : runtimeComposed ? (
            <Tag kind="info">runtime composed (connector-injected)</Tag>
          ) : (
            <Tag>default</Tag>
          )}
        </div>
        <div className="prompt-card-desc">{block.description}</div>
        <textarea
          className="prompt-card-text"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          spellCheck={false}
          rows={Math.min(20, Math.max(4, draft.split("\n").length + 1))}
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
              className="btn sm ghost"
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
        </div>
      </div>
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
