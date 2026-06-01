import { useEffect, useState } from "react";

import { deletePrompt, getPrompts, putPrompt } from "../../api/client";
import type { PromptBlockView } from "../../api/types";
import { Tag } from "../../components/Tag";
import { useApi } from "../../hooks/use-api";
import { dateLabel } from "../screen-utils";

export function PromptsScreen() {
  const api = useApi(getPrompts, []);

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
      <div className="prompts-list">
        {api.data.blocks.map((block) => (
          <PromptCard key={block.key} block={block} refetch={api.refetch} />
        ))}
      </div>
    </div>
  );
}

function PromptCard({
  block,
  refetch,
}: {
  block: PromptBlockView;
  refetch: () => Promise<void>;
}) {
  const [draft, setDraft] = useState<string>(block.current_text);
  const [busy, setBusy] = useState<"save" | "reset" | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDraft(block.current_text);
  }, [block.current_text, block.updated_at, block.overridden]);

  const dirty = draft !== block.current_text;
  const canSave = dirty && draft.trim().length > 0 && busy === null;
  const canReset = block.overridden && busy === null;

  async function save(): Promise<void> {
    setBusy("save");
    setError(null);
    try {
      await putPrompt(block.key, draft);
      await refetch();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Save failed");
    } finally {
      setBusy(null);
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
    <div className={`prompt-card ${block.overridden ? "overridden" : ""}`}>
      <div className="prompt-card-head">
        <div className="prompt-card-label">{block.label}</div>
        {block.overridden ? (
          <Tag kind="warn">overridden · {dateLabel(block.updated_at)}</Tag>
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
          onClick={() => void save()}
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
  );
}
