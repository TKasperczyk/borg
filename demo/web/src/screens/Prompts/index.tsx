import { useEffect, useMemo, useState } from "react";

import { getAssembledPrompt, getPrompts } from "../../api/client";
import type { PromptKey } from "../../api/types";
import { Empty } from "../../components/Empty";
import { ErrorState } from "../../components/ErrorState";
import { Loading } from "../../components/Loading";
import { Tag } from "../../components/Tag";
import { useApi } from "../../hooks/use-api";
import { AssembledPromptPane } from "./AssembledPromptPane";
import { PromptBlockList } from "./PromptBlockList";
import { PromptEditor } from "./PromptEditor";

export function PromptsScreen({ onDirtyChange }: { onDirtyChange?: (dirty: boolean) => void }) {
  const api = useApi(getPrompts, []);
  const previewApi = useApi(getAssembledPrompt, []);
  const [selectedKey, setSelectedKey] = useState<PromptKey | null>(null);
  const [drafts, setDrafts] = useState<Partial<Record<PromptKey, string>>>({});

  async function refetchAll(): Promise<void> {
    await Promise.all([api.refetch(), previewApi.refetch()]);
  }

  function setBlockDraft(key: PromptKey, text: string): void {
    setDrafts((current) => ({ ...current, [key]: text }));
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

  useEffect(() => {
    const blocks = api.data?.blocks;
    if (blocks === undefined || blocks.length === 0) {
      return;
    }
    const firstKey = blocks[0]!.key;

    setSelectedKey((current) =>
      current !== null && blocks.some((block) => block.key === current) ? current : firstKey,
    );
  }, [api.data]);

  const dirtyCount = useMemo(() => {
    const blocks = api.data?.blocks ?? [];
    return blocks.filter(
      (block) => (drafts[block.key] ?? block.current_text) !== block.current_text,
    ).length;
  }, [api.data, drafts]);

  useEffect(() => {
    onDirtyChange?.(dirtyCount > 0);
  }, [dirtyCount, onDirtyChange]);

  if (api.loading && api.data === null) {
    return <Loading>loading prompts</Loading>;
  }
  if (api.error !== null) {
    return <ErrorState onRetry={() => void api.refetch()}>{api.error.message}</ErrorState>;
  }
  if (api.data === null || api.data.blocks.length === 0) {
    return <Empty>no prompts available</Empty>;
  }

  const selectedBlock =
    api.data.blocks.find((block) => block.key === selectedKey) ?? api.data.blocks[0]!;

  return (
    <div className="prompts-screen full-page">
      <div className="page-head">
        <h1>prompt lab</h1>
        <span className="sep">/</span>
        <span className="desc">
          Edit the {api.data.blocks.length} static framing blocks that shape borg's system prompt.
        </span>
        {dirtyCount > 0 ? <Tag kind="warn">{dirtyCount} unsaved draft</Tag> : null}
      </div>
      <div className="prompt-lab-layout page-body">
        <PromptBlockList
          blocks={api.data.blocks}
          selectedKey={selectedBlock.key}
          onSelect={setSelectedKey}
        />
        <PromptEditor
          block={selectedBlock}
          draft={drafts[selectedBlock.key] ?? selectedBlock.current_text}
          onDraftChange={setBlockDraft}
          onDraftClear={clearBlockDraft}
          refetch={refetchAll}
        />
        <AssembledPromptPane
          data={previewApi.data}
          error={previewApi.error}
          loading={previewApi.loading}
        />
      </div>
    </div>
  );
}
