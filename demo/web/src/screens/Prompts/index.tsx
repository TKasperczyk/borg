import { useEffect, useState } from "react";

import { getAssembledPrompt, getPrompts } from "../../api/client";
import type { PromptKey } from "../../api/types";
import { useApi } from "../../hooks/use-api";
import { AssembledPromptPane } from "./AssembledPromptPane";
import { PromptBlockList } from "./PromptBlockList";
import { PromptEditor } from "./PromptEditor";

export function PromptsScreen() {
  const api = useApi(getPrompts, []);
  const previewApi = useApi(getAssembledPrompt, []);
  const [selectedKey, setSelectedKey] = useState<PromptKey | null>(null);

  async function refetchAll(): Promise<void> {
    await Promise.all([api.refetch(), previewApi.refetch()]);
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

  if (api.loading && api.data === null) {
    return <div className="notice">loading prompts</div>;
  }
  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
  }
  if (api.data === null || api.data.blocks.length === 0) {
    return <div className="notice">no prompts available</div>;
  }

  const selectedBlock =
    api.data.blocks.find((block) => block.key === selectedKey) ?? api.data.blocks[0]!;

  return (
    <div className="prompts-screen full-page">
      <div className="page-head">
        <h1>prompt lab</h1>
        <span className="sep">/</span>
        <span className="desc">
          Edit the 5 voice/posture/capabilities blocks that frame borg's system prompt.
        </span>
      </div>
      <div className="prompt-lab band-detail page-body">
        <PromptBlockList
          blocks={api.data.blocks}
          selectedKey={selectedBlock.key}
          onSelect={setSelectedKey}
        />
        <PromptEditor block={selectedBlock} refetch={refetchAll} />
        <AssembledPromptPane
          data={previewApi.data}
          error={previewApi.error}
          loading={previewApi.loading}
        />
      </div>
    </div>
  );
}
