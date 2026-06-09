import type { PromptBlockView, PromptKey } from "../../api/types";
import { Tag } from "../../components/Tag";
import { useInspector } from "../../components/Inspector/inspector-context";
import { dateLabel } from "../screen-utils";

export function PromptBlockList({
  blocks,
  selectedKey,
  onSelect,
}: {
  blocks: PromptBlockView[];
  selectedKey: PromptKey | null;
  onSelect: (key: PromptKey) => void;
}) {
  const inspector = useInspector();

  return (
    <aside className="list prompt-block-list" aria-label="prompt blocks">
      {blocks.map((block) => {
        const selected = block.key === selectedKey;
        return (
          <div
            key={block.key}
            className={`list-row prompt-block-row ${selected ? "selected" : ""}`}
            data-testid="prompt-block-row"
          >
            <button
              type="button"
              className="prompt-block-select"
              aria-current={selected ? "true" : undefined}
              onClick={() => onSelect(block.key)}
            >
              <span className="ttl">{block.label}</span>
              <span className="meta">
                <span>{block.key}</span>
                <Tag>{block.current_text_kind}</Tag>
                {block.overridden ? <Tag kind="warn">overridden</Tag> : null}
                <span>
                  updated_at {block.updated_at === null ? "-" : dateLabel(block.updated_at)}
                </span>
              </span>
            </button>
            <button
              type="button"
              className="btn sm ghost prompt-block-inspect"
              onClick={() =>
                inspector.openObject({ type: "prompt_block", id: block.key, hint: block })
              }
            >
              inspect
            </button>
          </div>
        );
      })}
    </aside>
  );
}
