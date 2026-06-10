import { useMemo, useRef, useState, type ReactNode } from "react";

import type { PromptAssembledResponse } from "../../api/types";
import { Empty } from "../../components/Empty";
import { ErrorState } from "../../components/ErrorState";
import { Loading } from "../../components/Loading";
import { Tag } from "../../components/Tag";
import { copyText } from "../../lib/clipboard";

type TextRange = {
  start: number;
  end: number;
  kind: "search" | "section";
};

type FocusedSection = {
  section: string;
  index: number;
};

function findMatchRanges(text: string, query: string): TextRange[] {
  if (query.length === 0) {
    return [];
  }

  const normalizedText = text.toLocaleLowerCase();
  const normalizedQuery = query.toLocaleLowerCase();
  const ranges: TextRange[] = [];
  let start = 0;
  let index = normalizedText.indexOf(normalizedQuery, start);
  while (index >= 0) {
    ranges.push({ start: index, end: index + normalizedQuery.length, kind: "search" });
    start = index + 1;
    index = normalizedText.indexOf(normalizedQuery, start);
  }
  return ranges;
}

function highlightedText(text: string, ranges: TextRange[]): ReactNode {
  if (ranges.length === 0) {
    return text;
  }

  const boundaries = new Set<number>([0, text.length]);
  for (const range of ranges) {
    boundaries.add(range.start);
    boundaries.add(range.end);
  }

  const sortedBoundaries = Array.from(boundaries).sort((left, right) => left - right);
  return sortedBoundaries.slice(0, -1).map((start, boundaryIndex) => {
    const end = sortedBoundaries[boundaryIndex + 1]!;
    const slice = text.slice(start, end);
    const activeKinds = ranges
      .filter((range) => range.start < end && range.end > start)
      .map((range) => range.kind);
    if (activeKinds.length === 0) {
      return <span key={`${start}-${end}`}>{slice}</span>;
    }
    const hasSearch = activeKinds.includes("search");
    const hasSection = activeKinds.includes("section");
    return (
      <mark
        key={`${start}-${end}`}
        className={`prompt-text-hit ${hasSearch ? "search" : ""} ${hasSection ? "section" : ""}`}
      >
        {slice}
      </mark>
    );
  });
}

export function AssembledPromptPane({
  data,
  error,
  loading,
}: {
  data: PromptAssembledResponse | null;
  error: Error | null;
  loading: boolean;
}) {
  const textRef = useRef<HTMLPreElement | null>(null);
  const [search, setSearch] = useState("");
  const [focusedSection, setFocusedSection] = useState<FocusedSection | null>(null);
  const [copyStatus, setCopyStatus] = useState<string | null>(null);

  const text = data?.text ?? "";
  const searchRanges = useMemo(() => findMatchRanges(text, search), [search, text]);
  const sectionRange = useMemo<TextRange | null>(() => {
    if (focusedSection === null) {
      return null;
    }
    return {
      start: focusedSection.index,
      end: focusedSection.index + focusedSection.section.length,
      kind: "section",
    };
  }, [focusedSection]);
  const highlightRanges = sectionRange === null ? searchRanges : [...searchRanges, sectionRange];
  const tokenEstimate = Math.ceil(text.length / 4);

  function focusSection(section: string): void {
    const index = text.indexOf(section);
    if (index < 0) {
      setFocusedSection(null);
      return;
    }

    setFocusedSection({ section, index });
    window.requestAnimationFrame(() => {
      const line = text.slice(0, index).split("\n").length - 1;
      if (textRef.current !== null) {
        textRef.current.scrollTop = Math.max(0, line * 18 - 18);
      }
    });
  }

  async function copyAll(): Promise<void> {
    setCopyStatus(null);
    try {
      await copyText(text);
      setCopyStatus("copied assembled prompt");
    } catch (cause) {
      setCopyStatus(cause instanceof Error ? cause.message : "Copy failed");
    }
  }

  return (
    <aside className="prompt-assembled-pane" aria-label="assembled prompt preview">
      <div className="prompt-preview-head">
        <div>
          <div className="prompt-preview-title">assembled framing preview</div>
          <div className="prompt-preview-note">
            static framing prompt -- the cacheable prefix; per-turn dynamic context (retrieval,
            evidence ledger, commitments, current message) is added at runtime and not shown here.
          </div>
        </div>
      </div>
      <div className="prompt-assembled-tools">
        <label className="prompt-search-label" htmlFor="assembled-prompt-search">
          search
        </label>
        <input
          id="assembled-prompt-search"
          className="prompt-search-input"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="case-insensitive; overlaps count"
          aria-label="search assembled prompt text"
        />
        <span className="prompt-search-count">
          {searchRanges.length} {searchRanges.length === 1 ? "match" : "matches"}
        </span>
        <span className="prompt-search-hint">case-insensitive; overlapping matches counted</span>
        <button
          type="button"
          className="btn sm ghost"
          disabled={text.length === 0}
          onClick={copyAll}
        >
          copy all
        </button>
      </div>
      <div className="prompt-token-row">
        <span data-testid="assembled-token-estimate">
          <Tag kind="info">approximate ~{tokenEstimate} tokens (chars/4, rough)</Tag>
        </span>
        {copyStatus === null ? null : <span className="prompt-copy-status">{copyStatus}</span>}
      </div>
      <div className="prompt-preview-body">
        {loading && data === null ? <Loading>loading assembled prompt</Loading> : null}
        {error === null ? null : <ErrorState>{error.message}</ErrorState>}
        {data === null ? (
          loading ? null : (
            <Empty>assembled prompt unavailable</Empty>
          )
        ) : (
          <>
            <div className="prompt-preview-sections" aria-label="assembled prompt sections">
              {data.sections.map((section) => (
                <button
                  key={section}
                  type="button"
                  className={`prompt-outline-chip ${
                    focusedSection?.section === section ? "active" : ""
                  }`}
                  onClick={() => focusSection(section)}
                >
                  {section}
                </button>
              ))}
            </div>
            <pre ref={textRef} className="prompt-preview-text">
              {highlightedText(data.text, highlightRanges)}
            </pre>
          </>
        )}
      </div>
    </aside>
  );
}
