import { useMemo, useRef, useState, type ReactNode } from "react";

import type { PromptAssembledResponse, PromptAssembledSegment } from "../../api/types";
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

type FocusedSection = Pick<PromptAssembledSegment, "id" | "start" | "end">;

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

function rangesForSlice(ranges: TextRange[], start: number, end: number): TextRange[] {
  return ranges
    .filter((range) => range.start < end && range.end > start)
    .map((range) => ({
      start: Math.max(range.start, start) - start,
      end: Math.min(range.end, end) - start,
      kind: range.kind,
    }));
}

function sortedSegments(data: PromptAssembledResponse): PromptAssembledSegment[] {
  return [...data.segments]
    .filter((segment) => segment.start >= 0 && segment.end >= segment.start)
    .sort((left, right) => left.start - right.start);
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
  const segmentRefs = useRef(new Map<string, HTMLSpanElement>());
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
      start: focusedSection.start,
      end: focusedSection.end,
      kind: "section",
    };
  }, [focusedSection]);
  const highlightRanges = sectionRange === null ? searchRanges : [...searchRanges, sectionRange];
  const tokenEstimate = Math.ceil(text.length / 4);

  function focusSection(segment: PromptAssembledSegment): void {
    setFocusedSection({ id: segment.id, start: segment.start, end: segment.end });
    window.requestAnimationFrame(() => {
      segmentRefs.current.get(segment.id)?.scrollIntoView({ block: "start", inline: "nearest" });
    });
  }

  function renderPreviewText(prompt: PromptAssembledResponse): ReactNode {
    const segments = sortedSegments(prompt);
    if (segments.length === 0) {
      return highlightedText(prompt.text, highlightRanges);
    }

    const nodes: ReactNode[] = [];
    let cursor = 0;
    for (const segment of segments) {
      const start = Math.min(segment.start, prompt.text.length);
      const end = Math.min(segment.end, prompt.text.length);
      if (start > cursor) {
        nodes.push(
          <span key={`gap-${cursor}-${start}`}>
            {highlightedText(
              prompt.text.slice(cursor, start),
              rangesForSlice(highlightRanges, cursor, start),
            )}
          </span>,
        );
      }

      nodes.push(
        <span
          key={segment.id}
          ref={(node) => {
            if (node === null) {
              segmentRefs.current.delete(segment.id);
            } else {
              segmentRefs.current.set(segment.id, node);
            }
          }}
          className={`prompt-preview-segment ${
            focusedSection?.id === segment.id ? "active" : ""
          }`.trim()}
          data-segment-id={segment.id}
        >
          {highlightedText(
            prompt.text.slice(start, end),
            rangesForSlice(highlightRanges, start, end),
          )}
        </span>,
      );
      cursor = Math.max(cursor, end);
    }

    if (cursor < prompt.text.length) {
      nodes.push(
        <span key={`gap-${cursor}-${prompt.text.length}`}>
          {highlightedText(
            prompt.text.slice(cursor),
            rangesForSlice(highlightRanges, cursor, prompt.text.length),
          )}
        </span>,
      );
    }

    return nodes;
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
        {data === null ? null : <Tag>{data.segments.length} assembled sections</Tag>}
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
              {sortedSegments(data).map((segment) => (
                <button
                  key={segment.id}
                  type="button"
                  className={`prompt-outline-chip ${
                    focusedSection?.id === segment.id ? "active" : ""
                  }`}
                  onClick={() => focusSection(segment)}
                >
                  {segment.label}
                </button>
              ))}
            </div>
            <pre className="prompt-preview-text">{renderPreviewText(data)}</pre>
          </>
        )}
      </div>
    </aside>
  );
}
