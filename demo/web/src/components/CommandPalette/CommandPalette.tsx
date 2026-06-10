import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";

import { getMemoryBand } from "../../api/client";
import type {
  MemoryBandDetail,
  MemoryBandId,
  ProceduralMemoryItem,
  SemanticMemoryEdge,
  SemanticMemoryNode,
} from "../../api/types";
import { useLiveCache } from "../../hooks/use-live-cache";
import {
  RAIL_ITEMS,
  routeChordLabel,
  type RouteId,
  type RouteNavigationOptions,
} from "../../routes";
import { previewLine } from "../SessionFleet";
import { shortId } from "../../screens/screen-utils";
import { useInspector } from "../Inspector/inspector-context";
import { resolveObjectType, type ObjectType } from "../Inspector/inspector-id";
import { objectRegistry } from "../Inspector/inspector-registry";

type CommandPaletteProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  setView: (view: RouteId, options?: RouteNavigationOptions) => void;
  setSessionId: (sessionId: string) => void;
  onOpenReset: () => void;
};

type CommandGroup = "Screens" | "Sessions" | "Open object" | "Memory" | "Actions";

type CommandResult = {
  id: string;
  group: CommandGroup;
  title: string;
  subtitle?: string;
  icon?: string;
  hint?: string;
  disabled?: boolean;
  run?: () => void;
};

type SearchableMemoryBand = Extract<MemoryBandId, "episodic" | "semantic" | "procedural">;

type MemoryHit = {
  id: string;
  band: SearchableMemoryBand;
  type: ObjectType;
  title: string;
  subtitle: string;
};

type MemorySearchState =
  | { status: "idle"; query: string; hits: MemoryHit[]; error: null }
  | { status: "loading"; query: string; hits: MemoryHit[]; error: null }
  | { status: "ready"; query: string; hits: MemoryHit[]; error: null }
  | { status: "error"; query: string; hits: MemoryHit[]; error: string };

const GROUP_ORDER: readonly CommandGroup[] = [
  "Screens",
  "Sessions",
  "Open object",
  "Memory",
  "Actions",
];
const SEARCHABLE_MEMORY_BANDS: readonly SearchableMemoryBand[] = [
  "episodic",
  "semantic",
  "procedural",
];
const MEMORY_SEARCH_LIMIT = 5;

function normalizeCatalogString(value: string): string {
  return value.toLocaleLowerCase().replace(/\s+/g, "");
}

function fuzzyIncludes(candidate: string, query: string): boolean {
  let queryIndex = 0;
  for (const char of candidate) {
    if (char === query[queryIndex]) {
      queryIndex += 1;
      if (queryIndex === query.length) {
        return true;
      }
    }
  }
  return query.length === 0;
}

function catalogMatches(query: string, strings: readonly string[]): boolean {
  const normalizedQuery = normalizeCatalogString(query);
  if (normalizedQuery.length === 0) {
    return true;
  }

  return strings.some((value) => {
    const normalizedValue = normalizeCatalogString(value);
    return (
      normalizedValue.includes(normalizedQuery) || fuzzyIncludes(normalizedValue, normalizedQuery)
    );
  });
}

function looksLikeUnresolvedObjectId(query: string): boolean {
  return /^\d+$/.test(query) || query.includes("_");
}

function memoryHitSubtitle(
  band: SearchableMemoryBand,
  typeLabel: string,
  id: string,
  detail: string,
): string {
  return `${band} · ${typeLabel} · ${shortId(id)} · ${detail}`;
}

function semanticNodeHit(node: SemanticMemoryNode): MemoryHit {
  return {
    id: node.id,
    band: "semantic",
    type: "semantic_node",
    title: node.label,
    subtitle: memoryHitSubtitle("semantic", "node", node.id, `${node.kind} · ${node.status}`),
  };
}

function semanticEdgeHit(edge: SemanticMemoryEdge): MemoryHit {
  return {
    id: edge.id,
    band: "semantic",
    type: "semantic_edge",
    title: `${shortId(edge.from_node_id)} → ${shortId(edge.to_node_id)}`,
    subtitle: memoryHitSubtitle("semantic", "edge", edge.id, String(edge.relation)),
  };
}

function proceduralHit(item: ProceduralMemoryItem): MemoryHit {
  return {
    id: item.id,
    band: "procedural",
    type: resolveObjectType(item.id) ?? "skill",
    title: item.applies_when,
    subtitle: memoryHitSubtitle("procedural", "skill", item.id, item.status),
  };
}

function hitsFromMemoryDetail(detail: MemoryBandDetail): MemoryHit[] {
  switch (detail.band) {
    case "episodic":
      return detail.items.map((item) => ({
        id: item.id,
        band: "episodic",
        type: "episode",
        title: item.title,
        subtitle: memoryHitSubtitle("episodic", "episode", item.id, item.audience ?? "global"),
      }));
    case "semantic":
      return [...detail.nodes.map(semanticNodeHit), ...detail.edges.map(semanticEdgeHit)];
    case "procedural":
      return detail.items.map(proceduralHit);
    default:
      return [];
  }
}

function activeOptionId(index: number): string {
  return `cmdp-option-${index}`;
}

export function CommandPalette({
  open,
  onOpenChange,
  setView,
  setSessionId,
  onOpenReset,
}: CommandPaletteProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const optionRefs = useRef<Array<HTMLDivElement | null>>([]);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const [memorySearch, setMemorySearch] = useState<MemorySearchState>({
    status: "idle",
    query: "",
    hits: [],
    error: null,
  });
  const { sessionsApi } = useLiveCache();
  const inspector = useInspector();
  const trimmedQuery = query.trim();
  const resolvedObjectType = trimmedQuery.length === 0 ? null : resolveObjectType(trimmedQuery);
  const unresolvedId =
    trimmedQuery.length > 0 &&
    resolvedObjectType === null &&
    looksLikeUnresolvedObjectId(trimmedQuery);
  const shouldSearchMemory =
    open && trimmedQuery.length > 0 && !unresolvedId && resolvedObjectType === null;

  useEffect(() => {
    if (!open) {
      return;
    }

    setQuery("");
    setActiveIndex(0);
    inputRef.current?.focus();
  }, [open]);

  useEffect(() => {
    if (!shouldSearchMemory) {
      setMemorySearch({ status: "idle", query: trimmedQuery, hits: [], error: null });
      return;
    }

    let cancelled = false;
    setMemorySearch({ status: "loading", query: trimmedQuery, hits: [], error: null });

    void Promise.all(
      SEARCHABLE_MEMORY_BANDS.map(async (band) =>
        getMemoryBand(band, { query: trimmedQuery, limit: MEMORY_SEARCH_LIMIT }),
      ),
    )
      .then((details) => {
        if (cancelled) {
          return;
        }
        setMemorySearch({
          status: "ready",
          query: trimmedQuery,
          hits: details.flatMap(hitsFromMemoryDetail),
          error: null,
        });
      })
      .catch((cause: unknown) => {
        if (cancelled) {
          return;
        }
        setMemorySearch({
          status: "error",
          query: trimmedQuery,
          hits: [],
          error: cause instanceof Error ? cause.message : "memory search failed",
        });
      });

    return () => {
      cancelled = true;
    };
  }, [shouldSearchMemory, trimmedQuery]);

  const results = useMemo<CommandResult[]>(() => {
    const catalogQuery = trimmedQuery;
    const screenResults: CommandResult[] = RAIL_ITEMS.filter((item) =>
      catalogMatches(catalogQuery, [
        `Go to ${item.title ?? item.label}`,
        item.title ?? item.label,
        item.label,
        item.short,
      ]),
    ).map((item) => ({
      id: `screen:${item.id}`,
      group: "Screens",
      title: `Go to ${item.title ?? item.label}`,
      subtitle: `${item.glyph} ${item.short}`,
      icon: item.glyph,
      hint: routeChordLabel(item),
      run: () => setView(item.id),
    }));

    const sessionResults: CommandResult[] = (sessionsApi.data?.sessions ?? [])
      .filter((session) =>
        catalogMatches(catalogQuery, [
          `Switch session ${session.label}`,
          session.label,
          session.audience_label,
        ]),
      )
      .map((session) => ({
        id: `session:${session.session_id}`,
        group: "Sessions",
        title: `Switch session: ${session.label}`,
        subtitle: previewLine(session),
        icon: "⇄",
        hint: shortId(session.session_id),
        run: () => setSessionId(session.session_id),
      }));

    const objectResults: CommandResult[] =
      resolvedObjectType === null
        ? unresolvedId
          ? [
              {
                id: `object-unresolved:${trimmedQuery}`,
                group: "Open object",
                title: "Object ID not resolvable",
                subtitle:
                  "specify a supported type prefix; numeric and non-sniffed ids need a typed opener",
                icon: "!",
                hint: "degraded",
                disabled: true,
              },
            ]
          : []
        : [
            {
              id: `object:${trimmedQuery}`,
              group: "Open object",
              title: `Open ${objectRegistry[resolvedObjectType].label} ${shortId(trimmedQuery)}`,
              subtitle: trimmedQuery,
              icon: "↗",
              hint: "inspect",
              run: () => inspector.openObject({ type: resolvedObjectType, id: trimmedQuery }),
            },
          ];

    const memoryResults: CommandResult[] =
      memorySearch.query === trimmedQuery
        ? [
            ...(memorySearch.status === "loading"
              ? [
                  {
                    id: `memory-loading:${trimmedQuery}`,
                    group: "Memory" as const,
                    title: "Searching memory",
                    subtitle: "episodic · semantic · procedural",
                    icon: "◇",
                    hint: "server",
                    disabled: true,
                  },
                ]
              : []),
            ...(memorySearch.status === "error"
              ? [
                  {
                    id: `memory-error:${trimmedQuery}`,
                    group: "Memory" as const,
                    title: "Memory search unavailable",
                    subtitle: memorySearch.error,
                    icon: "!",
                    hint: "degraded",
                    disabled: true,
                  },
                ]
              : []),
            ...memorySearch.hits.map((hit) => ({
              id: `memory:${hit.type}:${hit.id}`,
              group: "Memory" as const,
              title: `Open ${hit.title}`,
              subtitle: hit.subtitle,
              icon: "◇",
              hint: "inspect",
              run: () => inspector.openObject({ type: hit.type, id: hit.id }),
            })),
          ]
        : [];

    const actionCatalogItems: CommandResult[] = [
      {
        id: "action:create-commitment",
        group: "Actions",
        title: "Create commitment",
        subtitle: "open commitment flow",
        icon: "+",
        hint: "navigate",
        run: () => setView("governance", { governanceTab: "commitments" }),
      },
      {
        id: "action:run-dream-plan",
        group: "Actions",
        title: "Run dream plan",
        subtitle: "open dream plan flow",
        icon: "☾",
        hint: "navigate",
        run: () => setView("dream"),
      },
      {
        id: "action:open-assembled-prompt",
        group: "Actions",
        title: "Open assembled prompt",
        subtitle: "open prompt assembly screen",
        icon: "›",
        hint: "navigate",
        run: () => setView("prompts"),
      },
      {
        id: "action:reset-demo",
        group: "Actions",
        title: "Reset demo",
        subtitle: "open RESET confirmation",
        icon: "!",
        hint: "confirm",
        run: onOpenReset,
      },
    ];
    const actionCatalog = actionCatalogItems.filter((item) =>
      catalogMatches(catalogQuery, [item.title, item.subtitle ?? ""]),
    );

    return [
      ...screenResults,
      ...sessionResults,
      ...objectResults,
      ...memoryResults,
      ...actionCatalog,
    ];
  }, [
    inspector,
    memorySearch,
    onOpenReset,
    resolvedObjectType,
    sessionsApi.data?.sessions,
    setSessionId,
    setView,
    trimmedQuery,
    unresolvedId,
  ]);

  useEffect(() => {
    setActiveIndex(0);
  }, [query]);

  useEffect(() => {
    setActiveIndex((current) => {
      if (results.length === 0) {
        return 0;
      }
      return Math.min(current, results.length - 1);
    });
  }, [results.length]);

  useEffect(() => {
    const activeOption = optionRefs.current[activeIndex];
    if (typeof activeOption?.scrollIntoView === "function") {
      activeOption.scrollIntoView({ block: "nearest" });
    }
  }, [activeIndex, results.length]);

  function runResult(result: CommandResult | undefined): void {
    if (result === undefined || result.disabled || result.run === undefined) {
      return;
    }
    result.run();
    onOpenChange(false);
  }

  function containHandledKey(event: KeyboardEvent): void {
    event.preventDefault();
    event.stopPropagation();
    event.nativeEvent.stopImmediatePropagation();
  }

  function onKeyDown(event: KeyboardEvent): void {
    if (event.key === "Escape") {
      containHandledKey(event);
      onOpenChange(false);
      return;
    }

    if (event.key === "ArrowDown") {
      containHandledKey(event);
      setActiveIndex((current) => (results.length === 0 ? 0 : (current + 1) % results.length));
      return;
    }

    if (event.key === "ArrowUp") {
      containHandledKey(event);
      setActiveIndex((current) =>
        results.length === 0 ? 0 : (current - 1 + results.length) % results.length,
      );
      return;
    }

    if (event.key === "Enter") {
      containHandledKey(event);
      runResult(results[activeIndex]);
    }
  }

  if (!open) {
    return null;
  }

  let optionIndex = 0;
  const activeId = results.length === 0 ? undefined : activeOptionId(activeIndex);

  return (
    <div className="modal-backdrop cmdp-backdrop" onMouseDown={() => onOpenChange(false)}>
      <div
        className="modal-card cmdp-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="cmdp-title"
        onKeyDown={onKeyDown}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="modal-title cmdp-title" id="cmdp-title">
          command palette
        </div>
        <div className="cmdp-search">
          <input
            ref={inputRef}
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Type a command, session, id, or memory query"
            aria-label="Command palette search"
            aria-controls="cmdp-results"
            aria-activedescendant={activeId}
            autoComplete="off"
            spellCheck={false}
          />
        </div>
        <div className="cmdp-results" id="cmdp-results" role="listbox">
          {GROUP_ORDER.map((group) => {
            const groupResults = results.filter((result) => result.group === group);
            if (groupResults.length === 0) {
              return null;
            }

            return (
              <section className="cmdp-group" key={group} aria-label={group}>
                <div className="cmdp-group-label">{group}</div>
                {groupResults.map((result) => {
                  const index = optionIndex;
                  optionIndex += 1;
                  const active = index === activeIndex;
                  return (
                    <div
                      key={result.id}
                      ref={(element) => {
                        optionRefs.current[index] = element;
                      }}
                      id={activeOptionId(index)}
                      role="option"
                      aria-selected={active}
                      aria-disabled={result.disabled === true ? true : undefined}
                      className={`cmdp-row ${active ? "active" : ""} ${
                        result.disabled ? "disabled" : ""
                      }`}
                      onMouseEnter={() => setActiveIndex(index)}
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={() => runResult(result)}
                    >
                      <span className="cmdp-icon" aria-hidden="true">
                        {result.icon ?? "·"}
                      </span>
                      <span className="cmdp-copy">
                        <span className="cmdp-row-title">{result.title}</span>
                        {result.subtitle === undefined ? null : (
                          <span className="cmdp-row-subtitle">{result.subtitle}</span>
                        )}
                      </span>
                      {result.hint === undefined ? null : (
                        <span className="cmdp-hint">{result.hint}</span>
                      )}
                    </div>
                  );
                })}
              </section>
            );
          })}
          {results.length === 0 ? (
            <div className="cmdp-empty" role="status">
              no commands
            </div>
          ) : null}
        </div>
        <div className="modal-footer cmdp-footer">
          <span>
            <span className="kbd">↑</span>
            <span className="kbd">↓</span> move
          </span>
          <span>
            <span className="kbd">↵</span> run
          </span>
          <span>
            <span className="kbd">esc</span> close
          </span>
        </div>
      </div>
    </div>
  );
}
