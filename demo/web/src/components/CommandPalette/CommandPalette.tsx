import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";

import { ApiError, getDreamAudit, getMemoryBand, getReviews } from "../../api/client";
import type {
  MaintenanceAuditRow,
  MemoryBandDetail,
  MemoryBandId,
  ProceduralMemoryItem,
  ReviewRow,
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
import { isInternalId, shortId } from "../../screens/screen-utils";
import { useInspector } from "../Inspector/inspector-context";
import {
  resolveObjectType,
  type ObjectType,
  type PrefixedObjectType,
} from "../Inspector/inspector-id";
import { boundedAllSettled, objectRegistry } from "../Inspector/inspector-registry";

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
  | {
      status: "idle";
      query: string;
      hits: MemoryHit[];
      error: null;
      failures: MemorySearchFailure[];
    }
  | {
      status: "loading";
      query: string;
      hits: MemoryHit[];
      error: null;
      failures: MemorySearchFailure[];
    }
  | {
      status: "ready";
      query: string;
      hits: MemoryHit[];
      error: null;
      failures: MemorySearchFailure[];
    }
  | {
      status: "error";
      query: string;
      hits: MemoryHit[];
      error: string;
      failures: MemorySearchFailure[];
    };

type MemorySearchFailure = {
  band: SearchableMemoryBand;
  message: string;
};

type ObjectOpenCandidate = {
  type: ObjectType;
  id: string;
  subtitle?: string;
};

type ObjectLookupState =
  | { status: "idle"; query: string; type: null; error: null }
  | { status: "checking"; query: string; type: ObjectType; error: null }
  | { status: "found"; query: string; type: ObjectType; id: string; error: null }
  | { status: "missing"; query: string; type: ObjectType; error: null }
  | { status: "error"; query: string; type: ObjectType; error: string };

type NumericLookupState =
  | { status: "idle"; query: string; hits: ObjectOpenCandidate[]; failures: string[] }
  | { status: "loading"; query: string; hits: ObjectOpenCandidate[]; failures: string[] }
  | { status: "ready"; query: string; hits: ObjectOpenCandidate[]; failures: string[] };

type NumericLookupCache = {
  reviews: ReviewRow[] | null;
  dreamRows: MaintenanceAuditRow[] | null;
  failures: string[];
};

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
const MEMORY_SEARCH_DEBOUNCE_MS = 200;

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

function isNumericQuery(query: string): boolean {
  return /^\d+$/.test(query);
}

function containsDisplayEllipsis(query: string): boolean {
  return query.includes("…");
}

function isFullShapedPrefixedId(query: string, type: PrefixedObjectType): boolean {
  return resolveObjectType(query) === type && isInternalId(query);
}

function failureMessage(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

function cacheKey(candidate: Pick<ObjectOpenCandidate, "type" | "id">): string {
  return `${candidate.type}:${candidate.id}`;
}

function mergeObjectCandidates(
  current: readonly ObjectOpenCandidate[],
  nextCandidates: readonly ObjectOpenCandidate[],
): ObjectOpenCandidate[] {
  const byKey = new Map(current.map((candidate) => [cacheKey(candidate), candidate]));
  for (const candidate of nextCandidates) {
    byKey.set(cacheKey(candidate), candidate);
  }
  return [...byKey.values()];
}

function cachedCandidateMatches(query: string, candidate: ObjectOpenCandidate): boolean {
  if (containsDisplayEllipsis(query)) {
    const displayId = shortId(candidate.id);
    return query === displayId || query.includes(displayId);
  }
  return candidate.id === query || candidate.id.startsWith(query);
}

function memoryHitCandidate(hit: MemoryHit): ObjectOpenCandidate {
  return {
    type: hit.type,
    id: hit.id,
    subtitle: hit.subtitle,
  };
}

function reviewCandidate(row: ReviewRow): ObjectOpenCandidate {
  return {
    type: "review",
    id: String(row.id),
    subtitle: `${row.kind} · ${row.resolved_at === null ? "open" : "resolved"}`,
  };
}

function dreamAuditCandidate(row: MaintenanceAuditRow): ObjectOpenCandidate {
  return {
    type: "dream_audit",
    id: String(row.id),
    subtitle: `${row.process} · ${row.action}`,
  };
}

function candidatesFromNumericCache(
  query: string,
  cache: NumericLookupCache,
): ObjectOpenCandidate[] {
  const hits: ObjectOpenCandidate[] = [];
  const review = cache.reviews?.find((row) => String(row.id) === query);
  if (review !== undefined) {
    hits.push(reviewCandidate(review));
  }

  const numericId = Number(query);
  const dreamRow = Number.isFinite(numericId)
    ? cache.dreamRows?.find((row) => row.id === numericId)
    : undefined;
  if (dreamRow !== undefined) {
    hits.push(dreamAuditCandidate(dreamRow));
  }

  return hits;
}

function objectResult(candidate: ObjectOpenCandidate, inspector: ReturnType<typeof useInspector>) {
  const model = objectRegistry[candidate.type];
  return {
    id: `object:${candidate.type}:${candidate.id}`,
    group: "Open object" as const,
    title: `Open ${model.label} ${shortId(candidate.id)}`,
    subtitle: candidate.subtitle ?? candidate.id,
    icon: "↗",
    hint: "inspect",
    run: () => inspector.openObject({ type: candidate.type, id: candidate.id }),
  };
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
  const displayLabel = node.display_label?.trim() || null;
  return {
    id: node.id,
    band: "semantic",
    type: "semantic_node",
    title: displayLabel ?? `${node.kind} memory`,
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
  const memorySearchTimerRef = useRef<number | null>(null);
  const objectLookupTimerRef = useRef<number | null>(null);
  const numericLookupTimerRef = useRef<number | null>(null);
  const memorySearchSeqRef = useRef(0);
  const objectLookupSeqRef = useRef(0);
  const numericLookupSeqRef = useRef(0);
  const numericRowsCacheRef = useRef<NumericLookupCache | null>(null);
  const numericRowsLoadRef = useRef<Promise<NumericLookupCache> | null>(null);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const [memorySearch, setMemorySearch] = useState<MemorySearchState>({
    status: "idle",
    query: "",
    hits: [],
    error: null,
    failures: [],
  });
  const [objectLookup, setObjectLookup] = useState<ObjectLookupState>({
    status: "idle",
    query: "",
    type: null,
    error: null,
  });
  const [numericLookup, setNumericLookup] = useState<NumericLookupState>({
    status: "idle",
    query: "",
    hits: [],
    failures: [],
  });
  const [cachedObjectCandidates, setCachedObjectCandidates] = useState<ObjectOpenCandidate[]>([]);
  const { sessionsApi } = useLiveCache();
  const inspector = useInspector();
  const trimmedQuery = query.trim();
  const numericQuery = isNumericQuery(trimmedQuery);
  const ellipsizedQuery = containsDisplayEllipsis(trimmedQuery);
  const resolvedObjectType = trimmedQuery.length === 0 ? null : resolveObjectType(trimmedQuery);
  const fullResolvedObjectType =
    resolvedObjectType === null || !isFullShapedPrefixedId(trimmedQuery, resolvedObjectType)
      ? null
      : resolvedObjectType;
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
    numericRowsCacheRef.current = null;
    numericRowsLoadRef.current = null;
    inputRef.current?.focus();
  }, [open]);

  const cacheObjectCandidates = useCallback((candidates: readonly ObjectOpenCandidate[]) => {
    if (candidates.length === 0) {
      return;
    }
    setCachedObjectCandidates((current) => mergeObjectCandidates(current, candidates));
  }, []);

  const runMemorySearch = useCallback(
    async (searchQuery: string): Promise<void> => {
      const requestSeq = memorySearchSeqRef.current + 1;
      memorySearchSeqRef.current = requestSeq;
      setMemorySearch({
        status: "loading",
        query: searchQuery,
        hits: [],
        error: null,
        failures: [],
      });

      const results = await boundedAllSettled(SEARCHABLE_MEMORY_BANDS, async (band) =>
        getMemoryBand(band, { query: searchQuery, limit: MEMORY_SEARCH_LIMIT }),
      );

      if (memorySearchSeqRef.current !== requestSeq) {
        return;
      }

      const hits: MemoryHit[] = [];
      const failures: MemorySearchFailure[] = [];

      results.forEach((result, index) => {
        const band = SEARCHABLE_MEMORY_BANDS[index] as SearchableMemoryBand;
        if (result.status === "fulfilled") {
          hits.push(...hitsFromMemoryDetail(result.value));
        } else {
          failures.push({ band, message: failureMessage(result.reason) });
        }
      });

      cacheObjectCandidates(hits.map(memoryHitCandidate));

      if (failures.length === SEARCHABLE_MEMORY_BANDS.length) {
        setMemorySearch({
          status: "error",
          query: searchQuery,
          hits: [],
          error: failures.map((failure) => `${failure.band}: ${failure.message}`).join(" · "),
          failures,
        });
        return;
      }

      setMemorySearch({
        status: "ready",
        query: searchQuery,
        hits,
        error: null,
        failures,
      });
    },
    [cacheObjectCandidates],
  );

  const flushMemorySearch = useCallback(() => {
    if (!shouldSearchMemory) {
      return;
    }

    if (memorySearchTimerRef.current !== null) {
      window.clearTimeout(memorySearchTimerRef.current);
      memorySearchTimerRef.current = null;
    }

    void runMemorySearch(trimmedQuery);
  }, [runMemorySearch, shouldSearchMemory, trimmedQuery]);

  const runObjectLookup = useCallback(
    async (
      lookupQuery: string,
      lookupType: PrefixedObjectType,
    ): Promise<ObjectOpenCandidate | null> => {
      const requestSeq = objectLookupSeqRef.current + 1;
      objectLookupSeqRef.current = requestSeq;
      setObjectLookup({
        status: "checking",
        query: lookupQuery,
        type: lookupType,
        error: null,
      });

      try {
        const object = await objectRegistry[lookupType].fetch(lookupQuery, {
          sessionId: inspector.sessionId,
          audience: inspector.audience,
        });
        if (objectLookupSeqRef.current !== requestSeq) {
          return null;
        }

        if (object === null) {
          setObjectLookup({
            status: "missing",
            query: lookupQuery,
            type: lookupType,
            error: null,
          });
          return null;
        }

        const candidate = { type: lookupType, id: lookupQuery };
        cacheObjectCandidates([candidate]);
        setObjectLookup({
          status: "found",
          query: lookupQuery,
          type: lookupType,
          id: lookupQuery,
          error: null,
        });
        return candidate;
      } catch (caught) {
        if (objectLookupSeqRef.current !== requestSeq) {
          return null;
        }

        if (caught instanceof ApiError && caught.status === 404) {
          setObjectLookup({
            status: "missing",
            query: lookupQuery,
            type: lookupType,
            error: null,
          });
          return null;
        }

        setObjectLookup({
          status: "error",
          query: lookupQuery,
          type: lookupType,
          error: failureMessage(caught),
        });
        return null;
      }
    },
    [cacheObjectCandidates, inspector.audience, inspector.sessionId],
  );

  const loadNumericRows = useCallback(async (): Promise<NumericLookupCache> => {
    if (numericRowsCacheRef.current !== null) {
      return numericRowsCacheRef.current;
    }
    if (numericRowsLoadRef.current !== null) {
      return numericRowsLoadRef.current;
    }

    numericRowsLoadRef.current = Promise.allSettled([
      getReviews({ openOnly: false }),
      getDreamAudit(100),
    ]).then(([reviewsResult, dreamResult]) => {
      const failures: string[] = [];
      const cache: NumericLookupCache = {
        reviews: null,
        dreamRows: null,
        failures,
      };

      if (reviewsResult.status === "fulfilled") {
        cache.reviews = reviewsResult.value.rows;
      } else {
        failures.push(`reviews: ${failureMessage(reviewsResult.reason)}`);
      }

      if (dreamResult.status === "fulfilled") {
        cache.dreamRows = dreamResult.value.rows;
      } else {
        failures.push(`dream: ${failureMessage(dreamResult.reason)}`);
      }

      numericRowsCacheRef.current = cache;
      numericRowsLoadRef.current = null;
      return cache;
    });

    return numericRowsLoadRef.current;
  }, []);

  const runNumericLookup = useCallback(
    async (searchQuery: string): Promise<void> => {
      const requestSeq = numericLookupSeqRef.current + 1;
      numericLookupSeqRef.current = requestSeq;
      setNumericLookup({ status: "loading", query: searchQuery, hits: [], failures: [] });

      const cache = await loadNumericRows();
      if (numericLookupSeqRef.current !== requestSeq) {
        return;
      }

      const hits = candidatesFromNumericCache(searchQuery, cache);
      cacheObjectCandidates(hits);
      setNumericLookup({ status: "ready", query: searchQuery, hits, failures: cache.failures });
    },
    [cacheObjectCandidates, loadNumericRows],
  );

  useEffect(() => {
    if (!shouldSearchMemory) {
      if (memorySearchTimerRef.current !== null) {
        window.clearTimeout(memorySearchTimerRef.current);
        memorySearchTimerRef.current = null;
      }
      memorySearchSeqRef.current += 1;
      setMemorySearch({
        status: "idle",
        query: trimmedQuery,
        hits: [],
        error: null,
        failures: [],
      });
      return;
    }

    if (memorySearchTimerRef.current !== null) {
      window.clearTimeout(memorySearchTimerRef.current);
    }

    memorySearchTimerRef.current = window.setTimeout(() => {
      memorySearchTimerRef.current = null;
      void runMemorySearch(trimmedQuery);
    }, MEMORY_SEARCH_DEBOUNCE_MS);

    return () => {
      if (memorySearchTimerRef.current !== null) {
        window.clearTimeout(memorySearchTimerRef.current);
        memorySearchTimerRef.current = null;
      }
    };
  }, [runMemorySearch, shouldSearchMemory, trimmedQuery]);

  useEffect(() => {
    if (objectLookupTimerRef.current !== null) {
      window.clearTimeout(objectLookupTimerRef.current);
      objectLookupTimerRef.current = null;
    }

    if (!open || fullResolvedObjectType === null || trimmedQuery.length === 0) {
      objectLookupSeqRef.current += 1;
      setObjectLookup({ status: "idle", query: trimmedQuery, type: null, error: null });
      return;
    }

    objectLookupSeqRef.current += 1;
    setObjectLookup({
      status: "checking",
      query: trimmedQuery,
      type: fullResolvedObjectType,
      error: null,
    });

    objectLookupTimerRef.current = window.setTimeout(() => {
      objectLookupTimerRef.current = null;
      void runObjectLookup(trimmedQuery, fullResolvedObjectType);
    }, MEMORY_SEARCH_DEBOUNCE_MS);

    return () => {
      if (objectLookupTimerRef.current !== null) {
        window.clearTimeout(objectLookupTimerRef.current);
        objectLookupTimerRef.current = null;
      }
    };
  }, [fullResolvedObjectType, open, runObjectLookup, trimmedQuery]);

  useEffect(() => {
    if (!open || !numericQuery) {
      if (numericLookupTimerRef.current !== null) {
        window.clearTimeout(numericLookupTimerRef.current);
        numericLookupTimerRef.current = null;
      }
      numericLookupSeqRef.current += 1;
      setNumericLookup({ status: "idle", query: trimmedQuery, hits: [], failures: [] });
      return;
    }

    if (numericLookupTimerRef.current !== null) {
      window.clearTimeout(numericLookupTimerRef.current);
    }

    numericLookupSeqRef.current += 1;
    setNumericLookup({ status: "loading", query: trimmedQuery, hits: [], failures: [] });

    numericLookupTimerRef.current = window.setTimeout(() => {
      numericLookupTimerRef.current = null;
      void runNumericLookup(trimmedQuery);
    }, MEMORY_SEARCH_DEBOUNCE_MS);

    return () => {
      if (numericLookupTimerRef.current !== null) {
        window.clearTimeout(numericLookupTimerRef.current);
        numericLookupTimerRef.current = null;
      }
    };
  }, [numericQuery, open, runNumericLookup, trimmedQuery]);

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

    const sessionObjectCandidates: ObjectOpenCandidate[] = (sessionsApi.data?.sessions ?? []).map(
      (session) => ({
        type: "session",
        id: session.session_id,
        subtitle: previewLine(session),
      }),
    );
    const currentMemoryCandidates =
      memorySearch.query === trimmedQuery ? memorySearch.hits.map(memoryHitCandidate) : [];
    const availableCandidates = mergeObjectCandidates(
      [...sessionObjectCandidates, ...cachedObjectCandidates],
      currentMemoryCandidates,
    );
    const matchedCachedCandidates =
      trimmedQuery.length === 0
        ? []
        : availableCandidates.filter((candidate) =>
            cachedCandidateMatches(trimmedQuery, candidate),
          );
    const verifiedExactCandidates =
      objectLookup.query === trimmedQuery && objectLookup.status === "found"
        ? [{ type: objectLookup.type, id: objectLookup.id }]
        : [];
    const numericCandidates =
      numericLookup.query === trimmedQuery && numericLookup.status === "ready"
        ? numericLookup.hits
        : [];
    const objectCandidates = mergeObjectCandidates(
      mergeObjectCandidates(matchedCachedCandidates, verifiedExactCandidates),
      numericCandidates,
    );
    const objectResults: CommandResult[] = [
      ...objectCandidates.map((candidate) => objectResult(candidate, inspector)),
      ...(objectLookup.query === trimmedQuery &&
      objectLookup.status === "checking" &&
      objectCandidates.length === 0
        ? [
            {
              id: `object-checking:${trimmedQuery}`,
              group: "Open object" as const,
              title: "Checking object id",
              subtitle: trimmedQuery,
              icon: "↗",
              hint: "lookup",
              disabled: true,
            },
          ]
        : []),
      ...(objectLookup.query === trimmedQuery && objectLookup.status === "error"
        ? [
            {
              id: `object-error:${trimmedQuery}`,
              group: "Open object" as const,
              title: "Object lookup unavailable",
              subtitle: objectLookup.error,
              icon: "!",
              hint: "degraded",
              disabled: true,
            },
          ]
        : []),
      ...(numericLookup.query === trimmedQuery &&
      numericLookup.status === "loading" &&
      objectCandidates.length === 0
        ? [
            {
              id: `numeric-loading:${trimmedQuery}`,
              group: "Open object" as const,
              title: "Checking review and dream rows",
              subtitle: trimmedQuery,
              icon: "↗",
              hint: "lookup",
              disabled: true,
            },
          ]
        : []),
      ...(numericLookup.query === trimmedQuery &&
      numericLookup.status === "ready" &&
      numericLookup.failures.length > 0
        ? [
            {
              id: `numeric-partial:${trimmedQuery}`,
              group: "Open object" as const,
              title: "Partial numeric lookup",
              subtitle: numericLookup.failures.join(" · "),
              icon: "!",
              hint: "degraded",
              disabled: true,
            },
          ]
        : []),
      ...(ellipsizedQuery && objectCandidates.length === 0
        ? [
            {
              id: `object-ellipsis:${trimmedQuery}`,
              group: "Open object" as const,
              title: "Paste the full id",
              subtitle: "shortened display ids only resolve from loaded results",
              icon: "!",
              hint: "full id",
              disabled: true,
            },
          ]
        : []),
      ...(resolvedObjectType === null && unresolvedId && !numericQuery && !ellipsizedQuery
        ? [
            {
              id: `object-unresolved:${trimmedQuery}`,
              group: "Open object" as const,
              title: "Object ID not resolvable",
              subtitle: "specify a supported type prefix or use an existing loaded row",
              icon: "!",
              hint: "degraded",
              disabled: true,
            },
          ]
        : []),
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
            ...(memorySearch.failures.length > 0 && memorySearch.status !== "error"
              ? [
                  {
                    id: `memory-partial:${trimmedQuery}`,
                    group: "Memory" as const,
                    title: "Partial memory results",
                    subtitle: memorySearch.failures
                      .map((failure) => `${failure.band}: ${failure.message}`)
                      .join(" · "),
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
    cachedObjectCandidates,
    ellipsizedQuery,
    inspector,
    memorySearch,
    numericLookup,
    numericQuery,
    objectLookup,
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

  async function runEnter(): Promise<void> {
    flushMemorySearch();

    if (fullResolvedObjectType !== null) {
      if (objectLookupTimerRef.current !== null) {
        window.clearTimeout(objectLookupTimerRef.current);
        objectLookupTimerRef.current = null;
      }

      const existingCandidate =
        objectLookup.query === trimmedQuery && objectLookup.status === "found"
          ? { type: objectLookup.type, id: objectLookup.id }
          : null;
      const candidate =
        existingCandidate ?? (await runObjectLookup(trimmedQuery, fullResolvedObjectType));

      if (candidate !== null) {
        inspector.openObject({ type: candidate.type, id: candidate.id });
        onOpenChange(false);
      }
      return;
    }

    runResult(results[activeIndex]);
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
      void runEnter();
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
