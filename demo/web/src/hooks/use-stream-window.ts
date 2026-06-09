import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getStream } from "../api/client";
import type { StreamEntry, StreamEntryKind } from "../api/types";
import { mergeStreamEntriesForTurnGrouping, streamEntryAttachmentId } from "../lib/stream-grouping";
import { useLiveEventsContext } from "./live-context";

const DEFAULT_STREAM_WINDOW_LIMIT = 120;

export type UseStreamWindowInput = {
  sessionId: string;
  kinds?: readonly StreamEntryKind[];
  audience?: string;
  limit?: number;
  onAttachmentStatusesInvalidated?: (attachmentIds: readonly string[]) => void;
};

export type UseStreamWindowResult = {
  entries: StreamEntry[];
  loading: boolean;
  loadingOlder: boolean;
  error: Error | null;
  nextCursor: string | null;
  refetch: () => Promise<void>;
  loadOlder: () => Promise<void>;
};

function attachmentStatusInvalidationIds(entries: readonly StreamEntry[]): string[] {
  return [
    ...new Set(
      entries.flatMap((entry) => {
        if (entry.kind !== "user_image_attachment" && entry.kind !== "internal_event") {
          return [];
        }

        const id = streamEntryAttachmentId(entry);
        return id === undefined ? [] : [id];
      }),
    ),
  ];
}

function serverFilterMatches(
  entry: StreamEntry,
  input: Pick<UseStreamWindowInput, "sessionId" | "kinds" | "audience">,
): boolean {
  if (entry.session_id !== input.sessionId) {
    return false;
  }
  if (input.kinds !== undefined && !input.kinds.includes(entry.kind)) {
    return false;
  }
  if (input.audience !== undefined && entry.audience !== input.audience) {
    return false;
  }
  return true;
}

export function useStreamWindow({
  sessionId,
  kinds,
  audience,
  limit = DEFAULT_STREAM_WINDOW_LIMIT,
  onAttachmentStatusesInvalidated,
}: UseStreamWindowInput): UseStreamWindowResult {
  const live = useLiveEventsContext();
  const previousConnectionCountRef = useRef(live.connectionCount);
  const requestSeqRef = useRef(0);
  const olderRequestSeqRef = useRef(0);
  const mountedRef = useRef(true);
  const loadingOlderRef = useRef(false);
  const [entries, setEntries] = useState<StreamEntry[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const kindsKey = kinds?.join(",") ?? "";
  const normalizedKinds = useMemo(
    () => (kindsKey.length === 0 ? undefined : (kindsKey.split(",") as StreamEntryKind[])),
    [kindsKey],
  );
  const filterKey = `${sessionId}\n${kindsKey}\n${audience ?? ""}\n${limit}`;
  const filterKeyRef = useRef(filterKey);
  filterKeyRef.current = filterKey;

  useEffect(
    () => () => {
      mountedRef.current = false;
    },
    [],
  );

  const fetchTop = useCallback(async () => {
    const requestSeq = requestSeqRef.current + 1;
    const requestFilterKey = filterKey;
    requestSeqRef.current = requestSeq;
    setLoading(true);
    setError(null);

    try {
      const response = await getStream({
        session: sessionId,
        kinds: normalizedKinds,
        audience,
        limit,
      });
      if (
        !mountedRef.current ||
        requestSeqRef.current !== requestSeq ||
        filterKeyRef.current !== requestFilterKey
      ) {
        return;
      }
      const matchingEntries = response.entries.filter((entry) =>
        serverFilterMatches(entry, { sessionId, kinds: normalizedKinds, audience }),
      );
      setEntries(mergeStreamEntriesForTurnGrouping([], matchingEntries));
      setNextCursor(response.next_cursor);
    } catch (caught) {
      if (
        mountedRef.current &&
        requestSeqRef.current === requestSeq &&
        filterKeyRef.current === requestFilterKey
      ) {
        setError(caught instanceof Error ? caught : new Error(String(caught)));
      }
    } finally {
      if (
        mountedRef.current &&
        requestSeqRef.current === requestSeq &&
        filterKeyRef.current === requestFilterKey
      ) {
        setLoading(false);
      }
    }
  }, [audience, filterKey, limit, normalizedKinds, sessionId]);

  useEffect(() => {
    olderRequestSeqRef.current += 1;
    loadingOlderRef.current = false;
    setLoadingOlder(false);
    setEntries([]);
    setNextCursor(null);
    setError(null);
    void fetchTop();
  }, [fetchTop]);

  useEffect(() => {
    return live.subscribe((frame) => {
      if (frame.type !== "stream:append") {
        return;
      }
      const sessionEntries = frame.entries.filter((entry) => entry.session_id === sessionId);
      if (sessionEntries.length === 0) {
        return;
      }

      const invalidatedAttachmentIds = attachmentStatusInvalidationIds(sessionEntries);
      if (invalidatedAttachmentIds.length > 0) {
        onAttachmentStatusesInvalidated?.(invalidatedAttachmentIds);
      }

      const matchingEntries = sessionEntries.filter((entry) =>
        serverFilterMatches(entry, { sessionId, kinds: normalizedKinds, audience }),
      );
      if (matchingEntries.length === 0) {
        return;
      }

      setEntries((current) => mergeStreamEntriesForTurnGrouping(current, matchingEntries));
    });
  }, [audience, live, normalizedKinds, onAttachmentStatusesInvalidated, sessionId]);

  useEffect(() => {
    const previousConnectionCount = previousConnectionCountRef.current;
    previousConnectionCountRef.current = live.connectionCount;

    if (live.connectionCount <= 1 || live.connectionCount === previousConnectionCount) {
      return;
    }

    void fetchTop();
  }, [fetchTop, live.connectionCount]);

  const loadOlder = useCallback(async () => {
    if (nextCursor === null || loadingOlderRef.current) {
      return;
    }

    const requestSeq = requestSeqRef.current;
    const olderRequestSeq = olderRequestSeqRef.current + 1;
    const requestFilterKey = filterKey;
    olderRequestSeqRef.current = olderRequestSeq;
    loadingOlderRef.current = true;
    setLoadingOlder(true);
    setError(null);

    try {
      const response = await getStream({
        session: sessionId,
        kinds: normalizedKinds,
        audience,
        limit,
        before: nextCursor,
      });
      if (
        !mountedRef.current ||
        requestSeqRef.current !== requestSeq ||
        olderRequestSeqRef.current !== olderRequestSeq ||
        filterKeyRef.current !== requestFilterKey
      ) {
        return;
      }
      const matchingEntries = response.entries.filter((entry) =>
        serverFilterMatches(entry, { sessionId, kinds: normalizedKinds, audience }),
      );
      setEntries((current) => mergeStreamEntriesForTurnGrouping(current, matchingEntries));
      setNextCursor(response.next_cursor);
    } catch (caught) {
      if (
        mountedRef.current &&
        requestSeqRef.current === requestSeq &&
        olderRequestSeqRef.current === olderRequestSeq &&
        filterKeyRef.current === requestFilterKey
      ) {
        setError(caught instanceof Error ? caught : new Error(String(caught)));
      }
    } finally {
      if (olderRequestSeqRef.current === olderRequestSeq) {
        loadingOlderRef.current = false;
      }
      if (mountedRef.current && olderRequestSeqRef.current === olderRequestSeq) {
        setLoadingOlder(false);
      }
    }
  }, [audience, filterKey, limit, nextCursor, normalizedKinds, sessionId]);

  return {
    entries,
    loading,
    loadingOlder,
    error,
    nextCursor,
    refetch: fetchTop,
    loadOlder,
  };
}
