import { subscribeClientFetchErrors } from "../api/client";

export const MAX_CLIENT_ERRORS = 50;

export type ClientErrorLogEntry =
  | {
      id: string;
      source: "api";
      ts: number;
      endpoint: string;
      status?: number;
      message: string;
    }
  | {
      id: string;
      source: "boundary";
      ts: number;
      boundarySource: string;
      message: string;
    };

export type ClientErrorLogListener = () => void;

const listeners = new Set<ClientErrorLogListener>();
let entries: ClientErrorLogEntry[] = [];
let nextEntryId = 0;

function nextId(): string {
  nextEntryId += 1;
  return `client_err_${nextEntryId}`;
}

function notify(): void {
  for (const listener of listeners) {
    try {
      listener();
    } catch {
      // Client log observers must not affect transport/render behavior.
    }
  }
}

export function recordClientError(
  input:
    | {
        source: "api";
        endpoint: string;
        status?: number;
        message: string;
        ts?: number;
      }
    | {
        source: "boundary";
        boundarySource: string;
        message: string;
        ts?: number;
      },
): ClientErrorLogEntry {
  const entry =
    input.source === "api"
      ? {
          id: nextId(),
          source: "api" as const,
          ts: input.ts ?? Date.now(),
          endpoint: input.endpoint,
          status: input.status,
          message: input.message,
        }
      : {
          id: nextId(),
          source: "boundary" as const,
          ts: input.ts ?? Date.now(),
          boundarySource: input.boundarySource,
          message: input.message,
        };

  // In-memory only: this is a bounded browser-session transport/render failure log.
  entries = [entry, ...entries].slice(0, MAX_CLIENT_ERRORS);
  notify();
  return entry;
}

export function getClientErrors(): readonly ClientErrorLogEntry[] {
  return entries;
}

export function clearClientErrors(): void {
  entries = [];
  notify();
}

export function subscribeClientErrorLog(listener: ClientErrorLogListener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

subscribeClientFetchErrors((event) => {
  recordClientError({
    source: "api",
    endpoint: event.endpoint,
    status: event.status,
    message: event.message,
  });
});
