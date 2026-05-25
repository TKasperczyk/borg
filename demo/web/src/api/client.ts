import type {
  LedgerResponse,
  SharedStateResponse,
  StateSnapshot,
  StreamChatKind,
  StreamResponse,
  TurnRequest,
  TurnResponse
} from "./types";

const DEFAULT_API_BASE = "http://localhost:7740";

export type ApiErrorPayload = {
  status: number;
  message: string;
};

export class ApiError extends Error {
  readonly status: number;
  readonly payload: ApiErrorPayload;

  constructor(payload: ApiErrorPayload) {
    super(payload.message);
    this.name = "ApiError";
    this.status = payload.status;
    this.payload = payload;
  }
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

export function apiBase(): string {
  const configured = import.meta.env.VITE_BORG_API_BASE;
  return trimTrailingSlash(configured === undefined || configured.length === 0 ? DEFAULT_API_BASE : configured);
}

export function wsBase(): string {
  const configured = import.meta.env.VITE_BORG_WS_BASE;
  if (configured !== undefined && configured.length > 0) {
    return trimTrailingSlash(configured);
  }

  const base = apiBase();
  if (base.startsWith("https://")) {
    return `wss://${base.slice("https://".length)}`;
  }
  if (base.startsWith("http://")) {
    return `ws://${base.slice("http://".length)}`;
  }
  return base;
}

function apiUrl(path: string, params?: URLSearchParams): string {
  const url = new URL(path, `${apiBase()}/`);
  if (params !== undefined) {
    url.search = params.toString();
  }
  return url.toString();
}

async function fetchJson<T>(path: string, init?: RequestInit, params?: URLSearchParams): Promise<T> {
  const response = await fetch(apiUrl(path, params), {
    ...init,
    headers: {
      ...(init?.body === undefined ? {} : { "Content-Type": "application/json" }),
      ...init?.headers
    }
  });

  if (!response.ok) {
    let message = response.statusText;
    try {
      const body = (await response.json()) as { error?: { message?: string } };
      message = body.error?.message ?? message;
    } catch {
      // Keep the status text when the body is not JSON.
    }
    throw new ApiError({ status: response.status, message });
  }

  return (await response.json()) as T;
}

export async function getState(): Promise<StateSnapshot> {
  return fetchJson<StateSnapshot>("api/state");
}

export async function getStream(input: {
  audience?: string;
  kinds?: readonly StreamChatKind[];
  limit?: number;
  before?: string;
}): Promise<StreamResponse> {
  const params = new URLSearchParams();
  if (input.audience !== undefined) {
    params.set("audience", input.audience);
  }
  if (input.kinds !== undefined && input.kinds.length > 0) {
    params.set("kind", input.kinds.join(","));
  }
  if (input.limit !== undefined) {
    params.set("limit", String(input.limit));
  }
  if (input.before !== undefined) {
    params.set("before", input.before);
  }

  return fetchJson<StreamResponse>("api/stream", undefined, params);
}

export async function getLedger(turnId: string): Promise<LedgerResponse> {
  return fetchJson<LedgerResponse>(`api/turns/${encodeURIComponent(turnId)}/ledger`);
}

export async function getSharedState(audience: string): Promise<SharedStateResponse> {
  const params = new URLSearchParams({ audience });
  return fetchJson<SharedStateResponse>("api/shared-state", undefined, params);
}

export async function postTurn(input: TurnRequest): Promise<TurnResponse> {
  return fetchJson<TurnResponse>("api/turn", {
    method: "POST",
    body: JSON.stringify(input)
  });
}

export function liveUrl(): string {
  return `${wsBase()}/api/live`;
}

export function attachmentBytesUrl(attachmentId: string, audience?: string): string {
  const params = audience === undefined ? undefined : new URLSearchParams({ audience });
  return apiUrl(`api/attachments/${encodeURIComponent(attachmentId)}/bytes`, params);
}
