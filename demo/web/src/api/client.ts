import type {
  ApiState,
  LedgerResponse,
  SessionRecord,
  SessionsResponse,
  StreamResponse,
  TurnPostResponse,
  TurnsResponse,
} from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function responseMessage(response: Response): Promise<string> {
  const fallback = response.statusText || `HTTP ${response.status}`;
  try {
    const body = (await response.json()) as unknown;
    if (
      typeof body === "object" &&
      body !== null &&
      "message" in body &&
      typeof body.message === "string"
    ) {
      return body.message;
    }
  } catch {
    return fallback;
  }

  return fallback;
}

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, {
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new ApiError(response.status, await responseMessage(response));
  }

  return (await response.json()) as T;
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new ApiError(response.status, await responseMessage(response));
  }

  return (await response.json()) as T;
}

export function fetchState(session?: string): Promise<ApiState> {
  const params = new URLSearchParams();
  if (session !== undefined) {
    params.set("session", session);
  }

  const suffix = params.size === 0 ? "" : `?${params.toString()}`;
  return getJson<ApiState>(`/api/state${suffix}`);
}

export function fetchSessions(): Promise<SessionsResponse> {
  return getJson<SessionsResponse>("/api/sessions");
}

export function ensureOperatorSession(): Promise<SessionRecord> {
  return postJson<SessionRecord>("/api/sessions/operator", {});
}

export function fetchStream(session: string, limit = 100): Promise<StreamResponse> {
  const params = new URLSearchParams({ session, limit: String(limit) });
  return getJson<StreamResponse>(`/api/stream?${params.toString()}`);
}

export function fetchTurns(session: string, limit = 100): Promise<TurnsResponse> {
  const params = new URLSearchParams({ session, limit: String(limit) });
  return getJson<TurnsResponse>(`/api/turns?${params.toString()}`);
}

export function fetchLedger(turnId: string): Promise<LedgerResponse> {
  return getJson<LedgerResponse>(`/api/turns/${encodeURIComponent(turnId)}/ledger`);
}

export function postTurn(input: {
  message: string;
  external_message_id: string;
  session: string;
}): Promise<TurnPostResponse> {
  return postJson<TurnPostResponse>("/api/turn", input);
}
