import type {
  AttachmentMetadataResponse,
  AttachmentStatusItem,
  CommitmentEnforcement,
  CommitmentState,
  CommitmentsResponse,
  CreateGoalRequest,
  CreateGrowthMarkerRequest,
  CreateValueRequest,
  DreamApplyRequest,
  DreamApplyResponse,
  DreamAuditResponse,
  DreamPlanRequest,
  DreamPlanResponse,
  DreamStateResponse,
  GrowthMarker,
  IdentityGoal,
  IdentityResponse,
  IdentityValue,
  LedgerResponse,
  MemoryBandDetail,
  MemoryBandId,
  OpenQuestion,
  PatchGoalRequest,
  PatchOpenQuestionRequest,
  PatchReviewItemRequest,
  ReviewRow,
  MemoryBandsResponse,
  SemanticGraphResponse,
  SharedStateResponse,
  StateSnapshot,
  StreamEntryKind,
  StreamResponse,
  TurnRequest,
  TurnResponse,
} from "./types";

const DEFAULT_API_BASE = "";

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
  return trimTrailingSlash(
    configured === undefined || configured.length === 0 ? DEFAULT_API_BASE : configured,
  );
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

  // Same-origin: derive from the current page so WebSocket gets an absolute URL.
  if (base === "" && typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}`;
  }
  return base;
}

function apiUrl(path: string, params?: URLSearchParams): string {
  const base = apiBase();
  const origin =
    base.length > 0
      ? base
      : typeof window === "undefined"
        ? "http://localhost"
        : window.location.origin;
  const url = new URL(path, `${origin}/`);
  if (params !== undefined) {
    url.search = params.toString();
  }
  // When apiBase is empty, callers (and the test fixture) want the same-origin
  // relative URL the browser actually used, not the resolved absolute one.
  return base.length > 0 ? url.toString() : `${url.pathname}${url.search}${url.hash}`;
}

function isFormDataBody(body: BodyInit | null | undefined): body is FormData {
  return typeof FormData !== "undefined" && body instanceof FormData;
}

async function fetchJson<T>(
  path: string,
  init?: RequestInit,
  params?: URLSearchParams,
): Promise<T> {
  const response = await fetch(apiUrl(path, params), {
    ...init,
    headers: {
      ...(init?.body === undefined || isFormDataBody(init.body)
        ? {}
        : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
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
  kinds?: readonly StreamEntryKind[];
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

export async function getMemoryBands(): Promise<MemoryBandsResponse> {
  return fetchJson<MemoryBandsResponse>("api/memory/bands");
}

export async function getMemoryBand(id: MemoryBandId): Promise<MemoryBandDetail> {
  return fetchJson<MemoryBandDetail>(`api/memory/bands/${encodeURIComponent(id)}`);
}

export async function getSemanticGraph(limit = 300): Promise<SemanticGraphResponse> {
  return fetchJson<SemanticGraphResponse>(
    "api/semantic/graph",
    undefined,
    new URLSearchParams({ limit: String(limit) }),
  );
}

export async function getIdentity(): Promise<IdentityResponse> {
  return fetchJson<IdentityResponse>("api/identity");
}

export async function getCommitments(
  input: {
    state?: CommitmentState | "all";
    enforcement?: CommitmentEnforcement | "all";
    audience?: string;
  } = {},
): Promise<CommitmentsResponse> {
  const params = new URLSearchParams();
  if (input.state !== undefined) {
    params.set("state", input.state);
  }
  if (input.enforcement !== undefined && input.enforcement !== "all") {
    params.set("enforcement", input.enforcement);
  }
  if (input.audience !== undefined && input.audience.length > 0) {
    params.set("audience", input.audience);
  }

  return fetchJson<CommitmentsResponse>("api/commitments", undefined, params);
}

export async function getDreamAudit(limit = 50): Promise<DreamAuditResponse> {
  return fetchJson<DreamAuditResponse>(
    "api/dream/audit",
    undefined,
    new URLSearchParams({ limit: String(limit) }),
  );
}

export async function getDreamState(): Promise<DreamStateResponse> {
  return fetchJson<DreamStateResponse>("api/dream/state");
}

export async function postDreamPlan(input: DreamPlanRequest = {}): Promise<DreamPlanResponse> {
  return fetchJson<DreamPlanResponse>("api/dream/plan", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function postDreamApply(input: DreamApplyRequest = {}): Promise<DreamApplyResponse> {
  return fetchJson<DreamApplyResponse>("api/dream/apply", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function postValue(input: CreateValueRequest): Promise<IdentityValue> {
  return fetchJson<IdentityValue>("api/identity/values", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function postGoal(input: CreateGoalRequest): Promise<IdentityGoal> {
  return fetchJson<IdentityGoal>("api/identity/goals", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function patchGoal(id: string, input: PatchGoalRequest): Promise<IdentityGoal> {
  return fetchJson<IdentityGoal>(`api/identity/goals/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export async function postGrowthMarker(input: CreateGrowthMarkerRequest): Promise<GrowthMarker> {
  return fetchJson<GrowthMarker>("api/identity/growth-markers", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function patchOpenQuestion(
  id: string,
  input: PatchOpenQuestionRequest,
): Promise<OpenQuestion> {
  return fetchJson<OpenQuestion>(`api/identity/open-questions/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export async function patchReviewItem(
  id: number,
  input: PatchReviewItemRequest,
): Promise<ReviewRow> {
  return fetchJson<ReviewRow>(`api/dream/review/${encodeURIComponent(String(id))}`, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export async function getAttachmentMetadata(
  attachmentId: string,
): Promise<AttachmentMetadataResponse> {
  return fetchJson<AttachmentMetadataResponse>(
    `api/attachments/${encodeURIComponent(attachmentId)}`,
  );
}

export async function getAttachmentStatuses(
  attachmentIds: readonly string[],
): Promise<AttachmentStatusItem[]> {
  if (attachmentIds.length === 0) {
    return [];
  }

  return fetchJson<AttachmentStatusItem[]>(
    "api/attachments",
    undefined,
    new URLSearchParams({ ids: attachmentIds.join(",") }),
  );
}

export async function postTurn(input: TurnRequest): Promise<TurnResponse> {
  if (input.attachments !== undefined && input.attachments.length > 0) {
    const body = new FormData();
    body.set("message", input.message);
    body.set("audience", input.audience);
    if (input.stakes !== undefined) {
      body.set("stakes", input.stakes);
    }
    for (const attachment of input.attachments) {
      body.append("attachments[]", attachment, attachment.name);
    }

    return fetchJson<TurnResponse>("api/turn", {
      method: "POST",
      body,
    });
  }

  return fetchJson<TurnResponse>("api/turn", {
    method: "POST",
    body: JSON.stringify({
      message: input.message,
      audience: input.audience,
      stakes: input.stakes,
    }),
  });
}

export function liveUrl(): string {
  return `${wsBase()}/api/live`;
}

export function attachmentBytesUrl(attachmentId: string, audience?: string): string {
  const params = audience === undefined ? undefined : new URLSearchParams({ audience });
  return apiUrl(`api/attachments/${encodeURIComponent(attachmentId)}/bytes`, params);
}
