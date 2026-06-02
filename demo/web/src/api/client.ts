import type {
  AttachmentMetadataResponse,
  AttachmentStatusItem,
  CommitmentEnforcement,
  CommitmentItem,
  CommitmentState,
  CommitmentsResponse,
  CorrectionForgetResponse,
  CorrectionReviewsResponse,
  CorrectMemoryRequest,
  CreateCommitmentRequest,
  CreatorDirectivesResponse,
  CreatorDirectiveItem,
  CreatorDirectiveReconciliationRequest,
  CreatorDirectiveRevokeRequest,
  CreatorDirectiveSupersedeRequest,
  CreateGoalRequest,
  CreateGrowthMarkerRequest,
  CreateValueRequest,
  DreamApplyRequest,
  DreamApplyResponse,
  DreamAuditResponse,
  DreamPlanRequest,
  DreamPlanResponse,
  EntityBorgRole,
  EntityRecord,
  DreamStateResponse,
  GrowthMarker,
  IdentityGoal,
  IdentityResponse,
  IdentityValue,
  InvalidateSemanticEdgeRequest,
  LedgerResponse,
  MaintenanceAuditRow,
  MemoryBandDetail,
  MemoryBandId,
  OpenQuestion,
  PatchCorrectionReviewRequest,
  PatchGoalRequest,
  PatchOpenQuestionRequest,
  PatchReviewRequest,
  PatchReviewItemRequest,
  PromptBlockView,
  PromptBlocksResponse,
  PromptKey,
  RevokeCommitmentRequest,
  ReviewKind,
  ReviewRow,
  ReviewsResponse,
  MemoryBandsResponse,
  SemanticGraphResponse,
  SessionParticipationPolicy,
  SessionRecord,
  SessionsResponse,
  SharedStateResponse,
  StateSnapshot,
  StreamEntryKind,
  StreamResponse,
  TurnRequest,
  TurnResponse,
  TurnsResponse,
  WhyResponse,
} from "./types";

const DEFAULT_API_BASE = "";
export const RESET_CONFIRM_TOKEN = "RESET";

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

function addSessionParam(params: URLSearchParams, session?: string): void {
  if (session !== undefined && session.length > 0) {
    params.set("session", session);
  }
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

export async function getState(input: { session?: string } = {}): Promise<StateSnapshot> {
  const params = new URLSearchParams();
  addSessionParam(params, input.session);
  return fetchJson<StateSnapshot>("api/state", undefined, params);
}

export async function getSessions(): Promise<SessionsResponse> {
  return fetchJson<SessionsResponse>("api/sessions");
}

export async function setSessionPolicy(
  sessionId: string,
  policy: SessionParticipationPolicy,
  reason?: string,
): Promise<SessionRecord> {
  const trimmedReason = reason?.trim();

  return fetchJson<SessionRecord>(`api/sessions/${encodeURIComponent(sessionId)}/participation`, {
    method: "POST",
    body: JSON.stringify({
      policy,
      ...(trimmedReason === undefined || trimmedReason.length === 0
        ? {}
        : { reason: trimmedReason }),
    }),
  });
}

export async function getCreatorEntity(): Promise<EntityRecord | null> {
  return fetchJson<EntityRecord | null>("api/entities/creator");
}

export async function setEntityBorgRole(
  entityId: string,
  role: EntityBorgRole,
): Promise<EntityRecord> {
  return fetchJson<EntityRecord>(`api/entities/${encodeURIComponent(entityId)}/borg-role`, {
    method: "POST",
    body: JSON.stringify({ role }),
  });
}

export async function setCreatorByName(name: string): Promise<EntityRecord> {
  return fetchJson<EntityRecord>("api/entities/creator", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export async function openOperatorSession(): Promise<SessionRecord> {
  return fetchJson<SessionRecord>("api/sessions/operator", {
    method: "POST",
  });
}

export async function getStream(input: {
  session?: string;
  audience?: string;
  kinds?: readonly StreamEntryKind[];
  limit?: number;
  before?: string;
}): Promise<StreamResponse> {
  const params = new URLSearchParams();
  addSessionParam(params, input.session);
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

export async function getTurns(input: {
  session?: string;
  limit?: number;
  cursor?: string;
} = {}): Promise<TurnsResponse> {
  const params = new URLSearchParams();
  addSessionParam(params, input.session);
  if (input.limit !== undefined) {
    params.set("limit", String(input.limit));
  }
  if (input.cursor !== undefined) {
    params.set("cursor", input.cursor);
  }

  return fetchJson<TurnsResponse>("api/turns", undefined, params);
}

export async function getLedger(turnId: string): Promise<LedgerResponse> {
  return fetchJson<LedgerResponse>(`api/turns/${encodeURIComponent(turnId)}/ledger`);
}

export async function getSharedState(audience: string): Promise<SharedStateResponse> {
  const params = new URLSearchParams({ audience });
  return fetchJson<SharedStateResponse>("api/shared-state", undefined, params);
}

export async function getMemoryBands(
  input: { session?: string } = {},
): Promise<MemoryBandsResponse> {
  const params = new URLSearchParams();
  addSessionParam(params, input.session);
  return fetchJson<MemoryBandsResponse>("api/memory/bands", undefined, params);
}

export async function getMemoryBand(
  id: MemoryBandId,
  input: { session?: string } = {},
): Promise<MemoryBandDetail> {
  const params = new URLSearchParams();
  addSessionParam(params, input.session);
  return fetchJson<MemoryBandDetail>(
    `api/memory/bands/${encodeURIComponent(id)}`,
    undefined,
    params,
  );
}

export async function getSemanticGraph(limit = 300): Promise<SemanticGraphResponse> {
  return fetchJson<SemanticGraphResponse>(
    "api/semantic/graph",
    undefined,
    new URLSearchParams({ limit: String(limit) }),
  );
}

export async function getWhy(id: string): Promise<WhyResponse> {
  return fetchJson<WhyResponse>(`api/correction/${encodeURIComponent(id)}/why`);
}

export async function postCorrectionForget(id: string): Promise<CorrectionForgetResponse> {
  return fetchJson<CorrectionForgetResponse>(`api/correction/${encodeURIComponent(id)}/forget`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export async function postCorrectionCorrect(
  id: string,
  input: CorrectMemoryRequest,
): Promise<ReviewRow> {
  return fetchJson<ReviewRow>(`api/correction/${encodeURIComponent(id)}/correct`, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function postSemanticEdgeInvalidate(
  id: string,
  input: InvalidateSemanticEdgeRequest,
): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>(
    `api/correction/semantic-edges/${encodeURIComponent(id)}/invalidate`,
    {
      method: "POST",
      body: JSON.stringify(input),
    },
  );
}

export async function getCorrectionReviews(): Promise<CorrectionReviewsResponse> {
  return fetchJson<CorrectionReviewsResponse>("api/correction/reviews");
}

export async function getReviews(
  input: {
    openOnly?: boolean;
    kind?: ReviewKind;
  } = {},
): Promise<ReviewsResponse> {
  const params = new URLSearchParams();
  if (input.openOnly !== undefined) {
    params.set("open_only", input.openOnly ? "true" : "false");
  }
  if (input.kind !== undefined) {
    params.set("kind", input.kind);
  }
  return fetchJson<ReviewsResponse>("api/reviews", undefined, params);
}

export async function patchReview(id: number, input: PatchReviewRequest): Promise<ReviewRow> {
  return fetchJson<ReviewRow>(`api/reviews/${encodeURIComponent(String(id))}`, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export async function patchCorrectionReview(
  id: number,
  input: PatchCorrectionReviewRequest,
): Promise<ReviewRow> {
  return fetchJson<ReviewRow>(`api/correction/reviews/${encodeURIComponent(String(id))}`, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
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

export async function postCommitment(input: CreateCommitmentRequest): Promise<CommitmentItem> {
  return fetchJson<CommitmentItem>("api/commitments", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function postCommitmentRevoke(
  id: string,
  input: RevokeCommitmentRequest,
): Promise<CommitmentItem> {
  return fetchJson<CommitmentItem>(`api/commitments/${encodeURIComponent(id)}/revoke`, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function getCreatorDirectives(): Promise<CreatorDirectivesResponse> {
  return fetchJson<CreatorDirectivesResponse>("api/creator-directives");
}

export async function revokeCreatorDirective(
  id: string,
  reason: string,
): Promise<CreatorDirectiveItem> {
  const input: CreatorDirectiveRevokeRequest = { reason };
  return fetchJson<CreatorDirectiveItem>(
    `api/creator-directives/${encodeURIComponent(id)}/revoke`,
    {
      method: "POST",
      body: JSON.stringify(input),
    },
  );
}

export async function supersedeCreatorDirective(
  id: string,
  replacementId: string,
): Promise<CreatorDirectiveItem> {
  const input: CreatorDirectiveSupersedeRequest = { replacement_id: replacementId };
  return fetchJson<CreatorDirectiveItem>(
    `api/creator-directives/${encodeURIComponent(id)}/supersede`,
    {
      method: "POST",
      body: JSON.stringify(input),
    },
  );
}

export async function resolveCreatorDirectiveReconciliation(
  id: number,
  input: CreatorDirectiveReconciliationRequest,
): Promise<ReviewRow> {
  return fetchJson<ReviewRow>(
    `api/reviews/${encodeURIComponent(String(id))}/creator-directive-reconciliation`,
    {
      method: "POST",
      body: JSON.stringify(input),
    },
  );
}

export async function getDreamAudit(limit = 50): Promise<DreamAuditResponse> {
  return fetchJson<DreamAuditResponse>(
    "api/dream/audit",
    undefined,
    new URLSearchParams({ limit: String(limit) }),
  );
}

export async function revertDreamAudit(id: number): Promise<MaintenanceAuditRow> {
  return fetchJson<MaintenanceAuditRow>(
    `api/dream/audit/${encodeURIComponent(String(id))}/revert`,
    {
      method: "POST",
    },
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
    body.set("external_message_id", input.external_message_id);
    body.set("audience", input.audience);
    if (input.audience_entity_id !== undefined && input.audience_entity_id !== null) {
      body.set("audience_entity_id", input.audience_entity_id);
    }
    if (input.session !== undefined) {
      body.set("session", input.session);
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
      external_message_id: input.external_message_id,
      audience: input.audience,
      ...(input.audience_entity_id === undefined || input.audience_entity_id === null
        ? {}
        : { audience_entity_id: input.audience_entity_id }),
      session: input.session,
    }),
  });
}

export async function getPrompts(): Promise<PromptBlocksResponse> {
  return fetchJson<PromptBlocksResponse>("api/prompts");
}

export async function putPrompt(key: PromptKey, text: string): Promise<PromptBlockView> {
  return fetchJson<PromptBlockView>(`api/prompts/${encodeURIComponent(key)}`, {
    method: "PUT",
    body: JSON.stringify({ text }),
  });
}

export async function deletePrompt(key: PromptKey): Promise<PromptBlockView> {
  return fetchJson<PromptBlockView>(`api/prompts/${encodeURIComponent(key)}`, {
    method: "DELETE",
  });
}

export async function postAdminReset(): Promise<{ ok: true }> {
  return fetchJson<{ ok: true }>("api/admin/reset", {
    method: "POST",
    body: JSON.stringify({ confirm: RESET_CONFIRM_TOKEN }),
  });
}

export function liveUrl(): string {
  return `${wsBase()}/api/live`;
}

export function attachmentBytesUrl(attachmentId: string, audience: string): string {
  const params = new URLSearchParams({ audience });
  return apiUrl(`api/attachments/${encodeURIComponent(attachmentId)}/bytes`, params);
}
