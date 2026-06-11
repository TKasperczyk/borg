import type {
  AdminResetResponse,
  ApiState,
  AssembledPromptResponse,
  ActivityResponse,
  AutonomyStateResponse,
  BandDetailResponse,
  Commitment,
  CommitmentsResponse,
  CorrectionReviewPatchBody,
  CorrectionWhyResponse,
  CreatorDirective,
  CreatorDirectiveReconciliationBody,
  CreatorDirectivesResponse,
  DreamApplyResponse,
  DreamAuditResponse,
  DreamPlanResponse,
  DreamStateResponse,
  EntityRecord,
  EpisodeDetailResponse,
  GoalPatchBody,
  IdentityResponse,
  JournalResponse,
  LedgerResponse,
  MaintenanceAuditRow,
  MemoryBandId,
  MemoryBandsResponse,
  OpenQuestionPatchBody,
  OfflineProcessName,
  PromptBlock,
  PromptsResponse,
  ReviewGenericPatchBody,
  ReviewKind,
  ReviewRow,
  ReviewsResponse,
  SemanticEdgeDetailResponse,
  SemanticGraphResponse,
  SemanticNodeDetailResponse,
  SessionParticipationPolicy,
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
    if (
      typeof body === "object" &&
      body !== null &&
      "error" in body &&
      typeof body.error === "object" &&
      body.error !== null &&
      "message" in body.error &&
      typeof body.error.message === "string"
    ) {
      return body.error.message;
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

export async function patchJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "PATCH",
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

export async function putJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "PUT",
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

export async function deleteJson<T>(path: string): Promise<T> {
  const response = await fetch(path, {
    method: "DELETE",
    headers: {
      Accept: "application/json",
    },
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

export function fetchActivity(day?: string): Promise<ActivityResponse> {
  const params = new URLSearchParams();
  if (day !== undefined) {
    params.set("day", day);
  }
  const suffix = params.size === 0 ? "" : `?${params.toString()}`;
  return getJson<ActivityResponse>(`/api/activity${suffix}`);
}

export function fetchAutonomyState(): Promise<AutonomyStateResponse> {
  return getJson<AutonomyStateResponse>("/api/autonomy");
}

export function fetchJournal(limit = 10, day?: string): Promise<JournalResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (day !== undefined) {
    params.set("day", day);
  }
  return getJson<JournalResponse>(`/api/journal?${params.toString()}`);
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

export function fetchIdentity(): Promise<IdentityResponse> {
  return getJson<IdentityResponse>("/api/identity");
}

export function patchGoal(id: string, body: GoalPatchBody): Promise<unknown> {
  return patchJson<unknown>(`/api/identity/goals/${encodeURIComponent(id)}`, body);
}

export function patchOpenQuestion(id: string, body: OpenQuestionPatchBody): Promise<unknown> {
  return patchJson<unknown>(`/api/identity/open-questions/${encodeURIComponent(id)}`, body);
}

export function fetchCreatorDirectives(
  status: "active" | "revoked" | "superseded" | "all" = "active",
): Promise<CreatorDirectivesResponse> {
  return getJson<CreatorDirectivesResponse>(`/api/creator-directives?status=${status}`);
}

export function revokeCreatorDirective(id: string, reason: string): Promise<CreatorDirective> {
  return postJson<CreatorDirective>(`/api/creator-directives/${encodeURIComponent(id)}/revoke`, {
    reason,
  });
}

export function supersedeCreatorDirective(
  id: string,
  replacementId: string,
): Promise<CreatorDirective> {
  return postJson<CreatorDirective>(
    `/api/creator-directives/${encodeURIComponent(id)}/supersede`,
    { replacement_id: replacementId },
  );
}

export function fetchCommitments(
  state: "active" | "all" | "revoked" | "expired" = "active",
): Promise<CommitmentsResponse> {
  return getJson<CommitmentsResponse>(`/api/commitments?state=${state}`);
}

export function revokeCommitment(id: string, reason: string): Promise<Commitment> {
  return postJson<Commitment>(`/api/commitments/${encodeURIComponent(id)}/revoke`, {
    reason,
  });
}

export function fetchMemoryBands(session?: string): Promise<MemoryBandsResponse> {
  const params = new URLSearchParams();
  if (session !== undefined) {
    params.set("session", session);
  }
  const suffix = params.size === 0 ? "" : `?${params.toString()}`;
  return getJson<MemoryBandsResponse>(`/api/memory/bands${suffix}`);
}

export function fetchBandDetail(input: {
  band: MemoryBandId;
  session?: string;
  cursor?: string | null;
  limit?: number;
}): Promise<BandDetailResponse> {
  const params = new URLSearchParams({ limit: String(input.limit ?? 50) });
  if (input.session !== undefined) {
    params.set("session", input.session);
  }
  if (input.cursor !== undefined && input.cursor !== null) {
    params.set("cursor", input.cursor);
  }

  return getJson<BandDetailResponse>(
    `/api/memory/bands/${encodeURIComponent(input.band)}?${params.toString()}`,
  );
}

export function fetchSemanticGraph(limit = 40): Promise<SemanticGraphResponse> {
  return getJson<SemanticGraphResponse>(`/api/semantic/graph?limit=${limit}`);
}

export function fetchSemanticNode(id: string): Promise<SemanticNodeDetailResponse> {
  return getJson<SemanticNodeDetailResponse>(`/api/semantic/nodes/${encodeURIComponent(id)}`);
}

export function fetchSemanticEdge(id: string): Promise<SemanticEdgeDetailResponse> {
  return getJson<SemanticEdgeDetailResponse>(`/api/semantic/edges/${encodeURIComponent(id)}`);
}

export function fetchEpisode(id: string): Promise<EpisodeDetailResponse> {
  return getJson<EpisodeDetailResponse>(`/api/episodes/${encodeURIComponent(id)}`);
}

export function invalidateSemanticEdge(id: string, reason?: string): Promise<unknown> {
  return postJson<unknown>(`/api/correction/semantic-edges/${encodeURIComponent(id)}/invalidate`, {
    ...(reason === undefined || reason.trim().length === 0 ? {} : { reason }),
  });
}

export function fetchReviews(input: {
  openOnly?: boolean;
  kind?: Exclude<ReviewKind, "relationship_claim_ungrounded">;
} = {}): Promise<ReviewsResponse> {
  const params = new URLSearchParams();
  if (input.openOnly !== undefined) {
    params.set("open_only", input.openOnly ? "true" : "false");
  }
  if (input.kind !== undefined) {
    params.set("kind", input.kind);
  }
  const suffix = params.size === 0 ? "" : `?${params.toString()}`;
  return getJson<ReviewsResponse>(`/api/reviews${suffix}`);
}

export function patchReview(id: number, body: ReviewGenericPatchBody): Promise<ReviewRow> {
  return patchJson<ReviewRow>(`/api/reviews/${id}`, body);
}

export function postCreatorDirectiveReconciliation(
  id: number,
  body: CreatorDirectiveReconciliationBody,
): Promise<ReviewRow> {
  return postJson<ReviewRow>(
    `/api/reviews/${id}/creator-directive-reconciliation`,
    body,
  );
}

export function patchDreamReview(id: number, note?: string): Promise<ReviewRow> {
  return patchJson<ReviewRow>(`/api/dream/review/${id}`, {
    action: "dismiss",
    ...(note === undefined || note.trim().length === 0 ? {} : { note }),
  });
}

export function fetchCorrectionReviews(): Promise<ReviewsResponse> {
  return getJson<ReviewsResponse>("/api/correction/reviews");
}

export function fetchCorrectionWhy(id: string): Promise<CorrectionWhyResponse> {
  return getJson<CorrectionWhyResponse>(`/api/correction/${encodeURIComponent(id)}/why`);
}

export function patchCorrectionReview(
  id: number,
  body: CorrectionReviewPatchBody,
): Promise<ReviewRow> {
  return patchJson<ReviewRow>(`/api/correction/reviews/${id}`, body);
}

export function fetchDreamState(): Promise<DreamStateResponse> {
  return getJson<DreamStateResponse>("/api/dream/state");
}

export function fetchDreamAudit(limit = 50): Promise<DreamAuditResponse> {
  return getJson<DreamAuditResponse>(`/api/dream/audit?limit=${limit}`);
}

export function planDream(input: {
  processes?: OfflineProcessName[];
  budget?: number;
}): Promise<DreamPlanResponse> {
  return postJson<DreamPlanResponse>("/api/dream/plan", input);
}

export function applyDream(input: {
  processes?: OfflineProcessName[];
  budget?: number;
  plan_id?: string;
}): Promise<DreamApplyResponse> {
  return postJson<DreamApplyResponse>("/api/dream/apply", input);
}

export function revertDreamAudit(id: number): Promise<MaintenanceAuditRow> {
  return postJson<MaintenanceAuditRow>(`/api/dream/audit/${id}/revert`, {});
}

export function fetchPrompts(): Promise<PromptsResponse> {
  return getJson<PromptsResponse>("/api/prompts");
}

export function fetchAssembledPrompts(): Promise<AssembledPromptResponse> {
  return getJson<AssembledPromptResponse>("/api/prompts/assembled");
}

export function savePromptOverride(key: string, text: string): Promise<PromptBlock> {
  return putJson<PromptBlock>(`/api/prompts/${encodeURIComponent(key)}`, { text });
}

export function resetPromptOverride(key: string): Promise<PromptBlock> {
  return deleteJson<PromptBlock>(`/api/prompts/${encodeURIComponent(key)}`);
}

export function fetchCreatorEntity(): Promise<EntityRecord | null> {
  return getJson<EntityRecord | null>("/api/entities/creator");
}

export function setCreatorEntity(name: string): Promise<EntityRecord> {
  return postJson<EntityRecord>("/api/entities/creator", { name });
}

export function setSessionParticipation(
  sessionId: string,
  policy: SessionParticipationPolicy,
  reason?: string,
): Promise<SessionRecord> {
  return postJson<SessionRecord>(`/api/sessions/${encodeURIComponent(sessionId)}/participation`, {
    policy,
    ...(reason === undefined || reason.trim().length === 0 ? {} : { reason }),
  });
}

export function resetBorg(): Promise<AdminResetResponse> {
  return postJson<AdminResetResponse>("/api/admin/reset", { confirm: "RESET" });
}
