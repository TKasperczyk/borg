// Constants shared by deliberation prompt assembly, planning, and finalization.
export const DEFAULT_DELIBERATION_RESPONSE_MAX_TOKENS = 8_000;
export const DEFAULT_DELIBERATION_PLAN_MAX_TOKENS = 2_000;
export const DEFAULT_DELIBERATION_PLAN_CALL_TIMEOUT_MS = 12 * 60_000;
// Per-call output budget for being-cognition calls when adaptive thinking is on.
// Thinking tokens count against max_tokens, so the budget must hold the thinking
// AND the emission -- otherwise the model exhausts the budget mid-thought and
// emits no tool. Sized so high/xhigh effort completes with headroom (max effort
// is intentionally unsupported: it thinks without bound and never emits).
export const THINKING_DELIBERATION_MAX_TOKENS = 16_000;
// At 120K this single block was 55% of every finalizer call's turn context
// (~700K chars of plan-requested verification payloads per call, measured
// 2026-08-17) - the dominant share of live burn. Over-budget checks degrade
// to the explicit check_not_completed flag, never silent truncation, per the
// entity's consulted exact-or-flagged policy.
export const DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET = 32_000;
// A payload-less commitment verification row, including realistic handle IDs,
// disclosure, and enforcement fields, currently costs about 182 tokens in the
// conservative fixture. 64K therefore keeps 300-row commitment/goal-scale
// result sets complete with roughly 17% headroom while bounding the
// multi-thousand-row live failure mode.
export const DEFAULT_PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_TOKEN_BUDGET = 64_000;
export const DEFAULT_SEMANTIC_CONTEXT_BUDGET = 8_000;
