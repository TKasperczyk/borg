// Constants shared by deliberation prompt assembly, planning, and finalization.
export const DEFAULT_DELIBERATION_RESPONSE_MAX_TOKENS = 8_000;
export const DEFAULT_DELIBERATION_PLAN_MAX_TOKENS = 2_000;
// Per-call output budget for Sol-cognition calls when adaptive thinking is on.
// Thinking tokens count against max_tokens, so the budget must hold the thinking
// AND the emission -- otherwise the model exhausts the budget mid-thought and
// emits no tool. Sized so high/xhigh effort completes with headroom (max effort
// is intentionally unsupported: it thinks without bound and never emits).
export const THINKING_DELIBERATION_MAX_TOKENS = 16_000;
export const DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET = 120_000;
export const DEFAULT_SEMANTIC_CONTEXT_BUDGET = 8_000;
