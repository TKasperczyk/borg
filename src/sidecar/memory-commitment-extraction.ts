import { ConfigError } from "../util/errors.js";

export const MEMORY_COMMITMENT_EXTRACTION_ENABLED_ENV = "BORG_MEMORY_COMMITMENT_EXTRACTION_ENABLED";
/**
 * Post-usage token cap for each source entry. Because the provider reports
 * usage only after the call, exceeding this cap immediately dead-letters that
 * entry (with trace + error log) instead of silently discarding it or paying
 * for deterministic retries. Unset means usage is tracked without a cap.
 */
export const MEMORY_COMMITMENT_EXTRACTION_BUDGET_ENV = "BORG_MEMORY_COMMITMENT_EXTRACTION_BUDGET";

export function memoryCommitmentExtractionEnabledFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  const raw = env[MEMORY_COMMITMENT_EXTRACTION_ENABLED_ENV]?.trim().toLowerCase();

  if (raw === undefined || raw === "") {
    return true;
  }

  if (raw === "1" || raw === "true") {
    return true;
  }

  if (raw === "0" || raw === "false") {
    return false;
  }

  throw new ConfigError(`${MEMORY_COMMITMENT_EXTRACTION_ENABLED_ENV} must be true/false or 1/0`);
}

export function memoryCommitmentExtractionBudgetFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): number | null {
  const raw = env[MEMORY_COMMITMENT_EXTRACTION_BUDGET_ENV]?.trim();

  if (raw === undefined || raw === "") {
    return null;
  }

  const budget = Number(raw);

  if (!Number.isInteger(budget) || budget <= 0) {
    throw new ConfigError(`${MEMORY_COMMITMENT_EXTRACTION_BUDGET_ENV} must be a positive integer`);
  }

  return budget;
}
