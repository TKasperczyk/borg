// Argument parsers and validators shared by CLI command handlers.
import { commitmentTypeSchema } from "../../memory/commitments/index.js";
import { identityRecordTypeSchema } from "../../memory/identity/index.js";
import {
  goalStatusSchema,
  growthMarkerCategorySchema,
  openQuestionStatusSchema,
} from "../../memory/self/index.js";
import {
  reviewKindSchema,
  reviewResolutionSchema,
  semanticNodeKindSchema,
  semanticRelationSchema,
} from "../../memory/semantic/index.js";
import { OFFLINE_PROCESS_NAMES, type OfflineProcessName } from "../../offline/index.js";
import { CliError } from "./errors.js";

export function parseLimit(value: unknown, flag = "--limit"): number {
  const candidate = Array.isArray(value) ? value.at(-1) : value;

  if (!Number.isInteger(candidate) || candidate <= 0) {
    throw new CliError(`${flag} must be a positive integer`);
  }

  return candidate;
}

export function parsePriority(value: unknown, fallback = 0): number {
  if (value === undefined) {
    return fallback;
  }

  const candidate = Array.isArray(value) ? value.at(-1) : value;

  if (typeof candidate !== "number" || !Number.isFinite(candidate)) {
    throw new CliError("--priority must be a finite number");
  }

  return candidate;
}

export function parseRequiredText(value: unknown, flag: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError(`${flag} is required`);
  }

  return value.trim();
}

export function resolveEpisodeVisibilityOptions(commandOptions: Record<string, unknown>): {
  audience?: string | null;
  crossAudience?: boolean;
} {
  const audience =
    typeof commandOptions.audience === "string"
      ? parseRequiredText(commandOptions.audience, "--audience")
      : undefined;
  const crossAudience = commandOptions.all === true;

  if (audience !== undefined && crossAudience) {
    throw new CliError("--audience cannot be combined with --all");
  }

  return {
    audience,
    crossAudience,
  };
}

export function parseSinceToTimestamp(
  value: unknown,
  flag = "--since",
  nowMs = Date.now(),
): number | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError(
      `${flag} must be a duration like 1h, "now", or an epoch milliseconds timestamp`,
    );
  }

  const trimmed = value.trim();

  if (trimmed === "now") {
    return nowMs;
  }

  const absolute = Number(trimmed);

  if (Number.isFinite(absolute) && trimmed === String(absolute)) {
    return absolute;
  }

  const match = trimmed.match(/^(\d+)([smhd])$/);

  if (match === null) {
    throw new CliError(
      `${flag} must be a duration like 1h, "now", or an epoch milliseconds timestamp`,
    );
  }

  const amount = Number(match[1]);
  const unit = match[2];
  const multiplier =
    unit === "s" ? 1_000 : unit === "m" ? 60_000 : unit === "h" ? 3_600_000 : 86_400_000;

  return nowMs - amount * multiplier;
}

export function parseGoalStatus(value: unknown) {
  const parsed = goalStatusSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError("--status must be one of: active, done, abandoned, blocked", {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

export function parseGrowthMarkerCategory(value: unknown) {
  const parsed = growthMarkerCategorySchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError(
      `--category must be one of: ${growthMarkerCategorySchema.options.join(", ")}`,
      {
        cause: parsed.error,
      },
    );
  }

  return parsed.data;
}

export function parseOpenQuestionStatus(value: unknown) {
  const parsed = openQuestionStatusSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError(`--status must be one of: ${openQuestionStatusSchema.options.join(", ")}`, {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

export function parseSemanticNodeKind(value: unknown) {
  const parsed = semanticNodeKindSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError("--kind must be one of: concept, entity, proposition", {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

export function parseSemanticRelation(value: unknown) {
  const parsed = semanticRelationSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError("--relation must be a supported semantic relation", {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

export function parseCommitmentType(value: unknown) {
  const parsed = commitmentTypeSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError("--type must be one of: promise, boundary, rule, preference", {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

export function parseIdentityRecordType(value: unknown) {
  const parsed = identityRecordTypeSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError(
      `--record-type must be one of: ${identityRecordTypeSchema.options.join(", ")}`,
      {
        cause: parsed.error,
      },
    );
  }

  return parsed.data;
}

export function parseReviewKind(value: unknown) {
  const parsed = reviewKindSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError(`--kind must be one of: ${reviewKindSchema.options.join(", ")}`, {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

export function parseReviewResolution(value: unknown) {
  const parsed = reviewResolutionSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError(`--decision must be one of: ${reviewResolutionSchema.options.join(", ")}`, {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

export function parseIdList<T extends string>(
  value: unknown,
  itemParser: (value: string) => T,
  flag: string,
): T[] {
  if (value === undefined) {
    return [];
  }

  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError(`${flag} must be a comma-separated list`);
  }

  return value
    .split(",")
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
    .map((item) => itemParser(item));
}

export function parseStringList(value: unknown, flag: string): string[] {
  return parseIdList(value, (item) => item, flag);
}

export function parsePositiveInteger(value: unknown, flag: string): number {
  const candidate = Array.isArray(value) ? value.at(-1) : value;

  if (!Number.isInteger(candidate) || candidate <= 0) {
    throw new CliError(`${flag} must be a positive integer`);
  }

  return candidate;
}

export function parseOptionalPositiveInteger(value: unknown, flag: string): number | undefined {
  const candidate = Array.isArray(value) ? value.at(-1) : value;

  if (
    candidate === undefined ||
    candidate === null ||
    candidate === false ||
    (typeof candidate === "number" && Number.isNaN(candidate)) ||
    (typeof candidate === "string" && candidate.trim() === "")
  ) {
    return undefined;
  }

  if (typeof candidate !== "number" && typeof candidate !== "string") {
    return undefined;
  }

  return parsePositiveInteger(candidate, flag);
}

export function parseOptionalAsOf(value: unknown, flag = "--as-of"): number | undefined {
  const candidate = Array.isArray(value) ? value.at(-1) : value;

  if (
    candidate === undefined ||
    candidate === null ||
    candidate === false ||
    (typeof candidate === "number" && Number.isNaN(candidate)) ||
    (typeof candidate === "string" && candidate.trim() === "")
  ) {
    return undefined;
  }

  if (typeof candidate === "number" && Number.isFinite(candidate)) {
    return candidate;
  }

  if (typeof candidate !== "string") {
    throw new CliError(`${flag} must be an ISO timestamp or epoch milliseconds`);
  }

  const trimmed = candidate.trim();
  const epochMs = Number(trimmed);

  if (Number.isFinite(epochMs)) {
    return epochMs;
  }

  const parsed = Date.parse(trimmed);

  if (!Number.isFinite(parsed)) {
    throw new CliError(`${flag} must be an ISO timestamp or epoch milliseconds`);
  }

  return parsed;
}

export function parseFiniteNumber(value: unknown, label: string): number {
  const candidate = Array.isArray(value) ? value.at(-1) : value;
  const numeric =
    typeof candidate === "number"
      ? candidate
      : typeof candidate === "string" && candidate.trim() !== ""
        ? Number(candidate)
        : Number.NaN;

  if (!Number.isFinite(numeric)) {
    throw new CliError(`${label} must be a finite number`);
  }

  return numeric;
}

export function parseJsonObject(value: unknown, flag: string): Record<string, unknown> {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError(`${flag} must be a JSON object`);
  }

  try {
    const parsed = JSON.parse(value) as unknown;

    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new TypeError("value must be a JSON object");
    }

    return parsed as Record<string, unknown>;
  } catch (error) {
    throw new CliError(`${flag} must be a valid JSON object`, {
      cause: error,
    });
  }
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function parseStakes(value: unknown): "low" | "medium" | "high" | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (value === "low" || value === "medium" || value === "high") {
    return value;
  }

  throw new CliError("--stakes must be one of: low, medium, high");
}

export function parseBudget(value: unknown): number | undefined {
  if (value === undefined) {
    return undefined;
  }

  const candidate = Array.isArray(value) ? value.at(-1) : value;

  if (candidate === undefined) {
    return undefined;
  }

  if (typeof candidate === "string" && candidate.trim() !== "") {
    const parsed = Number(candidate);

    if (!Number.isInteger(parsed) || parsed <= 0) {
      throw new CliError("--budget must be a positive integer");
    }

    return parsed;
  }

  return parsePositiveInteger(candidate, "--budget");
}

export function parseOptionalPath(value: unknown, flag: string): string | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError(`${flag} must be a file path`);
  }

  return value.trim();
}

export function parseOfflineProcessName(value: unknown) {
  if (typeof value !== "string" || !OFFLINE_PROCESS_NAMES.includes(value as never)) {
    throw new CliError(`--process must be one of: ${OFFLINE_PROCESS_NAMES.join(", ")}`);
  }

  return value as OfflineProcessName;
}

export function parseOfflineProcessList(value: unknown) {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("--process must be a comma-separated list");
  }

  return [
    ...new Set(
      value
        .split(",")
        .map((item) => item.trim())
        .filter((item) => item.length > 0)
        .map((item) => parseOfflineProcessName(item)),
    ),
  ] satisfies OfflineProcessName[];
}
