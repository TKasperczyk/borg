// ID resolution helpers that turn raw CLI arguments into validated Borg IDs.
import {
  DEFAULT_SESSION_ID,
  parseAutobiographicalPeriodId,
  parseAuditId,
  parseCommitmentId,
  parseEpisodeId,
  parseGoalId,
  parseMaintenanceRunId,
  parseOpenQuestionId,
  parseSemanticNodeId,
  parseSessionId,
  parseSkillId,
  parseValueId,
} from "../../util/ids.js";
import { CliError } from "./errors.js";

export function resolveSessionId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    return DEFAULT_SESSION_ID;
  }

  try {
    return parseSessionId(value);
  } catch (error) {
    throw new CliError(`Invalid session id: ${value}`, {
      cause: error,
    });
  }
}

export function resolveEpisodeId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Episode id is required");
  }

  try {
    return parseEpisodeId(value);
  } catch (error) {
    throw new CliError(`Invalid episode id: ${value}`, {
      cause: error,
    });
  }
}

export function resolveSemanticNodeId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Semantic node id is required");
  }

  try {
    return parseSemanticNodeId(value);
  } catch (error) {
    throw new CliError(`Invalid semantic node id: ${value}`, {
      cause: error,
    });
  }
}

export function resolveCommitmentId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Commitment id is required");
  }

  try {
    return parseCommitmentId(value);
  } catch (error) {
    throw new CliError(`Invalid commitment id: ${value}`, {
      cause: error,
    });
  }
}

export function resolveGoalId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Goal id is required");
  }

  try {
    return parseGoalId(value);
  } catch (error) {
    throw new CliError(`Invalid goal id: ${value}`, {
      cause: error,
    });
  }
}

export function resolveValueId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Value id is required");
  }

  try {
    return parseValueId(value);
  } catch (error) {
    throw new CliError(`Invalid value id: ${value}`, {
      cause: error,
    });
  }
}

export function resolveAutobiographicalPeriodId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Period id is required");
  }

  try {
    return parseAutobiographicalPeriodId(value);
  } catch (error) {
    throw new CliError(`Invalid period id: ${value}`, {
      cause: error,
    });
  }
}

export function resolveOpenQuestionId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Open question id is required");
  }

  try {
    return parseOpenQuestionId(value);
  } catch (error) {
    throw new CliError(`Invalid open question id: ${value}`, {
      cause: error,
    });
  }
}

export function resolveSkillId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Skill id is required");
  }

  try {
    return parseSkillId(value);
  } catch (error) {
    throw new CliError(`Invalid skill id: ${value}`, {
      cause: error,
    });
  }
}

export function resolveMaintenanceRunId(value: unknown) {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Maintenance run id is required");
  }

  try {
    return parseMaintenanceRunId(value);
  } catch (error) {
    throw new CliError(`Invalid maintenance run id: ${value}`, {
      cause: error,
    });
  }
}

export function resolveAuditId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Audit id is required");
  }

  try {
    return parseAuditId(value);
  } catch (error) {
    throw new CliError(`Invalid audit id: ${value}`, {
      cause: error,
    });
  }
}
