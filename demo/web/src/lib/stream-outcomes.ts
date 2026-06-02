import type {
  AgentObservedStreamContent,
  FinalizerInvalidToolDiagnostic,
  GenerationSuppressionReason,
  StreamEntry,
} from "../api/types";
import type { TagKind } from "../components/Tag";

export type StreamOutcomeClass =
  | "deliberate-silence"
  | "emission-failed"
  | "guard-blocked"
  | "observed"
  | "unknown";

type SuppressedOutcomeClass = Exclude<StreamOutcomeClass, "observed" | "unknown">;
type ParsedAgentSuppressedStreamContent = {
  reason?: string;
  turn_id?: string;
  user_entry_id?: string;
  user_entry_ids?: string[];
  no_output_categories?: string[];
  primary_no_output_reason?: string;
  structural_no_output_flags?: string[];
  finalizer_invalid_tool?: FinalizerInvalidToolDiagnostic;
};

export type StreamOutcomeDescriptor = {
  outcomeClass: StreamOutcomeClass;
  label: string;
  tagKind: TagKind;
};

export type StreamOutcomeSummary = {
  outcome: StreamOutcomeDescriptor;
  reason: string | null;
  primaryNoOutputReason?: string;
  noOutputCategories: string[];
  structuralNoOutputFlags: string[];
  finalizerInvalidTool?: FinalizerInvalidToolDiagnostic;
};

export const SUPPRESSION_REASON_OUTCOME_CLASS = {
  generation_gate: "deliberate-silence",
  active_discourse_stop: "deliberate-silence",
  empty_finalizer: "emission-failed",
  finalizer_failed: "emission-failed",
  finalizer_no_output: "deliberate-silence",
  invalid_tool_after_regenerate: "emission-failed",
  manifest_no_output: "deliberate-silence",
  legacy_manifest_validation_failed_critical: "guard-blocked",
  manifest_validation_failed_critical: "guard-blocked",
  no_output_tool: "deliberate-silence",
  s2_planner_no_output: "deliberate-silence",
  closure_pressure_only: "deliberate-silence",
  closure_response_audit_failed_closed: "guard-blocked",
  commitment_violation: "guard-blocked",
  commitment_violation_after_regenerate: "guard-blocked",
  commitment_revision_failed: "guard-blocked",
  internal_identifier_leak: "guard-blocked",
  rewrite_unsupported_or_empty: "guard-blocked",
} as const satisfies Record<GenerationSuppressionReason, SuppressedOutcomeClass>;

export const STREAM_OUTCOME_DESCRIPTORS = {
  "deliberate-silence": {
    outcomeClass: "deliberate-silence",
    label: "deliberate silence",
    tagKind: "acc",
  },
  "emission-failed": {
    outcomeClass: "emission-failed",
    label: "emission failed",
    tagKind: "bad",
  },
  "guard-blocked": {
    outcomeClass: "guard-blocked",
    label: "guard blocked",
    tagKind: "warn",
  },
  observed: {
    outcomeClass: "observed",
    label: "observed",
    tagKind: "info",
  },
  unknown: {
    outcomeClass: "unknown",
    label: "unknown suppression",
    tagKind: "",
  },
} as const satisfies Record<StreamOutcomeClass, StreamOutcomeDescriptor>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const strings = value.filter((item): item is string => typeof item === "string");
  return strings.length === value.length ? strings : undefined;
}

function finalizerInvalidToolDiagnostic(
  value: unknown,
): FinalizerInvalidToolDiagnostic | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const { tool_name: toolName, reason, attempt } = value;
  if (
    typeof toolName !== "string" ||
    typeof reason !== "string" ||
    (attempt !== "initial" && attempt !== "regenerate")
  ) {
    return undefined;
  }

  return {
    tool_name: toolName,
    reason,
    attempt,
  };
}

export function isGenerationSuppressionReason(
  reason: unknown,
): reason is GenerationSuppressionReason {
  return typeof reason === "string" && Object.hasOwn(SUPPRESSION_REASON_OUTCOME_CLASS, reason);
}

export function outcomeForSuppressionReason(reason: unknown): StreamOutcomeDescriptor {
  if (!isGenerationSuppressionReason(reason)) {
    return STREAM_OUTCOME_DESCRIPTORS.unknown;
  }

  return STREAM_OUTCOME_DESCRIPTORS[SUPPRESSION_REASON_OUTCOME_CLASS[reason]];
}

export function agentSuppressedContent(
  content: unknown,
): ParsedAgentSuppressedStreamContent | null {
  if (!isRecord(content)) {
    return null;
  }

  const userEntryIds = stringArray(content.user_entry_ids);
  const noOutputCategories = stringArray(content.no_output_categories);
  const structuralNoOutputFlags = stringArray(content.structural_no_output_flags);
  const finalizerInvalidTool = finalizerInvalidToolDiagnostic(content.finalizer_invalid_tool);

  return {
    ...(typeof content.reason === "string" ? { reason: content.reason } : {}),
    ...(typeof content.turn_id === "string" ? { turn_id: content.turn_id } : {}),
    ...(typeof content.user_entry_id === "string" ? { user_entry_id: content.user_entry_id } : {}),
    ...(userEntryIds === undefined ? {} : { user_entry_ids: userEntryIds }),
    ...(noOutputCategories === undefined ? {} : { no_output_categories: noOutputCategories }),
    ...(typeof content.primary_no_output_reason === "string"
      ? { primary_no_output_reason: content.primary_no_output_reason }
      : {}),
    ...(structuralNoOutputFlags === undefined
      ? {}
      : { structural_no_output_flags: structuralNoOutputFlags }),
    ...(finalizerInvalidTool === undefined ? {} : { finalizer_invalid_tool: finalizerInvalidTool }),
  };
}

export function agentObservedContent(content: unknown): Partial<AgentObservedStreamContent> | null {
  if (!isRecord(content)) {
    return null;
  }

  return {
    ...(typeof content.reason === "string" ? { reason: content.reason } : {}),
    ...(typeof content.turn_id === "string" ? { turn_id: content.turn_id } : {}),
    ...(typeof content.user_entry_id === "string" ? { user_entry_id: content.user_entry_id } : {}),
    ...(stringArray(content.user_entry_ids) === undefined
      ? {}
      : { user_entry_ids: stringArray(content.user_entry_ids) }),
  };
}

export function streamOutcomeSummary(
  entry: Pick<StreamEntry, "kind" | "content">,
): StreamOutcomeSummary | null {
  if (entry.kind === "agent_observed") {
    const content = agentObservedContent(entry.content);

    return {
      outcome: STREAM_OUTCOME_DESCRIPTORS.observed,
      reason: content?.reason ?? null,
      noOutputCategories: [],
      structuralNoOutputFlags: [],
    };
  }

  if (entry.kind !== "agent_suppressed") {
    return null;
  }

  const content = agentSuppressedContent(entry.content);
  const reason = content?.reason ?? null;

  return {
    outcome: outcomeForSuppressionReason(reason),
    reason,
    ...(content?.primary_no_output_reason === undefined
      ? {}
      : { primaryNoOutputReason: content.primary_no_output_reason }),
    noOutputCategories: content?.no_output_categories ?? [],
    structuralNoOutputFlags: content?.structural_no_output_flags ?? [],
    ...(content?.finalizer_invalid_tool === undefined
      ? {}
      : { finalizerInvalidTool: content.finalizer_invalid_tool }),
  };
}
