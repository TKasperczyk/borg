import type {
  CommitmentCriticalDomain,
  CommitmentEnforcementClass,
  CommitmentKind,
  CommitmentType,
} from "../../memory/commitments/index.js";

export const CLASSIFICATION_DOWNGRADE_REASONS = [
  "internal_tool_hygiene_with_preference_type",
  "explicit_no_disclosure_without_boundary_kind",
  "process_norm_classified_critical",
  "preference_with_internal_tool_hygiene",
  "critical_without_domain",
] as const;

export type ClassificationDowngradeReason = (typeof CLASSIFICATION_DOWNGRADE_REASONS)[number];

export type ClassificationNormalizationResult = {
  enforcement_class: CommitmentEnforcementClass;
  critical_domain: CommitmentCriticalDomain | null;
  downgraded_from: {
    enforcement_class: "critical";
    critical_domain: CommitmentCriticalDomain | null;
  } | null;
  downgrade_reason: ClassificationDowngradeReason | null;
};

export type NormalizeCommitmentClassificationInput = {
  kind: CommitmentKind;
  type: CommitmentType;
  enforcement_class: CommitmentEnforcementClass;
  critical_domain: CommitmentCriticalDomain | null;
};

function downgrade(
  input: NormalizeCommitmentClassificationInput,
  reason: ClassificationDowngradeReason,
): ClassificationNormalizationResult {
  return {
    enforcement_class: "advisory",
    critical_domain: null,
    downgraded_from: {
      enforcement_class: "critical",
      critical_domain: input.critical_domain,
    },
    downgrade_reason: reason,
  };
}

export function normalizeCommitmentClassification(
  input: NormalizeCommitmentClassificationInput,
): ClassificationNormalizationResult {
  if (input.enforcement_class === "advisory") {
    return {
      enforcement_class: "advisory",
      critical_domain: null,
      downgraded_from: null,
      downgrade_reason: null,
    };
  }

  if (input.critical_domain === null) {
    return downgrade(input, "critical_without_domain");
  }

  if (input.kind === "process_norm") {
    return downgrade(input, "process_norm_classified_critical");
  }

  if (
    input.kind === "participant_preference" &&
    input.type === "preference" &&
    input.critical_domain === "internal_tool_hygiene"
  ) {
    return downgrade(input, "preference_with_internal_tool_hygiene");
  }

  if (input.type === "preference" && input.critical_domain === "internal_tool_hygiene") {
    return downgrade(input, "internal_tool_hygiene_with_preference_type");
  }

  if (
    input.critical_domain === "explicit_no_disclosure" &&
    input.kind !== "boundary" &&
    input.kind !== "audience_rule"
  ) {
    return downgrade(input, "explicit_no_disclosure_without_boundary_kind");
  }

  return {
    enforcement_class: "critical",
    critical_domain: input.critical_domain,
    downgraded_from: null,
    downgrade_reason: null,
  };
}
