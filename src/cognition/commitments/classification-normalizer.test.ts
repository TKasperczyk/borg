import { describe, expect, it } from "vitest";

import { normalizeCommitmentClassification } from "./classification-normalizer.js";

describe("normalizeCommitmentClassification", () => {
  it("downgrades participant preferences classified as internal-tool hygiene", () => {
    expect(
      normalizeCommitmentClassification({
        kind: "participant_preference",
        type: "preference",
        enforcement_class: "critical",
        critical_domain: "internal_tool_hygiene",
      }),
    ).toEqual({
      enforcement_class: "advisory",
      critical_domain: null,
      downgraded_from: {
        enforcement_class: "critical",
        critical_domain: "internal_tool_hygiene",
      },
      downgrade_reason: "preference_with_internal_tool_hygiene",
    });
  });

  it("downgrades any preference type classified as internal-tool hygiene", () => {
    expect(
      normalizeCommitmentClassification({
        kind: "boundary",
        type: "preference",
        enforcement_class: "critical",
        critical_domain: "internal_tool_hygiene",
      }),
    ).toMatchObject({
      enforcement_class: "advisory",
      critical_domain: null,
      downgrade_reason: "internal_tool_hygiene_with_preference_type",
    });
  });

  it("downgrades explicit no-disclosure classifications outside boundary or audience-rule kind", () => {
    expect(
      normalizeCommitmentClassification({
        kind: "participant_preference",
        type: "preference",
        enforcement_class: "critical",
        critical_domain: "explicit_no_disclosure",
      }),
    ).toMatchObject({
      enforcement_class: "advisory",
      critical_domain: null,
      downgrade_reason: "explicit_no_disclosure_without_boundary_kind",
    });
  });

  it("downgrades process norms classified as critical", () => {
    expect(
      normalizeCommitmentClassification({
        kind: "process_norm",
        type: "rule",
        enforcement_class: "critical",
        critical_domain: "safety",
      }),
    ).toMatchObject({
      enforcement_class: "advisory",
      critical_domain: null,
      downgrade_reason: "process_norm_classified_critical",
    });
  });

  it("downgrades critical classifications without a critical domain", () => {
    expect(
      normalizeCommitmentClassification({
        kind: "boundary",
        type: "boundary",
        enforcement_class: "critical",
        critical_domain: null,
      }),
    ).toEqual({
      enforcement_class: "advisory",
      critical_domain: null,
      downgraded_from: {
        enforcement_class: "critical",
        critical_domain: null,
      },
      downgrade_reason: "critical_without_domain",
    });
  });

  it("passes explicit advisory classifications through without a critical domain", () => {
    expect(
      normalizeCommitmentClassification({
        kind: "participant_preference",
        type: "preference",
        enforcement_class: "advisory",
        critical_domain: "privacy",
      }),
    ).toEqual({
      enforcement_class: "advisory",
      critical_domain: null,
      downgraded_from: null,
      downgrade_reason: null,
    });
  });

  it.each([
    {
      label: "boundary privacy",
      kind: "boundary",
      type: "boundary",
      critical_domain: "privacy",
    },
    {
      label: "boundary safety",
      kind: "boundary",
      type: "boundary",
      critical_domain: "safety",
    },
    {
      label: "boundary audience scope",
      kind: "boundary",
      type: "boundary",
      critical_domain: "audience_scope",
    },
    {
      label: "boundary explicit no-disclosure",
      kind: "boundary",
      type: "boundary",
      critical_domain: "explicit_no_disclosure",
    },
    {
      label: "boundary internal tool hygiene",
      kind: "boundary",
      type: "boundary",
      critical_domain: "internal_tool_hygiene",
    },
    {
      label: "audience-rule audience scope boundary",
      kind: "audience_rule",
      type: "boundary",
      critical_domain: "audience_scope",
    },
    {
      label: "audience-rule explicit no-disclosure boundary",
      kind: "audience_rule",
      type: "boundary",
      critical_domain: "explicit_no_disclosure",
    },
    {
      label: "audience-rule audience scope rule",
      kind: "audience_rule",
      type: "rule",
      critical_domain: "audience_scope",
    },
    {
      label: "boundary safety rule",
      kind: "boundary",
      type: "rule",
      critical_domain: "safety",
    },
  ] as const)("keeps legitimate critical classification: $label", (input) => {
    expect(
      normalizeCommitmentClassification({
        kind: input.kind,
        type: input.type,
        enforcement_class: "critical",
        critical_domain: input.critical_domain,
      }),
    ).toEqual({
      enforcement_class: "critical",
      critical_domain: input.critical_domain,
      downgraded_from: null,
      downgrade_reason: null,
    });
  });
});
