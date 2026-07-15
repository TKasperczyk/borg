import { describe, expect, it } from "vitest";

import {
  memoryCommitmentExtractionBudgetFromEnv,
  memoryCommitmentExtractionEnabledFromEnv,
} from "./memory-commitment-extraction.js";

describe("memory sidecar commitment extraction config", () => {
  it("defaults extraction on and accepts explicit boolean kill-switch values", () => {
    expect(memoryCommitmentExtractionEnabledFromEnv({})).toBe(true);
    expect(
      memoryCommitmentExtractionEnabledFromEnv({
        BORG_MEMORY_COMMITMENT_EXTRACTION_ENABLED: "true",
      }),
    ).toBe(true);
    expect(
      memoryCommitmentExtractionEnabledFromEnv({
        BORG_MEMORY_COMMITMENT_EXTRACTION_ENABLED: "1",
      }),
    ).toBe(true);
    expect(
      memoryCommitmentExtractionEnabledFromEnv({
        BORG_MEMORY_COMMITMENT_EXTRACTION_ENABLED: "false",
      }),
    ).toBe(false);
    expect(
      memoryCommitmentExtractionEnabledFromEnv({
        BORG_MEMORY_COMMITMENT_EXTRACTION_ENABLED: "0",
      }),
    ).toBe(false);
  });

  it("tracks without a cap by default and validates an optional positive budget", () => {
    expect(memoryCommitmentExtractionBudgetFromEnv({})).toBeNull();
    expect(
      memoryCommitmentExtractionBudgetFromEnv({
        BORG_MEMORY_COMMITMENT_EXTRACTION_BUDGET: "4096",
      }),
    ).toBe(4096);
    expect(() =>
      memoryCommitmentExtractionBudgetFromEnv({
        BORG_MEMORY_COMMITMENT_EXTRACTION_BUDGET: "0",
      }),
    ).toThrow("must be a positive integer");
  });

  it("rejects ambiguous kill-switch values", () => {
    expect(() =>
      memoryCommitmentExtractionEnabledFromEnv({
        BORG_MEMORY_COMMITMENT_EXTRACTION_ENABLED: "yes",
      }),
    ).toThrow("must be true/false or 1/0");
  });
});
