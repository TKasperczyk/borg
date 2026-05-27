import { readdirSync, readFileSync, statSync } from "node:fs";
import { relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { EXTRACTOR_MAX_TOKEN_LLM_LABELS, isExtractorMaxTokenLlmLabel } from "./extractor-labels.js";

const COGNITION_ROOT = resolve(fileURLToPath(new URL("../", import.meta.url)));

const EXTRACTOR_MAX_TOKEN_CONSUMER_LABELS = {
  "actions/action-state-extractor.ts": ["action-state-extractor", "action_state_extractor"],
  "commitments/corrective-preference-extractor.ts": [
    "corrective-preference-extractor",
    "corrective_preference_extractor",
  ],
  "creator-directives/extractor.ts": ["creator-directive-extractor", "creator_directive_extractor"],
  "frame-anomaly/classifier.ts": ["frame-anomaly-classifier", "frame_anomaly_classifier"],
  "generation/closure-loop.ts": ["closure-loop-classifier", "closure_loop_classifier"],
  "goals/goal-promotion-extractor.ts": ["goal-promotion-extractor", "goal_promotion_extractor"],
  "perception/entity-extractor.ts": ["entity_extractor", "perception-entity-fallback"],
  "perception/mode-detector.ts": ["mode_detector", "perception-mode-fallback"],
  "perception/temporal-cue.ts": ["perception-temporal-cue", "temporal_cue_extractor"],
  "procedural/context-extractor.ts": ["procedural-context", "procedural_context_extractor"],
  "turn-action/pending-action-judge.ts": ["pending-action-judge", "pending_action_judge"],
} as const satisfies Record<string, readonly string[]>;

function sourceFiles(dir: string): string[] {
  const files: string[] = [];

  for (const entry of readdirSync(dir)) {
    const path = resolve(dir, entry);
    const stat = statSync(path);

    if (stat.isDirectory()) {
      files.push(...sourceFiles(path));
      continue;
    }

    if (entry.endsWith(".ts") && !entry.endsWith(".test.ts")) {
      files.push(path);
    }
  }

  return files;
}

function extractorMaxTokenConsumerFiles(): string[] {
  return sourceFiles(COGNITION_ROOT)
    .filter((path) =>
      readFileSync(path, "utf8").includes("max_tokens: EXTRACTOR_MAX_TOKENS_DEFAULT"),
    )
    .map((path) => relative(COGNITION_ROOT, path))
    .sort((left, right) => left.localeCompare(right));
}

describe("extractor max-token labels", () => {
  it("registers every EXTRACTOR_MAX_TOKENS_DEFAULT consumer label", () => {
    expect(extractorMaxTokenConsumerFiles()).toEqual(
      Object.keys(EXTRACTOR_MAX_TOKEN_CONSUMER_LABELS).sort((left, right) =>
        left.localeCompare(right),
      ),
    );

    for (const labels of Object.values(EXTRACTOR_MAX_TOKEN_CONSUMER_LABELS)) {
      for (const label of labels) {
        expect(isExtractorMaxTokenLlmLabel(label), label).toBe(true);
      }
    }
  });

  it("keeps registry entries unique", () => {
    expect(new Set(EXTRACTOR_MAX_TOKEN_LLM_LABELS).size).toBe(
      EXTRACTOR_MAX_TOKEN_LLM_LABELS.length,
    );
  });
});
