import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { THRESHOLDS } from "../../scripts/similarity-distributions/inventory.js";
import { configSchema, loadConfig } from "./index.js";
import {
  BGE_SIMILARITY_MODEL,
  DEFAULT_SIMILARITY_PROFILES,
  QWEN_SIMILARITY_MODEL,
  logSimilarityProfile,
  similarityThresholds,
  type SimilarityThresholds,
} from "./similarity.js";

const inventoryKeys: Record<string, keyof SimilarityThresholds> = {
  action_persistence_duplicate: "actionPersistenceDuplicate",
  open_question_recall: "openQuestionRecall",
  commitment_evidence: "commitmentEvidence",
  action_thread: "actionThread",
  skill_selection: "skillSelection",
  consolidation_similarity: "consolidationSimilarity",
  consolidation_diameter: "consolidationDiameter",
  consolidation_temporal_bypass: "consolidationTemporalBypass",
  reflection_goal_and_tag_grouping: "reflectionGoalAndTagGrouping",
  skill_synthesis_duplicate: "skillSynthesisDuplicate",
  ruminator_duplicate: "ruminatorDuplicate",
  semantic_recall: "semanticRecall",
  semantic_revision: "semanticRevision",
  generation_repeated_input: "generationRepeatedInput",
  goal_promotion_duplicate: "goalPromotionDuplicate",
  open_question_duplicate_backstop: "openQuestionDuplicateBackstop",
  observed_event_topic: "observedEventTopic",
  semantic_duplicate_review: "semanticDuplicateReview",
  procedural_evidence_cluster: "proceduralEvidenceCluster",
  semantic_extraction_duplicate: "semanticExtractionDuplicate",
  pending_action_merge: "pendingActionMerge",
  reflection_insight_duplicate: "reflectionInsightDuplicate",
  BORG_RECALL_ABSTAIN_THRESHOLD: "recallAbstain",
};
const directories: string[] = [];
afterEach(() => {
  vi.restoreAllMocks();
  for (const dir of directories.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("similarity profiles", () => {
  it.each(THRESHOLDS)("preserves the former $id gate under Qwen", (gate) => {
    const thresholds = similarityThresholds(
      configSchema.parse({
        embedding: { model: QWEN_SIMILARITY_MODEL },
      }),
    );
    expect(thresholds[inventoryKeys[gate.id]!]).toBe(gate.value);
    expect(new Set(Object.values(inventoryKeys))).toEqual(new Set(Object.keys(thresholds)));
  });

  it("changes exactly the seven measured BGE gates and returns frozen values", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const qwen = similarityThresholds({ embedding: { model: QWEN_SIMILARITY_MODEL } });
    const bge = similarityThresholds(
      configSchema.parse({
        embedding: { model: BGE_SIMILARITY_MODEL },
      }),
    );
    expect(bge).toEqual({
      ...qwen,
      consolidationSimilarity: 0.76,
      consolidationDiameter: 0.24,
      consolidationTemporalBypass: 0.94,
      reflectionGoalAndTagGrouping: 0.76,
      semanticDuplicateReview: 0.87,
      semanticExtractionDuplicate: 0.84,
      reflectionInsightDuplicate: 0.84,
    });
    expect(Object.isFrozen(bge)).toBe(true);
    expect(Reflect.set(bge, "consolidationSimilarity", 0.1)).toBe(false);
    expect(warn).not.toHaveBeenCalled();
  });

  it("falls back to Qwen with a warning naming an unknown model", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const config = configSchema.parse({ embedding: { model: "test/unknown-similarity-model" } });
    expect(similarityThresholds(config)).toEqual(
      DEFAULT_SIMILARITY_PROFILES[QWEN_SIMILARITY_MODEL],
    );
    similarityThresholds(config);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0]?.[0]).toContain('model="test/unknown-similarity-model"');
    expect(warn.mock.calls[0]?.[0]).toContain(`profile="${QWEN_SIMILARITY_MODEL}" fallback=true`);
  });

  it("applies explicit overrides after complete custom profiles and legacy settings", () => {
    const config = configSchema.parse({
      embedding: { model: "test/custom" },
      similarity: {
        profiles: {
          "test/custom": {
            ...DEFAULT_SIMILARITY_PROFILES[QWEN_SIMILARITY_MODEL],
            actionThread: 0.6,
          },
        },
        overrides: { actionThread: 0, semanticRecall: 0.2 },
      },
      generation: { evidenceLedger: { actionThreadSimilarityThreshold: 0.7 } },
      procedural: { skillSelectionMinSimilarity: 0.4 },
    });
    expect(similarityThresholds(config)).toMatchObject({
      actionThread: 0,
      skillSelection: 0.4,
      semanticRecall: 0.2,
    });
    expect(config.similarity.profiles[BGE_SIMILARITY_MODEL]).toBeDefined();
    expect(similarityThresholds(config)).not.toBe(config.similarity.profiles["test/custom"]);
  });

  it("rejects incomplete profiles and misspelled override keys", () => {
    expect(
      configSchema.safeParse({
        similarity: { profiles: { "test/incomplete": { actionThread: 0.2 } } },
      }).success,
    ).toBe(false);
    expect(
      configSchema.safeParse({ similarity: { overrides: { actionThred: 0.2 } } }).success,
    ).toBe(false);
  });

  it("preserves all explicitly configured legacy gates without old defaults masking BGE", () => {
    const config = configSchema.parse({
      embedding: { model: BGE_SIMILARITY_MODEL },
      generation: { evidenceLedger: { actionThreadSimilarityThreshold: 0.1 } },
      procedural: { skillSelectionMinSimilarity: 0.2 },
      offline: {
        consolidator: {
          similarityThreshold: 0.3,
          maxClusterDiameter: 0.4,
          highSimilarityTemporalBypassThreshold: 0.5,
        },
        reflector: { goalSimilarityThreshold: 0.6 },
        proceduralSynthesizer: { dedupThreshold: 0.7 },
        ruminator: { duplicateSimilarityThreshold: 0.8 },
      },
    });
    expect(similarityThresholds(config)).toMatchObject({
      actionThread: 0.1,
      skillSelection: 0.2,
      consolidationSimilarity: 0.3,
      consolidationDiameter: 0.4,
      consolidationTemporalBypass: 0.5,
      reflectionGoalAndTagGrouping: 0.6,
      skillSynthesisDuplicate: 0.7,
      ruminatorDuplicate: 0.8,
      semanticExtractionDuplicate: 0.84,
    });
  });

  it("keeps environment overrides above tenant config and the selected profile", () => {
    const dir = mkdtempSync(join(tmpdir(), "similarity-config-"));
    directories.push(dir);
    writeFileSync(
      join(dir, "config.json"),
      JSON.stringify({
        embedding: { model: BGE_SIMILARITY_MODEL },
        similarity: { overrides: { actionThread: 0.2, recallAbstain: 0.4 } },
      }),
    );
    const config = loadConfig({
      dataDir: dir,
      env: {
        BORG_GENERATION_EVIDENCE_LEDGER_ACTION_THREAD_SIMILARITY_THRESHOLD: "0.91",
        BORG_PROCEDURAL_SKILL_SELECTION_MIN_SIMILARITY: "0.61",
        BORG_OFFLINE_REFLECTOR_GOAL_SIMILARITY_THRESHOLD: "0.81",
        BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_DEDUP_THRESHOLD: "0.89",
        BORG_OFFLINE_CONSOLIDATOR_SIMILARITY_THRESHOLD: "0.83",
        BORG_RECALL_ABSTAIN_THRESHOLD: "1.17",
      },
    });
    expect(config.similarity.overrides).toEqual({
      actionThread: 0.91,
      skillSelection: 0.61,
      reflectionGoalAndTagGrouping: 0.81,
      skillSynthesisDuplicate: 0.89,
      consolidationSimilarity: 0.83,
      recallAbstain: 1.17,
    });
    expect(similarityThresholds(config)).toMatchObject(config.similarity.overrides);
  });

  it.each(["invalid", "Infinity", "0", "-1", "1.5"])(
    "retains fused abstention env semantics (%s)",
    (raw) => {
      const dir = mkdtempSync(join(tmpdir(), "similarity-abstain-"));
      directories.push(dir);
      const config = loadConfig({ dataDir: dir, env: { BORG_RECALL_ABSTAIN_THRESHOLD: raw } });
      expect(config.similarity.overrides.recallAbstain).toBe(
        Number.isFinite(Number(raw)) ? Number(raw) : 0,
      );
    },
  );

  it.each(["borg open", "borg memory sidecar"])(
    "logs one startup profile line for %s",
    (context) => {
      const info = vi.spyOn(console, "info").mockImplementation(() => {});
      const config = configSchema.parse({
        embedding: { model: BGE_SIMILARITY_MODEL },
        similarity: { overrides: { recallAbstain: 1.17 } },
      });
      logSimilarityProfile(config, context);
      similarityThresholds(config);
      expect(info).toHaveBeenCalledExactlyOnceWith(
        `${context}: similarity model="${BGE_SIMILARITY_MODEL}" profile="${BGE_SIMILARITY_MODEL}" fallback=false overrides={"recallAbstain":1.17}`,
      );
    },
  );

  it("logs fallback at startup without a second accessor warning", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const config = configSchema.parse({
      embedding: { model: "test/unknown-at-startup" },
      similarity: { overrides: { actionThread: 0.4 } },
    });
    logSimilarityProfile(config, "borg open");
    expect(similarityThresholds(config).actionThread).toBe(0.4);
    expect(warn).toHaveBeenCalledExactlyOnceWith(
      `borg open: similarity model="test/unknown-at-startup" profile="${QWEN_SIMILARITY_MODEL}" fallback=true overrides={"actionThread":0.4}`,
    );
  });
});
