# Similarity calibration

Borg resolves its 23 similarity gates through `similarityThresholds(config)`.
Defaults live in `similarity.profiles`, keyed by the exact embedding model ID;
`similarity.overrides` supplies optional values that win over the selected profile.
Each profile is complete. The accessor returns a frozen typed object.

Library open uses the **effective embedding client identity validated by the bank
guard**, including when an injected client differs from `config.json`. The sidecar
logs its effective profile at startup; each tenant library open logs its own model,
profile, fallback flag, and applied overrides. Unknown models use the Qwen profile
and log a warning naming the unknown model. Dimension alone never selects a profile.

## Shipped profiles

`generative-apis/qwen3-embedding-8b` preserves every threshold from `fb25a5c8`.
`scw/bge-m3` changes only these seven values, rounded to two decimals from Phase A
measurements on the production `team-agent-ai` bank:

| Key | Qwen | BGE-M3 |
| --- | ---: | ---: |
| `consolidationSimilarity` | 0.82 | 0.76 |
| `consolidationDiameter` | 0.18 | 0.24 |
| `consolidationTemporalBypass` | 0.97 | 0.94 |
| `reflectionGoalAndTagGrouping` | 0.82 | 0.76 |
| `semanticDuplicateReview` | 0.90 | 0.87 |
| `semanticExtractionDuplicate` | 0.88 | 0.84 |
| `reflectionInsightDuplicate` | 0.88 | 0.84 |

The bank contained 1,321 episodes, 2,527 semantic nodes, and 32 open questions.
Open-question gates retain Qwen values because support is weak. Actions, skills,
and observed events were empty across production banks, so their gates also stay
unchanged. The 0.01 retrieval floors lie below the observed range and remain 0.01.
Episode pairs are only a proxy for reflection's goal/tag comparisons; review the
measurement caveats before extending this calibration.

`consolidationDiameter` is **1 − cosine**, not cosine. `recallAbstain` is the
existing **fused episode raw score**, not cosine: nonpositive values disable it,
and values above 1 are valid. No cosine calibration is applied to it. The retrieval
scoring weight `similarity: 0.7` is separate and unchanged.

## Overrides

For example, in a tenant's `config.json`:

```json
{
  "similarity": {
    "overrides": {
      "semanticDuplicateReview": 0.89,
      "recallAbstain": 1.17
    }
  }
}
```

Existing environment variables populate `similarity.overrides` and take precedence
over the tenant file:

| Environment variable | Key |
| --- | --- |
| `BORG_GENERATION_EVIDENCE_LEDGER_ACTION_THREAD_SIMILARITY_THRESHOLD` | `actionThread` |
| `BORG_PROCEDURAL_SKILL_SELECTION_MIN_SIMILARITY` | `skillSelection` |
| `BORG_OFFLINE_CONSOLIDATOR_SIMILARITY_THRESHOLD` | `consolidationSimilarity` |
| `BORG_OFFLINE_REFLECTOR_GOAL_SIMILARITY_THRESHOLD` | `reflectionGoalAndTagGrouping` |
| `BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_DEDUP_THRESHOLD` | `skillSynthesisDuplicate` |
| `BORG_RECALL_ABSTAIN_THRESHOLD` | `recallAbstain` |

Explicit values at the old config paths remain accepted as deprecated aliases;
they no longer supply defaults. Precedence is environment overrides, then
`similarity.overrides` in the file, then explicit legacy values, then the profile.
Use the new section for new configuration.

## Re-measure and add a model

Follow the [measurement pod recipe](../scripts/similarity-distributions/README.md#run-inside-the-pod)
on a quiescent tenant **copy** containing current and `lancedb.prev-<N>` vectors.
It sets the Paketo Node PATH and a writable cache, and runs both `--vectors current`
and `--vectors prev` with the same bank, seed, sample, and output directory.
The script reads stored vectors only; it never calls the embedding gateway.

Read `summary.md` and `report.json` under the bank's output directory. Compare
nearest-neighbor/random-pair distributions, duplicate examples, family strata,
population counts, and proposal caveats. Percentile matches preserve a population
rank; they are candidates for human review, not proven duplicate decisions.
The current proposal mapper is specifically Qwen-4096 → BGE-M3-1024. For another
model it still measures geometry, but intentionally withholds numeric proposals;
extend and test that explicit model/dimension mapping when calibrating a new pair.

Add the exact new model ID with a **complete** set of keys in
`src/config/similarity.ts` (or `similarity.profiles` in the tenant file). Start with
a full copy of the Qwen profile and change only values supported by reviewed
measurements. Add resolution/value tests and document the bank populations and
limitations. Use `similarityThresholds` for new gates and extend every profile;
do not add runtime threshold literals outside this module.
