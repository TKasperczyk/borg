import type {
  CompletedModelEvaluationResult,
  EmbeddingAbResults,
  GoldQuestionRank,
} from "./types.js";

function markdownCell(value: string): string {
  return value.replace(/\r?\n/g, " ").replace(/\|/g, "\\|");
}

function formatNumber(value: number | null, digits = 3): string {
  return value === null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function formatPercent(value: number | null): string {
  return value === null || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatLatency(value: number | null): string {
  return value === null ? "—" : `${value.toFixed(1)} ms`;
}

function crossRanks(ranks: readonly { episode_id: string; target_rank: number | null }[]): string {
  return ranks.map((rank) => `${rank.episode_id}:${rank.target_rank ?? "—"}`).join(", ");
}

function goldRankByIndex(ranks: readonly GoldQuestionRank[]): Map<number, GoldQuestionRank> {
  return new Map(ranks.map((rank) => [rank.index, rank]));
}

export function renderEmbeddingAbReport(results: EmbeddingAbResults): string {
  const completedModels = results.models.filter(
    (model): model is CompletedModelEvaluationResult => model.status === "completed",
  );
  const initializationFailures = results.models.filter(
    (model) => model.status === "initialization_failed",
  );
  const lines: string[] = [
    "# Embedding model A/B evaluation",
    "",
    `Generated: ${results.generated_at}`,
    "",
    "## Run summary",
    "",
    `Evaluated ${results.bank.active_episode_count} active/effectively-visible episodes (${results.bank.inactive_episode_count} inactive or superseded rows excluded). The source bank was read through Borg's episodic repository; all generated vectors and LLM artifacts were written under \`${markdownCell(results.inputs.out_dir)}\`.`,
    "",
    "| Setting | Value |",
    "| --- | --- |",
    `| Requested models | ${results.inputs.models.map(markdownCell).join("<br>")} |`,
    `| Participating models | ${results.comparison_corpus.participating_models.map(markdownCell).join("<br>") || "none"} |`,
    `| Source embedding dimensions | ${results.bank.source_embedding_dimensions} |`,
    `| Active corpus SHA-256 | \`${results.bank.active_corpus_sha256}\` |`,
    `| Common candidate corpus | ${results.comparison_corpus.common_episode_count}/${results.comparison_corpus.active_bank_episode_count} active episodes |`,
    `| Comparable to full-bank recall | ${results.comparison_corpus.comparative_metrics_comparable_to_full_bank_recall ? "yes" : "no"} |`,
    `| Gold generator | ${markdownCell(results.gateway.gold_generation_model ?? "disabled")} |`,
    `| Gold sample | ${results.gold_set.generated_question_count}/${results.gold_set.sampled_size} questions generated (requested ${results.gold_set.requested_size}) |`,
    `| Real queries | ${results.inputs.queries.length} |`,
    `| Judge | ${markdownCell(results.judging?.model ?? "disabled")} |`,
  ];

  if (!results.comparison_corpus.coverage_complete) {
    lines.push(
      "",
      `Coverage is incomplete: all comparative rankings use the ${results.comparison_corpus.common_episode_count}-episode intersection embedded by every participating model. Recall and other comparative metrics are **not comparable to full-bank recall**.`,
    );
  }

  if (initializationFailures.length > 0) {
    lines.push(
      "",
      `Partial run: ${initializationFailures.length} model initialization failure(s) were captured; completed models remain reported.`,
      "",
    );
    for (const model of initializationFailures) {
      lines.push(
        `- ${markdownCell(model.model)}: ${markdownCell(model.initialization_error.name)} — ${markdownCell(model.initialization_error.message)}`,
      );
    }
  }

  lines.push(
    "",
    "## Synthetic gold retrieval",
    "",
    "| Model | Status | Episode vector coverage | Gold query coverage | Recall@1 | Recall@3 | Recall@10 | MRR |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
  );

  for (const model of results.models) {
    if (model.status === "initialization_failed") {
      lines.push(
        `| ${markdownCell(model.model)} | initialization failed (${markdownCell(model.initialization_error.name)}) | — | — | — | — | — | — |`,
      );
      continue;
    }
    lines.push(
      `| ${markdownCell(model.model)} | completed | ${model.episode_vectors.available}/${model.episode_vectors.requested} | ${model.gold_question_vectors.available}/${model.gold_question_vectors.requested} | ${formatPercent(model.gold.metrics.recall_at_1)} | ${formatPercent(model.gold.metrics.recall_at_3)} | ${formatPercent(model.gold.metrics.recall_at_10)} | ${formatNumber(model.gold.metrics.mrr)} |`,
    );
  }

  if (results.gold_set.questions.length > 0) {
    const rankMaps = results.models.map((model) =>
      model.status === "completed" ? goldRankByIndex(model.gold.per_question) : null,
    );
    lines.push("", "### Per-question source ranks", "");
    lines.push(
      `| Gold item | Source episode | Question status | ${results.models
        .map((model) => `${markdownCell(model.model)} rank`)
        .join(" | ")} |`,
    );
    lines.push(`| ---: | --- | --- | ${results.models.map(() => "---:").join(" | ")} |`);
    for (const question of results.gold_set.questions) {
      const status =
        question.question === null ? `generation error: ${question.error?.name ?? "error"}` : "ok";
      lines.push(
        `| G${question.index} | ${markdownCell(question.source_episode_id)} | ${markdownCell(status)} | ${rankMaps
          .map((ranks) => String(ranks?.get(question.index)?.rank ?? "—"))
          .join(" | ")} |`,
      );
    }
  } else {
    lines.push("", "No synthetic gold questions were requested or generated.");
  }

  lines.push("", "## Real-query replay", "");
  if (results.inputs.queries.length === 0) {
    lines.push("No real queries were supplied.");
  } else {
    lines.push("| Query | Model | Top-10 episode IDs (cosine order) |", "| ---: | --- | --- |");
    for (const model of completedModels) {
      for (const query of model.real_queries) {
        const topTen =
          query.error === undefined
            ? query.top_10.map((candidate) => candidate.episode_id).join(", ")
            : `embedding error: ${query.error.name}`;
        lines.push(
          `| Q${query.query_index} | ${markdownCell(model.model)} | ${markdownCell(topTen)} |`,
        );
      }
    }

    if (results.replay_comparisons.length === 0) {
      lines.push("", "At least two models are required for pairwise replay comparisons.");
    } else {
      lines.push(
        "",
        "| Query | Model pair | Top-10 overlap | Left ranks of right top-3 | Right ranks of left top-3 |",
        "| ---: | --- | ---: | --- | --- |",
      );
      for (const comparison of results.replay_comparisons) {
        lines.push(
          `| Q${comparison.query_index} | ${markdownCell(comparison.left_model)} ↔ ${markdownCell(comparison.right_model)} | ${comparison.top_10_overlap_count}/${comparison.top_10_overlap_denominator} (${formatPercent(comparison.top_10_overlap_ratio)}) | ${markdownCell(crossRanks(comparison.left_ranks_of_right_top_3))} | ${markdownCell(crossRanks(comparison.right_ranks_of_left_top_3))} |`,
        );
      }
    }
  }

  if (results.judging !== null) {
    lines.push(
      "",
      "## LLM relevance judgment",
      "",
      `Judge model: ${markdownCell(results.judging.model)}. Each query's union of model top-5 candidates was rated once, then the same ratings were aggregated for each model.`,
      "",
      "| Model | Rated top-5 results | Mean relevance (0-3) |",
      "| --- | ---: | ---: |",
    );
    for (const model of results.judging.per_model) {
      lines.push(
        `| ${markdownCell(model.model)} | ${model.rated_result_count} | ${formatNumber(model.mean_relevance)} |`,
      );
    }
    const judgeErrors = results.judging.queries.filter((query) => query.error !== undefined);
    if (judgeErrors.length > 0) {
      lines.push("", `Judge failures: ${judgeErrors.length}/${results.judging.queries.length}.`);
    }
  }

  lines.push(
    "",
    "## Embedding call latency",
    "",
    "Timings cover gateway HTTP attempts made in this run, including failed and timed-out attempts. Cache hits make no gateway call, so use a fresh output directory when measuring latency again.",
    "",
    "| Model | Dimensions | Calls | Cache hits (episode/gold/real) | p50 | p95 | Max | Errors | Timeouts |",
    "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
  );
  for (const model of results.models) {
    if (model.status === "initialization_failed") {
      lines.push(
        `| ${markdownCell(model.model)} | — | ${model.latency.call_count} | — | ${formatLatency(model.latency.p50_ms)} | ${formatLatency(model.latency.p95_ms)} | ${formatLatency(model.latency.max_ms)} | ${model.latency.error_count} | ${model.latency.timeout_count} |`,
      );
      continue;
    }
    lines.push(
      `| ${markdownCell(model.model)} | ${model.dimensions} | ${model.latency.call_count} | ${model.episode_vectors.cache_hits}/${model.gold_question_vectors.cache_hits}/${model.real_query_vectors.cache_hits} | ${formatLatency(model.latency.p50_ms)} | ${formatLatency(model.latency.p95_ms)} | ${formatLatency(model.latency.max_ms)} | ${model.latency.error_count} | ${model.latency.timeout_count} |`,
    );
  }

  const embeddingFailures = completedModels.reduce(
    (count, model) =>
      count +
      model.episode_vectors.failed +
      model.gold_question_vectors.failed +
      model.real_query_vectors.failed,
    0,
  );
  if (embeddingFailures > 0) {
    lines.push(
      "",
      `Incomplete embedding coverage: ${embeddingFailures} item failure(s). Null gold ranks count as misses; inspect \`results.json\` for per-item errors and per-call timings.`,
    );
  }

  lines.push(
    "",
    "## Verdict template",
    "",
    "Given synthetic recall (___), real-query top-10 agreement and cross-ranks (___), optional judged relevance (___), and latency/stall behavior (___), the evidence [does / does not] support migrating the sidecar bank from ___ to ___. The main remaining uncertainty is ___, so the next action is ___.",
    "",
  );

  return lines.join("\n");
}
