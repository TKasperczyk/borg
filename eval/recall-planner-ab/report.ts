import type {
  RecallPlannerAbResults,
  RecallPlannerCaseRun,
  RecallPlannerLaneRank,
} from "./types.js";

function markdownCell(value: string): string {
  return value.replace(/\r?\n/g, " ").replace(/\|/g, "\\|");
}

function boundedCell(value: string, max = 180): string {
  const normalized = markdownCell(value);
  return normalized.length <= max ? normalized : `${normalized.slice(0, max)}…`;
}

function formatNumber(value: number | null, digits = 3): string {
  return value === null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function formatPercent(value: number | null): string {
  return value === null || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatLatency(value: number | null): string {
  return value === null || !Number.isFinite(value) ? "—" : `${value.toFixed(1)} ms`;
}

function laneLabel(lane: RecallPlannerLaneRank): string {
  const ranks = Object.entries(lane.expected_ranks)
    .map(([episodeId, rank]) => `${episodeId}:${rank ?? "—"}`)
    .join(",");
  return `${lane.intent_kind}/${lane.intent_id} [${ranks}]`;
}

function runStatus(run: RecallPlannerCaseRun): string {
  if (run.status === "failed") {
    return `failed: ${run.error.name}`;
  }
  return run.degradations.length === 0 ? "completed" : `degraded (${run.degradations.length})`;
}

export function renderRecallPlannerAbReport(results: RecallPlannerAbResults): string {
  const configurationById = new Map(
    results.configurations.map((configuration) => [configuration.id, configuration]),
  );
  const lines: string[] = [
    "# Recall query planner A/B evaluation",
    "",
    `Generated: ${results.generated_at}`,
    "",
    "## Run summary",
    "",
    "The copied bank was queried through Borg's production `RetrievalPipeline` with read-only SQLite/LanceDB handles and retrieval accounting disabled. The baseline is intentionally the raw FOCUS-blob vector lane only, with no planner, exact-term, typed, or recent candidate lane.",
    "",
    "| Setting | Value |",
    "| --- | --- |",
    `| Source bank | \`${markdownCell(results.inputs.data_dir)}\` |`,
    `| Cases | ${results.cases.length} (${results.generation.generated} generated this case set) |`,
    `| Active episodes | ${results.bank.active_episode_count}/${results.bank.all_episode_count} |`,
    `| Embedding model | ${markdownCell(results.inputs.embedding_model)} (${results.bank.embedding_dimensions}d) |`,
    `| Bank-configured embedding model | ${markdownCell(results.bank.configured_embedding_model)} |`,
    `| Recall planner model | ${markdownCell(results.inputs.planner_model)} |`,
    `| Configurations | ${results.configurations.map((item) => markdownCell(item.label)).join("<br>")} |`,
    `| Judge | ${markdownCell(results.judging?.model ?? "disabled")} |`,
    `| Active corpus SHA-256 | \`${results.bank.active_corpus_sha256}\` |`,
  ];

  if (results.bank.missing_expected_episode_ids.length > 0) {
    lines.push(
      "",
      `Warning: ${results.bank.missing_expected_episode_ids.length} expected episode ID(s) are absent from the active/effectively-visible corpus and therefore count as misses: ${results.bank.missing_expected_episode_ids.map(markdownCell).join(", ")}.`,
    );
  }

  if (results.generation.requested > 0) {
    const generationFailures = results.generation.records.filter((record) => record.case === null);
    lines.push(
      "",
      `Synthetic case generation produced ${results.generation.generated}/${results.generation.requested} requested cases using ${markdownCell(results.gateway.case_generation_model ?? "no model")}. ${generationFailures.length} generation failure(s) were retained in \`results.json\`. Prompt version: \`${results.generation.prompt_version}\`.`,
    );
  }

  lines.push(
    "",
    "## Retrieval metrics",
    "",
    "A case is a hit when any expected episode appears at or above the cutoff; MRR uses the best expected rank. Failed cases and absent expected IDs remain misses.",
    "",
    "| Configuration | Completed | Failed | Degraded | Recall@1 | Recall@3 | Recall@10 | MRR |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  );

  for (const summary of results.summaries) {
    const configuration = configurationById.get(summary.configuration_id);
    lines.push(
      `| ${markdownCell(configuration?.label ?? summary.configuration_id)} | ${summary.completed_case_count} | ${summary.failed_case_count} | ${summary.degraded_case_count} | ${formatPercent(summary.metrics.recall_at_1)} | ${formatPercent(summary.metrics.recall_at_3)} | ${formatPercent(summary.metrics.recall_at_10)} | ${formatNumber(summary.metrics.mrr)} |`,
    );
  }

  lines.push(
    "",
    "## Per-case results",
    "",
    "Lane ranks are the candidate order emitted by `retrieval.intent_candidates`; final rank is the post-fusion, post-MMR output order.",
    "",
    "| Case | Configuration | Status | Expected final rank | Lane expected ranks | Resolved query | Top-5 episode IDs |",
    "| --- | --- | --- | ---: | --- | --- | --- |",
  );

  for (const item of results.cases) {
    for (const configuration of results.configurations) {
      const run = results.runs.find(
        (entry) => entry.case_id === item.id && entry.configuration_id === configuration.id,
      );
      if (run === undefined) {
        continue;
      }
      lines.push(
        `| ${markdownCell(item.id)} | ${markdownCell(configuration.label)} | ${markdownCell(runStatus(run))} | ${run.best_expected_rank ?? "—"} | ${boundedCell(run.lane_ranks.map(laneLabel).join("; "), 260) || "—"} | ${boundedCell(run.planner_output?.resolved_query ?? "—")} | ${
          run.top_10
            .slice(0, 5)
            .map((candidate) => markdownCell(candidate.episode_id))
            .join(", ") || "—"
        } |`,
      );
    }
  }

  lines.push(
    "",
    "## Planner latency and embedding load",
    "",
    "Planner latency spans the traced recall-expansion call through parsed-plan completion (or degradation). Logical embedding calls include cache hits; gateway attempts and latency count physical HTTP attempts, including retries. Persistent cache hits intentionally make reruns cheaper.",
    "",
    "| Configuration | Planner cache hit/miss | Planner p50 | Planner p95 | Logical embedding calls/query | Gateway attempts/query | Gateway p95 | Embedding errors | Embedding timeouts |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  );

  for (const summary of results.summaries) {
    const configuration = configurationById.get(summary.configuration_id);
    lines.push(
      `| ${markdownCell(configuration?.label ?? summary.configuration_id)} | ${summary.planner_latency.cache_hit_count}/${summary.planner_latency.cache_miss_count} | ${formatLatency(summary.planner_latency.p50_ms)} | ${formatLatency(summary.planner_latency.p95_ms)} | ${formatNumber(summary.embedding.logical_calls_per_query, 2)} | ${formatNumber(summary.embedding.gateway_attempts_per_query, 2)} | ${formatLatency(summary.embedding.gateway_latency.p95_ms)} | ${summary.embedding.gateway_error_count} | ${summary.embedding.gateway_timeout_count} |`,
    );
  }

  if (results.judging !== null) {
    lines.push(
      "",
      "## Optional top-5 relevance judgment",
      "",
      "For each case, the union of every configuration's top five was rated once with the case CONTEXT, FOCUS, identity handles, and owner recent activity, then ratings were aggregated back over each configuration's own top five.",
      "",
      "| Configuration | Rated results | Mean relevance (0-3) |",
      "| --- | ---: | ---: |",
    );
    for (const summary of results.judging.per_configuration) {
      lines.push(
        `| ${markdownCell(configurationById.get(summary.configuration_id)?.label ?? summary.configuration_id)} | ${summary.rated_result_count} | ${formatNumber(summary.mean_relevance)} |`,
      );
    }
    const failures = results.judging.cases.filter((item) => item.error !== undefined).length;
    if (failures > 0) {
      lines.push("", `Judge failures: ${failures}/${results.judging.cases.length}.`);
    }
  }

  lines.push(
    "",
    "## Verdict template",
    "",
    "Given expected-episode Recall@1/3/10 and MRR (___), per-case lane/final-rank changes (___), resolved-query quality (___), optional top-5 judged relevance (___), planner latency (___), and embedding call/stall behavior (___), the evidence [does / does not] support using semantic variant count ___ instead of ___. The main remaining uncertainty is ___, so the next action is ___.",
    "",
  );

  return lines.join("\n");
}
