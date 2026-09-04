/*
 * Retrieval signal report -- measures how much ranking information each
 * retrieval signal actually carries on THIS deployment's corpus.
 *
 * Why this exists: `config.retrieval.attentionWeights.semantic` is fused
 * against a raw, un-normalized cosine similarity (unlike `heat`, which is
 * normalized before fusion). The spread of that cosine is a property of the
 * corpus, not of the code:
 *
 *   - a thematically diverse bank separates candidates widely, so a high
 *     `semantic` weight buys real discrimination;
 *   - a thematically narrow bank separates them by a few hundredths, where the
 *     same weight mostly displaces salience without replacing it.
 *
 * So the same weights can legitimately improve one deployment and degrade
 * another. Run this before changing them, and prefer evidence over a number
 * copied from another bank.
 *
 * Read-only: it parses trace files and touches neither the database nor any
 * running process. It is safe to run against a live deployment.
 *
 * PRIVACY: emits aggregate statistics only -- no episode titles, no query text,
 * no memory content. Output is safe to paste into a shared channel or an issue.
 *
 * Usage:
 *   pnpm retrieval:signal-report -- --data-dir <bank-dir>
 *   pnpm retrieval:signal-report -- --traces <dir-with-turns.jsonl>
 *   pnpm retrieval:signal-report -- --data-dir <dir> --json
 */
import { gunzipSync } from "node:zlib";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

// Lanes that carry a vector_score. The others (known_term, time, recent, ...)
// score similarity 0 by construction, so no semantic reweighting can move them
// -- including them would dilute every statistic below.
const SEMANTIC_LANES = new Set(["raw_text", "semantic_query"]);

// Below this within-query top1..top3 similarity gap, the ordering among the
// leaders is closer to noise than to signal.
const FLAT_SEPARATION_EPSILON = 0.02;
// Median top1..top3 gap under which a similarity-forward weight is hard to justify.
const NARROW_CORPUS_GAP = 0.05;
// ... and over which it clearly is.
const WIDE_CORPUS_GAP = 0.12;

type Candidate = { episode_id: string; score?: number; vector_score?: number };
type IntentEvent = {
  event?: string;
  intent_kind?: string;
  candidates?: Candidate[];
  candidate_texts?: { episode_id?: string; title?: string }[];
};

type Args = {
  tracesDir?: string;
  json: boolean;
};

function parseArgs(argv: readonly string[]): Args {
  const args: Args = { json: false };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--json") args.json = true;
    else if (a === "--traces") args.tracesDir = argv[++i];
    else if (a === "--data-dir") args.tracesDir = join(argv[++i] ?? "", "traces");
  }
  return args;
}

function readTraceLines(dir: string): string[] {
  if (!existsSync(dir)) {
    throw new Error(`traces directory not found: ${dir}`);
  }
  const lines: string[] = [];
  for (const name of readdirSync(dir).sort()) {
    if (!name.startsWith("turns.jsonl")) continue;
    const full = join(dir, name);
    try {
      const raw = name.endsWith(".gz")
        ? gunzipSync(readFileSync(full)).toString("utf8")
        : readFileSync(full, "utf8");
      for (const line of raw.split("\n")) {
        if (line.includes('"retrieval.intent_candidates"')) lines.push(line);
      }
    } catch {
      // A truncated tail on the live file is expected; skip what will not parse.
    }
  }
  return lines;
}

function quantile(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return Number.NaN;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * sorted.length)));
  return sorted[idx] ?? Number.NaN;
}

export type SignalReport = {
  intentsTotal: number;
  intentsSemantic: number;
  semanticSharePct: number;
  similarityMedian: number;
  similarityP10: number;
  similarityP90: number;
  gapTop1ToTop3Median: number;
  spreadTop1ToLastMedian: number;
  flatQueriesPct: number;
  clampSaturationPct: number;
  topOneConcentrationPct: number;
  similarityDisplacementPct: number;
  meanTop3OverlapOutOf3: number;
  verdict: string;
};

export function buildReport(lines: readonly string[]): SignalReport {
  let intentsTotal = 0;
  const events: IntentEvent[] = [];
  for (const line of lines) {
    let parsed: IntentEvent;
    try {
      parsed = JSON.parse(line) as IntentEvent;
    } catch {
      continue;
    }
    if (parsed.event !== "retrieval.intent_candidates") continue;
    intentsTotal += 1;
    const cands = parsed.candidates ?? [];
    if (!SEMANTIC_LANES.has(parsed.intent_kind ?? "") || cands.length < 2) continue;
    events.push(parsed);
  }

  const sims: number[] = [];
  const gaps: number[] = [];
  const spreads: number[] = [];
  const overlaps: number[] = [];
  const topOne = new Map<string, number>();
  let clamped = 0;
  let rows = 0;
  let displaced = 0;
  let flat = 0;

  for (const e of events) {
    const cands = e.candidates ?? [];
    const byScore = [...cands].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    const bySim = [...cands].sort((a, b) => (b.vector_score ?? 0) - (a.vector_score ?? 0));

    for (const c of cands) {
      rows += 1;
      if ((c.score ?? 0) >= 0.999) clamped += 1;
      if (typeof c.vector_score === "number" && c.vector_score > 0) sims.push(c.vector_score);
    }

    const ordered = bySim.map((c) => c.vector_score ?? 0);
    if (ordered.length >= 3) {
      const gap = (ordered[0] ?? 0) - (ordered[2] ?? 0);
      gaps.push(gap);
      if (gap < FLAT_SEPARATION_EPSILON) flat += 1;
    }
    if (ordered.length >= 2) {
      spreads.push((ordered[0] ?? 0) - (ordered[ordered.length - 1] ?? 0));
    }

    const scoreTop = byScore[0]?.episode_id;
    const simTop = bySim[0]?.episode_id;
    if (scoreTop !== undefined) topOne.set(scoreTop, (topOne.get(scoreTop) ?? 0) + 1);
    if (scoreTop !== undefined && simTop !== undefined && scoreTop !== simTop) displaced += 1;

    const a = new Set(byScore.slice(0, 3).map((c) => c.episode_id));
    let shared = 0;
    for (const c of bySim.slice(0, 3)) if (a.has(c.episode_id)) shared += 1;
    overlaps.push(shared);
  }

  const n = events.length;
  const pct = (x: number, d: number) => (d === 0 ? 0 : (100 * x) / d);
  const median = (xs: number[]) =>
    quantile(
      [...xs].sort((x, y) => x - y),
      0.5,
    );
  const sortedSims = [...sims].sort((x, y) => x - y);
  const gapMedian = median(gaps);
  const maxTopOne = [...topOne.values()].reduce((m, v) => Math.max(m, v), 0);

  const verdict =
    n === 0
      ? "No semantic-lane intents found -- nothing to judge. Check the traces path."
      : gapMedian < NARROW_CORPUS_GAP
        ? "NARROW corpus: similarity barely separates candidates here. A similarity-forward `semantic` weight will mostly displace salience without replacing its ranking signal. Prefer a lower `semantic` / higher `heat`."
        : gapMedian > WIDE_CORPUS_GAP
          ? "WIDE corpus: similarity separates candidates well. A similarity-forward `semantic` weight buys real discrimination here."
          : "MIXED corpus: similarity carries moderate signal. Change `semantic` in small steps and re-measure.";

  return {
    intentsTotal,
    intentsSemantic: n,
    semanticSharePct: pct(n, intentsTotal),
    similarityMedian: quantile(sortedSims, 0.5),
    similarityP10: quantile(sortedSims, 0.1),
    similarityP90: quantile(sortedSims, 0.9),
    gapTop1ToTop3Median: gapMedian,
    spreadTop1ToLastMedian: median(spreads),
    flatQueriesPct: pct(flat, gaps.length),
    clampSaturationPct: pct(clamped, rows),
    topOneConcentrationPct: pct(maxTopOne, n),
    similarityDisplacementPct: pct(displaced, n),
    meanTop3OverlapOutOf3:
      overlaps.length === 0 ? 0 : overlaps.reduce((s, v) => s + v, 0) / overlaps.length,
    verdict,
  };
}

function render(r: SignalReport): string {
  const f = (x: number, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : "n/a");
  return [
    "retrieval signal report",
    "=======================",
    `recall intents in traces        : ${r.intentsTotal}`,
    `  semantic-lane (analysable)    : ${r.intentsSemantic} (${f(r.semanticSharePct, 1)}%)`,
    `  other lanes are unaffected by the semantic weight (similarity is 0 there)`,
    "",
    "similarity distribution",
    `  median / p10 / p90            : ${f(r.similarityMedian)} / ${f(r.similarityP10)} / ${f(r.similarityP90)}`,
    "",
    "how well similarity separates candidates (the deciding number)",
    `  median gap  top1 -> top3      : ${f(r.gapTop1ToTop3Median, 4)}`,
    `  median spread top1 -> last    : ${f(r.spreadTop1ToLastMedian, 4)}`,
    `  queries with top3 within ${FLAT_SEPARATION_EPSILON} : ${f(r.flatQueriesPct, 1)}%`,
    "",
    "current-weight behaviour",
    `  candidate scores at 1.0 clamp : ${f(r.clampSaturationPct, 1)}%`,
    `  most frequent top-1 episode   : ${f(r.topOneConcentrationPct, 1)}% of queries`,
    `  top-1 != most-similar episode : ${f(r.similarityDisplacementPct, 1)}%`,
    `  mean top-3 overlap vs pure similarity: ${f(r.meanTop3OverlapOutOf3, 2)}/3`,
    "",
    `verdict: ${r.verdict}`,
  ].join("\n");
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  if (args.tracesDir === undefined) {
    console.error("usage: retrieval-signal-report --data-dir <bank-dir> [--traces <dir>] [--json]");
    process.exitCode = 1;
    return;
  }
  const lines = readTraceLines(args.tracesDir);
  const report = buildReport(lines);
  console.log(args.json ? JSON.stringify(report, null, 2) : render(report));
}

const invokedDirectly =
  process.argv[1] !== undefined && process.argv[1].includes("retrieval-signal-report");
if (invokedDirectly) {
  main();
}
