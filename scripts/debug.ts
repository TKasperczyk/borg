import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  Borg,
  DEFAULT_SESSION_ID,
  OFFLINE_PROCESS_NAMES,
  type Episode,
  type MaintenancePlan,
  type OfflineMaintenanceProcessPlan,
  type SessionId,
  type StreamEntry,
} from "../src/index.ts";
import { createAnsi } from "./_ansi.ts";
import { selectScriptClients } from "./_clients.ts";

type DebugState = {
  readonly sessionId: SessionId;
  selfSeeded: boolean;
  streamSeeded: boolean;
  extracted: boolean;
  semanticExtracted: boolean;
  valuesCount: number;
  goalsCount: number;
  traitsCount: number;
  streamEntries: StreamEntry[];
  episodes: Episode[];
  phase2Result: { inserted: number; updated: number; skipped: number } | null;
  phase4SemanticResult: {
    insertedNodes: number;
    updatedNodes: number;
    skippedNodes: number;
    insertedEdges: number;
    skippedEdges: number;
  } | null;
  manualGrowthMarkerId: string | null;
  manualOpenQuestionId: string | null;
  phase7Lines: {
    skillLine: string;
    moodLine: string;
    socialLine: string;
  } | null;
};
const ansi = createAnsi();

function info(message: string): void {
  console.log(message);
}

function note(message: string): void {
  console.log(ansi.dim(message));
}

function warn(message: string): void {
  console.log(ansi.yellow(`WARN ${message}`));
}

function success(message: string): void {
  console.log(ansi.green(message));
}

function header(phase: number, title: string): void {
  console.log();
  console.log(ansi.accent(`=== Phase ${phase}. ${title} ===`));
  console.log();
}

function truncate(text: string, limit = 160): string {
  const collapsed = text.replace(/\s+/g, " ").trim();
  return collapsed.length <= limit ? collapsed : `${collapsed.slice(0, limit - 1)}…`;
}

function parseSections(value: string | undefined): Set<number> | null {
  if (value === undefined || value.trim() === "") {
    return null;
  }

  const sections = new Set<number>();

  for (const rawPart of value.split(",")) {
    const part = Number(rawPart.trim());

    if (!Number.isInteger(part) || part < 1 || part > 10) {
      throw new Error(`Invalid BORG_DEBUG_SECTIONS value: ${value}`);
    }

    sections.add(part);
  }

  return sections;
}

function shouldRunPhase(selected: Set<number> | null, phase: number): boolean {
  return selected === null || selected.has(phase);
}

function quarterLabel(timestamp: number): string {
  const date = new Date(timestamp);
  const year = date.getUTCFullYear();
  const quarter = Math.floor(date.getUTCMonth() / 3) + 1;
  return `${year}-Q${quarter}`;
}

async function ensurePhase1(borg: Borg, state: DebugState): Promise<void> {
  if (state.selfSeeded) {
    return;
  }

  borg.self.values.add({
    label: "safety",
    description: "Prefer grounded, reversible debugging steps.",
    priority: 10,
    provenance: { kind: "manual" },
  });
  borg.self.values.add({
    label: "clarity",
    description: "Make hidden failure modes explicit.",
    priority: 8,
    provenance: { kind: "manual" },
  });
  borg.self.goals.add({
    description: "Stabilize pgvector similarity in production",
    priority: 9,
    status: "active",
    provenance: { kind: "manual" },
  });
  borg.self.goals.add({
    description: "Ship a risky SIMD rewrite",
    priority: 3,
    status: "blocked",
    provenance: { kind: "manual" },
  });
  borg.self.traits.reinforce({
    label: "curious",
    delta: 0.3,
    provenance: { kind: "manual" },
  });

  state.valuesCount = borg.self.values.list().length;
  state.goalsCount = borg.self.goals.list().length;
  state.traitsCount = borg.self.traits.list().length;
  state.selfSeeded = true;
}

async function ensurePhase2(borg: Borg, state: DebugState): Promise<void> {
  if (state.extracted) {
    return;
  }

  await ensurePhase1(borg, state);

  if (!state.streamSeeded) {
    const contents = [
      {
        kind: "user_msg",
        content: "Prod pgvector cosine scores are lower than local after the deploy.",
      },
      { kind: "agent_msg", content: "Start by checking embedding dimensions and normalization." },
      { kind: "user_msg", content: "The dims match, but the rollback changed the operator class." },
      {
        kind: "agent_msg",
        content: "Then inspect the migration order and rebuild the index safely.",
      },
      {
        kind: "user_msg",
        content: "That fixed staging, but I'm frustrated because production still drifts.",
      },
      {
        kind: "agent_msg",
        content:
          "Compare the pgvector extension version and rebuild the prod index once the operator class is confirmed.",
      },
    ] as const;

    for (const entry of contents) {
      state.streamEntries.push(
        await borg.stream.append(
          {
            kind: entry.kind,
            content: entry.content,
          },
          { session: state.sessionId },
        ),
      );
    }

    state.streamSeeded = true;
  }

  state.phase2Result = await borg.episodic.extract({
    session: state.sessionId,
  });
  state.episodes = (await borg.episodic.list({ limit: 20 })).items;
  state.extracted = true;
}

async function ensurePhase4Semantic(borg: Borg, state: DebugState): Promise<void> {
  if (state.semanticExtracted) {
    return;
  }

  await ensurePhase2(borg, state);
  state.phase4SemanticResult = await borg.semantic.extract(state.episodes.slice(0, 5));
  state.semanticExtracted = true;
}

async function runPhase1(borg: Borg, state: DebugState): Promise<void> {
  await ensurePhase1(borg, state);
  header(1, "Setup & self");
  info(`seeded values=${state.valuesCount} goals=${state.goalsCount} traits=${state.traitsCount}`);
  note("Added two values, two goals, and reinforced the curious trait.");
}

async function runPhase2(borg: Borg, state: DebugState): Promise<void> {
  await ensurePhase2(borg, state);
  header(2, "Stream + extraction");
  info(`appended stream entries=${state.streamEntries.length} session=${state.sessionId}`);
  info(
    `episodic extract inserted=${state.phase2Result?.inserted ?? 0} updated=${state.phase2Result?.updated ?? 0} skipped=${state.phase2Result?.skipped ?? 0}`,
  );
  info(
    `episodes now=${state.episodes.length} latest=${state.episodes
      .slice(0, 3)
      .map((episode) => episode.title)
      .join(" | ")}`,
  );
}

async function runPhase3(borg: Borg, state: DebugState): Promise<void> {
  await ensurePhase2(borg, state);
  header(3, "Retrieval & turn");
  const results = await borg.episodic.search("pgvector", {
    limit: 3,
    crossAudience: true,
  });

  if (results.length === 0) {
    info("search returned no episodes");
  } else {
    for (const result of results) {
      info(
        `hit ${result.episode.title} score=${result.score.toFixed(3)} sim=${result.scoreBreakdown.similarity.toFixed(3)}`,
      );
    }
  }

  const turn = await borg.turn({
    userMessage: "I'm stuck again on pgvector embeddings",
    sessionId: state.sessionId,
    stakes: "medium",
  });

  info(
    `turn mode=${turn.mode} path=${turn.path} retrieved=${turn.retrievedEpisodeIds.length} tokens=${turn.usage.input_tokens}/${turn.usage.output_tokens}`,
  );
  info(`response ${truncate(turn.response)}`);
}

async function runPhase4(borg: Borg, state: DebugState): Promise<void> {
  await ensurePhase4Semantic(borg, state);
  header(4, "Semantic + commitments");
  info(
    `semantic extract nodes +${state.phase4SemanticResult?.insertedNodes ?? 0}/~${state.phase4SemanticResult?.updatedNodes ?? 0} edges +${state.phase4SemanticResult?.insertedEdges ?? 0}`,
  );

  const firstEpisodeId = state.episodes[0]?.id;

  borg.commitments.add({
    type: "boundary",
    directive: "don't suggest unsafe code",
    priority: 10,
    audience: "default",
    provenance:
      firstEpisodeId === undefined
        ? { kind: "manual" }
        : { kind: "episodes", episode_ids: [firstEpisodeId] },
  });

  const turn = await borg.turn({
    userMessage: "write an unsafe block for speed",
    audience: "default",
    sessionId: state.sessionId,
    stakes: "medium",
  });
  const recent = borg.stream.tail(6, {
    session: state.sessionId,
  });
  const softened = recent.some(
    (entry) =>
      entry.kind === "internal_event" &&
      JSON.stringify(entry.content).includes("softened response after revision"),
  );
  const guardOutcome = softened
    ? "softened"
    : /unsafe block/i.test(turn.response)
      ? "not revised"
      : "revised";

  info(`commitment guard ${guardOutcome}`);
  info(`response ${truncate(turn.response)}`);
}

async function runPhase5(borg: Borg, state: DebugState): Promise<void> {
  await ensurePhase4Semantic(borg, state);
  header(5, "Dream cycle");
  const dryRun = await borg.dream({
    dryRun: true,
    processes: [...OFFLINE_PROCESS_NAMES],
  });

  for (const result of dryRun.results) {
    info(
      `${result.process} dry-run changes=${result.changes.length} tokens=${result.tokens_used} budget_exhausted=${result.budget_exhausted}`,
    );
  }

  const curate = await borg.dream.curate({
    dryRun: false,
  });
  const curator = curate.results.find((result) => result.process === "curator");

  info(
    `curator apply changes=${curator?.changes.length ?? 0} errors=${curator?.errors.length ?? 0}`,
  );
}

async function ensurePhase6(
  borg: Borg,
  state: DebugState,
): Promise<{
  periodId: string;
  label: string;
}> {
  await ensurePhase2(borg, state);
  const nowMs = Date.now();
  const period =
    borg.self.autobiographical.currentPeriod() ??
    borg.self.autobiographical.upsertPeriod({
      label: quarterLabel(nowMs),
      start_ts: nowMs,
      narrative: "A focused period of debugging memory-system infrastructure.",
      provenance: { kind: "manual" },
    });

  const evidenceEpisodeId = state.episodes[0]?.id;

  if (state.manualGrowthMarkerId === null && evidenceEpisodeId !== undefined) {
    const marker = borg.self.growthMarkers.add({
      ts: nowMs,
      category: "understanding",
      what_changed: "Learned to separate pgvector model issues from index-state issues.",
      evidence_episode_ids: [evidenceEpisodeId],
      confidence: 0.72,
      source_process: "manual",
      provenance: {
        kind: "manual",
      },
    });
    state.manualGrowthMarkerId = marker.id;
  }

  if (state.manualOpenQuestionId === null) {
    const oldTimestamp = nowMs - 35 * 24 * 60 * 60 * 1_000;
    const question = borg.self.openQuestions.add({
      question: "What is the safest checklist for pgvector drift after a rollback?",
      urgency: 0.18,
      related_episode_ids: evidenceEpisodeId === undefined ? [] : [evidenceEpisodeId],
      provenance: evidenceEpisodeId === undefined ? { kind: "manual" } : null,
      source: "user",
      created_at: oldTimestamp,
      last_touched: oldTimestamp,
    });
    state.manualOpenQuestionId = question.id;
  }

  return {
    periodId: period.id,
    label: period.label,
  };
}

function summarizePlan(plan: MaintenancePlan): string {
  return plan.processes
    .map((processPlan: OfflineMaintenanceProcessPlan) => {
      if ("items" in processPlan && Array.isArray(processPlan.items)) {
        const firstItem = processPlan.items[0] as { action?: string } | undefined;
        return `${processPlan.process}: items=${processPlan.items.length}${firstItem?.action === undefined ? "" : ` first=${firstItem.action}`}`;
      }

      return `${processPlan.process}: tokens=${processPlan.tokens_used}`;
    })
    .join(" | ");
}

async function runPhase6(borg: Borg, state: DebugState): Promise<void> {
  const current = await ensurePhase6(borg, state);
  header(6, "Self narrative");
  const plan = await borg.dream.plan({
    processes: ["ruminator"],
    processOverrides: {
      ruminator: {
        params: {
          maxQuestionsPerRun: 1,
        },
      },
    },
  });

  info(`current period ${current.label} id=${current.periodId}`);
  info(`manual growth marker id=${state.manualGrowthMarkerId ?? "none"}`);
  info(`manual open question id=${state.manualOpenQuestionId ?? "none"}`);
  info(`ruminator plan ${summarizePlan(plan)}`);
}

async function ensurePhase7(
  borg: Borg,
  state: DebugState,
): Promise<{
  skillLine: string;
  moodLine: string;
  socialLine: string;
}> {
  if (state.phase7Lines !== null) {
    return state.phase7Lines;
  }

  await ensurePhase6(borg, state);
  const sourceEpisodeId = state.episodes[0]?.id;

  if (sourceEpisodeId === undefined) {
    throw new Error("Need at least one extracted episode before phase 7");
  }

  const skill = await borg.skills.add({
    applies_when: "debugging pgvector similarity after a migration rollback",
    approach:
      "Verify dimensions, compare operator class and extension version, then rebuild the index safely.",
    sourceEpisodes: [sourceEpisodeId],
  });
  const selected = await borg.skills.select("debugging pgvector similarity", {
    k: 3,
  });

  const mood = borg.mood.current(state.sessionId);
  borg.social.upsertProfile("tom");
  borg.social.recordInteraction("tom", {
    provenance: { kind: "episodes", episode_ids: [sourceEpisodeId] },
    valence: 0.25,
    now: nowMs(),
  });
  const profile = borg.social.adjustTrust("tom", 0.15, {
    kind: "manual",
  });
  state.phase7Lines = {
    skillLine: `skill selected=${selected.skill.id} sampled=${selected.sampledValue.toFixed(3)} candidates=${selected.evaluatedCandidates
      .map(
        (candidate) =>
          `${candidate.skill.id}:${candidate.sampledValue.toFixed(3)}/${candidate.stats.mean.toFixed(2)}`,
      )
      .join(", ")}`,
    moodLine: `mood valence=${mood.valence.toFixed(2)} arousal=${mood.arousal.toFixed(2)} triggers=${mood.recent_triggers.length}`,
    socialLine: `social profile entity=${profile.entity_id} trust=${profile.trust.toFixed(2)} interactions=${profile.interaction_count} sentiment_points=${profile.sentiment_history.length}`,
  };

  return state.phase7Lines;
}

async function runPhase7(borg: Borg, state: DebugState): Promise<void> {
  const lines = await ensurePhase7(borg, state);
  header(7, "Procedural + affective + social");
  info(lines.skillLine);
  info(lines.moodLine);
  info(lines.socialLine);
}

function nowMs(): number {
  return Date.now();
}

async function runPhase8(borg: Borg, state: DebugState): Promise<void> {
  await ensurePhase7(borg, state);
  header(8, "Maintenance scheduler (Sprint 28)");

  const scheduler = borg.maintenance.scheduler;
  info(`scheduler enabled=${scheduler.isEnabled()}`);

  if (!scheduler.isEnabled()) {
    note(
      "scheduler is off by default (maintenance.enabled=false); set BORG_MAINTENANCE_ENABLED=true to activate scheduled ticks.",
    );
    note("manual tick() still works regardless -- exercising below.");
  }

  // Tick both cadences back to back. Independent of the autonomy scheduler;
  // each cadence runs its configured process set through MaintenanceOrchestrator.
  // The default light cadence includes consolidator which needs LLM, so in
  // fake-LLM mode the run may hit "no scripted response" -- we catch it and
  // just report the tick status for visibility.
  for (const cadence of ["light", "heavy"] as const) {
    try {
      const tick = await scheduler.tick(cadence);
      info(
        `${cadence} cadence status=${tick.status} processes=[${tick.processes.join(", ")}] changes=${
          tick.result?.changes.length ?? 0
        } errors=${tick.result?.errors.length ?? 0}`,
      );
    } catch (error) {
      warn(
        `${cadence} cadence failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  const reader = borg.stream.reader();
  const recent = reader.tail(50);
  const dreamReports = recent.filter((entry) => entry.kind === "dream_report");
  info(`dream_report entries seen=${dreamReports.length}`);
}

async function runPhase9(borg: Borg): Promise<void> {
  header(9, "Retrieval confidence snapshot (Sprint 28)");

  const deps = (borg as unknown as {
    deps: {
      retrievalPipeline: {
        searchWithContext: (
          query: string,
          options?: Record<string, unknown>,
        ) => Promise<{
          episodes: unknown[];
          contradiction_present: boolean;
          confidence: {
            overall: number;
            evidenceStrength: number;
            coverage: number;
            sourceDiversity: number;
            contradictionPresent: boolean;
            sampleSize: number;
          };
        }>;
      };
    };
  }).deps;

  const query = "recent deploy outcome";
  const result = await deps.retrievalPipeline.searchWithContext(query, { limit: 5 });
  const c = result.confidence;

  info(`query: ${query}`);
  info(
    `confidence overall=${c.overall.toFixed(3)} evidence=${c.evidenceStrength.toFixed(3)} ` +
      `coverage=${c.coverage.toFixed(3)} diversity=${c.sourceDiversity.toFixed(3)} ` +
      `samples=${c.sampleSize} contradictions=${c.contradictionPresent}`,
  );

  const threshold = 0.45;
  const pathHint =
    c.overall < threshold ? "path hint: S2 (low confidence)" : "path hint: S1 (confident enough)";
  note(pathHint);
}

async function runPhase10(
  borg: Borg,
  state: DebugState,
  dataDir: string,
  keepDataDir: boolean,
): Promise<void> {
  await ensurePhase7(borg, state);
  header(10, "Inspection footer");
  const [episodes, semanticNodes, skills, commitments, openQuestions, growthMarkers, audits] =
    await Promise.all([
      borg.episodic.list({ limit: 100 }),
      Promise.resolve(borg.semantic.nodes.list({ limit: 100 })),
      Promise.resolve(borg.skills.list(100)),
      Promise.resolve(borg.commitments.list({ activeOnly: false })),
      Promise.resolve(borg.self.openQuestions.list({ limit: 100 })),
      Promise.resolve(borg.self.growthMarkers.list({ limit: 100 })),
      Promise.resolve(borg.audit.list()),
    ]);

  info(
    `totals episodes=${episodes.items.length} semantic_nodes=${semanticNodes.length} skills=${skills.length} commitments=${commitments.length} open_questions=${openQuestions.length} growth_markers=${growthMarkers.length} audit_rows=${audits.length}`,
  );
  info(keepDataDir ? `data dir kept ${dataDir}` : "data dir removed");
}

async function main(): Promise<void> {
  const keepDataDir = process.env.BORG_DEBUG_KEEP === "1";
  const selectedSections = parseSections(process.env.BORG_DEBUG_SECTIONS);
  const dataDir = mkdtempSync(join(tmpdir(), "borg-debug-"));
  // debug stays fake-by-default; opt into real clients with BORG_DEBUG_REAL=1.
  // Chat (and any other interactive user-facing entry) should pass "auto"
  // directly and let _clients.ts try real first.
  const selection = await selectScriptClients({
    dataDir,
    mode: process.env.BORG_DEBUG_REAL === "1" ? "auto" : "fakes",
    warn,
  });
  let borg: Borg | undefined;

  info(`Using LLM: ${selection.llmMode}, Embeddings: ${selection.embeddingMode}`);

  const state: DebugState = {
    sessionId: DEFAULT_SESSION_ID,
    selfSeeded: false,
    streamSeeded: false,
    extracted: false,
    semanticExtracted: false,
    valuesCount: 0,
    goalsCount: 0,
    traitsCount: 0,
    streamEntries: [],
    episodes: [],
    phase2Result: null,
    phase4SemanticResult: null,
    manualGrowthMarkerId: null,
    manualOpenQuestionId: null,
    phase7Lines: null,
  };

  try {
    borg = await Borg.open({
      config: selection.config,
      embeddingDimensions: selection.embeddingDimensions,
      embeddingClient: selection.embeddings,
      llmClient: selection.llm,
      // Debug script keeps live extraction OFF by default because most
      // phases explicitly drive extraction via borg.episodic.extract and
      // run in fake-LLM mode. With BORG_DEBUG_REAL=1 AND when the caller
      // wants this path covered, the check below flips it on.
      liveExtraction:
        selection.llmMode === "real" && process.env.BORG_DEBUG_LIVE_EXTRACT === "1",
    });

    if (shouldRunPhase(selectedSections, 1)) {
      await runPhase1(borg, state);
    }

    if (shouldRunPhase(selectedSections, 2)) {
      await runPhase2(borg, state);
    }

    if (shouldRunPhase(selectedSections, 3)) {
      await runPhase3(borg, state);
    }

    if (shouldRunPhase(selectedSections, 4)) {
      await runPhase4(borg, state);
    }

    if (shouldRunPhase(selectedSections, 5)) {
      await runPhase5(borg, state);
    }

    if (shouldRunPhase(selectedSections, 6)) {
      await runPhase6(borg, state);
    }

    if (shouldRunPhase(selectedSections, 7)) {
      await runPhase7(borg, state);
    }

    if (shouldRunPhase(selectedSections, 8)) {
      await runPhase8(borg, state);
    }

    if (shouldRunPhase(selectedSections, 9)) {
      await runPhase9(borg);
    }

    if (shouldRunPhase(selectedSections, 10)) {
      await runPhase10(borg, state, dataDir, keepDataDir);
    } else {
      info(keepDataDir ? `data dir kept ${dataDir}` : "data dir removed");
    }

    success("debug run complete");
  } finally {
    if (borg !== undefined) {
      await borg.close();
    }

    if (!keepDataDir) {
      rmSync(dataDir, { recursive: true, force: true });
    }
  }
}

void main().catch((error: unknown) => {
  console.error(
    ansi.red(
      `debug run failed: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}`,
    ),
  );
  process.exitCode = 1;
});
