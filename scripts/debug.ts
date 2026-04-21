import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { getFreshCredentials } from "../src/auth/claude-oauth.ts";
import {
  AnthropicLLMClient,
  Borg,
  DEFAULT_CONFIG,
  DEFAULT_SESSION_ID,
  FakeEmbeddingClient,
  FakeLLMClient,
  OFFLINE_PROCESS_NAMES,
  OpenAICompatibleEmbeddingClient,
  loadConfig,
  type Config,
  type EmbeddingClient,
  type Episode,
  type LLMClient,
  type LLMCompleteOptions,
  type LLMCompleteResult,
  type MaintenancePlan,
  type OfflineMaintenanceProcessPlan,
  type SessionId,
  type StreamEntry,
} from "../src/index.ts";

type ClientMode = "real" | "fake";

type DebugClientSelection = {
  llm: LLMClient;
  embeddings: EmbeddingClient;
  llmMode: ClientMode;
  embeddingMode: ClientMode;
  config: Config;
  embeddingDimensions: number;
};

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

const COLOR_ENABLED = process.stdout.isTTY === true;

function ansi(text: string, code: string): string {
  return COLOR_ENABLED ? `\u001b[${code}m${text}\u001b[0m` : text;
}

function info(message: string): void {
  console.log(message);
}

function note(message: string): void {
  console.log(ansi(message, "2"));
}

function warn(message: string): void {
  console.log(ansi(`WARN ${message}`, "33"));
}

function success(message: string): void {
  console.log(ansi(message, "32"));
}

function header(phase: number, title: string): void {
  console.log();
  console.log(ansi(`=== Phase ${phase}. ${title} ===`, "1;36"));
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

    if (!Number.isInteger(part) || part < 1 || part > 8) {
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

function uniqueMatches(text: string, pattern: RegExp): string[] {
  return [...new Set([...text.matchAll(pattern)].map((match) => match[0]))];
}

function extractStreamIds(text: string): string[] {
  return uniqueMatches(text, /strm_[a-z0-9]{16}/gi);
}

function extractEpisodeIds(text: string): string[] {
  return uniqueMatches(text, /ep_[a-z0-9]{16}/gi);
}

function buildLlmResult(
  options: LLMCompleteOptions,
  text: string,
  toolCalls: LLMCompleteResult["tool_calls"] = [],
): LLMCompleteResult {
  const systemLength =
    typeof options.system === "string"
      ? options.system.length
      : (options.system?.reduce((sum, block) => sum + block.text.length, 0) ?? 0);
  const promptLength =
    systemLength + options.messages.reduce((sum, message) => sum + message.content.length, 0);

  return {
    text,
    input_tokens: Math.max(1, Math.ceil(promptLength / 4)),
    output_tokens: Math.max(
      1,
      Math.ceil(
        (text.length +
          toolCalls.reduce((sum, call) => sum + JSON.stringify(call.input).length, 0)) /
          4,
      ),
    ),
    stop_reason: toolCalls.length > 0 ? "tool_use" : "end_turn",
    tool_calls: toolCalls,
  };
}

function buildToolResult(options: LLMCompleteOptions, input: unknown): LLMCompleteResult {
  const toolName =
    options.tool_choice?.type === "tool" ? options.tool_choice.name : options.tools?.[0]?.name;

  if (toolName === undefined) {
    return buildLlmResult(options, JSON.stringify(input));
  }

  return buildLlmResult(options, "", [
    {
      id: `tool_${toolName.toLowerCase()}`,
      name: toolName,
      input,
    },
  ]);
}

class ScriptedDebugLLM implements LLMClient {
  private readonly inner = new FakeLLMClient();

  async complete(options: LLMCompleteOptions): Promise<LLMCompleteResult> {
    this.inner.pushResponse(() => this.respond(options));
    return this.inner.complete(options);
  }

  private respond(options: LLMCompleteOptions): LLMCompleteResult {
    const system = options.system ?? "";
    const userPrompt = options.messages.map((message) => message.content).join("\n\n");
    const prompt = `${system}\n\n${userPrompt}`;

    if (/You extract episodic memories from a stream chunk\./i.test(prompt)) {
      const streamIds = extractStreamIds(prompt);
      const episodes = [
        {
          title: "Pgvector similarity mismatch investigation",
          narrative:
            "Borg traced a similarity mismatch to deployment-specific index state. The conversation focused on comparing dimensions, operator class choices, and index rebuild history.",
          source_stream_ids: streamIds.slice(0, 2),
          participants: ["user", "borg"],
          location: null,
          tags: ["pgvector", "debugging", "embeddings"],
          confidence: 0.86,
          significance: 0.82,
        },
        {
          title: "Rollback changed the pgvector operator class",
          narrative:
            "A rollback altered the expected operator class and left scoring inconsistent. Borg narrowed the issue to migration order and safe index recreation.",
          source_stream_ids: streamIds.slice(2, 4),
          participants: ["user", "borg"],
          location: null,
          tags: ["pgvector", "rollback", "database"],
          confidence: 0.84,
          significance: 0.8,
        },
        {
          title: "Production drift remained after staging recovered",
          narrative:
            "Even after staging recovered, production still drifted until Borg compared extension version, normalization, and index freshness together. The discussion carried visible frustration but converged on a safer rebuild sequence.",
          source_stream_ids: streamIds.slice(4, 6),
          participants: ["user", "borg"],
          location: null,
          tags: ["pgvector", "production", "debugging"],
          confidence: 0.83,
          significance: 0.79,
        },
      ].filter((episode) => episode.source_stream_ids.length > 0);

      return buildToolResult(options, { episodes });
    }

    if (/Extract semantic knowledge from the provided episodes\./i.test(prompt)) {
      const episodeIds = extractEpisodeIds(prompt);

      return buildToolResult(options, {
        nodes: [
          {
            kind: "concept",
            label: "pgvector similarity drift",
            description:
              "A recurring mismatch where similar vectors score differently across environments.",
            aliases: ["pgvector drift"],
            confidence: 0.66,
            source_episode_ids: episodeIds.slice(0, 2),
          },
          {
            kind: "proposition",
            label: "Rollback-sensitive index state caused the mismatch",
            description:
              "The mismatch persisted until the operator class and index rebuild path were checked after rollback.",
            aliases: ["index rebuild after rollback"],
            confidence: 0.64,
            source_episode_ids: episodeIds.slice(0, 3),
          },
          {
            kind: "concept",
            label: "Safe rebuild workflow",
            description:
              "A conservative sequence that verifies extension version, operator class, and index freshness before making further changes.",
            aliases: ["safe pgvector rebuild"],
            confidence: 0.61,
            source_episode_ids: episodeIds.slice(1, 3),
          },
        ],
        edges: [
          {
            from_label: "pgvector similarity drift",
            to_label: "Rollback-sensitive index state caused the mismatch",
            relation: "supports",
            confidence: 0.63,
            evidence_episode_ids: episodeIds.slice(0, 3),
          },
          {
            from_label: "Safe rebuild workflow",
            to_label: "pgvector similarity drift",
            relation: "prevents",
            confidence: 0.58,
            evidence_episode_ids: episodeIds.slice(1, 3),
          },
        ],
      });
    }

    if (
      /Think briefly about what the assistant should verify, clarify, or compare before answering\./i.test(
        system,
      )
    ) {
      return buildLlmResult(
        options,
        "Verify the operator class, extension version, and whether the index was rebuilt after rollback before proposing the next step.",
      );
    }

    if (/Your previous response violated a commitment\./i.test(system)) {
      return buildLlmResult(
        options,
        "I won't suggest unsafe code here. Use profiling, safer data layout changes, or a targeted benchmark to decide what to optimize next.",
      );
    }

    if (/You are Borg, an agent with explicit memory and identity\./i.test(system)) {
      if (/unsafe block/i.test(userPrompt)) {
        return buildLlmResult(
          options,
          "Wrap the hot loop in an unsafe block for speed and skip the extra checks.",
        );
      }

      if (/stuck again on pgvector embeddings/i.test(userPrompt)) {
        return buildLlmResult(
          options,
          "Start by checking the operator class, extension version, and whether the rollback left an old index in place. If those line up, rebuild the index safely before blaming the embeddings.",
        );
      }

      return buildLlmResult(
        options,
        "I’d compare the grounded evidence first and then choose the safest next diagnostic step.",
      );
    }

    if (/Merge the redundant episodes into one grounded episode\./i.test(prompt)) {
      return buildToolResult(options, {
        title: "Consolidated pgvector mismatch investigation",
        narrative:
          "Across several debugging turns, Borg narrowed the pgvector mismatch to rollback-sensitive index state instead of the embedding model itself. The merged story preserves the operator-class checks, extension comparison, and safe rebuild sequence that stabilized similarity scoring.",
      });
    }

    if (
      /You propose low-confidence semantic propositions grounded in repeated episodic evidence\./i.test(
        system,
      )
    ) {
      const episodeIds = extractEpisodeIds(prompt);

      return buildToolResult(options, {
        label: "Pgvector rollbacks should trigger an index-state audit",
        description:
          "Repeated debugging episodes suggest that rollback-related similarity drift is best handled by auditing operator class, extension version, and rebuild state together.",
        confidence: 0.62,
        source_episode_ids: episodeIds.slice(0, Math.max(1, Math.min(episodeIds.length, 3))),
      });
    }

    if (
      /You update Borg's open questions conservatively and only from grounded evidence\./i.test(
        system,
      )
    ) {
      return buildToolResult(options, {
        resolution_note:
          "The newer episodes show that comparing operator class and rebuilding the index resolved the repeated pgvector mismatch.",
        growth_marker: {
          what_changed:
            "Borg now frames pgvector drift as an index-state problem instead of a vague embedding issue.",
          before_description: "It only suspected a generic similarity bug.",
          after_description:
            "It can name the rollback, operator-class, and rebuild checks that settle the issue.",
          confidence: 0.72,
          category: "understanding",
        },
      });
    }

    if (
      /You identify grounded autobiographical growth markers\. Return null when the evidence is weak\./i.test(
        system,
      )
    ) {
      const episodeIds = extractEpisodeIds(prompt);

      return buildToolResult(options, {
        observation: {
          category: "understanding",
          what_changed:
            "Borg became more systematic about separating embedding issues from index-state issues.",
          before_description: "It lumped the mismatch into generic pgvector confusion.",
          after_description:
            "It now uses a cleaner checklist around dimensions, operator class, and rebuild order.",
          confidence: 0.68,
          evidence_episode_ids: episodeIds.slice(0, 2),
        },
      });
    }

    if (
      /Check the memory item for misattribution, temporal drift, and identity inconsistency\./i.test(
        prompt,
      )
    ) {
      return buildToolResult(options, { flags: [] });
    }

    return buildLlmResult(options, "Acknowledged.");
  }
}

async function selectClients(dataDir: string): Promise<DebugClientSelection> {
  const loaded = loadConfig({ env: process.env, dataDir });
  const usingReal = process.env.BORG_DEBUG_REAL === "1";
  let llmMode: ClientMode = "fake";
  let embeddingMode: ClientMode = "fake";
  let llm: LLMClient = new ScriptedDebugLLM();
  let embeddings: EmbeddingClient = new FakeEmbeddingClient(4);
  let embeddingDimensions = 4;
  let oauthCredentialsAvailable = false;
  const apiKey = (process.env.ANTHROPIC_API_KEY ?? "").trim();

  if (usingReal) {
    const oauthCredentials = await getFreshCredentials({ env: process.env });
    oauthCredentialsAvailable = oauthCredentials !== null;

    if (oauthCredentials !== null) {
      try {
        llm = new AnthropicLLMClient({
          authMode: "oauth",
          env: process.env,
        });
        llmMode = "real";
      } catch (error) {
        warn(
          `real LLM unavailable (${error instanceof Error ? error.message : String(error)}); falling back to fake`,
        );
      }
    } else if (apiKey.length > 0) {
      try {
        llm = new AnthropicLLMClient({
          authMode: "api-key",
          apiKey,
          env: process.env,
        });
        llmMode = "real";
      } catch (error) {
        warn(
          `real LLM unavailable (${error instanceof Error ? error.message : String(error)}); falling back to fake`,
        );
      }
    } else {
      warn(
        "real LLM unavailable (tried ~/.claude/.credentials.json OAuth and ANTHROPIC_API_KEY); falling back to fake",
      );
    }

    try {
      const realEmbeddings = new OpenAICompatibleEmbeddingClient({
        baseUrl: loaded.embedding.baseUrl,
        apiKey: loaded.embedding.apiKey,
        model: loaded.embedding.model,
        dims: loaded.embedding.dims,
      });
      await realEmbeddings.embed("ping");
      embeddings = realEmbeddings;
      embeddingDimensions = loaded.embedding.dims;
      embeddingMode = "real";
    } catch (error) {
      warn(
        `real embeddings unavailable (${error instanceof Error ? error.message : String(error)}); falling back to fake`,
      );
    }
  }

  const config: Config = {
    ...DEFAULT_CONFIG,
    ...loaded,
    dataDir,
    perception: {
      ...DEFAULT_CONFIG.perception,
      ...loaded.perception,
      useLlmFallback: false,
    },
    affective: {
      ...DEFAULT_CONFIG.affective,
      ...loaded.affective,
      useLlmFallback: llmMode === "real" ? loaded.affective.useLlmFallback : false,
    },
    embedding: {
      ...DEFAULT_CONFIG.embedding,
      ...loaded.embedding,
      dims: embeddingDimensions,
    },
    anthropic: {
      ...DEFAULT_CONFIG.anthropic,
      ...loaded.anthropic,
      auth:
        llmMode === "real" && oauthCredentialsAvailable
          ? "oauth"
          : llmMode === "real" && apiKey.length > 0
            ? "api-key"
            : loaded.anthropic.auth,
      models: {
        ...DEFAULT_CONFIG.anthropic.models,
        ...loaded.anthropic.models,
      },
    },
    offline: {
      ...DEFAULT_CONFIG.offline,
      ...loaded.offline,
      consolidator: {
        ...DEFAULT_CONFIG.offline.consolidator,
        ...loaded.offline.consolidator,
      },
      reflector: {
        ...DEFAULT_CONFIG.offline.reflector,
        ...loaded.offline.reflector,
      },
      curator: {
        ...DEFAULT_CONFIG.offline.curator,
        ...loaded.offline.curator,
      },
      overseer: {
        ...DEFAULT_CONFIG.offline.overseer,
        ...loaded.offline.overseer,
      },
      ruminator: {
        ...DEFAULT_CONFIG.offline.ruminator,
        ...loaded.offline.ruminator,
      },
      selfNarrator: {
        ...DEFAULT_CONFIG.offline.selfNarrator,
        ...loaded.offline.selfNarrator,
      },
    },
  };

  return {
    llm,
    embeddings,
    llmMode,
    embeddingMode,
    config,
    embeddingDimensions,
  };
}

async function ensurePhase1(borg: Borg, state: DebugState): Promise<void> {
  if (state.selfSeeded) {
    return;
  }

  borg.self.values.add({
    label: "safety",
    description: "Prefer grounded, reversible debugging steps.",
    priority: 10,
  });
  borg.self.values.add({
    label: "clarity",
    description: "Make hidden failure modes explicit.",
    priority: 8,
  });
  borg.self.goals.add({
    description: "Stabilize pgvector similarity in production",
    priority: 9,
    status: "active",
  });
  borg.self.goals.add({
    description: "Ship a risky SIMD rewrite",
    priority: 3,
    status: "blocked",
  });
  borg.self.traits.reinforce("curious", 0.3);

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
    sourceEpisodeIds: firstEpisodeId === undefined ? undefined : [firstEpisodeId],
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
    });
    state.manualGrowthMarkerId = marker.id;
  }

  if (state.manualOpenQuestionId === null) {
    const oldTimestamp = nowMs - 35 * 24 * 60 * 60 * 1_000;
    const question = borg.self.openQuestions.add({
      question: "What is the safest checklist for pgvector drift after a rollback?",
      urgency: 0.18,
      related_episode_ids: evidenceEpisodeId === undefined ? [] : [evidenceEpisodeId],
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
    episode_id: sourceEpisodeId,
    valence: 0.25,
    now: nowMs(),
  });
  const profile = borg.social.adjustTrust("tom", 0.15);
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

async function runPhase8(
  borg: Borg,
  state: DebugState,
  dataDir: string,
  keepDataDir: boolean,
): Promise<void> {
  await ensurePhase7(borg, state);
  header(8, "Inspection footer");
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
  const selection = await selectClients(dataDir);
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
      await runPhase8(borg, state, dataDir, keepDataDir);
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
    ansi(
      `debug run failed: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}`,
      "31",
    ),
  );
  process.exitCode = 1;
});
