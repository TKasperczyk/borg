import { getFreshCredentials } from "../src/auth/claude-oauth.ts";
import {
  AnthropicLLMClient,
  DEFAULT_CONFIG,
  FakeEmbeddingClient,
  FakeLLMClient,
  OpenAICompatibleEmbeddingClient,
  loadConfig,
  type Config,
  type EmbeddingClient,
  type Episode,
  type LLMClient,
  type LLMCompleteOptions,
  type LLMCompleteResult,
} from "../src/index.ts";

export type ScriptClientMode = "real" | "fake";
export type ScriptClientSelectionMode = "auto" | "real" | "fakes";

export type ScriptClientSelection = {
  llm: LLMClient;
  embeddings: EmbeddingClient;
  llmMode: ScriptClientMode;
  embeddingMode: ScriptClientMode;
  config: Config;
  embeddingDimensions: number;
};

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

export class ScriptedDebugLLM implements LLMClient {
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

export async function selectScriptClients(options: {
  dataDir?: string;
  mode?: ScriptClientSelectionMode;
  env?: NodeJS.ProcessEnv;
  warn?: (message: string) => void;
}): Promise<ScriptClientSelection> {
  const env = options.env ?? process.env;
  const loaded = loadConfig({ env, dataDir: options.dataDir });
  // "auto" and "real" both attempt real clients (falling back to fakes with a
  // visible warning on failure). "fakes" opts out. Callers that want a
  // different default (e.g. the debug smoke script) resolve to "fakes" BEFORE
  // calling this helper rather than gating in here.
  const usingReal = options.mode !== "fakes";
  const warn = options.warn ?? (() => {});
  let llmMode: ScriptClientMode = "fake";
  let embeddingMode: ScriptClientMode = "fake";
  let llm: LLMClient = new ScriptedDebugLLM();
  let embeddings: EmbeddingClient = new FakeEmbeddingClient(4);
  let embeddingDimensions = 4;
  let oauthCredentialsAvailable = false;
  const apiKey = (env.ANTHROPIC_API_KEY ?? "").trim();

  if (usingReal) {
    const oauthCredentials = await getFreshCredentials({ env });
    oauthCredentialsAvailable = oauthCredentials !== null;

    if (oauthCredentials !== null) {
      try {
        llm = new AnthropicLLMClient({
          authMode: "oauth",
          env,
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
          env,
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
    dataDir: options.dataDir ?? loaded.dataDir,
    perception: {
      ...DEFAULT_CONFIG.perception,
      ...loaded.perception,
      // Match the affective pattern: only let the LLM classifier run when a
      // real LLM is wired. With fake LLMs (scripted tests/debug), forcing
      // LLM fallback would consume scripted responses meant for turns.
      useLlmFallback: llmMode === "real" ? loaded.perception.useLlmFallback : false,
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
    self: {
      ...DEFAULT_CONFIG.self,
      ...loaded.self,
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
