import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  Borg,
  FakeLLMClient,
  ManualClock,
  type LLMCompleteOptions,
  type LLMCompleteResult,
  type LLMConverseOptions,
  type LLMContentBlock,
  type LLMToolCall,
} from "../src/index.js";
import type { BorgDependencies } from "../src/borg/types.js";
import { createTestConfig, TestEmbeddingClient } from "../src/offline/test-support.js";
import type { RecallStateRepository, RetrievedContext } from "../src/retrieval/index.js";
import { createSessionId, type EntityId, type EpisodeId } from "../src/util/ids.js";

const MAYA_TERM = "Maya";
const AUDIENCE = "Tom";

type ScriptOptions = LLMCompleteOptions | LLMConverseOptions;
type RetrievalOptions = NonNullable<
  Parameters<BorgDependencies["retrievalPipeline"]["searchWithContext"]>[1]
>;
type RetrievalCall = {
  options: RetrievalOptions;
  result: RetrievedContext;
};
type RecallStateRow = {
  scope_key: string;
  state_json: string;
};

const tempDirs: string[] = [];

afterEach(() => {
  vi.restoreAllMocks();

  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop() as string, { recursive: true, force: true });
  }
});

function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "borg-cross-session-"));
  tempDirs.push(dir);
  return dir;
}

function toolNames(options: ScriptOptions): string[] {
  return options.tools?.map((tool) => tool.name) ?? [];
}

function blockText(block: LLMContentBlock): string {
  if (block.type === "text") {
    return block.text;
  }

  if (block.type === "tool_use") {
    return JSON.stringify(block.input);
  }

  return typeof block.content === "string"
    ? block.content
    : block.content.map((entry) => entry.text).join("\n");
}

function latestMessageText(options: ScriptOptions): string {
  const last = options.messages.at(-1);

  if (last === undefined) {
    return "";
  }

  if (typeof last.content === "string") {
    return last.content;
  }

  return last.content.map((block) => blockText(block)).join("\n");
}

function completeWithTool(call: LLMToolCall): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use",
    tool_calls: [call],
  };
}

function recallExpansion(namedTerms: string[]): LLMCompleteResult {
  return completeWithTool({
    id: "toolu_recall_expansion",
    name: "EmitRecallExpansion",
    input: {
      facets: [],
      named_terms: namedTerms,
    },
  });
}

function turnPlan(referencedEpisodeIds: readonly EpisodeId[] = []): LLMCompleteResult {
  return completeWithTool({
    id: "toolu_turn_plan",
    name: "EmitTurnPlan",
    input: {
      uncertainty: "",
      verification_steps: [],
      tensions: [],
      voice_note: "",
      emission_recommendation: "emit",
      referenced_episode_ids: [...referencedEpisodeIds],
      intents: [],
    },
  });
}

function stopCommitmentNone(): LLMCompleteResult {
  return completeWithTool({
    id: "toolu_stop_commitment",
    name: "EmitStopCommitmentClassification",
    input: {
      classification: "none",
      reason: "The response does not commit to future no-output behavior.",
      confidence: 0.98,
    },
  });
}

function emptyReflection(): LLMCompleteResult {
  return completeWithTool({
    id: "toolu_reflection",
    name: "EmitTurnReflection",
    input: {
      advanced_goals: [],
      procedural_outcomes: [],
      trait_demonstrations: [],
      intent_updates: [],
      step_outcomes: [],
      proposed_steps: [],
      open_questions: [],
    },
  });
}

function episodeExtraction(sourceStreamIds: readonly string[]): LLMCompleteResult {
  return completeWithTool({
    id: "toolu_episode_candidates",
    name: "EmitEpisodeCandidates",
    input: {
      episodes: [
        {
          title: "Maya ramen plan",
          narrative: "Tom described Maya as his partner and said she was making elaborate ramen.",
          source_stream_ids: [...sourceStreamIds],
          participants: [MAYA_TERM, AUDIENCE],
          location: null,
          tags: [MAYA_TERM, "ramen", "partner"],
          emotional_arc: null,
          confidence: 0.96,
          significance: 0.84,
        },
      ],
    },
  });
}

function emptyEpisodeExtraction(): LLMCompleteResult {
  return completeWithTool({
    id: "toolu_episode_candidates_empty",
    name: "EmitEpisodeCandidates",
    input: {
      episodes: [],
    },
  });
}

function extractorSourceIds(options: ScriptOptions): string[] {
  const entries: Array<{ id: string; kind: string }> = [];

  for (const line of latestMessageText(options).split("\n")) {
    try {
      const parsed = JSON.parse(line) as unknown;

      if (
        parsed !== null &&
        typeof parsed === "object" &&
        !Array.isArray(parsed) &&
        typeof (parsed as { id?: unknown }).id === "string" &&
        typeof (parsed as { kind?: unknown }).kind === "string"
      ) {
        entries.push({
          id: (parsed as { id: string }).id,
          kind: (parsed as { kind: string }).kind,
        });
      }
    } catch {
      // Non-JSON prompt lines are instructions, not stream entries.
    }
  }

  return entries
    .filter((entry) => entry.kind === "user_msg" || entry.kind === "agent_msg")
    .map((entry) => entry.id);
}

function createScriptedLlm(getKnownEpisodeId: () => EpisodeId | null): FakeLLMClient {
  let recallExpansionCalls = 0;
  let extractionCalls = 0;
  const dispatcher = (options: ScriptOptions): string | LLMCompleteResult => {
    const names = toolNames(options);

    if (names.includes("EmitRecallExpansion")) {
      recallExpansionCalls += 1;
      return recallExpansion(recallExpansionCalls === 3 ? [MAYA_TERM] : []);
    }

    if (names.includes("EmitEpisodeCandidates")) {
      extractionCalls += 1;
      return extractionCalls === 1
        ? episodeExtraction(extractorSourceIds(options))
        : emptyEpisodeExtraction();
    }

    if (names.includes("EmitTurnPlan")) {
      const episodeId = getKnownEpisodeId();
      return turnPlan(episodeId === null ? [] : [episodeId]);
    }

    if (names.includes("EmitStopCommitmentClassification")) {
      return stopCommitmentNone();
    }

    if (names.includes("EmitTurnReflection")) {
      return emptyReflection();
    }

    if (names.includes("EmitGenerationGateDecision")) {
      return completeWithTool({
        id: "toolu_generation_gate",
        name: "EmitGenerationGateDecision",
        input: {
          decision: "proceed",
          substantive: true,
          reason: "The turn has substantive content.",
          confidence: 0.99,
        },
      });
    }

    if (names.includes("ClassifyPendingAction")) {
      return completeWithTool({
        id: "toolu_pending_action",
        name: "ClassifyPendingAction",
        input: {
          classification: "non_action",
          reason: "The scripted scenario carries no future action.",
          confidence: 0.99,
        },
      });
    }

    return "Scripted Borg response.";
  };

  return new FakeLLMClient({
    responses: Array.from({ length: 100 }, () => dispatcher),
  });
}

function borgDeps(borg: Borg): BorgDependencies {
  return (borg as unknown as { deps: BorgDependencies }).deps;
}

function recallStateRepository(deps: BorgDependencies): Pick<RecallStateRepository, "load"> {
  return (
    deps.retrievalPipeline as unknown as {
      options: { recallStateRepository: Pick<RecallStateRepository, "load"> };
    }
  ).options.recallStateRepository;
}

function readRecallStateRows(deps: BorgDependencies): RecallStateRow[] {
  return deps.sqlite
    .prepare("SELECT scope_key, state_json FROM recall_state ORDER BY scope_key")
    .all() as RecallStateRow[];
}

function collectKeys(value: unknown): string[] {
  if (value === null || typeof value !== "object") {
    return [];
  }

  if (Array.isArray(value)) {
    return value.flatMap((item) => collectKeys(item));
  }

  return Object.entries(value).flatMap(([key, nested]) => [key, ...collectKeys(nested)]);
}

describe("cross-session recall_state integration", () => {
  it("carries episode handles across session rotation when audience stays stable", async () => {
    const dir = tempDir();
    const clock = new ManualClock(1_900_000_000_000);
    const sessionA = createSessionId();
    const sessionB = createSessionId();
    let sessionAEpisodeId: EpisodeId | null = null;
    const llm = createScriptedLlm(() => sessionAEpisodeId);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: dir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "problem_solving",
        },
        affective: {
          useLlmFallback: false,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "test-cognition",
            background: "test-background",
            extraction: "test-extraction",
            recallExpansion: "test-recall-expansion",
          },
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "test-embedding",
          dims: 4,
        },
        autonomy: {
          enabled: false,
        },
        maintenance: {
          enabled: false,
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new TestEmbeddingClient(),
      llmClient: llm,
      liveExtraction: true,
    });

    try {
      const deps = borgDeps(borg);
      const retrievalCalls: RetrievalCall[] = [];
      const originalSearch = deps.retrievalPipeline.searchWithContext.bind(deps.retrievalPipeline);
      vi.spyOn(deps.retrievalPipeline, "searchWithContext").mockImplementation(
        async (query, options) => {
          const result = await originalSearch(query, options);
          retrievalCalls.push({ options: options ?? {}, result });
          return result;
        },
      );
      const recallLoadSpy = vi.spyOn(recallStateRepository(deps), "load");

      await borg.turn({
        sessionId: sessionA,
        audience: AUDIENCE,
        userMessage: "Maya is my partner; she's making elaborate ramen tonight.",
        stakes: "low",
      });
      const ingestion = deps.streamIngestionCoordinator;
      expect(ingestion).toBeDefined();
      await ingestion!.ingest(sessionA, { minEntriesThreshold: 1 });

      const tomEntityId = deps.entityRepository.findByName(AUDIENCE);
      expect(tomEntityId).not.toBeNull();
      const extractedEpisodes = await deps.episodicRepository.searchByParticipantsOrTags(
        [MAYA_TERM],
        {
          audienceEntityId: tomEntityId,
          limit: 5,
        },
      );
      const sessionAEpisode = extractedEpisodes[0]?.episode;
      expect(sessionAEpisode).toBeDefined();
      expect(sessionAEpisode?.audience_entity_id).toBe(tomEntityId);
      expect(sessionAEpisode?.participants).toEqual(expect.arrayContaining([MAYA_TERM]));
      expect(sessionAEpisode?.tags).toEqual(expect.arrayContaining([MAYA_TERM]));
      sessionAEpisodeId = sessionAEpisode!.id;

      await borg.turn({
        sessionId: sessionA,
        audience: AUDIENCE,
        userMessage: "Let's talk about cloud storage for a moment.",
        stakes: "low",
      });
      await ingestion!.ingest(sessionA, { minEntriesThreshold: 1 });

      const stateAfterSessionA = recallStateRepository(deps).load(tomEntityId as EntityId);
      const sessionAHandle = stateAfterSessionA?.activeHandles.find(
        (item) => item.handle.source === "episode" && item.handle.episodeId === sessionAEpisodeId,
      );
      expect(stateAfterSessionA?.scopeKey).toBe(tomEntityId);
      expect(sessionAHandle?.firstSeenTurn).toBe(2);

      const episodeGetSpy = vi.spyOn(deps.episodicRepository, "get");
      recallLoadSpy.mockClear();
      retrievalCalls.length = 0;

      const turn3 = await borg.turn({
        sessionId: sessionB,
        audience: AUDIENCE,
        userMessage: "I think I told you about Maya earlier.",
        stakes: "low",
      });
      await ingestion!.ingest(sessionB, { minEntriesThreshold: 1 });

      const turn3Retrieval = retrievalCalls.find((call) => call.options.sessionId === sessionB);
      expect(turn3Retrieval).toBeDefined();
      const sessionAEvidence = turn3Retrieval!.result.evidence.find(
        (item) => item.provenance?.episodeId === sessionAEpisodeId,
      );
      const warmEvidence = turn3Retrieval!.result.evidence.find(
        (item) => item.source === "warm_recall" && item.provenance?.episodeId === sessionAEpisodeId,
      );
      const freshEvidence = turn3Retrieval!.result.evidence.find(
        (item) => item.source === "episode" && item.provenance?.episodeId === sessionAEpisodeId,
      );

      expect(sessionAEvidence?.provenance?.episodeId).toBe(sessionAEpisodeId);
      expect(warmEvidence ?? freshEvidence).toBeDefined();
      expect(episodeGetSpy).toHaveBeenCalledWith(sessionAEpisodeId);
      expect(turn3.retrievedEpisodeIds).toContain(sessionAEpisodeId);
      expect(turn3Retrieval!.result.episodes.map((item) => item.episode.id)).toContain(
        sessionAEpisodeId,
      );
      expect(recallLoadSpy).toHaveBeenCalledWith(tomEntityId);
      expect(recallLoadSpy).not.toHaveBeenCalledWith(sessionA);
      expect(recallLoadSpy).not.toHaveBeenCalledWith(sessionB);

      const rows = readRecallStateRows(deps);
      expect(rows.map((row) => row.scope_key)).toEqual([tomEntityId]);
      const storedState = JSON.parse(rows[0]!.state_json) as {
        scopeKey: string;
        activeHandles: Array<{
          handle: { source: string; episodeId?: string };
          firstSeenTurn: number;
        }>;
        lastRefreshTurn: number;
      };
      const storedHandle = storedState.activeHandles.find(
        (item) => item.handle.source === "episode" && item.handle.episodeId === sessionAEpisodeId,
      );
      const storedKeys = collectKeys(storedState);

      expect(storedState.scopeKey).toBe(tomEntityId);
      expect(storedState.lastRefreshTurn).toBe(3);
      expect(storedHandle?.firstSeenTurn).toBe(2);
      expect(storedKeys).not.toEqual(
        expect.arrayContaining(["text", "summary", "description", "content", "belief", "claim"]),
      );
      expect(rows[0]!.state_json).not.toContain("Maya ramen plan");
      expect(rows[0]!.state_json).not.toContain("elaborate ramen");
    } finally {
      await borg.close();
    }
  });
});
