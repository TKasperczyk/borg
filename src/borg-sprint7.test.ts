import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { Borg } from "./borg.js";
import { DEFAULT_CONFIG } from "./config/index.js";
import type { EmbeddingClient } from "./embeddings/index.js";
import { FakeLLMClient } from "./llm/index.js";
import { affectiveMigrations } from "./memory/affective/index.js";
import { commitmentMigrations } from "./memory/commitments/index.js";
import {
  EpisodicRepository,
  createEpisodesTableSchema,
  episodicMigrations,
  type Episode,
} from "./memory/episodic/index.js";
import { proceduralMigrations } from "./memory/procedural/index.js";
import { selfMigrations } from "./memory/self/index.js";
import { semanticMigrations } from "./memory/semantic/index.js";
import { socialMigrations } from "./memory/social/index.js";
import { offlineMigrations } from "./offline/index.js";
import { retrievalMigrations } from "./retrieval/index.js";
import { LanceDbStore } from "./storage/lancedb/index.js";
import { openDatabase } from "./storage/sqlite/index.js";
import { FixedClock } from "./util/clock.js";
import { DEFAULT_SESSION_ID, createEpisodeId, createStreamEntryId } from "./util/ids.js";

class RustEmbeddingClient implements EmbeddingClient {
  async embed(text: string): Promise<Float32Array> {
    return this.vector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.vector(text));
  }

  private vector(text: string): Float32Array {
    if (/rust|lifetime|borrow/i.test(text)) {
      return Float32Array.from([1, 0, 0, 0]);
    }

    return Float32Array.from([0, 1, 0, 0]);
  }
}

function createEpisode(overrides: Partial<Episode>): Episode {
  const nowMs = overrides.created_at ?? 1_000;

  return {
    id: overrides.id ?? createEpisodeId(),
    title: overrides.title ?? "Rust lifetimes",
    narrative: overrides.narrative ?? "A Rust lifetime episode.",
    participants: overrides.participants ?? ["user"],
    location: overrides.location ?? null,
    start_time: overrides.start_time ?? nowMs - 100,
    end_time: overrides.end_time ?? nowMs,
    source_stream_ids: overrides.source_stream_ids ?? [createStreamEntryId()],
    significance: overrides.significance ?? 0.7,
    tags: overrides.tags ?? ["rust", "debugging"],
    confidence: overrides.confidence ?? 0.8,
    lineage: overrides.lineage ?? {
      derived_from: [],
      supersedes: [],
    },
    emotional_arc: overrides.emotional_arc ?? null,
    embedding: overrides.embedding ?? Float32Array.from([1, 0, 0, 0]),
    created_at: nowMs,
    updated_at: overrides.updated_at ?? nowMs,
  };
}

function createEmptyReflectionResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_reflection_empty",
        name: "EmitTurnReflection",
        input: {
          advanced_goals: [],
          procedural_outcomes: [],
          trait_demonstrations: [],
          intent_updates: [],
        },
      },
    ],
  };
}

describe("Borg Sprint 7", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("carries mood across turns, prefers mood-congruent retrieval, and records skill success from follow-up", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new FixedClock(1_000);
    const embeddingClient = new RustEmbeddingClient();
    const sqlite = openDatabase(join(tempDir, "borg.db"), {
      migrations: [
        ...episodicMigrations,
        ...selfMigrations,
        ...affectiveMigrations,
        ...retrievalMigrations,
        ...semanticMigrations,
        ...commitmentMigrations,
        ...socialMigrations,
        ...proceduralMigrations,
        ...offlineMigrations,
      ],
    });
    const lance = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const episodesTable = await lance.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const episodicRepository = new EpisodicRepository({
      table: episodesTable,
      db: sqlite,
      clock,
    });

    const negativeEpisode = createEpisode({
      title: "Rust lifetime frustration",
      narrative: "The borrow checker errors felt frustrating and blocked progress.",
      emotional_arc: {
        start: { valence: -0.8, arousal: 0.7 },
        peak: { valence: -0.9, arousal: 0.8 },
        end: { valence: -0.5, arousal: 0.6 },
        dominant_emotion: "anger",
      },
    });
    const positiveEpisode = createEpisode({
      title: "Rust lifetime breakthrough",
      narrative: "The lifetime issue resolved cleanly and felt satisfying.",
      emotional_arc: {
        start: { valence: 0.3, arousal: 0.2 },
        peak: { valence: 0.8, arousal: 0.4 },
        end: { valence: 0.7, arousal: 0.3 },
        dominant_emotion: "joy",
      },
    });
    await episodicRepository.insert(negativeEpisode);
    await episodicRepository.insert(positiveEpisode);
    sqlite.close();
    await lance.close();

    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Try shrinking the borrow scope and introducing an intermediate binding.",
          input_tokens: 20,
          output_tokens: 20,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        createEmptyReflectionResponse(),
        {
          text: "Reuse the scoped-binding approach; it still fits the Rust lifetime issue.",
          input_tokens: 20,
          output_tokens: 20,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "",
          input_tokens: 20,
          output_tokens: 10,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_reflection",
              name: "EmitTurnReflection",
              input: {
                advanced_goals: [],
                procedural_outcomes: [
                  {
                    attempt_turn_counter: 1,
                    classification: "success",
                    evidence: "User confirmed the Rust lifetime error is fixed now.",
                    grounded: true,
                    skill_actually_applied: true,
                  },
                ],
              },
            },
          ],
        },
      ],
    });
    const borg = await Borg.open({
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "problem_solving",
        },
        affective: {
          ...DEFAULT_CONFIG.affective,
          useLlmFallback: false,
        },
        embedding: {
          ...DEFAULT_CONFIG.embedding,
          dims: 4,
        },
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient,
      llmClient: llm,
      liveExtraction: false,
    });

    try {
      const skill = await borg.skills.add({
        applies_when: "Rust lifetime debugging",
        approach: "Shrink borrow scopes and use intermediate bindings.",
        sourceEpisodes: [negativeEpisode.id],
      });

      const firstTurn = await borg.turn({
        userMessage: "I'm frustrated with Rust lifetime error E0597 again.",
      });
      const firstMood = borg.mood.current(DEFAULT_SESSION_ID);

      expect(firstTurn.retrievedEpisodeIds).toContain(negativeEpisode.id);
      expect(firstMood.valence).toBeLessThan(0);

      const secondTurn = await borg.turn({
        userMessage: "That Rust lifetime error E0597 is fixed now, great.",
      });
      const updatedSkill = borg.skills.get(skill.id);

      expect(secondTurn.retrievedEpisodeIds[0]).toBe(positiveEpisode.id);
      expect(updatedSkill?.alpha).toBe(2);
      expect(updatedSkill?.successes).toBe(1);
      const verificationDb = openDatabase(join(tempDir, "borg.db"), {
        migrations: [
          ...episodicMigrations,
          ...selfMigrations,
          ...affectiveMigrations,
          ...retrievalMigrations,
          ...semanticMigrations,
          ...commitmentMigrations,
          ...socialMigrations,
          ...proceduralMigrations,
          ...offlineMigrations,
        ],
      });

      try {
        expect(
          verificationDb
            .prepare(
              `
                SELECT context_key, alpha, beta, attempts, successes, failures
                FROM skill_context_stats
                WHERE skill_id = ?
              `,
            )
            .all(skill.id),
        ).toEqual([
          expect.objectContaining({
            context_key: "code_debugging:rust:unknown",
            alpha: 2,
            beta: 1,
            attempts: 1,
            successes: 1,
            failures: 0,
          }),
        ]);
      } finally {
        verificationDb.close();
      }
    } finally {
      await borg.close();
    }
  });
});
