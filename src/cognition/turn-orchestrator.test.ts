import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { Borg, FakeLLMClient, ManualClock, type LLMCompleteOptions } from "../index.js";
import type { BorgDependencies } from "../borg/types.js";
import type { ExecutiveStepsRepository } from "../executive/index.js";
import type { SelfSnapshot } from "./deliberation/deliberator.js";
import type { Episode, EpisodicRepository } from "../memory/episodic/index.js";
import { TestEmbeddingClient } from "../offline/test-support.js";
import {
  createEpisodeId,
  createStreamEntryId,
  type EntityId,
  type EpisodeId,
} from "../util/ids.js";

async function openTestBorg(tempDir: string, llm: FakeLLMClient, clock: ManualClock) {
  return Borg.open({
    config: {
      dataDir: tempDir,
      perception: {
        useLlmFallback: false,
        modeWhenLlmAbsent: "idle",
      },
      affective: {
        useLlmFallback: false,
      },
      embedding: {
        baseUrl: "http://localhost:1234/v1",
        apiKey: "test",
        model: "test-embed",
        dims: 4,
      },
      anthropic: {
        auth: "api-key",
        apiKey: "test",
        models: {
          cognition: "test-cognition",
          background: "test-background",
          extraction: "test-extraction",
        },
      },
    },
    clock,
    embeddingDimensions: 4,
    embeddingClient: new TestEmbeddingClient(),
    llmClient: llm,
    liveExtraction: false,
  });
}

function createEmptyReflectionResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_reflection",
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

function makeEpisode(input: {
  id: EpisodeId;
  now: number;
  audienceEntityId: EntityId | null;
  shared: boolean;
  title: string;
}): Episode {
  return {
    id: input.id,
    title: input.title,
    narrative: `${input.title} narrative.`,
    participants: ["Borg"],
    location: null,
    start_time: input.now,
    end_time: input.now,
    source_stream_ids: [createStreamEntryId()],
    significance: 0.7,
    tags: ["identity"],
    confidence: 0.9,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    emotional_arc: null,
    audience_entity_id: input.audienceEntityId,
    shared: input.shared,
    embedding: Float32Array.from([0, 0, 0, 1]),
    created_at: input.now,
    updated_at: input.now,
  };
}

function systemText(request: LLMCompleteOptions | undefined): string {
  const system = request?.system;

  if (typeof system === "string") {
    return system;
  }

  return system?.map((block) => block.text).join("\n") ?? "";
}

describe("TurnOrchestrator self snapshot audience visibility", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("does not surface self records backed only by another audience's private episodes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000_000);
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Public answer.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const internal = borg as unknown as {
        deps: Pick<BorgDependencies, "entityRepository"> & {
          episodicRepository: EpisodicRepository;
        };
      };
      const aliceEntityId = internal.deps.entityRepository.resolve("Alice");
      internal.deps.entityRepository.resolve("Bob");
      const alicePrivateEpisodeId = createEpisodeId();
      const publicEpisodeId = createEpisodeId();
      const now = clock.now();

      await internal.deps.episodicRepository.insert(
        makeEpisode({
          id: alicePrivateEpisodeId,
          now,
          audienceEntityId: aliceEntityId,
          shared: false,
          title: "Alice private identity evidence",
        }),
      );
      await internal.deps.episodicRepository.insert(
        makeEpisode({
          id: publicEpisodeId,
          now,
          audienceEntityId: null,
          shared: true,
          title: "Public identity evidence",
        }),
      );

      borg.self.values.add({
        label: "alice-private-value",
        description: "Alice-only value description.",
        priority: 10,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
      });
      borg.self.values.add({
        label: "public-value",
        description: "Public value description.",
        priority: 9,
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
      });
      borg.self.values.add({
        label: "mixed-visible-value",
        description: "Mixed public and private evidence stays visible.",
        priority: 8,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId, publicEpisodeId],
        },
      });
      borg.self.values.add({
        label: "manual-unscoped-value",
        description: "Manual self state has no audience-scoped evidence.",
        priority: 7,
        provenance: {
          kind: "manual",
        },
      });
      borg.self.goals.add({
        description: "alice-private-goal",
        priority: 10,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
      });
      borg.self.goals.add({
        description: "public-goal",
        priority: 9,
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
      });
      borg.self.traits.add({
        label: "alice-private-trait",
        delta: 0.4,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
        timestamp: now,
      });
      borg.self.traits.add({
        label: "public-trait",
        delta: 0.4,
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
        timestamp: now,
      });
      borg.self.traits.add({
        label: "mixed-visible-trait",
        delta: 0.2,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
        timestamp: now,
      });
      borg.self.traits.add({
        label: "mixed-visible-trait",
        delta: 0.2,
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
        timestamp: now + 1,
      });
      borg.self.growthMarkers.add({
        ts: now,
        category: "understanding",
        what_changed: "alice-private-growth",
        evidence_episode_ids: [alicePrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
      });
      borg.self.growthMarkers.add({
        ts: now + 1,
        category: "understanding",
        what_changed: "public-growth",
        evidence_episode_ids: [publicEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
      });
      borg.self.autobiographical.upsertPeriod({
        label: "alice-private-period",
        start_ts: now,
        narrative: "Alice-private period narrative.",
        key_episode_ids: [alicePrivateEpisodeId],
        themes: ["privacy"],
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
      });

      await borg.turn({
        userMessage: "Hello Bob.",
        audience: "Bob",
        stakes: "low",
      });

      const allRequestText = llm.requests.map((request) => JSON.stringify(request)).join("\n");
      const finalizerSystem = systemText(llm.requests[0]);

      expect(finalizerSystem).toContain("public-value");
      expect(finalizerSystem).toContain("mixed-visible-value");
      expect(finalizerSystem).toContain("manual-unscoped-value");
      expect(finalizerSystem).toContain("public-goal");
      expect(finalizerSystem).toContain("public-trait");
      expect(finalizerSystem).toContain("mixed-visible-trait");
      expect(finalizerSystem).toContain("public-growth");
      expect(allRequestText).not.toContain("alice-private-value");
      expect(allRequestText).not.toContain("Alice-only value description");
      expect(allRequestText).not.toContain("alice-private-goal");
      expect(allRequestText).not.toContain("alice-private-trait");
      expect(allRequestText).not.toContain("alice-private-growth");
      expect(allRequestText).not.toContain("alice-private-period");
      expect(allRequestText).not.toContain("Alice-private period narrative");
    } finally {
      await borg.close();
    }
  });

  it("keeps mixed, unscoped, matching-audience, and empty-evidence self records visible", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(2_000_000);
    const borg = await openTestBorg(tempDir, new FakeLLMClient(), clock);

    try {
      const internal = borg as unknown as {
        deps: Pick<
          BorgDependencies,
          "autobiographicalRepository" | "entityRepository" | "sqlite"
        > & {
          episodicRepository: EpisodicRepository;
          turnOrchestrator: {
            buildSelfSnapshot(audienceEntityId: EntityId | null): Promise<SelfSnapshot>;
          };
        };
      };
      const aliceEntityId = internal.deps.entityRepository.resolve("Alice");
      const bobEntityId = internal.deps.entityRepository.resolve("Bob");
      const alicePrivateEpisodeId = createEpisodeId();
      const bobPrivateEpisodeId = createEpisodeId();
      const publicEpisodeId = createEpisodeId();
      const now = clock.now();

      for (const episode of [
        makeEpisode({
          id: alicePrivateEpisodeId,
          now,
          audienceEntityId: aliceEntityId,
          shared: false,
          title: "Alice private snapshot evidence",
        }),
        makeEpisode({
          id: bobPrivateEpisodeId,
          now,
          audienceEntityId: bobEntityId,
          shared: false,
          title: "Bob private snapshot evidence",
        }),
        makeEpisode({
          id: publicEpisodeId,
          now,
          audienceEntityId: null,
          shared: true,
          title: "Public snapshot evidence",
        }),
      ]) {
        await internal.deps.episodicRepository.insert(episode);
      }

      borg.self.goals.add({
        description: "mixed-visible-goal",
        priority: 8,
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId, publicEpisodeId] },
      });
      borg.self.goals.add({
        description: "manual-visible-goal",
        priority: 7,
        provenance: { kind: "manual" },
      });
      borg.self.goals.add({
        description: "system-visible-goal",
        priority: 6,
        provenance: { kind: "system" },
      });
      borg.self.goals.add({
        description: "bob-private-visible-goal",
        priority: 5,
        provenance: { kind: "episodes", episode_ids: [bobPrivateEpisodeId] },
      });
      borg.self.traits.add({
        label: "manual-visible-trait",
        delta: 0.3,
        provenance: { kind: "manual" },
        timestamp: now,
      });
      borg.self.traits.add({
        label: "bob-private-visible-trait",
        delta: 0.3,
        provenance: { kind: "episodes", episode_ids: [bobPrivateEpisodeId] },
        timestamp: now,
      });
      borg.self.values.add({
        label: "bob-private-visible-value",
        description: "Bob-private evidence should be visible to Bob.",
        priority: 7,
        provenance: { kind: "episodes", episode_ids: [bobPrivateEpisodeId] },
      });
      const emptyValue = borg.self.values.add({
        label: "empty-evidence-visible-value",
        description: "Empty evidence should not scope the record.",
        priority: 6,
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId] },
      });
      const emptyTrait = borg.self.traits.add({
        label: "empty-evidence-visible-trait",
        delta: 0.3,
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId] },
        timestamp: now,
      });
      const emptyTraitId =
        emptyTrait.status === "applied" ? emptyTrait.record.id : emptyTrait.current.id;

      internal.deps.sqlite
        .prepare('UPDATE "values" SET evidence_episode_ids = ? WHERE id = ?')
        .run("[]", emptyValue.id);
      internal.deps.sqlite
        .prepare("UPDATE traits SET evidence_episode_ids = ? WHERE id = ?")
        .run("[]", emptyTraitId);

      borg.self.growthMarkers.add({
        ts: now,
        category: "understanding",
        what_changed: "mixed-visible-growth",
        evidence_episode_ids: [alicePrivateEpisodeId, publicEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId, publicEpisodeId] },
      });
      borg.self.growthMarkers.add({
        ts: now + 1,
        category: "understanding",
        what_changed: "manual-visible-growth",
        evidence_episode_ids: [alicePrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "manual" },
      });
      borg.self.growthMarkers.add({
        ts: now + 2,
        category: "understanding",
        what_changed: "system-visible-growth",
        evidence_episode_ids: [alicePrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "system" },
      });
      let snapshot = await internal.deps.turnOrchestrator.buildSelfSnapshot(bobEntityId);
      expect(snapshot.recentGrowthMarkers?.map((marker) => marker.what_changed)).toEqual([
        "system-visible-growth",
        "manual-visible-growth",
        "mixed-visible-growth",
      ]);

      borg.self.growthMarkers.add({
        ts: now + 3,
        category: "understanding",
        what_changed: "bob-private-visible-growth",
        evidence_episode_ids: [bobPrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "episodes", episode_ids: [bobPrivateEpisodeId] },
      });
      const emptyGrowth = borg.self.growthMarkers.add({
        ts: now + 4,
        category: "understanding",
        what_changed: "empty-evidence-visible-growth",
        evidence_episode_ids: [alicePrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId] },
      });
      internal.deps.sqlite
        .prepare("UPDATE growth_markers SET evidence_episode_ids = ? WHERE id = ?")
        .run("[]", emptyGrowth.id);

      for (const [label, provenance] of [
        ["public-visible-period", { kind: "episodes" as const, episode_ids: [publicEpisodeId] }],
        [
          "mixed-visible-period",
          { kind: "episodes" as const, episode_ids: [alicePrivateEpisodeId, publicEpisodeId] },
        ],
        ["manual-visible-period", { kind: "manual" as const }],
        ["system-visible-period", { kind: "system" as const }],
        [
          "bob-private-visible-period",
          { kind: "episodes" as const, episode_ids: [bobPrivateEpisodeId] },
        ],
        [
          "empty-evidence-visible-period",
          { kind: "episodes" as const, episode_ids: [alicePrivateEpisodeId] },
        ],
      ] as const) {
        internal.deps.autobiographicalRepository.upsertPeriod({
          label,
          start_ts: now,
          narrative: `${label} narrative.`,
          key_episode_ids: label.startsWith("empty")
            ? []
            : provenance.kind === "episodes"
              ? provenance.episode_ids
              : [alicePrivateEpisodeId],
          themes: ["visibility"],
          provenance,
        });
        snapshot = await internal.deps.turnOrchestrator.buildSelfSnapshot(bobEntityId);
        expect(snapshot.currentPeriod?.label).toBe(label);
      }

      snapshot = await internal.deps.turnOrchestrator.buildSelfSnapshot(bobEntityId);
      expect(snapshot.values.map((value) => value.label)).toEqual(
        expect.arrayContaining(["bob-private-visible-value", "empty-evidence-visible-value"]),
      );
      expect(snapshot.goals.map((goal) => goal.description)).toEqual(
        expect.arrayContaining([
          "mixed-visible-goal",
          "manual-visible-goal",
          "system-visible-goal",
          "bob-private-visible-goal",
        ]),
      );
      expect(snapshot.traits.map((trait) => trait.label)).toEqual(
        expect.arrayContaining([
          "manual-visible-trait",
          "bob-private-visible-trait",
          "empty-evidence-visible-trait",
        ]),
      );
      expect(snapshot.recentGrowthMarkers?.map((marker) => marker.what_changed)).toEqual(
        expect.arrayContaining(["bob-private-visible-growth", "empty-evidence-visible-growth"]),
      );
    } finally {
      await borg.close();
    }
  });

  it("selects an executive focus and renders it without dropping other active goals", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(3_000_000);
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Apollo answer.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      borg.self.goals.add({
        description: "Background maintenance",
        priority: 10,
        provenance: {
          kind: "system",
        },
      });
      borg.self.goals.add({
        description: "Apollo launch plan",
        priority: 9,
        provenance: {
          kind: "system",
        },
      });

      await borg.turn({
        userMessage: "Let's work on the Apollo launch plan.",
        stakes: "low",
      });

      const finalizerSystem = systemText(llm.requests[0]);
      const blockStart = finalizerSystem.indexOf("<borg_executive_focus>");
      const blockEnd = finalizerSystem.indexOf("</borg_executive_focus>");
      const executiveBlock = finalizerSystem.slice(blockStart, blockEnd);

      expect(blockStart).toBeGreaterThanOrEqual(0);
      expect(blockEnd).toBeGreaterThan(blockStart);
      expect(executiveBlock).toContain("Current driving goal: Apollo launch plan");
      expect(executiveBlock).toContain(
        "Use this as a bias, not an override of the user's request or commitments.",
      );
      expect(executiveBlock).not.toContain("Next step:");
      expect(executiveBlock).not.toContain("Background maintenance");
      expect(finalizerSystem).toContain("goals Background maintenance");
      expect(finalizerSystem).toContain("Apollo launch plan");
    } finally {
      await borg.close();
    }
  });

  it("renders the selected goal's top open executive step", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_000);
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Apollo step answer.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const internal = borg as unknown as {
        deps: {
          executiveStepsRepository: ExecutiveStepsRepository;
        };
      };
      borg.self.goals.add({
        description: "Background maintenance",
        priority: 10,
        provenance: {
          kind: "system",
        },
      });
      const selectedGoal = borg.self.goals.add({
        description: "Apollo launch plan",
        priority: 9,
        provenance: {
          kind: "system",
        },
      });
      const dueAt = clock.now() + 86_400_000;

      internal.deps.executiveStepsRepository.add({
        goalId: selectedGoal.id,
        description: "Inspect the launch readiness notes",
        kind: "research",
        dueAt,
        provenance: {
          kind: "system",
        },
      });

      await borg.turn({
        userMessage: "Let's work on the Apollo launch plan.",
        stakes: "low",
      });

      const finalizerSystem = systemText(llm.requests[0]);
      const blockStart = finalizerSystem.indexOf("<borg_executive_focus>");
      const blockEnd = finalizerSystem.indexOf("</borg_executive_focus>");
      const executiveBlock = finalizerSystem.slice(blockStart, blockEnd);

      expect(executiveBlock).toContain("Current driving goal: Apollo launch plan");
      expect(executiveBlock).toContain(
        `Next step: Inspect the launch readiness notes (kind: research, due: ${new Date(
          dueAt,
        ).toISOString()})`,
      );
    } finally {
      await borg.close();
    }
  });
});
