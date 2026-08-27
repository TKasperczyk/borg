import { afterEach, describe, expect, it, vi } from "vitest";

import type { LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../../tracing/tracer.js";
import {
  relationshipPrivateMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
} from "../../memory/common/disclosure-label.js";
import type { ReviewQueueItem } from "../../memory/review-queue/index.js";
import type { Episode } from "../../memory/episodic/index.js";
import {
  createEntityId,
  createEpisodeId,
  createSemanticNodeId,
  createStreamEntryId,
  type StreamEntryId,
} from "../../util/ids.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
} from "../test-support.js";
import { ReviewResolverProcess } from "./index.js";

const REVIEW_RESOLVER_TOOL_NAME = "EmitReviewResolverDecision";
const NEW_INSIGHT_REVIEW_RESOLVER_TOOL_NAME = "EmitNewInsightVerdict";
const SEMANTIC_PAIR_REVIEW_RESOLVER_TOOL_NAME = "EmitSemanticPairVerdict";

type OfflineHarness = Awaited<ReturnType<typeof createOfflineTestHarness>>;
type TraceEvent = { event: TurnTraceEventName } & TurnTraceData;

class ArrayTracer implements TurnTracer {
  readonly enabled = true;
  readonly includePayloads = true;
  readonly events: TraceEvent[] = [];

  emit(event: TurnTraceEventName, data: TurnTraceData): void {
    this.events.push({
      event,
      ...data,
    });
  }
}

function resolverResponse(input: Record<string, unknown>): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 7,
    output_tokens: 4,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_review_resolver",
        name: REVIEW_RESOLVER_TOOL_NAME,
        input,
      },
    ],
  };
}

function newInsightResolverResponse(input: Record<string, unknown>): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 7,
    output_tokens: 4,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_new_insight_review_resolver",
        name: NEW_INSIGHT_REVIEW_RESOLVER_TOOL_NAME,
        input,
      },
    ],
  };
}

function semanticPairResolverResponse(input: Record<string, unknown>): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 7,
    output_tokens: 4,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_semantic_pair_review_resolver",
        name: SEMANTIC_PAIR_REVIEW_RESOLVER_TOOL_NAME,
        input,
      },
    ],
  };
}

const DISTINCT_IDENTIFIER_REPLAY_FIXTURE = [
  {
    name: "distinct AININJAS tickets",
    labels: ["Ticket AININJAS-1110", "Ticket AININJAS-1111"],
    blocked: true,
  },
  {
    name: "distinct OPS tickets",
    labels: ["Ticket OPS-701 transition", "Ticket OPS-702 transition"],
    blocked: true,
  },
  {
    name: "distinct DEVX tickets",
    labels: ["DEVX-44 creation outcome", "DEVX-45 creation outcome"],
    blocked: true,
  },
  {
    name: "distinct autonomous run ids",
    labels: [
      "Autonomous Run f0bec94550ab5cd0e0d4408f710727fe",
      "Autonomous Run 8c16745d9ce140ed90c92f6ef06fb921",
    ],
    blocked: true,
  },
  {
    name: "distinct merge-request URLs",
    labels: [
      "MR (https://gitlab.example/Team/Project/-/merge_requests/34).",
      "MR https://gitlab.example/Team/Project/-/merge_requests/35",
    ],
    blocked: true,
  },
  {
    name: "distinct long digit runs",
    labels: ["Call record 48123456789", "Call record 48123456780"],
    blocked: true,
  },
  {
    name: "same ticket key",
    labels: ["ticket AININJAS-1088", "AININJAS-1088"],
    blocked: false,
  },
  {
    name: "same autonomous run id",
    labels: [
      "Run F0BEC94550AB5CD0E0D4408F710727FE",
      "Autonomous run f0bec94550ab5cd0e0d4408f710727fe",
    ],
    blocked: false,
  },
  {
    name: "same normalized URL path",
    labels: [
      "MR https://one.example/Team/Project/-/merge_requests/42?view=changes",
      "Merge request https://two.example/Team/Project/-/merge_requests/42/.",
    ],
    blocked: false,
  },
  {
    name: "shared independent batch id",
    labels: ["Ticket ABC-1 batch 123456789", "Ticket ABC-2 batch 123456789"],
    blocked: false,
  },
  {
    name: "one-sided identifier",
    labels: ["Ticket AININJAS-1090", "Ticket creation outcome"],
    blocked: false,
  },
  {
    name: "no identifiers",
    labels: ["Atlas platform", "Deployment platform"],
    blocked: false,
  },
] as const;

async function runResolver(harness: OfflineHarness, maxItemsPerPass = 3) {
  const process = new ReviewResolverProcess({
    db: harness.db,
    maxItemsPerPass,
  });

  return process.run(harness.createContext(), {});
}

async function insertSource(
  harness: OfflineHarness,
  content: string,
  episodeOverrides: Partial<Episode> = {},
) {
  const entry = await harness.streamWriter.append({
    kind: "user_msg",
    content,
  });
  const episode = await harness.episodicRepository.createEpisode(
    createEpisodeFixture({
      narrative: content,
      source_stream_ids: [entry.id],
      ...episodeOverrides,
    }),
  );

  return { entry, episode };
}

async function insertAssistantSource(harness: OfflineHarness, content: string) {
  const entry = await harness.streamWriter.append({
    kind: "agent_msg",
    content,
  });
  const episode = await harness.episodicRepository.createEpisode(
    createEpisodeFixture({
      narrative: content,
      source_stream_ids: [entry.id],
    }),
  );

  return { entry, episode };
}

function pendingNewInsightInsertRefs(input: {
  episodeIds: Episode["id"][];
  nodeId?: ReturnType<typeof createSemanticNodeId>;
  label?: string;
  description?: string;
  confidence?: number;
  clusterKey?: string;
  sourceDisclosureLabel?: MemoryDisclosureLabel;
}) {
  const nodeId = input.nodeId ?? createSemanticNodeId();
  const clusterKey = input.clusterKey ?? "cluster:new-insight";

  return {
    node_ids: [nodeId],
    episode_ids: input.episodeIds,
    evidence_cluster_key: clusterKey,
    evidence_cluster_size: input.episodeIds.length,
    ...(input.sourceDisclosureLabel === undefined
      ? {}
      : { source_disclosure_label: input.sourceDisclosureLabel }),
    reflector_pending_insight: {
      target: {
        mode: "insert" as const,
        node: {
          id: nodeId,
          kind: "proposition",
          label: input.label ?? "Rollback planning preference",
          description:
            input.description ?? "I treat rollback planning as important for release work.",
          domain: null,
          aliases: [],
          confidence: input.confidence ?? 0.5,
          source_episode_ids: input.episodeIds,
          created_at: 1_000_000,
          updated_at: 1_000_000,
          last_verified_at: 1_000_000,
          embedding: [0, 0, 1, 0],
          archived: false,
          superseded_by: null,
          status: "active",
          corrected_by: null,
          superseded_at: null,
        },
      },
      candidate_support_edges: [],
      evidence_cluster: {
        key: clusterKey,
        episode_ids: input.episodeIds,
        size: input.episodeIds.length,
      },
    },
  };
}

function overseerFlag(input: {
  kind: "misattribution" | "identity_inconsistency" | "temporal_drift";
  reason: string;
  sourceEntryId: StreamEntryId;
  sourceEpisodeId: Episode["id"];
  sourceEntryIds?: StreamEntryId[];
  sourceEpisodeIds?: Episode["id"][];
  patch?: Record<string, unknown>;
}) {
  const sourceEntryIds = input.sourceEntryIds ?? [input.sourceEntryId];
  const sourceEpisodeIds = input.sourceEpisodeIds ?? [input.sourceEpisodeId];

  return {
    kind: input.kind,
    flag_kind: input.kind,
    reason: input.reason,
    confidence: 0.9,
    ...(input.patch === undefined ? {} : { patch: input.patch }),
    source_assessment: "supports_flag",
    cited_stream_ids: sourceEntryIds,
    quoted_span: "deployment script",
    audience_entities: [],
    source_episode_ids: sourceEpisodeIds,
    source_stream_ids: sourceEntryIds,
  };
}

async function enqueueSemanticMisattribution(
  harness: OfflineHarness,
  options: {
    descriptionPatch?: string;
    aliasesPatch?: string[];
    sourceEpisodeIdsPatch?: Episode["id"][];
    sourceEpisodeOverrides?: Partial<Episode>;
  } = {},
) {
  const source = await insertSource(
    harness,
    "Ben wrote the deployment script and Alice reviewed it.",
    options.sourceEpisodeOverrides,
  );
  const node = await harness.semanticNodeRepository.insert(
    createSemanticNodeFixture({
      label: "Deployment script authorship",
      description: "Alice wrote the deployment script.",
      source_episode_ids: [source.episode.id],
    }),
  );
  const patch = {
    ...(options.descriptionPatch === undefined ? {} : { description: options.descriptionPatch }),
    ...(options.aliasesPatch === undefined ? {} : { aliases: options.aliasesPatch }),
    ...(options.sourceEpisodeIdsPatch === undefined
      ? {}
      : { source_episode_ids: options.sourceEpisodeIdsPatch }),
  };
  const item = harness.reviewQueueRepository.enqueue({
    kind: "misattribution",
    reason: "The node attributes the deployment script to the wrong actor.",
    refs: {
      target_type: "semantic_node",
      target_id: node.id,
      patch,
      evidence_stream_ids: [source.entry.id],
      overseer_flag: overseerFlag({
        kind: "misattribution",
        reason: "The node attributes the deployment script to the wrong actor.",
        sourceEntryId: source.entry.id,
        sourceEpisodeId: source.episode.id,
        patch,
      }),
    },
  });

  return {
    item,
    nodeId: node.id,
    sourceEntryId: source.entry.id,
    sourceEpisodeId: source.episode.id,
  };
}

async function enqueueEpisodeMisattribution(harness: OfflineHarness) {
  const source = await insertSource(harness, "Ben wrote the deployment helper.", {
    shared: true,
  });
  const targetAudienceId = createEntityId();
  const episode = await harness.episodicRepository.createEpisode(
    createEpisodeFixture({
      title: "Deployment helper authorship",
      narrative: "Alice wrote the deployment helper.",
      participants: ["Alice"],
      source_stream_ids: [source.entry.id],
      audience_entity_id: targetAudienceId,
      shared: false,
    }),
  );
  const patch = {
    participants: ["Ben"],
  };
  const item = harness.reviewQueueRepository.enqueue({
    kind: "misattribution",
    reason: "The episode attributes the deployment helper to the wrong actor.",
    refs: {
      target_type: "episode",
      target_id: episode.id,
      patch,
      evidence_stream_ids: [source.entry.id],
      overseer_flag: overseerFlag({
        kind: "misattribution",
        reason: "The episode attributes the deployment helper to the wrong actor.",
        sourceEntryId: source.entry.id,
        sourceEpisodeId: source.episode.id,
        patch,
      }),
    },
  });

  return {
    item,
    episodeId: episode.id,
    targetAudienceId,
    sourceEntryId: source.entry.id,
  };
}

async function enqueueTaintedSemanticMisattribution(
  harness: OfflineHarness,
  legitimateSource?: Awaited<ReturnType<typeof insertSource>>,
) {
  const tainted = await insertAssistantSource(
    harness,
    "Borg called Priya one of the three siblings.",
  );
  const evidenceSources = legitimateSource === undefined ? [tainted] : [tainted, legitimateSource];
  const patch = {
    description: "Nora and Julian are siblings; Priya is Nora's partner.",
  };
  const node = await harness.semanticNodeRepository.insert(
    createSemanticNodeFixture({
      label: "Family sibling group",
      description: "Nora, Julian, and Priya are the three siblings.",
      source_episode_ids: evidenceSources.map((source) => source.episode.id),
    }),
  );
  const item = harness.reviewQueueRepository.enqueue({
    kind: "misattribution",
    reason: "The node may have laundered Borg's own sibling label.",
    refs: {
      target_type: "semantic_node",
      target_id: node.id,
      patch,
      evidence_stream_ids: evidenceSources.map((source) => source.entry.id),
      reviewed_assistant_stream_entry_id: tainted.entry.id,
      overseer_flag: overseerFlag({
        kind: "misattribution",
        reason: "The node may have laundered Borg's own sibling label.",
        sourceEntryId: tainted.entry.id,
        sourceEpisodeId: tainted.episode.id,
        sourceEntryIds: evidenceSources.map((source) => source.entry.id),
        sourceEpisodeIds: evidenceSources.map((source) => source.episode.id),
        patch,
      }),
    },
  });

  return {
    item,
    node,
    taintedSourceEntryId: tainted.entry.id,
    legitimateSourceEntryId: legitimateSource?.entry.id ?? null,
  };
}

describe("review resolver process", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();

    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("resolves accepted semantic-node misattribution repairs by superseding instead of inline-patching embedded text", async () => {
    const llm = new FakeLLMClient();
    const tracer = new ArrayTracer();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, nodeId, sourceEntryId, sourceEpisodeId } = await enqueueSemanticMisattribution(
      harness,
      {
        descriptionPatch: "Ben wrote the deployment script.",
        sourceEpisodeOverrides: {
          audience_entity_id: createEntityId(),
          shared: false,
        },
      },
    );
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The cited source says Ben wrote the script.",
        cited_stream_ids: [sourceEntryId],
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const node = await harness.semanticNodeRepository.get(nodeId);
    const promptPayload = JSON.parse(String(llm.requests[0]?.messages[0]?.content ?? "{}")) as {
      review?: {
        disclosure_label?: { disclosure_class?: string };
        refs?: {
          overseer_flag?: unknown;
        };
      };
      target?: {
        content?: {
          disclosure?: string;
          source_episode_ids?: string[];
          disclosure_label?: { disclosure_class?: string };
        };
      };
      source_bundle?: {
        overseer_flag?: {
          source_episode_ids?: string[];
          disclosure_label?: { disclosure_class?: string };
        };
      };
    };

    expect(result.errors).toEqual([]);
    expect(resolved).toMatchObject({
      resolved_at: expect.any(Number),
      resolution: "accept",
    });
    expect(resolved?.refs.__borg_review_resolver_repair).toMatchObject({
      mode: "repair_via_supersede",
      corrected_by: sourceEntryId,
    });
    expect(node).toMatchObject({
      status: "superseded",
      corrected_by: sourceEntryId,
      description: "Alice wrote the deployment script.",
    });
    expect(promptPayload.target).toMatchObject({
      content: {
        source_episode_ids: [sourceEpisodeId],
        disclosure_label: {
          disclosure_class: "relationship_private",
        },
      },
    });
    expect(promptPayload.target?.content?.disclosure).toContain(
      "disclosure_class=relationship_private",
    );
    expect(promptPayload.source_bundle?.overseer_flag).toMatchObject({
      source_episode_ids: [sourceEpisodeId],
      disclosure_label: {
        disclosure_class: "relationship_private",
      },
    });
    expect(promptPayload.review).toMatchObject({
      disclosure_label: {
        disclosure_class: "relationship_private",
      },
    });
    expect(promptPayload.review?.refs?.overseer_flag).toBeUndefined();
    expect(
      tracer.events.find((event) => event.event === "semantic_node.status.transitioned"),
    ).toMatchObject({
      nodeId,
      toStatus: "superseded",
      correctedBy: sourceEntryId,
      source: "review_resolver",
    });
    expect(
      tracer.events.find((event) => event.event === "review_resolver.decision.completed"),
    ).toMatchObject({
      review_id: item.id,
      kind: "misattribution",
      verdict: "accept_repair",
      applied_resolution: "repair_via_supersede",
    });
  });

  it("resolves vector-only duplicate candidates by superseding after LLM compatibility judgment", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const source = await insertSource(
      harness,
      "Atlas platform and deployment platform refer to the same deployment service.",
    );
    const winner = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Atlas platform",
        description: "Atlas is the deployment service.",
        confidence: 0.9,
        source_episode_ids: [source.episode.id],
      }),
    );
    const loser = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Deployment platform",
        description: "The deployment platform is Atlas.",
        confidence: 0.6,
        source_episode_ids: [source.episode.id],
      }),
    );
    const item = harness.reviewQueueRepository.enqueue({
      kind: "duplicate",
      reason: "Vector-only semantic merge candidate with similarity 0.920",
      refs: {
        node_ids: [loser.id, winner.id],
        node_labels: [loser.label, winner.label],
        duplicate_subtype: "vector_only_merge_candidate",
        vector_similarity: 0.92,
        source_overlap: {
          candidate_source_episode_ids: [source.episode.id],
          matched_source_episode_ids: [source.episode.id],
          overlapping_source_episode_ids: [source.episode.id],
          overlap_count: 1,
        },
      },
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The supplied node records describe the same deployment service.",
        support_basis: "direct_user_or_source",
        cited_stream_ids: [loser.id, winner.id],
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const storedWinner = await harness.semanticNodeRepository.get(winner.id);
    const storedLoser = await harness.semanticNodeRepository.get(loser.id);
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
    const promptPayload = JSON.parse(prompt) as {
      review?: {
        disclosure_label?: { disclosure_class?: string };
        refs?: { overseer_flag?: unknown };
      };
    };

    expect(result.errors).toEqual([]);
    expect(result.candidate_stats).toMatchObject({
      accepted: 1,
      rejected: 0,
    });
    expect(resolved).toMatchObject({
      resolved_at: expect.any(Number),
      resolution: "supersede",
    });
    expect(storedWinner?.status).toBe("active");
    expect(storedLoser).toMatchObject({
      status: "superseded",
      corrected_by: winner.id,
    });
    expect(llm.requests[0]?.tools?.[0]?.name).toBe(REVIEW_RESOLVER_TOOL_NAME);
    expect(prompt).toContain("vector_only_merge_candidate");
    expect(prompt).toContain("Do not populate stream citations");
    expect(prompt).toContain('"disclosure_label"');
    expect(prompt).toContain('"disclosure_class": "public"');
    expect(promptPayload.review?.disclosure_label).toMatchObject({
      disclosure_class: "public",
    });
    expect(promptPayload.review?.refs?.overseer_flag).toBeUndefined();
  });

  it.each([
    { autonomous: false, mode: "default", path: "vector" },
    { autonomous: true, mode: "autonomous", path: "vector" },
    { autonomous: false, mode: "default", path: "semantic-pair" },
    { autonomous: true, mode: "autonomous", path: "semantic-pair" },
  ] as const)(
    "replays the distinct-identifier matrix through the $path path in $mode mode",
    async ({ autonomous, path }) => {
      const llm = new FakeLLMClient();
      const tracer = new ArrayTracer();
      const harness = await createOfflineTestHarness({
        llmClient: llm,
        reviewOpenQuestionExtractor: null,
        tracer,
        configOverrides: {
          offline: {
            reviewResolver: {
              autonomous,
            },
          },
        },
      });
      cleanup.push(harness.cleanup);
      const replayRows: Array<{
        fixture: (typeof DISTINCT_IDENTIFIER_REPLAY_FIXTURE)[number];
        item: ReviewQueueItem;
        nodeIds: [ReturnType<typeof createSemanticNodeId>, ReturnType<typeof createSemanticNodeId>];
        sourceEpisodeId: Episode["id"];
      }> = [];

      for (const fixture of DISTINCT_IDENTIFIER_REPLAY_FIXTURE) {
        const sourceEpisode = await harness.episodicRepository.createEpisode(
          createEpisodeFixture({
            narrative: `Sanitized replay evidence for ${fixture.name}.`,
          }),
        );
        const first = await harness.semanticNodeRepository.insert(
          createSemanticNodeFixture({
            label: fixture.labels[0],
            description: `Sanitized replay node A for ${fixture.name}.`,
            source_episode_ids: [sourceEpisode.id],
          }),
        );
        const second = await harness.semanticNodeRepository.insert(
          createSemanticNodeFixture({
            label: fixture.labels[1],
            description: `Sanitized replay node B for ${fixture.name}.`,
            source_episode_ids: [sourceEpisode.id],
          }),
        );
        const item = harness.reviewQueueRepository.enqueue({
          kind: "duplicate",
          reason:
            path === "vector"
              ? "Vector-only semantic merge candidate with similarity 0.990"
              : "Sanitized semantic-pair replay candidate",
          refs: {
            node_ids: [first.id, second.id],
            node_labels: [first.label, second.label],
            ...(path === "vector"
              ? {
                  duplicate_subtype: "vector_only_merge_candidate" as const,
                  vector_similarity: 0.99,
                  source_overlap: {
                    candidate_source_episode_ids: [sourceEpisode.id],
                    matched_source_episode_ids: [sourceEpisode.id],
                    overlapping_source_episode_ids: [sourceEpisode.id],
                    overlap_count: 1,
                  },
                }
              : {}),
          },
        });
        replayRows.push({
          fixture,
          item,
          nodeIds: [first.id, second.id],
          sourceEpisodeId: sourceEpisode.id,
        });

        if (!fixture.blocked) {
          llm.pushResponse(
            path === "vector"
              ? resolverResponse({
                  verdict: "dismiss_false_positive",
                  reason: "The replay pair reached the vector LLM judge.",
                })
              : semanticPairResolverResponse({
                  decision: "keep_both",
                  rationale: "The replay pair reached the semantic-pair LLM judge.",
                  confidence: "high",
                }),
          );
        }
      }

      const episodeGetMany = vi.spyOn(harness.episodicRepository, "getMany");
      const result = await runResolver(harness, DISTINCT_IDENTIFIER_REPLAY_FIXTURE.length);
      const requestPrompts = llm.requests.map((request) =>
        String(request.messages[0]?.content ?? ""),
      );
      const fallthroughCount = DISTINCT_IDENTIFIER_REPLAY_FIXTURE.filter(
        (fixture) => !fixture.blocked,
      ).length;
      const evidenceReadEpisodeIds = new Set(
        episodeGetMany.mock.calls.flatMap(([episodeIds]) => episodeIds),
      );

      expect(result.errors).toEqual([]);
      expect(llm.requests).toHaveLength(fallthroughCount);
      expect(
        llm.requests.every(
          (request) =>
            request.tools?.[0]?.name ===
            (path === "vector"
              ? REVIEW_RESOLVER_TOOL_NAME
              : SEMANTIC_PAIR_REVIEW_RESOLVER_TOOL_NAME),
        ),
      ).toBe(true);

      for (const { fixture, item, nodeIds, sourceEpisodeId } of replayRows) {
        const pairReachedLlm = requestPrompts.some(
          (prompt) => prompt.includes(fixture.labels[0]) && prompt.includes(fixture.labels[1]),
        );
        const resolved = harness.reviewQueueRepository.get(item.id);
        const stored = await harness.semanticNodeRepository.getMany(nodeIds);

        expect(
          stored.map((node) => node?.status),
          fixture.name,
        ).toEqual(["active", "active"]);

        if (fixture.blocked) {
          const queueTrace = tracer.events.find(
            (event) =>
              event.event === "review_queue.completed" &&
              event.item_id === item.id &&
              event.resolution === "keep_both",
          );

          expect(pairReachedLlm, fixture.name).toBe(false);
          expect(evidenceReadEpisodeIds.has(sourceEpisodeId), fixture.name).toBe(false);
          expect(resolved, fixture.name).toMatchObject({
            resolved_at: expect.any(Number),
            resolution: "keep_both",
          });
          expect(queueTrace, fixture.name).toMatchObject({
            decision_reason: expect.stringContaining("machine-identifier integrity guard"),
          });
        } else {
          expect(pairReachedLlm, fixture.name).toBe(true);
          expect(evidenceReadEpisodeIds.has(sourceEpisodeId), fixture.name).toBe(true);
          expect(resolved?.resolution, fixture.name).toBe(
            path === "vector" ? "dismiss" : "keep_both",
          );
        }
      }
    },
  );

  it("fails closed to unknown labels for review and overseer flags without source episodes", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const source = await insertSource(harness, "Ben wrote the deployment script.");
    const node = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Deployment script authorship",
        description: "Alice wrote the deployment script.",
        source_episode_ids: [source.episode.id],
      }),
    );
    const item = harness.reviewQueueRepository.enqueue({
      kind: "misattribution",
      reason: "The node attributes the deployment script to the wrong actor.",
      refs: {
        target_type: "semantic_node",
        target_id: node.id,
        patch: {
          description: "Ben wrote the deployment script.",
        },
        evidence_stream_ids: [source.entry.id],
        overseer_flag: {
          kind: "misattribution",
          flag_kind: "misattribution",
          reason: "The node attributes the deployment script to the wrong actor.",
          confidence: 0.9,
          patch: {
            description: "Ben wrote the deployment script.",
          },
          source_assessment: "supports_flag",
          cited_stream_ids: [source.entry.id],
          quoted_span: "deployment script",
          audience_entities: [],
          source_stream_ids: [source.entry.id],
        },
      },
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "dismiss_false_positive",
        reason: "The supplied source does not require a repair.",
        cited_stream_ids: [],
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const promptPayload = JSON.parse(String(llm.requests[0]?.messages[0]?.content ?? "{}")) as {
      review?: {
        disclosure_label?: { disclosure_class?: string };
        refs?: { overseer_flag?: unknown };
      };
      source_bundle?: {
        overseer_flag?: {
          source_episode_ids?: string[];
          disclosure_label?: { disclosure_class?: string };
        };
      };
    };

    expect(result.errors).toEqual([]);
    expect(resolved?.resolution).toBe("dismiss");
    expect(promptPayload.source_bundle?.overseer_flag?.source_episode_ids).toBeUndefined();
    expect(promptPayload.source_bundle?.overseer_flag?.disclosure_label).toMatchObject({
      disclosure_class: "unknown",
    });
    expect(promptPayload.review?.disclosure_label).toMatchObject({
      disclosure_class: "unknown",
    });
    expect(promptPayload.review?.refs?.overseer_flag).toBeUndefined();
  });

  it("fails closed to unknown labels for vector-duplicate review rows without source labels", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const missingEpisodeId = createEpisodeId();
    const first = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Atlas platform",
        description: "Atlas is the deployment service.",
        source_episode_ids: [missingEpisodeId],
      }),
    );
    const second = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Deployment platform",
        description: "Deployment platform refers to Atlas.",
        source_episode_ids: [missingEpisodeId],
      }),
    );
    const item = harness.reviewQueueRepository.enqueue({
      kind: "duplicate",
      reason: "Vector-only semantic merge candidate with similarity 0.910",
      refs: {
        node_ids: [first.id, second.id],
        node_labels: [first.label, second.label],
        duplicate_subtype: "vector_only_merge_candidate",
        vector_similarity: 0.91,
        source_overlap: {
          candidate_source_episode_ids: [missingEpisodeId],
          matched_source_episode_ids: [missingEpisodeId],
          overlapping_source_episode_ids: [missingEpisodeId],
          overlap_count: 1,
        },
      },
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "dismiss_false_positive",
        reason: "The supplied node records are not compatible duplicates.",
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const promptPayload = JSON.parse(String(llm.requests[0]?.messages[0]?.content ?? "{}")) as {
      review?: {
        disclosure_label?: { disclosure_class?: string };
        refs?: { overseer_flag?: unknown };
      };
    };

    expect(result.errors).toEqual([]);
    expect(resolved?.resolution).toBe("dismiss");
    expect(promptPayload.review?.disclosure_label).toMatchObject({
      disclosure_class: "unknown",
    });
    expect(promptPayload.review?.refs?.overseer_flag).toBeUndefined();
  });

  it("leaves identifiers found only in aliases to the vector duplicate LLM", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const source = await insertSource(
      harness,
      "Atlas platform and Atlas expedition planning are different contexts.",
    );
    const first = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Atlas platform",
        description: "Atlas is the deployment service.",
        aliases: ["AININJAS-1110"],
        confidence: 0.9,
        source_episode_ids: [source.episode.id],
      }),
    );
    const second = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Atlas expedition planning",
        description: "Atlas is an expedition planning codename.",
        aliases: ["AININJAS-1111"],
        confidence: 0.8,
        source_episode_ids: [source.episode.id],
      }),
    );
    const item = harness.reviewQueueRepository.enqueue({
      kind: "duplicate",
      reason: "Vector-only semantic merge candidate with similarity 0.910",
      refs: {
        node_ids: [first.id, second.id],
        node_labels: [first.label, second.label],
        duplicate_subtype: "vector_only_merge_candidate",
        vector_similarity: 0.91,
        source_overlap: {
          candidate_source_episode_ids: [source.episode.id],
          matched_source_episode_ids: [source.episode.id],
          overlapping_source_episode_ids: [source.episode.id],
          overlap_count: 1,
        },
      },
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "dismiss_false_positive",
        reason: "The supplied node records describe different things.",
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const storedFirst = await harness.semanticNodeRepository.get(first.id);
    const storedSecond = await harness.semanticNodeRepository.get(second.id);

    expect(result.errors).toEqual([]);
    expect(result.candidate_stats).toMatchObject({
      accepted: 1,
      rejected: 0,
    });
    expect(resolved).toMatchObject({
      resolved_at: expect.any(Number),
      resolution: "dismiss",
    });
    expect(storedFirst?.status).toBe("active");
    expect(storedSecond?.status).toBe("active");
    expect(llm.requests).toHaveLength(1);
  });

  it("resolves contradiction supersede verdicts with a validated semantic-pair winner", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const source = await insertSource(
      harness,
      "Atlas launch plan replaced the older deployment-platform wording.",
    );
    const winner = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Atlas launch plan",
        description: "Atlas is the current deployment launch plan.",
        confidence: 0.9,
        source_episode_ids: [source.episode.id],
      }),
    );
    const loser = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Deployment platform launch plan",
        description: "The deployment platform launch plan is the current Atlas plan.",
        confidence: 0.5,
        source_episode_ids: [source.episode.id],
      }),
    );
    const item = harness.reviewQueueRepository.enqueue({
      kind: "contradiction",
      reason: "Direct contradiction edge recorded for review",
      refs: {
        node_ids: [loser.id, winner.id],
        node_labels: [loser.label, winner.label],
      },
    });
    llm.pushResponse(
      semanticPairResolverResponse({
        decision: "supersede",
        winner_node_id: winner.id,
        rationale: "The Atlas node is better grounded and more current.",
        confidence: "high",
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const storedWinner = await harness.semanticNodeRepository.get(winner.id);
    const storedLoser = await harness.semanticNodeRepository.get(loser.id);
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
    const promptPayload = JSON.parse(prompt) as {
      review?: {
        disclosure_label?: { disclosure_class?: string };
        refs?: { overseer_flag?: unknown };
      };
    };

    expect(result.errors).toEqual([]);
    expect(resolved).toMatchObject({
      resolved_at: expect.any(Number),
      resolution: "supersede",
    });
    expect(storedWinner?.status).toBe("active");
    expect(storedLoser).toMatchObject({
      status: "superseded",
      corrected_by: winner.id,
    });
    expect(llm.requests[0]?.tools?.[0]?.name).toBe(SEMANTIC_PAIR_REVIEW_RESOLVER_TOOL_NAME);
    expect(prompt).toContain("evidence_by_node");
    expect(prompt).toContain("Atlas launch plan replaced");
    expect(prompt).toContain('"disclosure_label"');
    expect(prompt).toContain('"disclosure_class": "public"');
    expect(promptPayload.review?.disclosure_label).toMatchObject({
      disclosure_class: "public",
    });
    expect(promptPayload.review?.refs?.overseer_flag).toBeUndefined();
  });

  it("keeps invalid semantic-pair winners out of the handler and marks needs_manual", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const source = await insertSource(harness, "Atlas and rollback notes need review.");
    const first = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Atlas note",
        description: "Atlas is the deployment note.",
        source_episode_ids: [source.episode.id],
      }),
    );
    const second = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Rollback note",
        description: "Rollback is the deployment note.",
        source_episode_ids: [source.episode.id],
      }),
    );
    const invalidWinnerId = createSemanticNodeId();
    const item = harness.reviewQueueRepository.enqueue({
      kind: "contradiction",
      reason: "Direct contradiction edge recorded for review",
      refs: {
        node_ids: [first.id, second.id],
      },
    });
    const resolveSpy = vi.spyOn(harness.reviewQueueRepository, "resolve");
    llm.pushResponse(
      semanticPairResolverResponse({
        decision: "supersede",
        winner_node_id: invalidWinnerId,
        rationale: "The judge named a node outside the reviewed pair.",
        confidence: "medium",
      }),
    );

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);
    const storedFirst = await harness.semanticNodeRepository.get(first.id);
    const storedSecond = await harness.semanticNodeRepository.get(second.id);

    expect(result.errors).toEqual([]);
    expect(resolveSpy).not.toHaveBeenCalled();
    expect(open).toMatchObject({
      resolved_at: null,
      resolution: null,
    });
    expect(open?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "needs_manual",
      reason: "semantic_pair_winner_out_of_pair",
    });
    expect(storedFirst?.status).toBe("active");
    expect(storedSecond?.status).toBe("active");
  });

  it("routes non-vector duplicate pairs through the semantic-pair judge for keep_both", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const source = await insertSource(
      harness,
      "Atlas the service and Atlas the trip are separate.",
    );
    const first = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Atlas service",
        description: "Atlas is a deployment service.",
        source_episode_ids: [source.episode.id],
      }),
    );
    const second = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Atlas trip",
        description: "Atlas is a travel planning codename.",
        source_episode_ids: [source.episode.id],
      }),
    );
    const item = harness.reviewQueueRepository.enqueue({
      kind: "duplicate",
      reason: "Semantic pair duplicate review",
      refs: {
        node_ids: [first.id, second.id],
        node_labels: [first.label, second.label],
      },
    });
    llm.pushResponse(
      semanticPairResolverResponse({
        decision: "keep_both",
        rationale: "The nodes describe different contexts.",
        confidence: "high",
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(result.errors).toEqual([]);
    expect(resolved).toMatchObject({
      resolved_at: expect.any(Number),
      resolution: "keep_both",
    });
    expect(llm.requests[0]?.tools?.[0]?.name).toBe(SEMANTIC_PAIR_REVIEW_RESOLVER_TOOL_NAME);
    expect(prompt).not.toContain("vector_match");
    expect(prompt).not.toContain("Do not populate stream citations");
  });

  it("dismisses semantic-pair false positives without mutating either node", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const source = await insertSource(harness, "Atlas and rollback notes are compatible.");
    const first = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Atlas note",
        description: "Atlas is a deployment service.",
        source_episode_ids: [source.episode.id],
      }),
    );
    const second = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Rollback note",
        description: "Rollback planning supports Atlas deployments.",
        source_episode_ids: [source.episode.id],
      }),
    );
    const item = harness.reviewQueueRepository.enqueue({
      kind: "contradiction",
      reason: "Direct contradiction edge recorded for review",
      refs: {
        node_ids: [first.id, second.id],
      },
    });
    llm.pushResponse(
      semanticPairResolverResponse({
        decision: "dismiss",
        rationale: "The review flag is spurious.",
        confidence: "high",
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const storedFirst = await harness.semanticNodeRepository.get(first.id);
    const storedSecond = await harness.semanticNodeRepository.get(second.id);

    expect(result.errors).toEqual([]);
    expect(resolved).toMatchObject({
      resolved_at: expect.any(Number),
      resolution: "dismiss",
    });
    expect(storedFirst?.status).toBe("active");
    expect(storedSecond?.status).toBe("active");
  });

  it("accepts grounded new insight proposals through the existing review handler", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const sourceAudienceId = createEntityId();
    const source = await insertSource(
      harness,
      "I repeatedly ask for rollback plans before risky release changes.",
      {
        audience_entity_id: sourceAudienceId,
        shared: false,
      },
    );
    const nodeId = createSemanticNodeId();
    const item = harness.reviewQueueRepository.enqueue({
      kind: "new_insight",
      reason: "New low-confidence insight extracted from cluster:release",
      refs: pendingNewInsightInsertRefs({
        nodeId,
        episodeIds: [source.episode.id],
        label: "Rollback planning preference",
        description: "I value rollback planning before risky release changes.",
      }),
    });
    llm.pushResponse(
      newInsightResolverResponse({
        decision: "accept",
        confidence: "high",
        rationale: "The supplied episode directly grounds a useful self-memory.",
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const stored = await harness.semanticNodeRepository.get(nodeId);
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
    const promptPayload = JSON.parse(prompt) as {
      review?: {
        disclosure?: string;
        disclosure_label?: {
          disclosure_class?: string;
          private_to_entity_ids?: string[];
        };
      };
      evidence_cluster?: {
        disclosure?: string;
        disclosure_label?: {
          disclosure_class?: string;
          private_to_entity_ids?: string[];
        };
      };
    };

    expect(result.errors).toEqual([]);
    expect(resolved).toMatchObject({
      resolved_at: expect.any(Number),
      resolution: "accept",
    });
    expect(stored).toMatchObject({
      id: nodeId,
      label: "Rollback planning preference",
      description: "I value rollback planning before risky release changes.",
    });
    expect(llm.requests[0]?.tools?.[0]?.name).toBe(NEW_INSIGHT_REVIEW_RESOLVER_TOOL_NAME);
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: NEW_INSIGHT_REVIEW_RESOLVER_TOOL_NAME,
    });
    expect(prompt).toContain("evidence_fetch_bound");
    expect(prompt).toContain("I repeatedly ask for rollback plans");
    expect(promptPayload.review).toMatchObject({
      disclosure_label: expect.objectContaining({
        disclosure_class: "relationship_private",
        private_to_entity_ids: [sourceAudienceId],
      }),
    });
    expect(promptPayload.review?.disclosure).toContain("disclosure_class=relationship_private");
    expect(promptPayload.evidence_cluster).toMatchObject({
      disclosure_label: expect.objectContaining({
        disclosure_class: "relationship_private",
        private_to_entity_ids: [sourceAudienceId],
      }),
    });
    expect(promptPayload.evidence_cluster?.disclosure).toContain(
      "disclosure_class=relationship_private",
    );
  });

  it("dismisses noisy new insight proposals without inserting the pending node", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const source = await insertSource(harness, "The release sync covered several unrelated notes.");
    const nodeId = createSemanticNodeId();
    const item = harness.reviewQueueRepository.enqueue({
      kind: "new_insight",
      reason: "New low-confidence insight extracted from cluster:noisy",
      refs: pendingNewInsightInsertRefs({
        nodeId,
        episodeIds: [source.episode.id],
        label: "Release sync identity shift",
        description: "I have a stable identity shift around release syncs.",
      }),
    });
    llm.pushResponse(
      newInsightResolverResponse({
        decision: "dismiss",
        confidence: "high",
        rationale: "The supplied evidence does not ground the proposed self-memory.",
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const stored = await harness.semanticNodeRepository.get(nodeId);

    expect(result.errors).toEqual([]);
    expect(resolved).toMatchObject({
      resolved_at: expect.any(Number),
      resolution: "dismiss",
    });
    expect(stored).toBeNull();
  });

  it("keeps persisted private source disclosure on new insight prompt rows", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const persistedAudienceId = createEntityId();
    const source = await insertSource(harness, "Public evidence mentioned rollback planning.", {
      shared: true,
    });
    const nodeId = createSemanticNodeId();
    const item = harness.reviewQueueRepository.enqueue({
      kind: "new_insight",
      reason: "New low-confidence insight extracted from cluster:persisted-private",
      refs: pendingNewInsightInsertRefs({
        nodeId,
        episodeIds: [source.episode.id],
        sourceDisclosureLabel: relationshipPrivateMemoryDisclosureLabel([persistedAudienceId]),
      }),
    });
    llm.pushResponse(
      newInsightResolverResponse({
        decision: "dismiss",
        confidence: "high",
        rationale: "The supplied evidence does not ground the proposed self-memory.",
      }),
    );

    const result = await runResolver(harness);
    const promptPayload = JSON.parse(String(llm.requests[0]?.messages[0]?.content ?? "{}")) as {
      review?: {
        disclosure_label?: {
          disclosure_class?: string;
          private_to_entity_ids?: string[];
        };
      };
      evidence_cluster?: {
        disclosure_label?: {
          disclosure_class?: string;
          private_to_entity_ids?: string[];
        };
      };
      proposed_insight?: {
        disclosure_label?: {
          disclosure_class?: string;
          private_to_entity_ids?: string[];
        };
      };
    };

    expect(result.errors).toEqual([]);
    expect(promptPayload.review?.disclosure_label).toMatchObject({
      disclosure_class: "relationship_private",
      private_to_entity_ids: [persistedAudienceId],
    });
    expect(promptPayload.review?.disclosure_label?.disclosure_class).not.toBe("public");
    expect(promptPayload.evidence_cluster?.disclosure_label).toMatchObject({
      disclosure_class: "relationship_private",
      private_to_entity_ids: [persistedAudienceId],
    });
    expect(promptPayload.evidence_cluster?.disclosure_label?.disclosure_class).not.toBe("public");
    expect(promptPayload.proposed_insight?.disclosure_label).toMatchObject({
      disclosure_class: "relationship_private",
      private_to_entity_ids: [persistedAudienceId],
    });
    expect(promptPayload.proposed_insight?.disclosure_label?.disclosure_class).not.toBe("public");
  });

  it("keeps current-node private disclosure on new insight update proposed rows", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const currentAudienceId = createEntityId();
    const currentSource = await insertSource(harness, "Private current source for Atlas.", {
      audience_entity_id: currentAudienceId,
      shared: false,
    });
    const patchSource = await insertSource(harness, "Public patch source for Atlas.", {
      shared: true,
    });
    const node = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Atlas deployment preference",
        description: "I prefer Atlas deployment rollback planning.",
        source_episode_ids: [currentSource.episode.id],
      }),
    );
    const item = harness.reviewQueueRepository.enqueue({
      kind: "new_insight",
      reason: "Updated low-confidence insight extracted from cluster:update-private",
      refs: {
        node_ids: [node.id],
        episode_ids: [patchSource.episode.id],
        evidence_cluster_key: "cluster:update-private",
        evidence_cluster_size: 1,
        reflector_pending_insight: {
          target: {
            mode: "update" as const,
            node_id: node.id,
            patch: {
              description: "I strongly prefer rollback planning before Atlas deployments.",
              confidence: 0.64,
              source_episode_ids: [patchSource.episode.id],
              last_verified_at: 1_000_000,
              embedding: [0, 0, 1, 0],
              archived: false,
            },
          },
          candidate_support_edges: [],
          evidence_cluster: {
            key: "cluster:update-private",
            episode_ids: [patchSource.episode.id],
            size: 1,
          },
        },
      },
    });
    llm.pushResponse(
      newInsightResolverResponse({
        decision: "dismiss",
        confidence: "high",
        rationale: "The supplied evidence does not ground the proposed update.",
      }),
    );

    const result = await runResolver(harness);
    const promptPayload = JSON.parse(String(llm.requests[0]?.messages[0]?.content ?? "{}")) as {
      proposed_insight?: {
        description?: string;
        disclosure_label?: {
          disclosure_class?: string;
          private_to_entity_ids?: string[];
        };
      };
    };

    expect(result.errors).toEqual([]);
    expect(promptPayload.proposed_insight).toMatchObject({
      description: "I strongly prefer rollback planning before Atlas deployments.",
      disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [currentAudienceId],
      },
    });
    expect(promptPayload.proposed_insight?.disclosure_label?.disclosure_class).not.toBe("public");
  });

  it("keeps ambiguous new insight proposals open with a resolver diagnostic", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const source = await insertSource(
      harness,
      "The release sync mentioned rollback planning once.",
    );
    const nodeId = createSemanticNodeId();
    const item = harness.reviewQueueRepository.enqueue({
      kind: "new_insight",
      reason: "New low-confidence insight extracted from cluster:ambiguous",
      refs: pendingNewInsightInsertRefs({
        nodeId,
        episodeIds: [source.episode.id],
      }),
    });
    llm.pushResponse(
      newInsightResolverResponse({
        decision: "needs_manual",
        confidence: "medium",
        rationale: "The evidence is too ambiguous to decide automatically.",
      }),
    );

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);
    const stored = await harness.semanticNodeRepository.get(nodeId);

    expect(result.errors).toEqual([]);
    expect(open).toMatchObject({
      resolved_at: null,
      resolution: null,
    });
    expect(open?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "needs_manual",
      reason: "The evidence is too ambiguous to decide automatically.",
      process: "review-resolver",
    });
    expect(stored).toBeNull();
  });

  it("processes new insight proposals only up to the configured cap", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const items: ReviewQueueItem[] = [];

    for (let index = 0; index < 4; index += 1) {
      const source = await insertSource(harness, `Release note ${index} mentioned rollback plans.`);
      items.push(
        harness.reviewQueueRepository.enqueue({
          kind: "new_insight",
          reason: `New low-confidence insight extracted from cluster:${index}`,
          refs: pendingNewInsightInsertRefs({
            nodeId: createSemanticNodeId(),
            episodeIds: [source.episode.id],
            clusterKey: `cluster:${index}`,
          }),
        }),
      );
      llm.pushResponse(
        newInsightResolverResponse({
          decision: "dismiss",
          confidence: "high",
          rationale: "The proposal should not be preserved.",
        }),
      );
    }

    const result = await runResolver(harness, 2);
    const resolved = items.filter(
      (item) => harness.reviewQueueRepository.get(item.id)?.resolved_at !== null,
    );
    const open = items.filter(
      (item) => harness.reviewQueueRepository.get(item.id)?.resolved_at === null,
    );

    expect(result.changes).toHaveLength(2);
    expect(resolved).toHaveLength(2);
    expect(open).toHaveLength(2);
    expect(llm.requests).toHaveLength(2);
  });

  it("leaves new insight proposals open when the review resolver budget is exhausted", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
      configOverrides: {
        offline: {
          reviewResolver: {
            budget: 1_000,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);
    const source = await insertSource(harness, "I want evidence before retaining memories.");
    const nodeId = createSemanticNodeId();
    const item = harness.reviewQueueRepository.enqueue({
      kind: "new_insight",
      reason: "New low-confidence insight extracted from cluster:budget",
      refs: pendingNewInsightInsertRefs({
        nodeId,
        episodeIds: [source.episode.id],
      }),
    });
    llm.pushResponse(
      newInsightResolverResponse({
        decision: "accept",
        confidence: "high",
        rationale: "The supplied episode grounds the proposed self-memory.",
      }),
    );

    const process = new ReviewResolverProcess({ db: harness.db, maxItemsPerPass: 3 });
    const plan = await process.plan(harness.createContext(), { budget: 10 });
    expect(plan.budget).toBe(10);
    const result = await process.apply(harness.createContext(), plan);
    const open = harness.reviewQueueRepository.get(item.id);
    const stored = await harness.semanticNodeRepository.get(nodeId);

    expect(result.budget_exhausted).toBe(true);
    expect(result.tokens_used).toBe(11);
    expect(result.errors).toHaveLength(1);
    expect(open).toMatchObject({
      resolved_at: null,
      resolution: null,
    });
    expect(stored).toBeNull();
  });

  it("keeps needs_manual reviews open with a resolver diagnostic", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, nodeId } = await enqueueSemanticMisattribution(harness, {
      descriptionPatch: "Ben wrote the deployment script.",
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "needs_manual",
        reason: "The source needs a human canonicalization decision.",
        cited_stream_ids: [],
      }),
    );

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);
    const node = await harness.semanticNodeRepository.get(nodeId);

    expect(result.changes).toHaveLength(1);
    expect(open).toMatchObject({
      resolved_at: null,
      resolution: null,
    });
    expect(open?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "needs_manual",
      reason: "The source needs a human canonicalization decision.",
      process: "review-resolver",
    });
    expect(node?.status).toBe("active");
  });

  it("does not accept a repair supported only by the assistant output under review", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, node, taintedSourceEntryId } =
      await enqueueTaintedSemanticMisattribution(harness);
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The cited utterance says Priya was treated as a sibling.",
        cited_stream_ids: [taintedSourceEntryId],
      }),
    );

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);
    const storedNode = await harness.semanticNodeRepository.get(node.id);
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(result.candidate_stats).toMatchObject({
      proposed: 1,
      accepted: 0,
    });
    expect(open).toMatchObject({
      resolved_at: null,
      resolution: null,
    });
    expect(open?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "needs_manual",
      reason: "tainted_assistant_output_under_review_cannot_independently_support_claim",
    });
    expect(storedNode?.status).toBe("active");
    expect(prompt).toContain("evidence_hierarchy");
    expect(prompt).toContain("assistant_output_under_review");
    expect(prompt).toContain("cannot independently support the claim");
    expect(prompt).toContain('"disclosure_label"');
    expect(prompt).toContain('"disclosure_class": "unknown"');
  });

  it("does not accept a zero-citation repair", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, nodeId } = await enqueueSemanticMisattribution(harness, {
      descriptionPatch: "Ben wrote the deployment script.",
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The repair is asserted without a citation.",
        cited_stream_ids: [],
      }),
    );

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);
    const node = await harness.semanticNodeRepository.get(nodeId);

    expect(result.candidate_stats).toMatchObject({
      accepted: 0,
    });
    expect(open?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "needs_manual",
      reason: "accept_repair_requires_loaded_non_tainted_citation",
    });
    expect(node?.status).toBe("active");
  });

  it("keeps overseer accept_repair gated when no cited stream is loaded and non-tainted", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, nodeId } = await enqueueSemanticMisattribution(harness, {
      descriptionPatch: "Ben wrote the deployment script.",
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The repair cites no loaded non-tainted stream entry.",
        support_basis: "direct_user_or_source",
        cited_stream_ids: [createStreamEntryId()],
      }),
    );

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);
    const node = await harness.semanticNodeRepository.get(nodeId);

    expect(result.errors).toEqual([]);
    expect(result.candidate_stats).toMatchObject({
      accepted: 0,
    });
    expect(open?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "needs_manual",
      reason: "accept_repair_requires_loaded_non_tainted_citation",
    });
    expect(node?.status).toBe("active");
  });

  it("does not accept a repair cited by tainted and unloaded sources only", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, node, taintedSourceEntryId } =
      await enqueueTaintedSemanticMisattribution(harness);
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The repair cites one tainted source and one unloaded source.",
        cited_stream_ids: [taintedSourceEntryId, createStreamEntryId()],
      }),
    );

    await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);
    const storedNode = await harness.semanticNodeRepository.get(node.id);

    expect(open?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "needs_manual",
      reason: "tainted_assistant_output_under_review_cannot_independently_support_claim",
    });
    expect(storedNode?.status).toBe("active");
  });

  it("accepts a tainted citation only when a loaded non-tainted citation also supports it", async () => {
    const llm = new FakeLLMClient();
    const tracer = new ArrayTracer();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const legitimate = await insertSource(
      harness,
      "Nora and Julian are siblings; Priya is Nora's partner.",
    );
    const { item, node, taintedSourceEntryId, legitimateSourceEntryId } =
      await enqueueTaintedSemanticMisattribution(harness, legitimate);

    if (legitimateSourceEntryId === null) {
      throw new Error("expected legitimate source entry id");
    }

    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The user source supports the repair independently of the tainted utterance.",
        cited_stream_ids: [taintedSourceEntryId, legitimateSourceEntryId],
        support_basis: "mixed",
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const storedNode = await harness.semanticNodeRepository.get(node.id);

    expect(result.candidate_stats).toMatchObject({
      accepted: 1,
    });
    expect(resolved?.resolution).toBe("accept");
    expect(resolved?.refs.__borg_review_resolver_repair).toMatchObject({
      mode: "repair_via_supersede",
      corrected_by: legitimateSourceEntryId,
    });
    expect(storedNode?.status).toBe("superseded");
    expect(storedNode?.corrected_by).toBe(legitimateSourceEntryId);
  });

  it("allows manual resolution after a needs_manual diagnostic is present", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, episodeId } = await enqueueEpisodeMisattribution(harness);
    llm.pushResponse(
      resolverResponse({
        verdict: "needs_manual",
        reason: "The episode needs a human attribution check.",
        cited_stream_ids: [],
      }),
    );

    await runResolver(harness);
    await expect(harness.reviewQueueRepository.resolve(item.id, "accept")).resolves.toMatchObject({
      resolution: "accept",
    });
    const episode = await harness.episodicRepository.get(episodeId);

    expect(episode?.participants).toEqual(["Ben"]);
  });

  it("routes aliases-only semantic-node patches through supersede", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, nodeId, sourceEntryId } = await enqueueSemanticMisattribution(harness, {
      aliasesPatch: ["deployment script author"],
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The alias update changes embedded semantic text.",
        cited_stream_ids: [sourceEntryId],
      }),
    );

    await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const node = await harness.semanticNodeRepository.get(nodeId);

    expect(resolved?.resolution).toBe("accept");
    expect(resolved?.refs.__borg_review_resolver_repair).toMatchObject({
      mode: "repair_via_supersede",
    });
    expect(node?.status).toBe("superseded");
    expect(node?.aliases).toEqual([]);
  });

  it("routes source_episode_ids-only semantic-node patches through supersede", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const replacementSource = await insertSource(
      harness,
      "Ben wrote the deployment script during the release cleanup.",
    );
    const { item, nodeId, sourceEntryId } = await enqueueSemanticMisattribution(harness, {
      sourceEpisodeIdsPatch: [replacementSource.episode.id],
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "Changing source episodes changes the semantic basis of the node.",
        cited_stream_ids: [sourceEntryId],
      }),
    );

    await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const node = await harness.semanticNodeRepository.get(nodeId);

    expect(resolved?.refs.__borg_review_resolver_repair).toMatchObject({
      mode: "repair_via_supersede",
    });
    expect(node?.status).toBe("superseded");
    expect(node?.source_episode_ids).not.toEqual([replacementSource.episode.id]);
  });

  it("still applies non-semantic episode patches through the existing handler", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, episodeId, targetAudienceId, sourceEntryId } =
      await enqueueEpisodeMisattribution(harness);
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The source clearly identifies Ben as the helper author.",
        cited_stream_ids: [sourceEntryId],
      }),
    );

    await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const episode = await harness.episodicRepository.get(episodeId);
    const promptPayload = JSON.parse(String(llm.requests[0]?.messages[0]?.content ?? "{}")) as {
      review?: {
        disclosure_label?: { disclosure_class?: string };
      };
      target?: {
        content?: {
          id?: string;
          narrative?: string;
          disclosure?: string;
          disclosure_label?: {
            disclosure_class?: string;
            private_to_entity_ids?: string[];
          };
        };
      };
      source_bundle?: {
        overseer_flag?: {
          disclosure?: string;
          disclosure_label?: {
            disclosure_class?: string;
            private_to_entity_ids?: string[];
          };
        };
      };
    };

    expect(resolved?.resolution).toBe("accept");
    expect(episode?.participants).toEqual(["Ben"]);
    expect(promptPayload.target?.content).toMatchObject({
      id: episodeId,
      narrative: "Alice wrote the deployment helper.",
      disclosure_label: expect.objectContaining({
        disclosure_class: "relationship_private",
        private_to_entity_ids: [targetAudienceId],
      }),
    });
    expect(promptPayload.target?.content?.disclosure).toContain(
      "disclosure_class=relationship_private",
    );
    expect(promptPayload.target?.content?.disclosure_label?.disclosure_class).not.toBe("public");
    expect(promptPayload.source_bundle?.overseer_flag).toMatchObject({
      disclosure_label: expect.objectContaining({
        disclosure_class: "relationship_private",
        private_to_entity_ids: [targetAudienceId],
      }),
    });
    expect(promptPayload.source_bundle?.overseer_flag?.disclosure).toContain(
      "disclosure_class=relationship_private",
    );
    expect(promptPayload.source_bundle?.overseer_flag?.disclosure_label?.disclosure_class).not.toBe(
      "public",
    );
    expect(promptPayload.review?.disclosure_label).toMatchObject({
      disclosure_class: "relationship_private",
    });
    expect(promptPayload.review?.disclosure_label?.disclosure_class).not.toBe("public");
  });

  it("labels semantic-edge targets and overseer flags in the generic resolver prompt", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const source = await insertSource(
      harness,
      "The Atlas migration dependency was only valid during the rollout window.",
      {
        audience_entity_id: createEntityId(),
        shared: false,
      },
    );
    const from = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Atlas migration",
        description: "Atlas migration planning is active.",
        source_episode_ids: [source.episode.id],
      }),
    );
    const to = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Rollout window",
        description: "The rollout window constrained Atlas migration work.",
        source_episode_ids: [source.episode.id],
      }),
    );
    const edge = harness.semanticEdgeRepository.addEdge({
      from_node_id: from.id,
      to_node_id: to.id,
      relation: "supports",
      confidence: 0.72,
      evidence_episode_ids: [source.episode.id],
      created_at: 1_000_000,
      last_verified_at: 1_000_000,
    });
    const item = harness.reviewQueueRepository.enqueue({
      kind: "temporal_drift",
      reason: "Semantic edge temporal validity needs review.",
      refs: {
        target_type: "semantic_edge",
        target_kind: "semantic_edge",
        target_id: edge.id,
        suggested_valid_to: 1_010_000,
        reason: "The dependency has expired.",
        overseer_flag: overseerFlag({
          kind: "temporal_drift",
          reason: "Semantic edge temporal validity needs review.",
          sourceEntryId: source.entry.id,
          sourceEpisodeId: source.episode.id,
        }),
      },
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "dismiss_false_positive",
        reason: "The supplied edge can remain active.",
        cited_stream_ids: [],
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const promptPayload = JSON.parse(String(llm.requests[0]?.messages[0]?.content ?? "{}")) as {
      target?: {
        content?: {
          disclosure?: string;
          evidence_episode_ids?: string[];
          disclosure_label?: { disclosure_class?: string };
        };
      };
      source_bundle?: {
        overseer_flag?: {
          source_episode_ids?: string[];
          disclosure_label?: { disclosure_class?: string };
        };
      };
    };

    expect(result.errors).toEqual([]);
    expect(resolved?.resolution).toBe("dismiss");
    expect(promptPayload.target).toMatchObject({
      content: {
        evidence_episode_ids: [source.episode.id],
        disclosure_label: {
          disclosure_class: "relationship_private",
        },
      },
    });
    expect(promptPayload.target?.content?.disclosure).toContain(
      "disclosure_class=relationship_private",
    );
    expect(promptPayload.source_bundle?.overseer_flag).toMatchObject({
      source_episode_ids: [source.episode.id],
      disclosure_label: {
        disclosure_class: "relationship_private",
      },
    });
  });

  it("fails open and emits degraded when the LLM call fails", async () => {
    const llm = new FakeLLMClient();
    const tracer = new ArrayTracer();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item } = await enqueueSemanticMisattribution(harness, {
      descriptionPatch: "Ben wrote the deployment script.",
    });

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);

    expect(open?.resolved_at).toBeNull();
    expect(result.errors).toHaveLength(1);
    expect(tracer.events.find((event) => event.event === "review_resolver.degraded")).toMatchObject(
      {
        review_id: item.id,
      },
    );
  });

  it("fails open when the repair handler rejects the accepted repair", async () => {
    const llm = new FakeLLMClient();
    const tracer = new ArrayTracer();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, sourceEntryId } = await enqueueEpisodeMisattribution(harness);
    vi.spyOn(harness.episodicRepository, "update").mockRejectedValue(
      new Error("repair handler failed"),
    );
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The alias patch is supported by the source.",
        cited_stream_ids: [sourceEntryId],
      }),
    );

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);

    expect(open?.resolved_at).toBeNull();
    expect(result.errors).toHaveLength(1);
    expect(tracer.events.find((event) => event.event === "review_resolver.degraded")).toMatchObject(
      {
        review_id: item.id,
        reason: "repair handler failed",
      },
    );
  });

  it("rolls back semantic supersede when review queue finalization fails", async () => {
    const llm = new FakeLLMClient();
    const tracer = new ArrayTracer();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, nodeId, sourceEntryId } = await enqueueSemanticMisattribution(harness, {
      descriptionPatch: "Ben wrote the deployment script.",
    });
    const originalPrepare = harness.db.prepare.bind(harness.db);
    vi.spyOn(harness.db, "prepare").mockImplementation((sql: string) => {
      if (sql.indexOf("UPDATE review_queue") >= 0 && sql.indexOf("resolved_at = ?") >= 0) {
        return {
          run() {
            throw new Error("queue update failed");
          },
        } as never;
      }

      return originalPrepare(sql);
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The cited source says Ben wrote the script.",
        cited_stream_ids: [sourceEntryId],
      }),
    );

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);
    const node = await harness.semanticNodeRepository.get(nodeId);

    expect(result.errors).toHaveLength(1);
    expect(open?.resolved_at).toBeNull();
    expect(node?.status).toBe("active");
  });

  it("processes at most the configured cap per pass", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const items: ReviewQueueItem[] = [];

    for (let index = 0; index < 5; index += 1) {
      const source = await insertSource(
        harness,
        `Ben wrote deployment helper ${index}; Alice reviewed it.`,
      );
      const episode = await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          title: `Deployment helper ${index}`,
          narrative: `Alice wrote deployment helper ${index}.`,
          participants: ["Alice"],
          source_stream_ids: [source.entry.id],
        }),
      );
      items.push(
        harness.reviewQueueRepository.enqueue({
          kind: "misattribution",
          reason: "Episode participant attribution needs review.",
          refs: {
            target_type: "episode",
            target_id: episode.id,
            patch: {
              participants: ["Ben"],
            },
            evidence_stream_ids: [source.entry.id],
            overseer_flag: overseerFlag({
              kind: "misattribution",
              reason: "Episode participant attribution needs review.",
              sourceEntryId: source.entry.id,
              sourceEpisodeId: source.episode.id,
              patch: {
                participants: ["Ben"],
              },
            }),
          },
        }),
      );
      llm.pushResponse(
        resolverResponse({
          verdict: "dismiss_false_positive",
          reason: "The cited source does not require a repair.",
          cited_stream_ids: [],
        }),
      );
    }

    const result = await runResolver(harness, 3);
    const resolved = items.filter(
      (item) => harness.reviewQueueRepository.get(item.id)?.resolved_at !== null,
    );
    const open = items.filter(
      (item) => harness.reviewQueueRepository.get(item.id)?.resolved_at === null,
    );

    expect(result.changes).toHaveLength(3);
    expect(resolved).toHaveLength(3);
    expect(open).toHaveLength(2);
    expect(llm.requests).toHaveLength(3);
  });

  it("leaves identity inconsistency reviews for manual resolution", async () => {
    const llm = new FakeLLMClient();
    const tracer = new ArrayTracer();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const evidenceEpisode = createEpisodeFixture().id;
    const value = harness.valuesRepository.add({
      label: "groundedness",
      description: "Stay grounded in cited evidence.",
      priority: 5,
      provenance: {
        kind: "manual",
      },
    });
    const item = harness.reviewQueueRepository.enqueue({
      kind: "identity_inconsistency",
      reason: "The value has new supporting evidence.",
      refs: {
        target_type: "value",
        target_id: value.id,
        repair_op: "reinforce",
        evidence_episode_ids: [evidenceEpisode],
        proposed_provenance: {
          kind: "offline",
          process: "overseer",
        },
      },
    });

    await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);

    expect(open?.resolved_at).toBeNull();
    expect(open?.refs.__borg_review_resolver_diagnostic).toBeUndefined();
    expect(
      tracer.events.find((event) => event.event === "review_resolver.decision.completed"),
    ).toBeUndefined();
    expect(llm.requests).toHaveLength(0);
  });
});

describe("review resolver autonomous mode", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
    vi.restoreAllMocks();
  });

  const AUTONOMOUS_OVERRIDES = {
    offline: {
      reviewResolver: {
        autonomous: true,
        maxNeedsManualAttempts: 2,
      },
    },
  };

  function identityResolverResponse(input: Record<string, unknown>): LLMCompleteResult {
    return {
      text: "",
      input_tokens: 7,
      output_tokens: 4,
      stop_reason: "tool_use",
      tool_calls: [
        {
          id: "toolu_identity_review_resolver",
          name: "EmitIdentityInconsistencyVerdict",
          input,
        },
      ],
    };
  }

  it("resolves identity_inconsistency items when autonomous", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
      configOverrides: AUTONOMOUS_OVERRIDES,
    });
    cleanup.push(harness.cleanup);
    const value = harness.valuesRepository.add({
      label: "groundedness",
      description: "Stay grounded in cited evidence.",
      priority: 5,
      provenance: {
        kind: "manual",
      },
    });
    const item = harness.reviewQueueRepository.enqueue({
      kind: "identity_inconsistency",
      reason: "The value has new supporting evidence.",
      refs: {
        target_type: "value",
        target_id: value.id,
        repair_op: "reinforce",
        evidence_episode_ids: [createEpisodeFixture().id],
        proposed_provenance: {
          kind: "offline",
          process: "overseer",
        },
      },
    });
    llm.pushResponse(
      identityResolverResponse({
        verdict: "accept_repair",
        reason: "The reinforcement is supported by the stated evidence.",
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);

    expect(result.changes).toHaveLength(1);
    expect(resolved?.resolved_at).not.toBeNull();
    expect(resolved?.resolution).toBe("accept");
    expect(llm.requests).toHaveLength(1);
  });

  it("retries needs_manual items and terminally dismisses at the attempt cap", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
      configOverrides: AUTONOMOUS_OVERRIDES,
    });
    cleanup.push(harness.cleanup);
    const { item } = await enqueueSemanticMisattribution(harness, {
      descriptionPatch: "Ben wrote the deployment script.",
    });

    llm.pushResponse(
      resolverResponse({
        verdict: "needs_manual",
        reason: "First pass could not decide.",
        cited_stream_ids: [],
      }),
    );
    await runResolver(harness);
    const stamped = harness.reviewQueueRepository.get(item.id);

    expect(stamped?.resolved_at).toBeNull();
    expect(stamped?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "needs_manual",
      attempts: 1,
    });

    llm.pushResponse(
      resolverResponse({
        verdict: "needs_manual",
        reason: "Second pass could not decide either.",
        cited_stream_ids: [],
      }),
    );
    await runResolver(harness);
    const finalized = harness.reviewQueueRepository.get(item.id);

    expect(finalized?.resolved_at).not.toBeNull();
    expect(finalized?.resolution).toBe("dismiss");
    expect(finalized?.refs.__borg_review_resolver_repair).toMatchObject({
      mode: "resolver_finalized_without_handler",
      bypass_handler_reason: "autonomous_needs_manual_exhausted",
    });
    expect(llm.requests).toHaveLength(2);
  });

  it("keeps stamped items parked when autonomous is off", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item } = await enqueueSemanticMisattribution(harness, {
      descriptionPatch: "Ben wrote the deployment script.",
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "needs_manual",
        reason: "Needs a human canonicalization decision.",
        cited_stream_ids: [],
      }),
    );

    await runResolver(harness);
    await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);

    expect(open?.resolved_at).toBeNull();
    expect(open?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "needs_manual",
      attempts: 1,
    });
    expect(llm.requests).toHaveLength(1);
  });

  // A repair whose apply throws used to leave the item open with nothing counted,
  // so the resolver re-picked and re-threw it forever (ai-prod: 5 edge-closure
  // items x every 4-hourly run). Missing targets and malformed refs never reach
  // apply -- the resolver pre-rejects those without the handler -- so the throw is
  // forced where the wedge class actually lives: an apply-time invariant failure
  // on a valid item, simulated at the storage boundary.
  it("stamps a bounded-retry attempt when the accepted repair throws", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
      configOverrides: {
        offline: { reviewResolver: { autonomous: true, maxNeedsManualAttempts: 3 } },
      },
    });
    cleanup.push(harness.cleanup);
    const { item, sourceEntryId } = await enqueueEpisodeMisattribution(harness);
    vi.spyOn(harness.episodicRepository, "update").mockImplementation(() => {
      throw new Error("simulated storage failure during repair");
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The cited source supports the repair.",
        cited_stream_ids: [sourceEntryId],
      }),
    );

    const result = await runResolver(harness);

    expect(result.errors).toHaveLength(1);
    const after = harness.reviewQueueRepository.get(item.id);
    expect(after?.resolved_at).toBeNull();
    expect(after?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "apply_failed",
      attempts: 1,
    });
  });

  it("terminally dismisses an item whose apply keeps throwing once attempts are exhausted", async () => {
    // The prod wedge shape exactly: a semantic_edge repair is SQLITE-scoped, so a
    // throwing apply rolls back and leaves the refs pristine -- the next pass
    // re-judges and re-throws identically, forever, unless the attempts cap ends it.
    // (Cross-store repairs self-limit differently: their leftover applying-state
    // makes the next pass pre-reject the item as malformed.)
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
      configOverrides: {
        offline: { reviewResolver: { autonomous: true, maxNeedsManualAttempts: 2 } },
      },
    });
    cleanup.push(harness.cleanup);
    const source = await insertSource(harness, "The support interval ended at the rollback.", {
      shared: true,
    });
    const first = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Atlas deploy pipeline",
        description: "Pipeline node for the drift closure fixture.",
        source_episode_ids: [source.episode.id],
      }),
    );
    const second = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Atlas rollback window",
        description: "Rollback node for the drift closure fixture.",
        source_episode_ids: [source.episode.id],
      }),
    );
    const edge = harness.semanticEdgeRepository.addEdge({
      from_node_id: first.id,
      to_node_id: second.id,
      relation: "supports",
      confidence: 0.7,
      evidence_episode_ids: [source.episode.id],
      created_at: 1_000,
      last_verified_at: 1_000,
    });
    const item = harness.reviewQueueRepository.enqueue({
      kind: "temporal_drift",
      reason: "support interval should close",
      refs: {
        target_type: "semantic_edge",
        target_kind: "semantic_edge",
        target_id: edge.id,
        suggested_valid_to: 5_000,
        reason: "support interval should close",
        overseer_flag: overseerFlag({
          kind: "temporal_drift",
          reason: "support interval should close",
          sourceEntryId: source.entry.id,
          sourceEpisodeId: source.episode.id,
        }),
      },
    });
    vi.spyOn(harness.semanticEdgeRepository, "invalidateEdge").mockImplementation(() => {
      throw new Error("simulated invariant failure during edge closure");
    });

    for (const _pass of [1, 2]) {
      llm.pushResponse(
        resolverResponse({
          verdict: "accept_repair",
          reason: "The cited source supports closing the interval.",
          cited_stream_ids: [source.entry.id],
        }),
      );
      await runResolver(harness);
    }

    const after = harness.reviewQueueRepository.get(item.id);
    expect(after?.resolved_at).not.toBeNull();
    expect(after?.resolution).toBe("dismiss");
    expect(after?.refs.__borg_review_resolver_repair).toMatchObject({
      bypass_handler_reason: "autonomous_apply_failure_exhausted",
    });
  });
});
