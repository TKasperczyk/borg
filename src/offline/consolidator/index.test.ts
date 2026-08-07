import { afterEach, describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { buildConsolidationCoverageHash } from "../../memory/episodic/index.js";
import { memoryDisclosureLabelFromEpisodeAccess } from "../../retrieval/index.js";
import { createConsolidationFamilyId } from "../../util/ids.js";
import { SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE } from "../../util/self-memory-voice.js";

import {
  createEpisodeFixture,
  createOfflineTestHarness,
  TestEmbeddingClient,
  type OfflineTestHarness,
} from "../test-support.js";
import { ConsolidatorProcess } from "./index.js";

const CONSOLIDATION_TOOL_NAME = "EmitConsolidation";

function createConsolidationResponse(title: string, narrative: string) {
  return {
    text: "",
    input_tokens: 20,
    output_tokens: 15,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: CONSOLIDATION_TOOL_NAME,
        input: { title, narrative },
      },
    ],
  };
}

function createProcess(harness: OfflineTestHarness) {
  return new ConsolidatorProcess({
    episodicRepository: harness.episodicRepository,
    registry: harness.registry,
  });
}

function consolidationEmbeddingText(
  title: string,
  narrative: string,
  tags = ["planning"],
  participants = ["team"],
): string {
  return `${title}\n${narrative}\n${tags.join(" ")}\n${participants.join(" ")}`;
}

function consolidationCoverageHashForTest(sourceStreamIds: readonly string[]): string {
  return buildConsolidationCoverageHash([...sourceStreamIds, "consolidation_policy_version:1"]);
}

function unitVector(degrees: number): number[] {
  const radians = (degrees * Math.PI) / 180;
  return [Math.cos(radians), Math.sin(radians), 0, 0];
}

describe("consolidator process", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("creates a family version, hides covered raws without archiving them, and supports reversal", async () => {
    const outcomeLine = "OUTCOME fp=consolidation-receipt role=planner tenant=tenant_42";
    const decisionLine = "decision=create";
    const ticketActionLine = "ticket=AININJAS-1187 action=created";
    const teamsCardLine = "action=teams_card";
    const uncheckedMergedNarrative = [
      "The team merged two overlapping planning notes into one grounded summary.",
      "decision=create:ticket",
      ticketActionLine,
    ].join("\n");
    const llm = new FakeLLMClient({
      responses: [
        createConsolidationResponse("Merged planning incident", uncheckedMergedNarrative),
        createConsolidationResponse("Merged planning incident", uncheckedMergedNarrative),
      ],
    });
    const embeddingClient = new TestEmbeddingClient();
    const embedSpy = vi.spyOn(embeddingClient, "embed");
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient,
    });
    cleanup.push(harness.cleanup);

    const first = createEpisodeFixture(
      {
        title: "Sprint planning note",
        narrative: [
          "The team planned the sprint and listed the deploy checklist.",
          outcomeLine,
          ticketActionLine,
        ].join("\n"),
        tags: ["planning", "deploy"],
        created_at: 10_000,
        updated_at: 10_000,
      },
      [0, 1, 0, 0],
    );
    const second = createEpisodeFixture(
      {
        title: "Sprint planning follow-up",
        narrative: [
          "The same planning session captured the deploy checklist again.",
          decisionLine,
          teamsCardLine,
          ticketActionLine,
        ].join("\n"),
        tags: ["planning", "deploy"],
        created_at: 20_000,
        updated_at: 20_000,
      },
      [0, 0.99, 0, 0],
    );

    await harness.episodicRepository.createEpisode(first);
    await harness.episodicRepository.createEpisode(second);
    harness.episodicRepository.updateStats(first.id, {
      tier: "T2",
    });

    const process = createProcess(harness);

    const dryRun = await process.run(harness.createContext(), {
      dryRun: true,
    });
    expect(dryRun.changes).toHaveLength(1);
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: CONSOLIDATION_TOOL_NAME,
    });
    expect(String(llm.requests[0]?.messages[0]?.content ?? "")).toContain(
      "Messages with kind agent_msg are your own; write your own actions, statements, and decisions in first person; refer to every other sender by name or stable handle.",
    );
    expect((await harness.episodicRepository.listAll()).map((episode) => episode.id)).toEqual([
      second.id,
      first.id,
    ]);

    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });
    expect(result.errors).toEqual([]);
    expect(result.tokens_used).toBe(35);

    const family = harness.episodicRepository.listConsolidationFamilies()[0];
    const members = harness.episodicRepository.listConsolidationMembers(family!.family_id);
    const merged = await harness.episodicRepository.get(family!.current_version_episode_id);

    expect(merged).toMatchObject({
      title: "Merged planning incident",
      episode_kind: "consolidation_version",
      consolidation_family_id: family!.family_id,
      consolidation_coverage_hash: family!.coverage_hash,
    });
    expect(merged?.lineage.derived_from).toEqual([first.id, second.id]);
    expect(merged?.lineage.supersedes).toEqual([first.id, second.id]);
    expect(merged?.narrative.split(/\r\n|\n|\r/u)).toEqual([
      "The team merged two overlapping planning notes into one grounded summary.",
      "decision=create:ticket",
      ticketActionLine,
      outcomeLine,
      decisionLine,
      teamsCardLine,
    ]);
    expect(embedSpy).toHaveBeenCalledTimes(2);
    expect(embedSpy).toHaveBeenNthCalledWith(
      2,
      [
        "Merged planning incident",
        "The team merged two overlapping planning notes into one grounded summary.",
        outcomeLine,
        "planning deploy",
        "team",
      ].join("\n"),
    );
    expect(members.map((member) => member.raw_episode_id).sort()).toEqual(
      [first.id, second.id].sort(),
    );
    expect(harness.episodicRepository.getStats(merged!.id)?.tier).toBe("T2");
    expect(harness.episodicRepository.getStats(first.id)?.archived).toBe(false);
    expect(harness.episodicRepository.getStats(second.id)?.archived).toBe(false);
    expect(await harness.episodicRepository.get(first.id)).toBeNull();
    expect(await harness.episodicRepository.get(second.id)).toBeNull();
    expect(await harness.episodicRepository.get(first.id, { includeArchived: true })).toBeDefined();

    const auditEntry = harness.auditLog.list({ process: "consolidator" })[0];
    expect(auditEntry?.action).toBe("consolidate");
    expect(String(llm.requests[0]?.messages[0]?.content ?? "")).toContain(
      "copy that complete line verbatim",
    );
    expect(String(llm.requests[0]?.messages[0]?.content ?? "")).toContain("ticket=<X> action=<Y>");
    expect(String(llm.requests[0]?.messages[0]?.content ?? "")).toContain("action=teams_card");

    await harness.auditLog.revert(auditEntry!.id, "test");

    expect(await harness.episodicRepository.get(merged!.id)).toBeNull();
    expect(harness.episodicRepository.listConsolidationFamilies()).toEqual([]);
    expect(harness.episodicRepository.listConsolidationMembers()).toEqual([]);
    expect(harness.episodicRepository.getStats(first.id)?.archived).toBe(false);
    expect(harness.episodicRepository.getStats(second.id)?.archived).toBe(false);
    expect((await harness.episodicRepository.get(first.id))?.id).toBe(first.id);
    expect((await harness.episodicRepository.get(second.id))?.id).toBe(second.id);
  });

  it("reaches a fixed point and skips stale saved coverage on apply", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createConsolidationResponse(
          "Merged deploy prep",
          "Two overlapping deploy-prep notes were merged into one grounded episode.",
        ),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);
    const first = createEpisodeFixture(
      {
        title: "Deploy prep one",
        narrative: "The team reviewed rollback steps before the Atlas deploy.",
        tags: ["deploy", "atlas"],
        created_at: 10_000,
        updated_at: 10_000,
      },
      [1, 0, 0, 0],
    );
    const second = createEpisodeFixture(
      {
        title: "Deploy prep two",
        narrative: "The Atlas deploy checklist repeated the rollback prep discussion.",
        tags: ["deploy", "atlas"],
        created_at: 20_000,
        updated_at: 20_000,
      },
      [0.99, 0, 0, 0],
    );

    await harness.episodicRepository.createEpisode(first);
    await harness.episodicRepository.createEpisode(second);

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());

    expect(plan.items).toHaveLength(1);
    expect(llm.requests).toHaveLength(1);

    const firstApply = await process.apply(harness.createContext(), plan);
    const episodeCount = (await harness.episodicRepository.listAll()).length;
    const familyCount = harness.episodicRepository.listConsolidationFamilies().length;
    const memberCount = harness.episodicRepository.listConsolidationMembers().length;

    expect(firstApply.changes).toHaveLength(1);

    const staleApply = await process.apply(harness.createContext(), plan);
    expect(staleApply.changes).toEqual([]);
    expect((await harness.episodicRepository.listAll()).length).toBe(episodeCount);
    expect(harness.episodicRepository.listConsolidationFamilies()).toHaveLength(familyCount);
    expect(harness.episodicRepository.listConsolidationMembers()).toHaveLength(memberCount);

    const secondPlan = await process.plan(harness.createContext());
    expect(secondPlan.items).toEqual([]);
    expect(llm.requests).toHaveLength(1);
    expect((await harness.episodicRepository.listAll()).length).toBe(episodeCount);
    expect(harness.episodicRepository.listConsolidationFamilies()).toHaveLength(familyCount);
  });

  it("drops a stale saved plan when any planned raw has become covered", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createConsolidationResponse(
          "Merged stale cluster",
          "Two overlapping stale-plan raws were merged into one grounded episode.",
        ),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);
    const first = createEpisodeFixture(
      {
        title: "Stale raw one",
        created_at: 10_000,
        updated_at: 10_000,
      },
      [1, 0, 0, 0],
    );
    const second = createEpisodeFixture(
      {
        title: "Stale raw two",
        created_at: 20_000,
        updated_at: 20_000,
      },
      [0.99, 0, 0, 0],
    );

    await harness.episodicRepository.createEpisode(first);
    await harness.episodicRepository.createEpisode(second);

    const process = createProcess(harness);
    const stalePlan = await process.plan(harness.createContext());

    expect(stalePlan.items).toHaveLength(1);

    const familyId = createConsolidationFamilyId();
    const coverageHash = consolidationCoverageHashForTest(first.source_stream_ids);
    const version = createEpisodeFixture(
      {
        title: "Externally covered raw one",
        narrative: "A concurrent consolidation covered the first raw episode.",
        source_stream_ids: first.source_stream_ids,
        lineage: {
          derived_from: [first.id],
          supersedes: [first.id],
        },
        episode_kind: "consolidation_version",
        consolidation_family_id: familyId,
        consolidation_coverage_hash: coverageHash,
        created_at: 30_000,
        updated_at: 30_000,
      },
      [1, 0, 0, 0],
    );

    await harness.episodicRepository.createEpisode(version);
    harness.episodicRepository.createConsolidationFamily({
      familyId,
      currentVersionEpisodeId: version.id,
      coverageHash,
      policyVersion: 1,
      members: [
        {
          raw_episode_id: first.id,
          source_stream_ids: first.source_stream_ids,
          added_by_version_episode_id: version.id,
        },
      ],
    });

    const episodeCount = (await harness.episodicRepository.listAll()).length;
    const familyCount = harness.episodicRepository.listConsolidationFamilies().length;
    const memberCount = harness.episodicRepository.listConsolidationMembers().length;
    const result = await process.apply(harness.createContext(), stalePlan);

    expect(result.changes).toEqual([]);
    expect((await harness.episodicRepository.listAll()).length).toBe(episodeCount);
    expect(harness.episodicRepository.listConsolidationFamilies()).toHaveLength(familyCount);
    expect(harness.episodicRepository.listConsolidationMembers()).toHaveLength(memberCount);
  });

  it("rejects an apply plan whose new-raw set omits a covered source episode", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createConsolidationResponse(
          "Merged coverage gap",
          "Two overlapping raws were merged into one grounded episode.",
        ),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);
    const first = createEpisodeFixture(
      {
        title: "Coverage raw one",
        created_at: 10_000,
        updated_at: 10_000,
      },
      [1, 0, 0, 0],
    );
    const second = createEpisodeFixture(
      {
        title: "Coverage raw two",
        created_at: 20_000,
        updated_at: 20_000,
      },
      [0.99, 0, 0, 0],
    );

    await harness.episodicRepository.createEpisode(first);
    await harness.episodicRepository.createEpisode(second);

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());

    expect(plan.items).toHaveLength(1);
    expect(plan.items[0]!.new_raw_episode_ids).toHaveLength(2);

    // Tamper a saved plan so one source raw is dropped from the new-raw set.
    // Applying it would leave that raw with no consolidation_members row while
    // the version's coverage still claims it -- the double-count gap.
    const tampered = {
      ...plan,
      items: [
        {
          ...plan.items[0]!,
          new_raw_episode_ids: [plan.items[0]!.new_raw_episode_ids[0]!],
        },
      ],
    };

    await expect(process.apply(harness.createContext(), tampered)).rejects.toMatchObject({
      code: "CONSOLIDATOR_PLAN_INVALID",
    });
    expect(harness.episodicRepository.listConsolidationFamilies()).toHaveLength(0);
  });

  it("keeps complete-link cohesion bounded so transitive chains do not percolate", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createConsolidationResponse("Merged chain one", "A bounded chain pair was merged."),
        createConsolidationResponse("Merged chain two", "A second bounded chain pair was merged."),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      configOverrides: {
        offline: {
          consolidator: {
            similarityThreshold: 0.9,
            maxClusterDiameter: 0.02,
            maxClustersPerRun: 10,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);
    const episodes = [0, 10, 20, 30].map((degrees, index) =>
      createEpisodeFixture(
        {
          title: `Chain ${index}`,
          tags: ["chain"],
          created_at: 10_000 + index * 1_000,
          updated_at: 10_000 + index * 1_000,
        },
        unitVector(degrees),
      ),
    );

    for (const episode of episodes) {
      await harness.episodicRepository.createEpisode(episode);
    }

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());

    expect(plan.items).toHaveLength(2);
    expect(plan.items.map((item) => item.source_episode_ids)).toEqual(
      expect.arrayContaining([
        expect.arrayContaining([episodes[0]!.id, episodes[1]!.id]),
        expect.arrayContaining([episodes[2]!.id, episodes[3]!.id]),
      ]),
    );
    expect(plan.items.some((item) => item.source_episode_ids.length === 4)).toBe(false);
  });

  it("attaches a single new raw to an active family by creating a new current version", async () => {
    const firstTitle = "Merged family v1";
    const firstNarrative = "The first two planning notes became the initial family version.";
    const secondTitle = "Merged family v2";
    const secondNarrative = "A new planning note extended the existing family version.";
    const llm = new FakeLLMClient({
      responses: [
        createConsolidationResponse(firstTitle, firstNarrative),
        createConsolidationResponse(secondTitle, secondNarrative),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [consolidationEmbeddingText(firstTitle, firstNarrative), [1, 0, 0, 0]],
          [consolidationEmbeddingText(secondTitle, secondNarrative), [1, 0, 0, 0]],
        ]),
      ),
    });
    cleanup.push(harness.cleanup);
    const first = createEpisodeFixture(
      {
        title: "Planning raw one",
        tags: ["planning"],
        created_at: 10_000,
        updated_at: 10_000,
      },
      [1, 0, 0, 0],
    );
    const second = createEpisodeFixture(
      {
        title: "Planning raw two",
        tags: ["planning"],
        created_at: 20_000,
        updated_at: 20_000,
      },
      [0.99, 0, 0, 0],
    );

    await harness.episodicRepository.createEpisode(first);
    await harness.episodicRepository.createEpisode(second);

    const process = createProcess(harness);
    const firstRun = await process.run(harness.createContext(), {
      dryRun: false,
    });
    const family = harness.episodicRepository.listConsolidationFamilies()[0]!;
    const firstVersionId = family.current_version_episode_id;
    const firstCoverageHash = family.coverage_hash;
    const firstPolicyVersion = family.policy_version;
    const third = createEpisodeFixture(
      {
        title: "Planning raw three",
        tags: ["planning"],
        created_at: 30_000,
        updated_at: 30_000,
      },
      [1, 0, 0, 0],
    );

    expect(firstRun.changes).toHaveLength(1);
    await harness.episodicRepository.createEpisode(third);

    const extensionPlan = await process.plan(harness.createContext());
    const prompt = String(llm.requests[1]?.messages[0]?.content ?? "");

    expect(extensionPlan.items).toHaveLength(1);
    expect(extensionPlan.items[0]).toMatchObject({
      family_id: family.family_id,
      previous_current_version_episode_id: firstVersionId,
      new_raw_episode_ids: [third.id],
    });
    expect(extensionPlan.items[0]?.source_episode_ids.sort()).toEqual(
      [first.id, second.id, third.id].sort(),
    );
    expect(extensionPlan.items[0]?.source_episode_ids).not.toContain(firstVersionId);
    expect(prompt).toContain("Previous current consolidation context");
    expect(prompt).toContain("Planning raw three");

    const extensionResult = await process.apply(harness.createContext(), extensionPlan);
    const updatedFamily = harness.episodicRepository.listConsolidationFamilies()[0]!;
    const members = harness.episodicRepository.listConsolidationMembers(family.family_id);
    const secondVersion = await harness.episodicRepository.get(
      updatedFamily.current_version_episode_id,
    );

    expect(extensionResult.changes).toHaveLength(1);
    expect(harness.episodicRepository.listConsolidationFamilies()).toHaveLength(1);
    expect(updatedFamily.current_version_episode_id).not.toBe(firstVersionId);
    expect(updatedFamily.coverage_hash).not.toBe(firstCoverageHash);
    expect(secondVersion).toMatchObject({
      title: secondTitle,
      episode_kind: "consolidation_version",
      consolidation_family_id: family.family_id,
      consolidation_coverage_hash: updatedFamily.coverage_hash,
    });
    expect(members.map((member) => member.raw_episode_id).sort()).toEqual(
      [first.id, second.id, third.id].sort(),
    );
    expect(
      members.find((member) => member.raw_episode_id === third.id)?.added_by_version_episode_id,
    ).toBe(secondVersion!.id);
    expect(harness.episodicRepository.getStats(first.id)?.archived).toBe(false);
    expect(harness.episodicRepository.getStats(second.id)?.archived).toBe(false);
    expect(harness.episodicRepository.getStats(third.id)?.archived).toBe(false);
    expect(await harness.episodicRepository.get(first.id)).toBeNull();
    expect(await harness.episodicRepository.get(second.id)).toBeNull();
    expect(await harness.episodicRepository.get(third.id)).toBeNull();
    expect(await harness.episodicRepository.get(firstVersionId)).toBeNull();
    expect(harness.episodicRepository.getStats(firstVersionId)?.archived).toBe(false);
    expect((await harness.episodicRepository.get(secondVersion!.id))?.id).toBe(secondVersion!.id);

    const postExtensionPlan = await process.plan(harness.createContext());
    expect(postExtensionPlan.items).toEqual([]);
    expect(llm.requests).toHaveLength(2);

    const extensionAudit = harness.auditLog
      .list({ process: "consolidator" })
      .find((entry) => entry.reversal.versionEpisodeId === secondVersion!.id);
    expect(extensionAudit?.reversal).toMatchObject({
      previousCoverageHash: firstCoverageHash,
      previousPolicyVersion: firstPolicyVersion,
    });

    await harness.auditLog.revert(extensionAudit!.id, "test");

    const restoredFamily = harness.episodicRepository.listConsolidationFamilies()[0]!;
    expect(restoredFamily.current_version_episode_id).toBe(firstVersionId);
    expect(restoredFamily.coverage_hash).toBe(firstCoverageHash);
    expect(restoredFamily.policy_version).toBe(firstPolicyVersion);
    expect(await harness.episodicRepository.get(secondVersion!.id)).toBeNull();
  });

  it("does not attach a raw that only matches the family summary outside raw-member diameter", async () => {
    const summaryTitle = "Summary-shaped anchor";
    const summaryNarrative = "The summary embedding points away from one raw member.";
    const llm = new FakeLLMClient({
      responses: [createConsolidationResponse(summaryTitle, summaryNarrative)],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(
        new Map([[consolidationEmbeddingText(summaryTitle, summaryNarrative), [0, 1, 0, 0]]]),
      ),
      configOverrides: {
        offline: {
          consolidator: {
            similarityThreshold: 0.8,
            maxClusterDiameter: 0.05,
            maxClustersPerRun: 10,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);
    const first = createEpisodeFixture(
      {
        title: "Raw member one",
        created_at: 10_000,
        updated_at: 10_000,
      },
      [1, 0, 0, 0],
    );
    const second = createEpisodeFixture(
      {
        title: "Raw member two",
        created_at: 20_000,
        updated_at: 20_000,
      },
      [0.99, 0, 0, 0],
    );
    const later = createEpisodeFixture(
      {
        title: "Summary-near raw",
        created_at: 30_000,
        updated_at: 30_000,
      },
      [0, 1, 0, 0],
    );

    await harness.episodicRepository.createEpisode(first);
    await harness.episodicRepository.createEpisode(second);

    const process = createProcess(harness);
    const firstRun = await process.run(harness.createContext(), {
      dryRun: false,
    });

    expect(firstRun.changes).toHaveLength(1);

    await harness.episodicRepository.createEpisode(later);

    const plan = await process.plan(harness.createContext());

    expect(plan.items).toEqual([]);
    expect(llm.requests).toHaveLength(1);
    expect((await harness.episodicRepository.get(later.id))?.id).toBe(later.id);
  });

  it("merges overlapping cross-scope raws into one private combined family version", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createConsolidationResponse(
          "Merged architecture pattern",
          "Public, Alice-private, and Bob-private notes were merged into one grounded architecture memory.",
        ),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);
    const alice = harness.entityRepository.resolve("Alice");
    const bob = harness.entityRepository.resolve("Bob");
    const selfEntityId = harness.entityRepository.resolve("self", {
      kind: "self",
      provenance: "assistant_seeded",
    });
    const sourceEpisodes = [
      createEpisodeFixture(
        {
          title: "Public architecture note one",
          tags: ["architecture"],
          audience_entity_id: null,
          shared: true,
          created_at: 10_000,
          updated_at: 10_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Alice-only architecture note",
          tags: ["architecture"],
          audience_entity_id: alice,
          shared: false,
          created_at: 20_000,
          updated_at: 20_000,
        },
        [0.99, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Bob-only architecture note",
          tags: ["architecture"],
          audience_entity_id: bob,
          shared: false,
          created_at: 30_000,
          updated_at: 30_000,
        },
        [0.98, 0, 0, 0],
      ),
    ];

    for (const episode of sourceEpisodes) {
      await harness.episodicRepository.createEpisode(episode);
    }

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });
    const episodes = await harness.episodicRepository.listAll();
    const merged = episodes.find((episode) => episode.title === "Merged architecture pattern");
    const expectedPrivateTo = [alice, bob].sort();

    expect(result.changes).toHaveLength(1);
    expect(merged?.episode_kind).toBe("consolidation_version");
    expect(merged?.lineage.derived_from).toEqual(sourceEpisodes.map((episode) => episode.id));
    expect(merged?.lineage.supersedes).toEqual(sourceEpisodes.map((episode) => episode.id));
    expect(merged?.audience_entity_id).toBeNull();
    expect(merged?.origin_audience_entity_ids).toEqual(expectedPrivateTo);
    expect(merged?.shared).toBe(false);
    expect(
      sourceEpisodes.map((episode) => harness.episodicRepository.getStats(episode.id)?.archived),
    ).toEqual([false, false, false]);
    expect(memoryDisclosureLabelFromEpisodeAccess(merged!)).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: expectedPrivateTo,
      privateToEntityIds: expectedPrivateTo,
      publicToEntityIds: [],
    });
    expect(llm.requests).toHaveLength(1);
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
    expect(prompt).toContain("disclosure_class=relationship_private");
    expect(prompt).toContain("I can use this internally");
    expect(prompt).toContain(`I am the self entity ${selfEntityId};`);
    expect(prompt).toContain(SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE);
    expect(prompt).toContain("I keep the title topic-neutral and scannable");
    expect(prompt).toContain(alice);
    expect(prompt).toContain(bob);
  });

  it("halts further llm work after budget exhaustion", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          ...createConsolidationResponse("Merged cluster A", "Merged cluster A."),
          input_tokens: 40,
          output_tokens: 30,
        },
        {
          ...createConsolidationResponse("Merged cluster B", "Merged cluster B."),
          input_tokens: 40,
          output_tokens: 30,
        },
        {
          ...createConsolidationResponse("Merged cluster C", "Merged cluster C."),
          input_tokens: 40,
          output_tokens: 30,
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);

    const episodes = [
      createEpisodeFixture(
        {
          title: "Alpha one",
          tags: ["alpha"],
          created_at: 1_000,
          updated_at: 1_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Alpha two",
          tags: ["alpha"],
          created_at: 2_000,
          updated_at: 2_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Beta one",
          tags: ["beta"],
          created_at: 3_000,
          updated_at: 3_000,
        },
        [0, 1, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Beta two",
          tags: ["beta"],
          created_at: 4_000,
          updated_at: 4_000,
        },
        [0, 1, 0, 0],
      ),
      createEpisodeFixture(
        {
          title: "Gamma one",
          tags: ["gamma"],
          created_at: 5_000,
          updated_at: 5_000,
        },
        [0, 0, 1, 0],
      ),
      createEpisodeFixture(
        {
          title: "Gamma two",
          tags: ["gamma"],
          created_at: 6_000,
          updated_at: 6_000,
        },
        [0, 0, 1, 0],
      ),
    ];

    for (const episode of episodes) {
      await harness.episodicRepository.createEpisode(episode);
    }

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {
      dryRun: false,
      budget: 100,
    });

    expect(result.budget_exhausted).toBe(true);
    expect(llm.requests).toHaveLength(2);
    expect(result.changes).toHaveLength(1);
  });
});
