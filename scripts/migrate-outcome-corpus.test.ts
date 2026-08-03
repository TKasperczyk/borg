import { createHash } from "node:crypto";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  buildConsolidationCoverageHash,
  type Episode,
  type EpisodeStats,
} from "../src/memory/episodic/index.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
  TestEmbeddingClient,
  type OfflineTestHarness,
} from "../src/offline/test-support.js";
import {
  createConsolidationFamilyId,
  createEpisodeId,
  createMaintenanceRunId,
  createSemanticNodeId,
  type ConsolidationFamilyId,
} from "../src/util/ids.js";
import {
  DEFAULT_OUTCOME_CORPUS_GRAMMAR,
  formatOutcomeCorpusMigrationReport,
  main,
  migrateOutcomeCorpus,
  outcomeCorpusMigrationExitCode,
  parseOutcomeActionRecords,
  parseOutcomeCorpusCliArgs,
  renderOutcomeRollup,
  type OutcomeCorpusSpecification,
  type ScheduledOutcomeSource,
} from "./migrate-outcome-corpus.js";

function bodySha256(episode: Episode): string {
  return createHash("sha256").update(`${episode.title}\n${episode.narrative}`).digest("hex");
}

async function createFamily(
  harness: OfflineTestHarness,
  members: readonly Episode[],
  input: { title: string; narrative: string; tags?: string[] },
): Promise<{ familyId: ConsolidationFamilyId; version: Episode }> {
  const familyId = createConsolidationFamilyId();
  const sourceStreamIds = members.flatMap((member) => member.source_stream_ids);
  const coverageHash = buildConsolidationCoverageHash(sourceStreamIds);
  const version = {
    ...createEpisodeFixture({
      id: createEpisodeId(),
      title: input.title,
      narrative: input.narrative,
      tags: input.tags,
      source_stream_ids: sourceStreamIds,
      lineage: {
        derived_from: members.map((member) => member.id),
        supersedes: members.map((member) => member.id),
      },
    }),
    episode_kind: "consolidation_version",
    consolidation_family_id: familyId,
    consolidation_coverage_hash: coverageHash,
  } satisfies Episode;

  await harness.episodicRepository.createEpisode(version);
  harness.episodicRepository.createConsolidationFamily({
    familyId,
    currentVersionEpisodeId: version.id,
    coverageHash,
    policyVersion: 1,
    members: members.map((member) => ({
      raw_episode_id: member.id,
      source_stream_ids: member.source_stream_ids,
      added_by_version_episode_id: version.id,
    })),
  });

  return { familyId, version };
}

async function createSafeSpecification(
  harness: OfflineTestHarness,
): Promise<OutcomeCorpusSpecification> {
  const keepSource = createEpisodeFixture({
    title: "Safe specification source",
    narrative: "Stable source outside the migration corpus.",
  });
  await harness.episodicRepository.createEpisode(keepSource);
  const keep = await createFamily(harness, [keepSource], {
    title: "Safe explicit keep",
    narrative: "Stable explicit keep version outside the migration corpus.",
  });
  const punishmentSemanticNodeId = createSemanticNodeId();
  await harness.semanticNodeRepository.insert(
    createSemanticNodeFixture({
      id: punishmentSemanticNodeId,
      source_episode_ids: [keep.version.id],
    }),
  );

  return {
    toxicEpisodeSpecs: [],
    explicitKeepEpisodeId: keep.version.id,
    explicitKeepBodySha256: bodySha256(keep.version),
    punishmentSemanticNodeId,
    scheduledFpSelfChecks: [],
    grammar: DEFAULT_OUTCOME_CORPUS_GRAMMAR,
  };
}

describe("OUTCOME corpus migration", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();

    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("defaults to dry-run and accepts flag or positional data directories", () => {
    expect(parseOutcomeCorpusCliArgs(["--data-dir", "/tmp/example-bank"])).toMatchObject({
      help: false,
      apply: false,
      dataDir: "/tmp/example-bank",
    });
    expect(parseOutcomeCorpusCliArgs(["/tmp/example-bank", "--apply"])).toMatchObject({
      help: false,
      apply: true,
      dataDir: "/tmp/example-bank",
    });
  });

  it("runs dry-run from the existing Lance schema without EMBEDDING_DIMS", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);
    const stdout = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);

    await expect(main(["--data-dir", harness.tempDir], {})).resolves.toBe(1);
    expect(stdout).toHaveBeenCalledWith(expect.stringContaining("mode=dry-run"));
    expect(stderr.mock.calls.flatMap((call) => call).join("\n")).not.toContain(
      "EMBEDDING_DIMS is required",
    );
  });

  it("rejects apply when provider dimensions disagree with the Lance schema", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);
    vi.spyOn(process.stderr, "write").mockImplementation(() => true);

    await expect(
      main(["--data-dir", harness.tempDir, "--apply"], { EMBEDDING_DIMS: "5" }),
    ).rejects.toThrow(
      "EMBEDDING_DIMS=5 does not match the existing episodes LanceDB schema dimension 4",
    );
  });

  it("parses decision-first and ticket-first action grammars without losing records", () => {
    const records = parseOutcomeActionRecords([
      "decision=created:AININJAS-1200 action=created ticket=AININJAS-1200 summary=First incident decision=transition:AININJAS-1201:Ready_for_dev action=transition ticket=AININJAS-1201 transition=Ready for dev verdict=handoff approved run=abc",
      "ticket=AININJAS-1202 action=created summary=Second incident ticket=AININJAS-1203 action=mr mr=https://example.test/mr/3",
      "decision=teams_card:posted action=teams_card teams_card=yes card_count=1",
    ]);

    expect(records).toEqual([
      {
        action: "created",
        decision: "created:AININJAS-1200",
        ticket: "AININJAS-1200",
        summary: "First incident",
      },
      {
        action: "transition",
        decision: "transition:AININJAS-1201:Ready_for_dev",
        ticket: "AININJAS-1201",
        transition: "Ready for dev",
        verdict: "handoff approved",
      },
      {
        action: "created",
        ticket: "AININJAS-1202",
        summary: "Second incident",
      },
      {
        action: "mr",
        ticket: "AININJAS-1203",
        mr: "https://example.test/mr/3",
      },
      {
        action: "teams_card",
        decision: "teams_card:posted",
        teamsCard: "yes",
        cardCount: "1",
      },
    ]);
  });

  it("renders deterministic 2-4 sentence prose and full-line-deduplicated token appendices", () => {
    const header =
      "[triage] OUTCOME fp=scheduled:triage:team-agent-ai role=triage tenant=team-agent-ai";
    const firstLines = [
      header,
      "decision=created:AININJAS-1200 action=created ticket=AININJAS-1200 summary=First incident",
      "ticket=AININJAS-1201 action=transition transition=Ready for dev verdict=handoff approved",
    ];
    const secondLines = [
      header,
      "ticket=AININJAS-1202 action=mr mr=https://example.test/mr/2",
      "action=teams_card",
    ];
    const stats = (episodeId: EpisodeStats["episode_id"]): EpisodeStats => ({
      episode_id: episodeId,
      retrieval_count: 0,
      use_count: 0,
      last_retrieved: null,
      win_rate: 0,
      tier: "T1",
      promoted_at: 0,
      promoted_from: null,
      gist: null,
      gist_generated_at: null,
      last_decayed_at: null,
      heat_multiplier: 1,
      valence_mean: 0,
      archived: false,
    });
    const firstEpisode = createEpisodeFixture({
      narrative: firstLines.join("\n"),
      start_time: Date.UTC(2026, 7, 1, 8),
    });
    const secondEpisode = createEpisodeFixture({
      narrative: secondLines.join("\n"),
      start_time: Date.UTC(2026, 7, 1, 9),
    });
    const source = (
      episode: typeof firstEpisode,
      protectedLines: string[],
    ): ScheduledOutcomeSource => ({
      episode,
      stats: stats(episode.id),
      role: "triage",
      utcDay: "2026-08-01",
      fingerprints: ["scheduled:triage:team-agent-ai"],
      protectedLines,
      actions: parseOutcomeActionRecords(protectedLines),
    });

    const rendered = renderOutcomeRollup({
      role: "triage",
      utcDay: "2026-08-01",
      sources: [source(firstEpisode, firstLines), source(secondEpisode, secondLines)],
    });

    expect(rendered.prose).toBe(
      "On 2026-08-01, triage created AININJAS-1200 (First incident). The triage role transitioned AININJAS-1201 to Ready for dev (handoff approved). The triage role opened merge requests for AININJAS-1202 (https://example.test/mr/2); and posted 1 Teams card notification.",
    );
    expect(rendered.prose.split(/(?<=\.) (?=The )/u)).toHaveLength(3);
    expect(rendered.protectedLines).toEqual([...firstLines, ...secondLines.slice(1)]);
    expect(rendered.narrative).toBe(
      `${rendered.prose}\n${[...firstLines, ...secondLines.slice(1)].join("\n")}`,
    );
  });

  it("builds a multi-member dry-run plan without mutating the corpus", async () => {
    const harness: OfflineTestHarness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);
    const header =
      "[aiops] OUTCOME fp=scheduled:aiops:team-agent-ai role=aiops tenant=team-agent-ai";

    for (const [offset, ticket] of ["AININJAS-1200", "AININJAS-1201"].entries()) {
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          narrative: `${header}\nticket=${ticket} action=created summary=Incident ${offset + 1}`,
          start_time: Date.UTC(2026, 7, 1, 8 + offset),
          end_time: Date.UTC(2026, 7, 1, 8 + offset, 30),
        }),
      );
    }

    const beforeIds = (await harness.episodicRepository.listAll()).map((episode) => episode.id);
    const report = await migrateOutcomeCorpus({
      db: harness.db,
      episodicRepository: harness.episodicRepository,
      auditLog: harness.auditLog,
      clock: harness.clock,
      runId: createMaintenanceRunId(),
    });

    expect(report).toMatchObject({
      dryRun: true,
      rawOutcomeEpisodeCount: 2,
      scheduledOutcomeSourceCount: 2,
      groupCount: 1,
      multiMemberGroupCount: 1,
      singletonGroupCount: 0,
      rollupsCreated: [],
      auditRowsWritten: 0,
    });
    expect(report.groups[0]?.sources).toHaveLength(2);
    expect((await harness.episodicRepository.listAll()).map((episode) => episode.id)).toEqual(
      beforeIds,
    );
    expect(harness.episodicRepository.listConsolidationFamilies()).toEqual([]);
    expect(harness.auditLog.list()).toEqual([]);
  });

  it("excludes a scheduled family whose current version is the explicit keep", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);
    const source = createEpisodeFixture({
      narrative:
        "OUTCOME fp=scheduled:triage:test role=triage tenant=test\nticket=TEST-KEEP action=created",
      start_time: Date.UTC(2026, 7, 1, 8),
      end_time: Date.UTC(2026, 7, 1, 8, 30),
    });
    await harness.episodicRepository.createEpisode(source);
    const keep = await createFamily(harness, [source], {
      title: "Protected current version",
      narrative: "This current version is explicitly protected from family dissolution.",
    });
    const punishmentSemanticNodeId = createSemanticNodeId();
    await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        id: punishmentSemanticNodeId,
        source_episode_ids: [keep.version.id],
      }),
    );
    const report = await migrateOutcomeCorpus({
      db: harness.db,
      episodicRepository: harness.episodicRepository,
      auditLog: harness.auditLog,
      clock: harness.clock,
      runId: createMaintenanceRunId(),
      specification: {
        toxicEpisodeSpecs: [],
        explicitKeepEpisodeId: keep.version.id,
        explicitKeepBodySha256: bodySha256(keep.version),
        punishmentSemanticNodeId,
        scheduledFpSelfChecks: [],
        grammar: DEFAULT_OUTCOME_CORPUS_GRAMMAR,
      },
    });

    expect(report.legacyFamiliesToDissolve).toEqual([]);
    expect(report.unsafeFamilies).toEqual([
      expect.objectContaining({
        familyId: keep.familyId,
        reason: "contains_explicit_keep",
      }),
    ]);
    expect(report.unsafeItems).toContainEqual(
      expect.stringContaining("contains explicit-keep episode"),
    );
    expect(outcomeCorpusMigrationExitCode(report)).toBe(1);
    expect(
      await harness.episodicRepository.get(keep.version.id, { includeArchived: true }),
    ).not.toBe(null);
  });

  it("refuses an unaudited rollup artifact as unsafe partial state", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);
    const specification = await createSafeSpecification(harness);
    const sources = [0, 1].map((offset) =>
      createEpisodeFixture({
        narrative: `OUTCOME fp=scheduled:aiops:test role=aiops tenant=test\nticket=TEST-PARTIAL-${offset} action=created`,
        start_time: Date.UTC(2026, 7, 1, 8 + offset),
        end_time: Date.UTC(2026, 7, 1, 8 + offset, 30),
      }),
    );
    for (const source of sources) {
      await harness.episodicRepository.createEpisode(source);
    }
    const partial = await createFamily(harness, sources, {
      title: "Unaudited rollup-shaped version",
      narrative:
        "OUTCOME fp=scheduled:aiops:test role=aiops tenant=test\nticket=TEST-PARTIAL-0 action=created",
      tags: ["outcome-rollup", "outcome-corpus-rollup-v1"],
    });

    const report = await migrateOutcomeCorpus({
      db: harness.db,
      episodicRepository: harness.episodicRepository,
      auditLog: harness.auditLog,
      clock: harness.clock,
      runId: createMaintenanceRunId(),
      specification,
    });

    expect(report.unsafeItems).toContainEqual(
      expect.stringContaining(
        `rollup family ${partial.familyId} version ${partial.version.id} has no corresponding migration audit row`,
      ),
    );
    expect(report.unsafeItems).toContainEqual(expect.stringContaining("pre-surgery backup"));
    expect(report.rollupsCreated).toEqual([]);
    expect(outcomeCorpusMigrationExitCode(report)).toBe(1);
  });

  it("restores a dissolved family payload and aborts after an injected cross-store failure", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);
    const specification = await createSafeSpecification(harness);
    const sources = [0, 1].map((offset) =>
      createEpisodeFixture({
        narrative: `OUTCOME fp=scheduled:drafter:test role=drafter tenant=test\nticket=TEST-FAIL-${offset} action=created`,
        start_time: Date.UTC(2026, 7, 1, 10 + offset),
        end_time: Date.UTC(2026, 7, 1, 10 + offset, 30),
      }),
    );
    for (const source of sources) {
      await harness.episodicRepository.createEpisode(source);
    }
    const legacy = await createFamily(harness, sources, {
      title: "Fault-injection legacy family",
      narrative: "The current version payload must be restored after failure.",
    });
    const revert = harness.episodicRepository.revertConsolidationVersion.bind(
      harness.episodicRepository,
    );
    vi.spyOn(harness.episodicRepository, "revertConsolidationVersion").mockImplementationOnce(
      async (input) => {
        await revert(input);
        throw new Error("injected post-dissolution failure");
      },
    );

    await expect(
      migrateOutcomeCorpus(
        {
          db: harness.db,
          episodicRepository: harness.episodicRepository,
          auditLog: harness.auditLog,
          embeddingClient: new TestEmbeddingClient(),
          clock: harness.clock,
          runId: createMaintenanceRunId(),
          specification,
        },
        { apply: true },
      ),
    ).rejects.toThrow(/payload_restore_attempt=succeeded.*pre-surgery backup/u);
    expect(
      harness.episodicRepository.getConsolidationFamily(legacy.familyId)
        ?.current_version_episode_id,
    ).toBe(legacy.version.id);
    expect(
      await harness.episodicRepository.get(legacy.version.id, { includeArchived: true }),
    ).not.toBeNull();
  });

  it("reports unsafe synthetic corpus state, then applies with exact-set acceptance and reruns as a no-op", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);
    const outcome = (role: string, ticket: string, hour: number): Episode =>
      createEpisodeFixture({
        title: `${role} ${ticket}`,
        narrative: [
          `[${role}] OUTCOME fp=job|scheduled:${role}:test role=${role} tenant=test`,
          `ticket=${ticket} action=created summary=Synthetic ${ticket}`,
        ].join("\n"),
        start_time: Date.UTC(2026, 7, 1, hour),
        end_time: Date.UTC(2026, 7, 1, hour, 30),
      });
    const aiops = [outcome("aiops", "TEST-100", 8), outcome("aiops", "TEST-101", 9)];
    const drafter = [outcome("drafter", "TEST-200", 10), outcome("drafter", "TEST-201", 11)];
    const mixedTarget = outcome("triage", "TEST-300", 12);
    const mixedUnrelated = createEpisodeFixture({
      title: "Unrelated mixed-family member",
      narrative: "This record must remain unrelated to the OUTCOME surgery.",
      start_time: Date.UTC(2026, 7, 1, 12),
      end_time: Date.UTC(2026, 7, 1, 12, 30),
    });
    const malformed = createEpisodeFixture({
      title: "Malformed scheduled record",
      narrative: "OUTCOME fp=job|scheduled:aiops:test\nticket=TEST-400 action=created",
      start_time: Date.UTC(2026, 7, 1, 13),
      end_time: Date.UTC(2026, 7, 1, 13, 30),
    });
    const intentionalSurvivor = createEpisodeFixture({
      title: "Manual outcome survivor",
      narrative: "OUTCOME fp=job|manual:operator:test\ndecision=retain action=retain",
      start_time: Date.UTC(2026, 7, 1, 14),
      end_time: Date.UTC(2026, 7, 1, 14, 30),
    });
    const keepSource = createEpisodeFixture({
      title: "Keep source",
      narrative: "Stable source for the explicitly kept consolidation.",
    });

    for (const episode of [
      ...aiops,
      ...drafter,
      mixedTarget,
      mixedUnrelated,
      malformed,
      intentionalSurvivor,
      keepSource,
    ]) {
      await harness.episodicRepository.createEpisode(episode);
    }

    const safeLegacy = await createFamily(harness, drafter, {
      title: "Legacy drafter family",
      narrative: "Legacy consolidation that the migration may safely dissolve.",
    });
    const mixedLegacy = await createFamily(harness, [mixedTarget, mixedUnrelated], {
      title: "Mixed legacy family",
      narrative: "Legacy consolidation mixing targeted and unrelated members.",
    });
    const keepFamily = await createFamily(harness, [keepSource], {
      title: "Explicit keep version",
      narrative: "This exact consolidation version must survive the migration.",
    });
    const punishmentSemanticNodeId = createSemanticNodeId();
    await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        id: punishmentSemanticNodeId,
        source_episode_ids: [keepFamily.version.id],
      }),
    );
    const specification: OutcomeCorpusSpecification = {
      toxicEpisodeSpecs: [],
      explicitKeepEpisodeId: keepFamily.version.id,
      explicitKeepBodySha256: bodySha256(keepFamily.version),
      punishmentSemanticNodeId,
      scheduledFpSelfChecks: [],
      grammar: {
        ...DEFAULT_OUTCOME_CORPUS_GRAMMAR,
        outcomeFingerprintPattern: /OUTCOME fp=(job\|\S+)/gu,
        scheduledFingerprintPattern: /^job\|scheduled:/u,
      },
    };
    const baseDependencies = {
      db: harness.db,
      episodicRepository: harness.episodicRepository,
      auditLog: harness.auditLog,
      clock: harness.clock,
      specification,
    };
    const beforeDryRunIds = (await harness.episodicRepository.listAll())
      .map((episode) => episode.id)
      .sort();
    const dryRun = await migrateOutcomeCorpus({
      ...baseDependencies,
      runId: createMaintenanceRunId(),
    });
    const dryRunText = formatOutcomeCorpusMigrationReport(dryRun);

    expect(dryRun.dryRun).toBe(true);
    expect(dryRun.malformedScheduledOutcomeEpisodeIds).toEqual([malformed.id]);
    expect(dryRun.unsafeFamilies).toEqual([
      expect.objectContaining({
        familyId: mixedLegacy.familyId,
        reason: "mixed_membership",
        members: expect.arrayContaining([
          { episodeId: mixedTarget.id, intendedState: "roll_up:triage/2026-08-01" },
          { episodeId: mixedUnrelated.id, intendedState: "preserve_unrelated" },
        ]),
      }),
    ]);
    expect(dryRun.legacyFamiliesToDissolve.map((item) => item.family.family_id)).toEqual([
      safeLegacy.familyId,
    ]);
    expect(dryRunText).toContain(`malformed_scheduled episode=${malformed.id}`);
    expect(dryRunText).toContain(
      `unsafe_family_member family=${mixedLegacy.familyId} episode=${mixedUnrelated.id} intended_state=preserve_unrelated`,
    );
    expect(outcomeCorpusMigrationExitCode(dryRun)).toBe(1);
    expect(
      (await harness.episodicRepository.listAll()).map((episode) => episode.id).sort(),
    ).toEqual(beforeDryRunIds);

    // The operator must resolve every dry-run hazard before --apply. The test
    // explicitly separates the mixed family and removes the malformed fixture.
    await harness.episodicRepository.revertConsolidationVersion({
      familyId: mixedLegacy.familyId,
      versionEpisodeId: mixedLegacy.version.id,
      previousCurrentVersionEpisodeId: null,
      previousCoverageHash: null,
      previousPolicyVersion: null,
    });
    await harness.episodicRepository.delete(malformed.id);

    const applied = await migrateOutcomeCorpus(
      {
        ...baseDependencies,
        embeddingClient: new TestEmbeddingClient(),
        runId: createMaintenanceRunId(),
      },
      { apply: true },
    );

    expect(applied.dryRun).toBe(false);
    expect(applied.unsafeItems).toEqual([]);
    expect(applied.rollupsCreated).toHaveLength(2);
    expect(applied.expectedVisibleOutcomeEpisodeIds).toHaveLength(4);
    expect(applied.missingVisibleOutcomeEpisodeIdsAfter).toEqual([]);
    expect(applied.extraVisibleOutcomeEpisodeIdsAfter).toEqual([]);
    expect(applied.actualVisibleOutcomeEpisodeIdsAfter).toEqual(
      applied.expectedVisibleOutcomeEpisodeIds,
    );
    expect(applied.customAuditRowsWritten).toBe(3);
    expect(applied.customAuditRowsWithNoReverser).toBe(3);
    expect(outcomeCorpusMigrationExitCode(applied)).toBe(0);

    const idsAfterApply = (await harness.episodicRepository.listAll())
      .map((episode) => episode.id)
      .sort();
    const auditCountAfterApply = harness.auditLog.list().length;
    const secondApply = await migrateOutcomeCorpus(
      {
        ...baseDependencies,
        embeddingClient: new TestEmbeddingClient(),
        runId: createMaintenanceRunId(),
      },
      { apply: true },
    );

    expect(secondApply.dryRun).toBe(false);
    expect(secondApply.unsafeItems).toEqual([]);
    expect(secondApply.legacyFamiliesToDissolve).toEqual([]);
    expect(secondApply.rollupsCreated).toEqual([]);
    expect(secondApply.versionsReembedded).toEqual([]);
    expect(secondApply.auditRowsWritten).toBe(0);
    expect(secondApply.missingVisibleOutcomeEpisodeIdsAfter).toEqual([]);
    expect(secondApply.extraVisibleOutcomeEpisodeIdsAfter).toEqual([]);
    expect(outcomeCorpusMigrationExitCode(secondApply)).toBe(0);
    expect(
      (await harness.episodicRepository.listAll()).map((episode) => episode.id).sort(),
    ).toEqual(idsAfterApply);
    expect(harness.auditLog.list()).toHaveLength(auditCountAfterApply);
  });

  it("dissolves an acknowledged mixed family and rejects unknown acknowledgments", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);
    const target = createEpisodeFixture({
      title: "triage TEST-500",
      narrative: [
        "[triage] OUTCOME fp=job|scheduled:triage:test role=triage tenant=test",
        "ticket=TEST-500 action=created summary=Synthetic TEST-500",
      ].join("\n"),
      start_time: Date.UTC(2026, 7, 2, 8),
      end_time: Date.UTC(2026, 7, 2, 8, 30),
    });
    const sibling = createEpisodeFixture({
      title: "triage TEST-501",
      narrative: [
        "[triage] OUTCOME fp=job|scheduled:triage:test role=triage tenant=test",
        "ticket=TEST-501 action=created summary=Synthetic TEST-501",
      ].join("\n"),
      start_time: Date.UTC(2026, 7, 2, 9),
      end_time: Date.UTC(2026, 7, 2, 9, 30),
    });
    const unrelated = createEpisodeFixture({
      title: "Unrelated member",
      narrative: "Unrelated narrative that must return to visibility.",
      start_time: Date.UTC(2026, 7, 2, 9),
      end_time: Date.UTC(2026, 7, 2, 9, 30),
    });

    for (const episode of [target, sibling, unrelated]) {
      await harness.episodicRepository.createEpisode(episode);
    }

    const mixed = await createFamily(harness, [target, sibling, unrelated], {
      title: "Mixed family",
      narrative: "Mixed consolidation of targeted and unrelated members.",
    });
    const keepSource = createEpisodeFixture({
      title: "Keep source",
      narrative: "Stable keep source.",
    });
    await harness.episodicRepository.createEpisode(keepSource);
    const keepFamily = await createFamily(harness, [keepSource], {
      title: "Keep version",
      narrative: "Kept consolidation version.",
    });
    const punishmentSemanticNodeId = createSemanticNodeId();
    await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        id: punishmentSemanticNodeId,
        source_episode_ids: [keepFamily.version.id],
      }),
    );
    const specification: OutcomeCorpusSpecification = {
      toxicEpisodeSpecs: [],
      explicitKeepEpisodeId: keepFamily.version.id,
      explicitKeepBodySha256: bodySha256(keepFamily.version),
      punishmentSemanticNodeId,
      scheduledFpSelfChecks: [],
      grammar: {
        ...DEFAULT_OUTCOME_CORPUS_GRAMMAR,
        outcomeFingerprintPattern: /OUTCOME fp=(job\|\S+)/gu,
        scheduledFingerprintPattern: /^job\|scheduled:/u,
      },
    };
    const baseDependencies = {
      db: harness.db,
      episodicRepository: harness.episodicRepository,
      auditLog: harness.auditLog,
      clock: harness.clock,
      specification,
    };

    const unacknowledged = await migrateOutcomeCorpus({
      ...baseDependencies,
      runId: createMaintenanceRunId(),
    });

    expect(unacknowledged.unsafeFamilies).toHaveLength(1);
    expect(unacknowledged.unsafeItems.join(" ")).toContain("--dissolve-mixed-family");

    const unknownAck = await migrateOutcomeCorpus({
      ...baseDependencies,
      runId: createMaintenanceRunId(),
      acknowledgedMixedFamilyIds: ["cfam_does_not_exist"],
    });

    expect(unknownAck.unsafeItems.join(" ")).toContain(
      "did not match any mixed family",
    );

    const applied = await migrateOutcomeCorpus(
      {
        ...baseDependencies,
        embeddingClient: new TestEmbeddingClient(),
        runId: createMaintenanceRunId(),
        acknowledgedMixedFamilyIds: [mixed.familyId],
      },
      { apply: true },
    );

    expect(applied.dryRun).toBe(false);
    expect(applied.unsafeFamilies).toEqual([]);
    expect(applied.acknowledgedMixedItems.join(" ")).toContain(unrelated.id);
    expect(applied.rollupsCreated).toHaveLength(1);
    expect(outcomeCorpusMigrationExitCode(applied)).toBe(0);
    const statsById = new Map(
      harness.episodicRepository.listStats().map((stats) => [stats.episode_id, stats]),
    );
    expect(statsById.get(unrelated.id)?.archived).toBe(false);
    const visibleIds = new Set(
      (await harness.episodicRepository.listEffectivelyVisibleEpisodeIds?.() ?? []) as string[],
    );
    if (visibleIds.size > 0) {
      expect(visibleIds.has(unrelated.id)).toBe(true);
    }
  });

});
