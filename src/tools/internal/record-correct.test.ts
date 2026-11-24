import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resolveFinalizerNonTerminalTools } from "../../cognition/deliberation/autonomous-finalizer-tools.js";
import { compactEvidenceLedger } from "../../cognition/evidence-ledger/compaction.js";
import { renderEvidenceLedger } from "../../cognition/evidence-ledger/finalizer-ledger.js";
import {
  buildRecordEvidenceLedger,
  createRecordEvidenceFixture,
} from "../../cognition/record-evidence-test-support.js";
import {
  recordCorrectionTextHash,
  type RecordCorrectionResult,
} from "../../memory/identity/annotations.js";
import {
  EVIDENCE_GUIDANCE,
  episodeEvidencePromptRow,
  renderEvidencePromptJson,
  withEpisodeEvidenceContext,
  withSemanticNodeEvidenceContext,
} from "../../memory/common/episode-evidence.js";
import { RecordCorrectionService } from "../../memory/lifecycle-ops/correct-record.js";
import { MAX_RESOLUTION_CORRECTIONS } from "../../memory/self/resolution-corrections.js";
import { recordTestAnnotation } from "../../offline/evidence-test-support.js";
import { createOfflineTestHarness } from "../../offline/test-support.js";
import { StreamReader, StreamWriter } from "../../stream/index.js";
import { CallbackTracer, type CallbackTraceEntry } from "../../tracing/tracer.js";
import { ManualClock } from "../../util/clock.js";
import {
  createEntityId,
  createEpisodeId,
  createGoalId,
  createSessionId,
  createStreamEntryId,
} from "../../util/ids.js";
import { ToolDispatcher } from "../dispatcher.js";
import { createRecordIdResolver } from "../record-id-resolver.js";
import { createEpisodicGetTool } from "./episodic-get.js";
import { createIdentityEventsListForCognitionTool } from "./identity-events-list.js";
import { createRecordCorrectTool } from "./record-correct.js";
import { createSemanticWalkTool } from "./semantic-walk.js";

describe("record corrections from the autonomous tool", () => {
  let h: Awaited<ReturnType<typeof createOfflineTestHarness>>;
  let fixture: ReturnType<typeof createRecordEvidenceFixture>;
  let dispatcher: ToolDispatcher;
  let service: RecordCorrectionService;
  let clock: ManualClock;
  let traces: CallbackTraceEntry[];
  let sessionId: ReturnType<typeof createSessionId>;

  beforeEach(async () => {
    clock = new ManualClock(1_700_000_000_123);
    h = await createOfflineTestHarness({ clock });
    fixture = createRecordEvidenceFixture();
    traces = [];
    sessionId = createSessionId();
    for (const { episode } of fixture.episodes) await h.episodicRepository.createEpisode(episode);
    await h.semanticNodeRepository.insert(fixture.node);
    const entryIndex = h.createContext().entryIndex!;
    const createStreamWriter = (id: typeof sessionId) =>
      new StreamWriter({
        sessionId: id,
        dataDir: h.tempDir,
        entryIndex,
        clock,
      });
    service = new RecordCorrectionService({
      ...h,
      entryIndex,
      tracer: new CallbackTracer({ sink: (entry) => traces.push(entry) }),
    });
    dispatcher = new ToolDispatcher({
      createStreamWriter,
      recordIdResolver: createRecordIdResolver(h.db),
      clock,
    })
      .register(createRecordCorrectTool({ recordCorrections: service, createStreamWriter }))
      .register(
        createIdentityEventsListForCognitionTool({
          listEvents: (options) => h.identityService.listEvents(options),
        }),
      );
  });
  afterEach(async () => {
    vi.restoreAllMocks();
    await h?.cleanup();
  });

  const target = () => fixture.episodes[0]!.episode;
  const evidence = () => [{ id: fixture.episodes[1]!.episode.id, basis: "own_read" as const }];
  const annotations = () =>
    h.identityEventRepository.listAnnotationsForTargets([
      { record_type: "episode", record_id: target().id },
    ])[0]!.annotations;
  const logs = () => new StreamReader({ dataDir: h.tempDir, sessionId }).tail(100);
  const dispatch = (patch: Record<string, unknown> = {}, session = sessionId) =>
    dispatcher.dispatch({
      toolName: "tool.records.correct",
      origin: "autonomous",
      turnOrigin: "autonomous",
      turnId: "synthetic-wake",
      sessionId: session,
      input: {
        target_id: target().id,
        correction_note: "確定していない。",
        evidence: evidence(),
        ...patch,
      },
    });

  it("offers the write on wakes and commits both typed targets to readable identity history", async () => {
    const tool = dispatcher
      .listTools("autonomous")
      .find(({ name }) => name === "tool.records.correct")!;
    expect(tool.description).toHaveLength(290);
    expect(`- ${tool.name}: ${tool.menuSummary}\n`).toHaveLength(87);
    expect(
      resolveFinalizerNonTerminalTools({ dispatcher, turnOrigin: "autonomous" }).map(
        (tool) => tool.name,
      ),
    ).toContain("tool.records.correct");
    expect(
      resolveFinalizerNonTerminalTools({ dispatcher, turnOrigin: "user" }).map((tool) => tool.name),
    ).not.toContain("tool.records.correct");
    const source = await h.streamWriter.append({ kind: "thought", content: "Synthetic evidence" });
    const note = "  未確認だった。\n🧭  ";
    const result = await dispatch({
      target_id: target().id.slice(0, 7),
      correction_note: note,
      corrects_span: "\n原文。",
      evidence: [
        { id: evidence()[0]!.id.slice(0, 7), basis: "own_read" },
        { id: source.id, basis: "reported", reported_by: "  別の読者  " },
      ],
    });
    expect(result).toMatchObject({
      ok: true,
      output: {
        seq: 1,
        total_count: 1,
        shown_count: 1,
        id_resolutions: [
          { field: "target_id", resolved_id: target().id },
          { field: "evidence.0.id", resolved_id: evidence()[0]!.id },
        ],
        correction: {
          text: note,
          corrects_span: "\n原文。",
          writer: {
            author: "borg",
            turn_id: "synthetic-wake",
            turn_kind: "autonomous",
            path: "tool",
          },
          created_at: clock.now(),
          created_at_iso: new Date(clock.now()).toISOString(),
          original_record: { record_type: "episode", record_id: target().id },
          target_text_hash: recordCorrectionTextHash(target().narrative),
          evidence: [
            evidence()[0],
            { id: source.id, basis: "reported", reported_by: "  別の読者  " },
          ],
        },
      },
    });
    expect(
      await dispatch({ target_id: fixture.node.id.slice(0, 9), corrects_span: "原文。" }),
    ).toMatchObject({ ok: true });
    expect(logs().filter((entry) => entry.kind === "internal_event")).toMatchObject([
      {
        session_id: sessionId,
        turn_id: "synthetic-wake",
        content: { event: "episode_annotation", target_ids: [target().id], correction: note },
      },
      {
        session_id: sessionId,
        turn_id: "synthetic-wake",
        content: { event: "episode_annotation", target_ids: [fixture.node.id] },
      },
    ]);
    const history = await dispatcher.dispatch({
      toolName: "tool.identityEvents.listForCognition",
      origin: "autonomous",
      sessionId,
      input: { limit: 10 },
    });
    expect(history).toMatchObject({
      ok: true,
      output: {
        events: [
          {
            action: "record_self_correction",
            record_type: "semantic_node",
            record_id: fixture.node.id,
          },
          { action: "record_self_correction", record_type: "episode", record_id: target().id },
        ],
      },
    });
    expect(annotations()[0]).toMatchObject({ text: note, self_correction: { seq: 1 } });
  });

  it.each([
    ["empty text", { correction_note: "" }, "TOOL_INPUT_INVALID"],
    ["blank text", { correction_note: "  \n" }, "TOOL_INPUT_INVALID"],
    ["601 UTF-16 units", { correction_note: "界".repeat(599) + "🧭" }, "TOOL_INPUT_INVALID"],
    ["wrong target type", { target_id: createGoalId() }, "TOOL_INPUT_INVALID"],
    ["missing target", { target_id: createEpisodeId() }, "RECORD_CORRECTION_TARGET_NOT_FOUND"],
    ["absent span", { corrects_span: " 原文。 " }, "RECORD_CORRECTION_SPAN_NOT_FOUND"],
    [
      "missing episode evidence",
      { evidence: [{ id: createEpisodeId(), basis: "own_read" }] },
      "RECORD_CORRECTION_EVIDENCE_NOT_FOUND",
    ],
    [
      "missing stream evidence",
      { evidence: [{ id: createStreamEntryId(), basis: "own_read" }] },
      "RECORD_CORRECTION_EVIDENCE_NOT_FOUND",
    ],
    ["empty evidence", { evidence: [] }, "TOOL_INPUT_INVALID"],
    [
      "missing reporter",
      { evidence: [{ id: createEpisodeId(), basis: "reported" }] },
      "TOOL_INPUT_INVALID",
    ],
    [
      "missing earlier correction",
      { targets_correction_seq: 1 },
      "RECORD_CORRECTION_SEQUENCE_NOT_FOUND",
    ],
    ["model-supplied stamp", { author: "borg" }, "TOOL_INPUT_INVALID"],
  ])("rejects %s before either annotation write", async (_case, patch, errorCode) => {
    expect(await dispatch(patch)).toMatchObject({ ok: false, errorCode });
    expect(logs().filter((entry) => entry.kind === "internal_event")).toEqual([]);
    expect(h.identityService.listEvents()).toEqual([]);
  });

  it.each([
    ["rewrite", "RECORD_CORRECTION_SPAN_NOT_FOUND"],
    ["delete", "RECORD_CORRECTION_TARGET_NOT_FOUND"],
  ])(
    "rejects a node %s during source disclosure lookup before either write",
    async (change, errorCode) => {
      const source = await h.streamWriter.append({
        kind: "thought",
        content: "Synthetic evidence",
      });
      const getMany = h.episodicRepository.getMany.bind(h.episodicRepository);
      const lookup = vi.spyOn(h.episodicRepository, "getMany").mockImplementation(async (ids) => {
        if (ids.includes(fixture.node.source_episode_ids[0]!)) {
          if (change === "rewrite") {
            await h.semanticNodeRepository.update(fixture.node.id, {
              description: "Rewritten node.",
            });
          } else {
            await h.semanticNodeRepository.delete(fixture.node.id);
          }
        }
        return getMany(ids);
      });
      expect(
        await dispatch({
          target_id: fixture.node.id,
          corrects_span: "原文。",
          evidence: [{ id: source.id, basis: "own_read" }],
        }),
      ).toMatchObject({ ok: false, errorCode });
      expect(lookup).toHaveBeenCalledWith(fixture.node.source_episode_ids);
      expect(logs().filter((entry) => entry.kind === "internal_event")).toEqual([]);
      expect(h.identityService.listEvents()).toEqual([]);
    },
  );

  it.each([
    ["a correction text containing a lone surrogate", { correction_note: "Draft \ud83e" }],
    ["a quote of a single surrogate from an emoji", { corrects_span: "\ud83e" }],
  ])("rejects %s before either write", async (_case, patch) => {
    await h.episodicRepository.update(target().id, { narrative: "Synthetic observation 🧭" });
    expect(await dispatch(patch)).toMatchObject({ ok: false, errorCode: "TOOL_INPUT_INVALID" });
    expect(logs().filter((entry) => entry.kind === "internal_event")).toEqual([]);
    expect(h.identityService.listEvents()).toEqual([]);
  });

  it("serializes concurrent appends and validates a quote against the earlier correction on the same record", async () => {
    const first = "界".repeat(598) + "🧭";
    const [left, right] = await Promise.all([
      dispatch({ correction_note: first }),
      dispatch({ correction_note: "Separate correction" }, createSessionId()),
    ]);
    expect(left.ok && right.ok).toBe(true);
    if (!left.ok) throw new Error(left.error);
    expect(
      annotations()
        .map((row) => row.self_correction!.seq)
        .sort(),
    ).toEqual([1, 2]);
    const firstSeq = (left.output as RecordCorrectionResult).seq;
    expect(
      await dispatch({ targets_correction_seq: firstSeq, corrects_span: "界🧭" }),
    ).toMatchObject({
      ok: true,
      output: {
        seq: 3,
        correction: {
          targets_correction_seq: firstSeq,
          quoted_text_hash: recordCorrectionTextHash(first),
        },
      },
    });
    expect(
      await dispatch({ target_id: fixture.node.id, targets_correction_seq: firstSeq }),
    ).toMatchObject({
      ok: false,
      errorCode: "RECORD_CORRECTION_SEQUENCE_NOT_FOUND",
    });
    expect(annotations().find((row) => row.text === first)?.text).toBe(first);
  });

  it("reserves the newest self correction beside operator rows and retains the count under page pressure", async () => {
    const text = "界".repeat(600);
    await dispatch({ correction_note: "Older self correction" });
    await dispatch({ correction_note: "Middle self correction" });
    for (const label of ["First operator correction", "Second operator correction"]) {
      recordTestAnnotation(h.identityEventRepository, target().id, label, { ts: clock.now() + 1 });
    }
    expect(await dispatch({ correction_note: text })).toMatchObject({
      ok: true,
      output: { total_count: 3, shown_count: 1 },
    });
    const ledger = await buildRecordEvidenceLedger(fixture.input, h);
    const entry = ledger.sections
      .flatMap((section) => section.entries)
      .find((row) => row.id === `episode:${target().id}`);
    expect(entry?.evidence_context).toMatchObject({
      annotations: [
        { author: "borg", seq: 3, text },
        { text: "Second operator correction", authority: "Synthetic operator" },
      ],
      self_correction_count: 3,
      annotations_omitted: 3,
    });
    const compacted = compactEvidenceLedger(ledger, {
      sectionOptions: { episodes: { maxTokens: 180, maxEntries: 3 } },
    }).ledger;
    const compactedEntry = compacted.sections
      .find((section) => section.id === "episodes")
      ?.entries.find((row) => row.id === entry?.id);
    expect(compactedEntry?.evidence_context).toMatchObject({ self_correction_count: 3 });
    expect(
      compactedEntry?.evidence_context?.annotations?.some((row) => row.author === "borg") ?? false,
    ).toBe(false);
    expect(renderEvidenceLedger(compacted)).toContain("self_correction_count: 3");
  });

  it.each(["episode", "semantic_node"] as const)(
    "reports a changed %s and whether the quote remains",
    async (type) => {
      const recordId = type === "episode" ? target().id : fixture.node.id;
      const patch = async (text: string) =>
        type === "episode"
          ? h.episodicRepository.update(target().id, { narrative: text })
          : h.semanticNodeRepository.update(fixture.node.id, { description: text });
      const context = { origin: "autonomous", sessionId } as const;
      const read = async () => {
        if (type === "episode") {
          const get = createEpisodicGetTool({
            recordEvidenceReader: h,
            episodicRepository: h.episodicRepository,
          });
          return get.outputSchema.parse(await get.invoke({ episode_id: target().id }, context))
            .episode;
        }
        const walk = createSemanticWalkTool({
          recordEvidenceReader: h,
          walkGraph: async () => [
            { node: (await h.semanticNodeRepository.get(fixture.node.id))!, edgePath: [] },
          ],
        });
        return walk.outputSchema.parse(
          await walk.invoke({ node_id: fixture.node.id, relation: "supports" }, context),
        ).steps[0]?.node;
      };
      expect(await dispatch({ target_id: recordId, corrects_span: "原文。" })).toMatchObject({
        ok: true,
      });
      expect(await read()).toMatchObject({
        annotations: [{ author: "borg", target_changed: false }],
      });
      await patch(`${"changed ".repeat(80)}原文。`);
      expect(await read()).toMatchObject({
        annotations: [{ author: "borg", target_changed: true, quoted_span_present: true }],
      });
      await patch("The quoted words were removed.");
      expect(await read()).toMatchObject({
        annotations: [{ author: "borg", target_changed: true, quoted_span_present: false }],
      });
    },
  );

  it("keeps a short operator annotation in the tool and offline envelopes when the reserved correction is too large", async () => {
    const episode = (await h.episodicRepository.update(target().id, {
      origin_audience_entity_ids: Array.from({ length: 220 }, () => createEntityId()),
      shared: false,
    }))!;
    const correction = await dispatch({ correction_note: "界".repeat(600) });
    expect(correction).toMatchObject({
      ok: true,
      output: { seq: 1, total_count: 1, shown_count: 1 },
    });
    const operator = "Audited observation.";
    recordTestAnnotation(h.identityEventRepository, episode.id, operator);

    const get = createEpisodicGetTool({
      recordEvidenceReader: h,
      episodicRepository: h.episodicRepository,
    });
    const payload = get.outputSchema.parse(
      await get.invoke({ episode_id: episode.id }, { origin: "autonomous", sessionId }),
    );
    const [contextual] = await withEpisodeEvidenceContext(h, [episode]);
    const offline = JSON.parse(
      renderEvidencePromptJson({
        evidence_guidance: EVIDENCE_GUIDANCE,
        episode: episodeEvidencePromptRow(contextual!),
      }),
    );
    for (const row of [payload.episode, offline.episode]) {
      expect(row).toMatchObject({
        narrative: episode.narrative,
        annotations: [{ text: operator, authority: "Synthetic operator" }],
        annotations_omitted: 1,
        self_correction_count: 1,
      });
    }
  });

  it.each([false, true])(
    "keeps an earlier correction's quote after a parent change (earlier retracted: %s)",
    async (retracted) => {
      const nodeId = fixture.node.id;
      const quote = "Only in the earlier correction. 🧭";
      const first = await dispatch({ target_id: nodeId, correction_note: quote });
      if (!first.ok) throw new Error(first.error);
      expect(
        await dispatch({ target_id: nodeId, targets_correction_seq: 1, corrects_span: quote }),
      ).toMatchObject({ ok: true, output: { seq: 2 } });
      if (retracted) {
        h.identityEventRepository.record({
          record_type: "semantic_node",
          record_id: nodeId,
          action: "synthetic_retraction",
          provenance: { kind: "manual" },
          new_value: {
            stream_entry_id: createStreamEntryId(),
            annotation: {
              event: "episode_annotation_retracted",
              target_ids: [nodeId],
              reverses_stream_entry_id: (first.output as RecordCorrectionResult).correction
                .stream_entry_id,
            },
          },
        });
      }
      const updated = await h.semanticNodeRepository.update(nodeId, {
        description: "New description.",
      });
      const [projected] = withSemanticNodeEvidenceContext(h, [updated!]);
      expect(projected!.evidence_context?.annotations?.[0]).toMatchObject({
        seq: 2,
        targets_correction_seq: 1,
        target_changed: true,
        quoted_span_present: true,
      });
      expect(projected!.evidence_context?.self_correction_count).toBe(retracted ? 1 : 2);
    },
  );

  it("keeps retracted sequences, deduplicates repeats, and enforces the lifetime cap", async () => {
    expect(await dispatch()).toMatchObject({ ok: true });
    const [first] = annotations();
    h.identityEventRepository.record({
      record_type: "episode",
      record_id: target().id,
      action: "synthetic_retraction",
      provenance: { kind: "manual" },
      new_value: {
        stream_entry_id: createStreamEntryId(),
        annotation: {
          event: "episode_annotation_retracted",
          target_ids: [target().id],
          reverses_stream_entry_id: first!.stream_entry_id,
        },
      },
    });
    expect(annotations()).toEqual([]);
    expect(await dispatch({ evidence: [...evidence(), ...evidence()] })).toMatchObject({
      ok: true,
      output: { seq: 1, total_count: 0, shown_count: 0 },
    });
    for (let seq = 2; seq <= MAX_RESOLUTION_CORRECTIONS; seq++) {
      expect(await dispatch({ correction_note: `Synthetic correction ${seq}` })).toMatchObject({
        ok: true,
        output: { seq },
      });
    }
    expect(await dispatch({ correction_note: "Over the lifetime limit" })).toMatchObject({
      ok: false,
      errorCode: "RECORD_CORRECTION_LIMIT",
    });
    expect(logs().filter((entry) => entry.kind === "internal_event")).toHaveLength(
      MAX_RESOLUTION_CORRECTIONS,
    );
    expect(annotations()).toHaveLength(MAX_RESOLUTION_CORRECTIONS - 1);
  });

  it("surfaces a failed identity insert once and leaves the orphan out of the annotation reader", async () => {
    vi.spyOn(h.identityEventRepository, "record").mockImplementationOnce(() => {
      throw new Error("Fixture identity failure");
    });
    expect(await dispatch()).toMatchObject({
      ok: false,
      errorCode: "RECORD_CORRECTION_IDENTITY_FAILED",
    });
    expect(annotations()).toEqual([]);
    const orphans = logs().filter((entry) => entry.kind === "internal_event");
    expect(orphans).toHaveLength(1);
    expect(orphans[0]?.content).not.toHaveProperty("self_correction.seq");
    expect(traces).toEqual([
      expect.objectContaining({
        event: "memory.annotations.write_failed",
        reason: "identity_append_failed",
      }),
    ]);
    expect(await dispatch()).toMatchObject({ ok: true, output: { seq: 1 } });
  });
});
