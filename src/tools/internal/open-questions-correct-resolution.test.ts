import { openQuestionResolutionIdentity } from "../../memory/self/resolution-corrections.js";
import { EvidenceLedgerBuilder } from "../../cognition/evidence-ledger/builder.js";
import { renderEvidenceLedger } from "../../cognition/evidence-ledger/renderer.js";
import { compactEvidenceLedger } from "../../cognition/evidence-ledger/compaction.js";
import { renderSection } from "../../cognition/evidence-ledger/section-rendering.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";
import { StreamReader } from "../../stream/index.js";
import { createWorkingMemory } from "../../memory/working/index.js";
import {
  memoryDisclosurePayloadFields,
  openQuestionMemoryDisclosureLabel,
} from "../../memory/common/disclosure-serializers.js";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createBorgFacades } from "../../borg/facade.js";
import type { BorgDependencies } from "../../borg/types.js";
import { resolveFinalizerNonTerminalTools } from "../../cognition/deliberation/autonomous-finalizer-tools.js";
import { renderPlanRequestedVerificationRetrieval } from "../../cognition/deliberation/prompt/retrieval.js";
import { openQuestionStateMetadata } from "../../cognition/evidence-ledger/open-question-handles.js";
import { formatAutonomyTriggerContext } from "../../cognition/autonomy-trigger.js";
import { CorrectionService } from "../../correction/service.js";
import {
  publicMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
} from "../../memory/common/disclosure-label.js";
import { IdentityCasMismatchError } from "../../util/errors.js";
import {
  formatResolutionCorrections,
  MAX_RESOLUTION_CORRECTIONS,
  RESOLUTION_CORRECTION_TOOL_NAME,
} from "../../memory/self/resolution-corrections.js";
import { openQuestionResolutionCorrectionInputSchema } from "../../memory/self/types.js";
import { createEpisodeFixture, createOfflineTestHarness } from "../../offline/test-support.js";
import { ManualClock } from "../../util/clock.js";
import {
  createEntityId,
  createEpisodeId,
  createOpenQuestionId,
  createStreamEntryId,
  DEFAULT_SESSION_ID,
} from "../../util/ids.js";
import { headTailTextExcerpt } from "../../util/text-excerpt.js";
import { ToolDispatcher } from "../dispatcher.js";
import { createRecordIdResolver } from "../record-id-resolver.js";
import { createOpenQuestionsCorrectResolutionTool } from "./open-questions-correct-resolution.js";
import { createOpenQuestionsCreateTool } from "./open-questions-create.js";
import { createOpenQuestionsResolveTool } from "./open-questions-resolve.js";
import { createIdentityEventsListForCognitionTool } from "./identity-events-list.js";

const context = {
  sessionId: DEFAULT_SESSION_ID,
  origin: "autonomous",
  turnOrigin: "autonomous",
  turnId: "turn-correction",
} as const;
const originalNote = `  HEAD 原文\n${"middle ".repeat(2_000)}\nTAIL  `;
const validInput = {
  open_question_id: createOpenQuestionId(),
  correction_note: "Correction",
  evidence: [{ id: createEpisodeId(), basis: "own_read" as const }],
};

describe("correction input structure", () => {
  it.each([
    { correction_note: "  " },
    { correction_note: "x".repeat(2_001) },
    { corrects_span: " " },
    { targets_correction_seq: 0 },
    { targets_correction_seq: -1 },
    { targets_correction_seq: 1.5 },
    { evidence: [] },
    { evidence: [{ id: "wrong", basis: "own_read" }] },
    { evidence: [{ id: createEpisodeId(), basis: "inferred" }] },
    { evidence: [{ id: createEpisodeId(), basis: "reported" }] },
    { evidence: [{ id: createEpisodeId(), basis: "reported", reported_by: "  " }] },
    { evidence: [{ id: createEpisodeId(), basis: "reported", reported_by: "x".repeat(201) }] },
    { evidence: [{ id: createEpisodeId(), basis: "own_read", reported_by: "Someone" }] },
    { evidence: [{ id: createEpisodeId(), basis: "own_read", extra: true }] },
    { writer: { author: "someone" } },
    { created_at: 1 },
    { seq: 2 },
    { resolution_version: 2 },
    { disclosure_label: publicMemoryDisclosureLabel() },
    { open_question_id: "bad" },
  ])("rejects malformed fields %j", (patch) => {
    expect(
      openQuestionResolutionCorrectionInputSchema.safeParse({ ...validInput, ...patch }).success,
    ).toBe(false);
  });
  it("trims only outer whitespace, accepts bounds and a null target", () => {
    const input = openQuestionResolutionCorrectionInputSchema.parse({
      ...validInput,
      correction_note: `  ${"字".repeat(2_000)}  `,
      corrects_span: "  A\n B  ",
      targets_correction_seq: null,
      evidence: [
        { id: createStreamEntryId(), basis: "reported", reported_by: ` ${"名".repeat(200)} ` },
      ],
    });
    expect(input.correction_note).toHaveLength(2_000);
    expect(input.corrects_span).toBe("A\n B");
    expect(input.evidence[0]).toHaveProperty("reported_by", "名".repeat(200));
  });
});

describe("resolved question corrections", () => {
  let h: Awaited<ReturnType<typeof createOfflineTestHarness>>;
  let clock: ManualClock;
  let question: ReturnType<typeof h.openQuestionsRepository.resolve>;
  let episode: ReturnType<typeof createEpisodeFixture>;
  let streamId: ReturnType<typeof createStreamEntryId>;
  let tool: ReturnType<typeof createOpenQuestionsCorrectResolutionTool>;
  const input = () => ({
    open_question_id: question.id,
    correction_note: "First correction 原文",
    corrects_span: "HEAD 原文",
    evidence: [{ id: episode.id, basis: "own_read" as const }],
  });
  const invoke = (patch: Partial<Parameters<typeof tool.invoke>[0]> = {}) =>
    tool.invoke({ ...input(), ...patch }, context);

  beforeEach(async () => {
    clock = new ManualClock(Date.parse("2026-11-16T10:11:12.345Z"));
    h = await createOfflineTestHarness({ clock });
    episode = createEpisodeFixture({ shared: true, audience_entity_id: null });
    await h.episodicRepository.createEpisode(episode);
    streamId = (await h.streamWriter.append({ kind: "thought", content: "Evidence." })).id;
    const open = h.openQuestionsRepository.add({
      question: "What was established?",
      urgency: 0.5,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    question = h.openQuestionsRepository.resolve(open.id, {
      resolution_note: originalNote,
      resolution_evidence_episode_ids: [episode.id],
      resolution_disclosure_label: publicMemoryDisclosureLabel(),
    });
    clock.advance(123);
    tool = createOpenQuestionsCorrectResolutionTool({
      identityService: h.identityService,
      openQuestionsRepository: h.openQuestionsRepository,
      episodicRepository: h.episodicRepository,
      entryIndex: h.createContext().entryIndex!,
    });
  });
  afterEach(async () => {
    vi.restoreAllMocks();
    await h?.cleanup();
  });

  it("corrects a resolved question by short handle and echoes question and evidence resolutions", async () => {
    const dispatcher = new ToolDispatcher({
      recordIdResolver: createRecordIdResolver(h.db),
      createStreamWriter: () => h.streamWriter,
      clock,
    });
    dispatcher.register(tool);
    const questionHandle = question.id.slice(0, 7);
    const episodeHandle = episode.id.slice(0, 7);
    const result = await dispatcher.dispatch({
      toolName: tool.name,
      origin: "autonomous",
      audienceEntityId: createEntityId(),
      sessionId: DEFAULT_SESSION_ID,
      input: {
        ...input(),
        open_question_id: questionHandle,
        evidence: [{ id: episodeHandle, basis: "own_read" }],
      },
    });
    expect(result).toMatchObject({
      ok: true,
      output: {
        openQuestion: { id: question.id, status: "resolved", resolution_correction_count: 1 },
        id_resolutions: [
          { field: "open_question_id", input: questionHandle, resolved_id: question.id },
          { field: "evidence.0.id", input: episodeHandle, resolved_id: episode.id },
        ],
      },
    });
    expect(
      h.openQuestionsRepository.get(question.id)?.resolution_corrections?.[0]?.evidence[0]?.id,
    ).toBe(episode.id);
  });

  it("appends byte-preserving, versioned, audited corrections and targets an earlier correction", async () => {
    const first = tool.outputSchema.parse(await invoke()).openQuestion;
    expect(first).toMatchObject({
      resolution_note: originalNote,
      record_version: question.record_version! + 1,
      last_touched: clock.now(),
      resolution_correction_count: 1,
    });
    expect(first.resolution_corrections?.[0]).toMatchObject({
      seq: 1,
      created_at: clock.now(),
      created_at_iso: new Date(clock.now()).toISOString(),
      targets_correction_seq: null,
      writer: {
        author: "borg",
        turn_id: context.turnId,
        turn_kind: "autonomous",
        path: "tool",
        provenance: {
          kind: "online_reflector",
          process: RESOLUTION_CORRECTION_TOOL_NAME,
          evidence_episode_ids: [episode.id],
        },
      },
    });
    clock.advance(10);
    await invoke({
      correction_note: "Second correction",
      corrects_span: "First correction",
      targets_correction_seq: 1,
    });
    const stored = h.openQuestionsRepository.get(question.id)!;
    expect(stored).toMatchObject({
      resolution_note: originalNote,
      resolved_at: question.resolved_at,
      resolution_evidence_episode_ids: question.resolution_evidence_episode_ids,
      resolution_evidence_stream_entry_ids: question.resolution_evidence_stream_entry_ids,
      resolution_disclosure_label: question.resolution_disclosure_label,
      record_version: question.record_version! + 2,
    });
    expect(stored.resolution_corrections?.map((c) => c.seq)).toEqual([1, 2]);
    const events = h.identityEventRepository
      .list({ recordType: "open_question", recordId: question.id })
      .filter((e) => e.action === "correct_resolution");
    expect(events).toHaveLength(2);
    expect(events[0]).toMatchObject({
      ts: clock.now(),
      old_value: { record_version: question.record_version! + 1 },
      new_value: {
        resolution_note: originalNote,
        resolution_corrections: stored.resolution_corrections,
      },
      provenance: { kind: "online_reflector", process: RESOLUTION_CORRECTION_TOOL_NAME },
    });
    expect(
      h.openQuestionsRepository.list({ status: "resolved" })[0]?.resolution_corrections,
    ).toEqual(stored.resolution_corrections);
    const writer = stored.resolution_corrections![0]!.writer;
    const { open_question_id: _id, ...fields } = input();
    expect(() =>
      h.identityService.correctOpenQuestionResolution(
        question.id,
        { ...fields, correction_note: "Stale write" },
        writer,
        unknownMemoryDisclosureLabel(),
        {
          expectedVersion: question.record_version,
          expectedResolution: openQuestionResolutionIdentity(
            h.openQuestionsRepository.get(question.id)!,
          ),
        },
      ),
    ).toThrow(IdentityCasMismatchError);
    expect(h.openQuestionsRepository.get(question.id)).toEqual(stored);
  });

  it.each(["Different note", originalNote])(
    "rejects an in-flight resolution change through the direct tool (%#. case)",
    async (note) => {
      const getMany = h.episodicRepository.getMany.bind(h.episodicRepository);
      let replacement = question;
      let eventsAfterReplacement = h.identityEventRepository.list({ recordId: question.id });
      vi.spyOn(h.episodicRepository, "getMany").mockImplementationOnce(async (ids) => {
        h.openQuestionsRepository.reopenForReversal(question.id);
        clock.set(question.resolved_at!);
        replacement = h.openQuestionsRepository.resolve(question.id, {
          resolution_note: note,
          resolution_evidence_episode_ids: [episode.id],
          resolution_disclosure_label: unknownMemoryDisclosureLabel(),
        });
        eventsAfterReplacement = h.identityEventRepository.list({ recordId: question.id });
        return getMany(ids);
      });
      await expect(invoke()).rejects.toMatchObject({ code: "resolution_changed" });
      expect(replacement.resolved_at).toBe(question.resolved_at);
      expect(replacement.resolution_version).not.toBe(question.resolution_version);
      expect(h.openQuestionsRepository.get(question.id)).toEqual(replacement);
      expect(h.identityEventRepository.list({ recordId: question.id })).toEqual(
        eventsAfterReplacement,
      );
    },
  );

  it("allows another append to the same generation during evidence lookup", async () => {
    const getMany = h.episodicRepository.getMany.bind(h.episodicRepository);
    vi.spyOn(h.episodicRepository, "getMany").mockImplementationOnce(async (ids) => {
      await invoke({ correction_note: "Concurrent correction" });
      return getMany(ids);
    });
    await invoke();
    const stored = h.openQuestionsRepository.get(question.id)!;
    expect(stored.resolution_version).toBe(question.resolution_version);
    expect(stored.record_version).toBe(question.record_version! + 2);
    expect(stored.resolution_corrections?.map((row) => row.text)).toEqual([
      "Concurrent correction",
      input().correction_note,
    ]);
  });

  it("stamps a new generation through the resolve tool", async () => {
    h.openQuestionsRepository.reopenForReversal(question.id);
    const resolve = createOpenQuestionsResolveTool({
      identityService: h.identityService,
      episodicRepository: h.episodicRepository,
      entryIndex: h.createContext().entryIndex!,
      disclosureLabelForEvidence: async () => publicMemoryDisclosureLabel(),
    });
    const result = await resolve.invoke(
      {
        open_question_id: question.id,
        resolution_note: originalNote,
        resolution_evidence_episode_ids: [episode.id],
        resolution_evidence_stream_entry_ids: [],
      },
      context,
    );
    expect(result.openQuestion.resolution_version).toBe(result.openQuestion.record_version);
    expect(result.openQuestion.resolution_version).not.toBe(question.resolution_version);
  });

  it("rolls back the row when the identity event cannot be recorded", async () => {
    vi.spyOn(h.identityEventRepository, "record").mockImplementation(() => {
      throw new Error("audit unavailable");
    });
    await expect(invoke()).rejects.toThrow("audit unavailable");
    expect(h.openQuestionsRepository.get(question.id)).toEqual(question);
  });

  it("treats the evidence as a set for exact repeats, including at the 20 cap", async () => {
    const evidence = [
      { id: episode.id, basis: "own_read" as const },
      { id: streamId, basis: "reported" as const, reported_by: "Reader" },
    ];
    await tool.invoke({ ...input(), evidence }, context);
    const before = h.openQuestionsRepository.get(question.id);
    const auditBefore = h.identityEventRepository.list({ recordId: question.id });
    clock.advance(500);
    await tool.invoke(
      { ...input(), evidence: [...evidence].reverse().concat(evidence[0]!) },
      context,
    );
    expect(h.openQuestionsRepository.get(question.id)).toEqual(before);
    expect(h.identityEventRepository.list({ recordId: question.id })).toEqual(auditBefore);
    for (let i = 1; i < MAX_RESOLUTION_CORRECTIONS; i++)
      await invoke({ correction_note: `Correction ${i}` });
    const capped = h.openQuestionsRepository.get(question.id);
    await expect(invoke({ correction_note: "21st" })).rejects.toMatchObject({
      code: "OPEN_QUESTION_CORRECTION_LIMIT",
    });
    await tool.invoke({ ...input(), evidence }, context);
    expect(h.openQuestionsRepository.get(question.id)).toEqual(capped);
  });

  it("rejects missing questions, open/abandoned questions, missing targets and missing references", async () => {
    await expect(invoke({ open_question_id: createOpenQuestionId() })).rejects.toMatchObject({
      code: "OPEN_QUESTION_NOT_FOUND",
    });
    const open = h.openQuestionsRepository.add({
      question: "Still open?",
      urgency: 0.5,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    await expect(invoke({ open_question_id: open.id })).rejects.toMatchObject({
      code: "OPEN_QUESTION_CORRECTION_NOT_RESOLVED",
    });
    h.openQuestionsRepository.abandon(open.id, "No longer pursued");
    await expect(invoke({ open_question_id: open.id })).rejects.toMatchObject({
      code: "OPEN_QUESTION_CORRECTION_NOT_RESOLVED",
    });
    await expect(invoke({ targets_correction_seq: 1 })).rejects.toMatchObject({
      code: "OPEN_QUESTION_CORRECTION_TARGET_NOT_FOUND",
    });
    for (const id of [createEpisodeId(), createStreamEntryId()]) {
      await expect(invoke({ evidence: [{ id, basis: "own_read" }] })).rejects.toMatchObject({
        code: "OPEN_QUESTION_CITATION_NOT_FOUND",
        message: expect.stringContaining(id),
      });
    }
    const mismatch = await invoke({ corrects_span: "absent" }).catch((error: Error) => error);
    expect(mismatch).toMatchObject({
      code: "OPEN_QUESTION_CORRECTION_SPAN_NOT_FOUND",
      message: expect.stringContaining(JSON.stringify(originalNote)),
    });
    expect(mismatch).toHaveProperty("message", expect.stringContaining('"disclosure_label"'));
    await invoke();
    await expect(
      invoke({ targets_correction_seq: 1, corrects_span: "HEAD" }),
    ).rejects.toMatchObject({
      code: "OPEN_QUESTION_CORRECTION_SPAN_NOT_FOUND",
      message: expect.stringContaining(JSON.stringify(input().correction_note)),
    });
  });

  it("derives private and unknown labels from archived episodes and inactive streams, without audience gates", async () => {
    const audience = createEntityId();
    const privateEpisode = createEpisodeFixture({
      audience_entity_id: audience,
      origin_audience_entity_ids: [audience],
      shared: false,
    });
    await h.episodicRepository.createEpisode(privateEpisode);
    h.episodicRepository.archiveEpisode(privateEpisode.id, {
      caller: "test",
      reason: "archive",
      process: "curator",
    });
    h.db.prepare("UPDATE stream_entry_index SET active = 0 WHERE entry_id = ?").run(streamId);
    const privateResult = await tool.invoke(
      {
        ...input(),
        evidence: [{ id: privateEpisode.id, basis: "reported", reported_by: " Reader " }],
      },
      { ...context, audienceEntityId: createEntityId() },
    );
    expect(privateResult.openQuestion.resolution_corrections?.[0]).toMatchObject({
      disclosure_label: { disclosureClass: "relationship_private", privateToEntityIds: [audience] },
      evidence: [{ reported_by: "Reader" }],
    });
    expect(privateResult.openQuestion.disclosure_label.disclosure_class).not.toBe("public");
    const mixed = await tool.invoke(
      {
        ...input(),
        correction_note: "Mixed",
        evidence: [
          { id: episode.id, basis: "own_read" },
          { id: streamId, basis: "own_read" },
        ],
      },
      context,
    );
    expect(mixed.openQuestion.resolution_corrections?.[0]?.disclosure_label.disclosureClass).toBe(
      "unknown",
    );
    expect(mixed.openQuestion.disclosure_label.disclosure_class).toBe("unknown");
  });

  it("offers the direct tool alongside autonomous resolve and leaves conversation writes to reflection", () => {
    const dispatcher = new ToolDispatcher({
      createStreamWriter: () => h.streamWriter,
      clock,
    }).register(tool);
    expect(
      resolveFinalizerNonTerminalTools({ dispatcher, turnOrigin: "autonomous" }).map((t) => t.name),
    ).toContain(RESOLUTION_CORRECTION_TOOL_NAME);
    for (const turnOrigin of ["user", "directed_outbound"] as const)
      expect(
        resolveFinalizerNonTerminalTools({ dispatcher, turnOrigin }).map((t) => t.name),
      ).not.toContain(RESOLUTION_CORRECTION_TOOL_NAME);
  });

  it("keeps both corrections outside long note excerpts on pages, recall and API/tool outputs", async () => {
    await invoke();
    const result = await invoke({
      correction_note: "Second correction",
      targets_correction_seq: 1,
      corrects_span: "First correction",
    });
    const stored = h.openQuestionsRepository.get(question.id)!;
    const metadata = formatResolutionCorrections({
      resolution_corrections: stored.resolution_corrections,
    });
    expect(metadata.resolution_corrections?.map((c) => c.seq)).toEqual([2, 1]);
    const excerpt = headTailTextExcerpt(originalNote, 80);
    const page = {
      resolution_note_excerpt: { head: excerpt.head, tail: excerpt.tail },
      ...metadata,
    };
    expect(excerpt.exact).toBe(false);
    expect(page.resolution_correction_count).toBe(2);
    expect(
      formatResolutionCorrections(
        { resolution_corrections: stored.resolution_corrections },
        { maxChars: 0 },
      ),
    ).toEqual({
      resolution_correction_count: 2,
      resolution_corrections: [],
      resolution_corrections_omitted: { count: 2, total: 2, reason: "budget" },
    });
    expect(
      formatResolutionCorrections({ resolution_note: originalNote, resolution_corrections: [] }),
    ).toEqual({ resolution_note: originalNote });
    expect(openQuestionStateMetadata(stored, clock.now())).toMatchObject({
      resolution_note: originalNote,
      ...metadata,
    });
    const source = {
      evidence: [],
      episodes: [],
      semantic: {
        matched_node_ids: [],
        matched_nodes: [],
        supports: [],
        contradicts: [],
        categories: [],
        support_hits: [],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      },
      open_questions: [stored],
    };
    const retrieval = renderPlanRequestedVerificationRetrieval(source, 100_000, 10_000);
    expect(retrieval).toContain('resolution_correction_count="2"');
    expect(retrieval).toContain("Second correction");
    expect(retrieval).toContain("First correction");
    const omitted = renderPlanRequestedVerificationRetrieval(source, 0, 10_000);
    expect(omitted).toContain('resolution_correction_count="2"');
    expect(omitted).toContain("resolution_corrections_omitted=");
    expect(omitted).toContain('payload_status="check_not_completed_budget"');
    expect(tool.outputSchema.parse(result).openQuestion).toMatchObject(metadata);
    const create = createOpenQuestionsCreateTool({ createOpenQuestion: () => stored });
    expect(
      create.outputSchema.parse(await create.invoke({ question: stored.question }, context))
        .openQuestion,
    ).toMatchObject(metadata);
    const resolve = createOpenQuestionsResolveTool({
      identityService: {
        resolveOpenQuestion: () => ({ status: "requires_review", current: stored }),
      },
      episodicRepository: h.episodicRepository,
      entryIndex: h.createContext().entryIndex!,
      disclosureLabelForEvidence: async () => unknownMemoryDisclosureLabel(),
    });
    expect(
      resolve.outputSchema.parse(
        await resolve.invoke(
          {
            open_question_id: stored.id,
            resolution_note: "Resolution",
            resolution_evidence_episode_ids: [episode.id],
          },
          context,
        ),
      ).openQuestion,
    ).toMatchObject(metadata);
    const facade = createBorgFacades({
      ...h,
      actionRepository: h.actionRepository,
    } as unknown as BorgDependencies);
    expect(facade.self.openQuestions.list({ status: "resolved" })[0]).toMatchObject({
      resolution_note: originalNote,
      ...metadata,
    });
    const why = await new CorrectionService(h).why(stored.id);
    expect(why.record).toMatchObject(metadata);
    const event = h.identityEventRepository.list({ recordId: stored.id })[0]!;
    expect(facade.identity.listEvents({ recordId: stored.id })[0]?.new_value).toMatchObject(
      metadata,
    );
    const history = createIdentityEventsListForCognitionTool({
      listEvents: () => [{ ...event, old_value: null }],
    });
    const row = history.outputSchema.parse(await history.invoke({}, context)).events[0]!;
    expect(row.change.excerpt_exact).toBe(false);
    expect(row.resolution_correction_snapshots?.new).toMatchObject(metadata);
    const wake = formatAutonomyTriggerContext({
      source_name: "scheduled_reflection",
      source_type: "trigger",
      event_id: "test-wake",
      sort_ts: clock.now(),
      payload: { recent_identity_events: [row] },
    });
    expect(wake).toContain('"resolution_correction_count": 2');
    expect(wake).toContain("First correction");
    expect(wake).toContain("Second correction");
    const boundedWake = formatAutonomyTriggerContext(
      {
        source_name: "scheduled_reflection",
        source_type: "trigger",
        event_id: "test-wake",
        sort_ts: clock.now(),
        payload: { recent_identity_events: [row] },
      },
      4_000,
    );
    expect(boundedWake.length).toBeLessThanOrEqual(4_000);
    expect(boundedWake).toContain('"resolution_correction_count": 2');
    expect(boundedWake).toContain('"resolution_corrections_omitted"');
    expect(boundedWake).toContain('"total": 2');
  });

  it("defers ledger correction omissions to compaction while identity pages retain their own allowance", async () => {
    await invoke({ correction_note: "a".repeat(2_000) });
    await invoke({ correction_note: "b".repeat(2_000) });
    const stored = h.openQuestionsRepository.get(question.id)!;
    const metadata = openQuestionStateMetadata(stored, clock.now());
    expect(metadata).toMatchObject({
      resolution_correction_count: 2,
      resolution_corrections: [{ seq: 2 }, { seq: 1 }],
    });
    expect(metadata).not.toHaveProperty("resolution_corrections_omitted");
    const history = createIdentityEventsListForCognitionTool({
      listEvents: () => h.identityEventRepository.list({ recordId: stored.id }),
    });
    const snapshot = (await history.invoke({}, context)).events[0]!.resolution_correction_snapshots!
      .new;
    expect(snapshot).toMatchObject({
      resolution_correction_count: 2,
      resolution_corrections_omitted: { count: 1, total: 2 },
    });
    expect(formatResolutionCorrections(snapshot, { maxChars: 0 })).toMatchObject({
      resolution_correction_count: 2,
      resolution_corrections_omitted: { count: 2, total: 2 },
    });
  });
  it.each(["zero", "two", "omitted"])(
    "renders final note surfaces with %s corrections",
    async (kind) => {
      if (kind !== "zero") {
        await invoke({
          correction_note: kind === "omitted" ? "a".repeat(2_000) : "First correction",
        });
        await invoke({
          correction_note: kind === "omitted" ? "b".repeat(2_000) : "Second correction",
        });
      }
      const stored = h.openQuestionsRepository.get(question.id)!;
      const event = h.identityEventRepository.record({
        record_type: "open_question",
        record_id: stored.id,
        action: "presentation_test",
        old_value: null,
        new_value: stored,
        provenance: { kind: "manual" },
      });
      const history = createIdentityEventsListForCognitionTool({ listEvents: () => [event] });
      const historyRow = history.outputSchema.parse(await history.invoke({}, context)).events[0]!;
      expect(historyRow.change.excerpt_exact).toBe(false);
      const wake = formatAutonomyTriggerContext({
        source_name: "scheduled_reflection",
        source_type: "trigger",
        event_id: "presentation",
        sort_ts: clock.now(),
        payload: { recent_identity_events: [historyRow] },
      });
      const builder = new EvidenceLedgerBuilder({
        createStreamReader: (sessionId) => new StreamReader({ dataDir: h.tempDir, sessionId }),
        relationalSlotRepository: { list: () => [] },
        actionRepository: { list: () => [] },
        currentSessionTranscriptTokenBudget: 50_000,
        openQuestionsRepository: {
          findByHandles: () => [stored],
          countRuminations: () => 0,
          countByStatus: () => ({ open: 0, resolved: 1, abandoned: 0 }),
        },
      });
      const ledger = await builder.build({
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: null,
        currentUserMessage: stored.id,
        workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, clock.now()),
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        retrievedSemantic: null,
        openQuestions: [],
        pendingCorrections: [],
        frameAnomaly: null,
      });
      // This deliberately long note needs a larger section; corrections share its remaining space.
      const compacted = compactEvidenceLedger(ledger, {
        sectionOptions: { open_questions: { maxTokens: 5_000 } },
      });
      expect(compacted.traceSummary.budgetOutcome).toEqual({ status: "within_budget" });
      expect(
        estimatePromptTokens(
          renderSection(
            compacted.ledger.sections.find((section) => section.id === "open_questions")!,
          ),
        ),
      ).toBeLessThanOrEqual(5_000);
      const renderedLedger = renderEvidenceLedger(compacted.ledger)!;
      const facade = createBorgFacades(h as unknown as BorgDependencies);
      const apiRecord = facade.self.openQuestions.list({ status: "resolved" })[0]!;
      const why = await new CorrectionService(h).why(stored.id);
      const create = createOpenQuestionsCreateTool({ createOpenQuestion: () => stored });
      const created = create.outputSchema.parse(
        await create.invoke({ question: stored.question }, context),
      ).openQuestion;
      const resolve = createOpenQuestionsResolveTool({
        identityService: {
          resolveOpenQuestion: () => ({ status: "requires_review", current: stored }),
        },
        episodicRepository: h.episodicRepository,
        entryIndex: h.createContext().entryIndex!,
        disclosureLabelForEvidence: async () => unknownMemoryDisclosureLabel(),
      });
      const returned = resolve.outputSchema.parse(
        await resolve.invoke(
          {
            open_question_id: stored.id,
            resolution_note: "Resolution",
            resolution_evidence_episode_ids: [episode.id],
          },
          context,
        ),
      ).openQuestion;
      const spanError = (await invoke({
        corrects_span: "missing quotation",
        correction_note: "retry",
      }).catch((error: Error) => error)) as Error;
      const target = JSON.parse(
        spanError.message.slice(spanError.message.indexOf("Target: ") + "Target: ".length),
      );
      expect(target.text).toBe(originalNote);
      const fullRecords = [
        apiRecord,
        why.record,
        created,
        returned,
        target,
        facade.identity.listEvents({ recordId: stored.id })[0]!.new_value,
      ];
      if (kind === "zero") {
        const legacy = { ...stored };
        delete legacy.resolution_corrections;
        expect(JSON.stringify(apiRecord)).toBe(JSON.stringify(legacy));
        const legacyEvent = { ...event, new_value: legacy };
        const legacyHistory = createIdentityEventsListForCognitionTool({
          listEvents: () => [legacyEvent],
        });
        const legacyRow = (await legacyHistory.invoke({}, context)).events[0]!;
        expect(JSON.stringify(historyRow)).toBe(JSON.stringify(legacyRow));
        expect(wake).toBe(
          formatAutonomyTriggerContext({
            source_name: "scheduled_reflection",
            source_type: "trigger",
            event_id: "presentation",
            sort_ts: clock.now(),
            payload: { recent_identity_events: [legacyRow] },
          }),
        );
        expect(JSON.stringify(created)).toBe(
          JSON.stringify(
            create.outputSchema.parse({
              openQuestion: {
                ...legacy,
                ...memoryDisclosurePayloadFields(openQuestionMemoryDisclosureLabel(legacy)),
              },
            }).openQuestion,
          ),
        );
        for (const output of [
          ...fullRecords.map((value) => JSON.stringify(value)),
          renderedLedger,
          wake,
        ])
          expect(output).not.toContain("resolution_correction");
      } else {
        for (const record of fullRecords) {
          expect(record).toMatchObject({
            resolution_correction_count: 2,
            resolution_corrections: [{ seq: 2 }, { seq: 1 }],
          });
        }
        expect(renderedLedger).toContain('"resolution_correction_count":2');
        expect(historyRow.resolution_correction_snapshots?.new).toMatchObject({
          resolution_correction_count: 2,
        });
        expect(wake).toContain('"resolution_correction_count": 2');
        if (kind === "omitted") {
          expect(renderedLedger).toContain(
            '"resolution_corrections_omitted":{"count":1,"total":2,"reason":"budget"}',
          );
          expect(historyRow.resolution_correction_snapshots?.new).toMatchObject({
            resolution_corrections_omitted: { count: 1, total: 2, reason: "budget" },
          });
          expect(wake).toContain('"resolution_corrections_omitted"');
        } else {
          for (const output of [renderedLedger, JSON.stringify(historyRow), wake]) {
            expect(output).toContain("First correction");
            expect(output).toContain("Second correction");
          }
        }
      }
    },
  );
});
