import { canonicalizeOpenQuestionWithSharedStateEntry } from "../lifecycle-ops/canonicalize.js";
import { resolveOpenQuestionsForCompletedAction } from "../actions/open-question-resolution.js";
import { makeActionRecord } from "../../test-support/factories/memory.js";
import { makeSharedStateEntry } from "../../test-support/factories/shared-state.js";
import {
  isSameOpenQuestionResolution,
  openQuestionResolutionIdentity,
} from "./resolution-corrections.js";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TurnLifecycleTracker } from "../../cognition/lifecycle/turn-lifecycle-tracker.js";
import { createWorkingMemory } from "../working/index.js";
import { AssociatorProcess } from "../../offline/associator/index.js";
import { RuminatorProcess, ruminatorPlanSchema } from "../../offline/ruminator/index.js";
import { createEpisodeFixture, createOfflineTestHarness } from "../../offline/test-support.js";
import { ManualClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { IdentityCasMismatchError } from "../../util/errors.js";
import { publicMemoryDisclosureLabel } from "../common/disclosure-label.js";
import { correctOpenQuestionResolutionThroughIdentityService } from "../lifecycle-ops/correct-resolution.js";
import { OpenQuestionsRepository } from "./open-questions.js";
import type { OpenQuestion } from "./types.js";

describe("corrections stay with their resolution through reversals", () => {
  let h: Awaited<ReturnType<typeof createOfflineTestHarness>>;
  let open: OpenQuestion;
  let resolved: OpenQuestion;
  let episode: ReturnType<typeof createEpisodeFixture>;
  let ruminator: RuminatorProcess;
  const clock = new ManualClock(10_000);
  const correct = (text = "Corrected reading", id = open.id) =>
    correctOpenQuestionResolutionThroughIdentityService({
      input: {
        open_question_id: id,
        correction_note: text,
        evidence: [{ id: episode.id, basis: "own_read" }],
      },
      identityService: h.identityService,
      openQuestionsRepository: h.openQuestionsRepository,
      episodicRepository: h.episodicRepository,
      entryIndex: h.createContext().entryIndex!,
      turnId: "correction-turn",
      turnOrigin: "autonomous",
      path: "tool",
    });
  beforeEach(async () => {
    h = await createOfflineTestHarness({ clock });
    episode = createEpisodeFixture({ shared: true });
    await h.episodicRepository.createEpisode(episode);
    open = h.openQuestionsRepository.add({
      question: "Which reading?",
      urgency: 0.5,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    resolved = h.openQuestionsRepository.resolve(open.id, {
      resolution_note: "Original reading",
      resolution_evidence_episode_ids: [episode.id],
    });
    new AssociatorProcess({ ...h, clock });
    ruminator = new RuminatorProcess(h);
  });
  afterEach(async () => {
    vi.restoreAllMocks();
    await h?.cleanup();
  });

  async function reverse(path: string, snapshot: OpenQuestion) {
    if (path === "turn") {
      const working = createWorkingMemory(DEFAULT_SESSION_ID, clock.now());
      const tracker = new TurnLifecycleTracker({
        workingMemoryStore: {
          load: () => working,
          save: (value) => value,
          recordPendingActionMerges: vi.fn(),
        },
        actionRepository: { delete: vi.fn() },
        executiveStepsRepository: { delete: vi.fn(), restore: vi.fn() },
        goalsRepository: { remove: vi.fn(), restore: vi.fn() },
        openQuestionsRepository: h.openQuestionsRepository,
        episodicRepository: h.episodicRepository,
        relationalSlotRepository: { restore: vi.fn() },
      });
      tracker.trackReflectionEffects({
        createdActionIds: [],
        createdExecutiveStepIds: [],
        createdOpenQuestionIds: [],
        updatedExecutiveSteps: [],
        updatedGoals: [],
        retiredGoalIds: [],
        resolvedOpenQuestions: [snapshot],
        updatedEpisodeStats: [],
      });
      expect(
        await tracker.cleanupAbortedTurnState({
          turnId: "correction-turn",
          sessionId: DEFAULT_SESSION_ID,
        }),
      ).toEqual([]);
    } else {
      const associator = path === "associator";
      const audit = h.auditLog.record({
        run_id: h.createContext().runId,
        process: associator ? "associator" : "ruminator",
        action: associator ? "open_question" : "merge_duplicate",
        targets: { question_id: snapshot.id },
        reversal: associator
          ? { mode: "reinforced", previous: snapshot }
          : {
              [path === "ruminator_primary" ? "previous_primary" : "previous_duplicate"]: snapshot,
            },
      });
      expect((await h.auditLog.revert(audit.id))?.reverted_at).not.toBeNull();
    }
  }

  describe.each(["turn", "associator", "ruminator_primary", "ruminator_duplicate"])(
    "%s restore",
    (path) => {
      it.each(["legacy", "stale"])(
        "preserves committed corrections from a %s snapshot of the same note",
        async (kind) => {
          if (kind === "legacy") {
            h.db
              .prepare("UPDATE open_questions SET resolution_version = NULL WHERE id = ?")
              .run(open.id);
          }
          const snapshot = { ...h.openQuestionsRepository.get(open.id)! };
          if (kind === "legacy") {
            delete snapshot.resolution_corrections;
            delete snapshot.resolution_version;
          }
          h.openQuestionsRepository.update(open.id, {
            resolution_disclosure_label: publicMemoryDisclosureLabel(),
            resolution_evidence_episode_ids: [],
            resolution_evidence_stream_entry_ids: [
              (await h.streamWriter.append({ kind: "thought", content: "Evidence" })).id,
            ],
          });
          await correct();
          const before = h.openQuestionsRepository.get(open.id)!;
          clock.advance(10);
          await reverse(path, snapshot);
          const after = h.openQuestionsRepository.get(open.id)!;
          expect(openQuestionResolutionIdentity(after)).toEqual(
            openQuestionResolutionIdentity(before),
          );
          expect(after.resolution_evidence_episode_ids).toEqual(
            before.resolution_evidence_episode_ids,
          );
          expect(after.resolution_evidence_stream_entry_ids).toEqual(
            before.resolution_evidence_stream_entry_ids,
          );
          expect(after.resolution_disclosure_label).toEqual(before.resolution_disclosure_label);
          expect(after.resolution_corrections).toEqual(before.resolution_corrections);
          expect(after.record_version).toBe(before.record_version! + 1);
          expect(h.identityEventRepository.list({ recordId: open.id })[0]).toMatchObject({
            action: "restore",
            old_value: before,
            new_value: after,
          });
        },
      );
      it.each(["open", "different_note", "legacy_generation"])(
        "archives the full departing resolution when restoring %s",
        async (kind) => {
          await correct();
          const before = h.openQuestionsRepository.get(open.id)!;
          const snapshot =
            kind === "open"
              ? open
              : kind === "different_note"
                ? { ...resolved, resolution_note: "Earlier resolution" }
                : { ...resolved };
          if (kind === "legacy_generation") delete snapshot.resolution_version;
          await reverse(path, snapshot);
          const after = h.openQuestionsRepository.get(open.id)!;
          expect(after).toEqual({
            ...snapshot,
            resolution_version: snapshot.resolution_version ?? null,
            resolution_corrections: [],
            record_version: before.record_version! + 1,
            last_touched: clock.now(),
          });
          expect(h.identityEventRepository.list({ recordId: open.id })[0]).toMatchObject({
            action: "restore",
            old_value: before,
            new_value: after,
          });
        },
      );
    },
  );

  it.each(["resolve", "abandon"])(
    "reopen followed by %s cannot retarget old corrections",
    async (next) => {
      await correct();
      const before = h.openQuestionsRepository.get(open.id)!;
      const audit = h.auditLog.record({
        run_id: h.createContext().runId,
        process: "ruminator",
        action: "resolve",
        targets: { question_id: open.id },
        reversal: { previous: open },
      });
      await h.auditLog.revert(audit.id);
      const reopened = h.openQuestionsRepository.get(open.id)!;
      expect(reopened).toMatchObject({
        status: "open",
        resolution_version: null,
        resolution_note: null,
        resolution_corrections: [],
      });
      expect(h.identityEventRepository.list({ recordId: open.id })[0]).toMatchObject({
        action: "reopen",
        old_value: before,
        new_value: reopened,
      });
      if (next === "resolve") {
        h.openQuestionsRepository.resolve(open.id, {
          resolution_note: "New resolution",
          resolution_evidence_episode_ids: [episode.id],
        });
        const corrected = await correct("New correction");
        expect(corrected.resolution_corrections?.map((item) => [item.seq, item.text])).toEqual([
          [1, "New correction"],
        ]);
      } else {
        expect(h.openQuestionsRepository.abandon(open.id, "Closed without answer")).toMatchObject({
          status: "abandoned",
          resolution_note: null,
          resolution_corrections: [],
        });
      }
    },
  );

  it("rejects a correction after a concurrent reopen, and a stale version after re-resolution", async () => {
    const getMany = h.episodicRepository.getMany.bind(h.episodicRepository);
    vi.spyOn(h.episodicRepository, "getMany").mockImplementationOnce(async (ids) => {
      h.openQuestionsRepository.reopenForReversal(open.id);
      return getMany(ids);
    });
    await expect(correct()).rejects.toMatchObject({
      code: "resolution_changed",
    });
    h.openQuestionsRepository.resolve(open.id, {
      resolution_note: "New resolution",
      resolution_evidence_episode_ids: [episode.id],
    });
    expect(() =>
      h.identityService.correctOpenQuestionResolution(
        open.id,
        { correction_note: "Stale", evidence: [{ id: episode.id, basis: "own_read" }] },
        {
          author: "borg",
          turn_id: "stale",
          turn_kind: "autonomous",
          path: "tool",
          provenance: { kind: "manual" },
        },
        publicMemoryDisclosureLabel(),
        {
          expectedVersion: resolved.record_version,
          expectedResolution: openQuestionResolutionIdentity(
            h.openQuestionsRepository.get(open.id)!,
          ),
        },
      ),
    ).toThrow(IdentityCasMismatchError);
    expect(h.openQuestionsRepository.get(open.id)?.resolution_corrections).toEqual([]);
  });

  it("reuses the uncommitted sequence after audit failure and atomically refuses unaudited departures", async () => {
    const eventsBefore = h.identityEventRepository.list({ recordId: open.id });
    const realRecord = h.identityEventRepository.record.bind(h.identityEventRepository);
    const record = vi.spyOn(h.identityEventRepository, "record").mockImplementationOnce((event) => {
      realRecord(event);
      throw new Error("audit unavailable");
    });
    await expect(correct()).rejects.toThrow("audit unavailable");
    expect(h.openQuestionsRepository.get(open.id)).toEqual(resolved);
    expect(h.identityEventRepository.list({ recordId: open.id })).toEqual(eventsBefore);
    const corrected = await correct();
    expect(corrected.resolution_corrections?.[0]?.seq).toBe(1);
    record.mockImplementationOnce((event) => {
      realRecord(event);
      throw new Error("restore audit unavailable");
    });
    expect(() => h.openQuestionsRepository.restore(open)).toThrow("restore audit unavailable");
    expect(h.openQuestionsRepository.get(open.id)).toEqual(corrected);
    record.mockImplementationOnce((event) => {
      realRecord(event);
      throw new Error("reopen audit unavailable");
    });
    expect(() => h.openQuestionsRepository.reopenForReversal(open.id)).toThrow(
      "reopen audit unavailable",
    );
    expect(h.openQuestionsRepository.get(open.id)).toEqual(corrected);
    const unaudited = new OpenQuestionsRepository({ db: h.db, clock });
    for (const change of [
      () => unaudited.restore(open),
      () => unaudited.reopenForReversal(open.id),
    ]) {
      expect(change).toThrow(
        expect.objectContaining({ code: "OPEN_QUESTION_CORRECTION_AUDIT_REQUIRED" }),
      );
      expect(h.openQuestionsRepository.get(open.id)).toEqual(corrected);
    }
  });

  it("restores A1 after B and A2 with identical notes and clocks without mixing lifecycle metadata", async () => {
    const a1 = await correct("A1 correction");
    for (const note of ["B", a1.resolution_note!]) {
      h.openQuestionsRepository.reopenForReversal(open.id);
      clock.set(a1.resolved_at!);
      h.openQuestionsRepository.resolve(open.id, {
        resolution_note: note,
        resolution_evidence_episode_ids: [episode.id],
        resolution_disclosure_label: publicMemoryDisclosureLabel(),
      });
      await correct(`${note} correction`);
    }
    const a2 = h.openQuestionsRepository.get(open.id)!;
    expect(a2.resolution_note).toBe(a1.resolution_note);
    expect(a2.resolved_at).toBe(a1.resolved_at);
    expect(a2.resolution_version).not.toBe(a1.resolution_version);
    expect(isSameOpenQuestionResolution(a2, openQuestionResolutionIdentity(a1))).toBe(false);
    const after = h.openQuestionsRepository.restore(a1);
    expect(after).toEqual({
      ...a1,
      record_version: a2.record_version! + 1,
      last_touched: clock.now(),
    });
    expect(h.openQuestionsRepository.get(open.id)).toEqual(after);
    expect(h.identityEventRepository.list({ recordId: open.id })[0]).toMatchObject({
      action: "restore",
      old_value: a2,
      new_value: after,
    });
  });

  it.each([
    [" padded ", "padded", false],
    ["line\nend", "line end", false],
    [null, "", false],
    ["", null, false],
    [null, null, true],
    ["", "", true],
    [" padded ", " padded ", true],
  ])("compares legacy note bytes %j and %j exactly", async (note, restoredNote, same) => {
    h.db
      .prepare(
        "UPDATE open_questions SET resolution_version = NULL, resolution_note = ? WHERE id = ?",
      )
      .run(note, open.id);
    const snapshot = { ...h.openQuestionsRepository.get(open.id)!, resolution_note: restoredNote };
    delete snapshot.resolution_version;
    const current = await correct();
    const restored = h.openQuestionsRepository.restore(snapshot);
    expect(restored.resolution_version).toBeNull();
    expect(restored.resolution_note).toBe(restoredNote);
    expect(restored.resolution_corrections).toEqual(same ? current.resolution_corrections : []);
    expect(h.identityEventRepository.list({ recordId: open.id })[0]?.old_value).toEqual(current);
  });

  it("stamps fresh resolutions, preserves the generation on ordinary updates/corrections and clears on reopen", async () => {
    expect(resolved.resolution_version).toBe(resolved.record_version);
    const corrected = await correct();
    const updated = h.openQuestionsRepository.update(open.id, { urgency: 0.8 });
    expect(updated.resolution_version).toBe(resolved.resolution_version);
    expect(updated.resolution_corrections).toEqual(corrected.resolution_corrections);
    const patched = h.identityService.updateOpenQuestion(
      open.id,
      { resolution_note: "Repaired note" },
      { kind: "manual" },
      { throughReview: true },
    );
    expect(patched.status).toBe("applied");
    const repaired = h.openQuestionsRepository.get(open.id)!;
    expect(repaired.resolution_version).toBe(repaired.record_version);
    expect(repaired.resolution_corrections).toEqual([]);
    expect(
      h.identityEventRepository
        .list({ recordId: open.id })
        .some((event) => JSON.stringify(event.old_value) === JSON.stringify(updated)),
    ).toBe(true);
    const reopened = h.openQuestionsRepository.update(open.id, { status: "open" });
    expect(reopened).toMatchObject({
      resolution_version: null,
      resolution_note: null,
      resolved_at: null,
    });
    const resolvedByPatch = h.openQuestionsRepository.update(open.id, {
      status: "resolved",
      resolution_note: "Patch resolution",
      resolved_at: clock.now(),
      resolution_evidence_episode_ids: [episode.id],
    });
    expect(resolvedByPatch.resolution_version).toBe(resolvedByPatch.record_version);
  });

  it.each(["canonicalization", "completed_action"])("stamps the generation via %s", (path) => {
    h.openQuestionsRepository.reopenForReversal(open.id);
    if (path === "canonicalization") {
      expect(
        canonicalizeOpenQuestionWithSharedStateEntry({
          openQuestionId: open.id,
          repository: h.openQuestionsRepository,
          entry: makeSharedStateEntry(),
        }).status,
      ).toBe("success");
    } else {
      expect(
        resolveOpenQuestionsForCompletedAction({
          action: makeActionRecord({
            state: "completed",
            open_question_id: open.id,
            provenance_episode_ids: [episode.id],
          }),
          openQuestionsRepository: h.openQuestionsRepository,
          identityService: h.identityService,
        }),
      ).toHaveLength(1);
    }
    const current = h.openQuestionsRepository.get(open.id)!;
    expect(current.status).toBe("resolved");
    expect(current.resolution_version).toBe(current.record_version);
  });

  it("rolls back both restored rows and audit inserts when the second row fails after its insert", async () => {
    const second = h.openQuestionsRepository.add({
      question: "Second?",
      urgency: 0.4,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    h.openQuestionsRepository.resolve(second.id, {
      resolution_note: "Second note",
      resolution_evidence_episode_ids: [episode.id],
    });
    await correct();
    await correct("Second correction", second.id);
    const before = [
      h.openQuestionsRepository.get(open.id),
      h.openQuestionsRepository.get(second.id),
    ];
    const events = h.identityEventRepository.list();
    const audit = h.auditLog.record({
      run_id: h.createContext().runId,
      process: "ruminator",
      action: "merge_duplicate",
      targets: { question_id: open.id },
      reversal: { previous_primary: open, previous_duplicate: second },
    });
    const record = h.identityEventRepository.record.bind(h.identityEventRepository);
    vi.spyOn(h.identityEventRepository, "record").mockImplementation((event) => {
      const result = record(event);
      if (event.record_id === second.id) throw new Error("after second insert");
      return result;
    });
    await expect(h.auditLog.revert(audit.id)).rejects.toThrow("after second insert");
    expect([
      h.openQuestionsRepository.get(open.id),
      h.openQuestionsRepository.get(second.id),
    ]).toEqual(before);
    expect(h.identityEventRepository.list()).toEqual(events);
    expect(h.auditLog.get(audit.id)?.reverted_at).toBeNull();
  });

  it("renders the actual corrected record on ruminator replay separately from the historical plan", async () => {
    const longNote = `HEAD ${"middle ".repeat(2_000)} TAIL`;
    h.openQuestionsRepository.reopenForReversal(open.id);
    const plan = ruminatorPlanSchema.parse({
      process: "ruminator",
      tokens_used: 0,
      items: [
        {
          action: "resolve",
          question_id: open.id,
          previous: open,
          resolution_evidence_episode_ids: [episode.id],
          resolution_note: longNote,
          resolution_disclosure_label: publicMemoryDisclosureLabel(),
          growth_marker: null,
        },
      ],
    });
    const initial = await ruminator.apply(h.createContext(), plan);
    const generation = h.openQuestionsRepository.get(open.id)!;
    expect(generation.resolution_version).toBe(generation.record_version);
    const initialText = JSON.stringify(initial.changes[0]?.preview);
    expect(initialText).not.toContain("resolution_correction");
    await correct("First correction");
    await correct("Second correction");
    const result = await ruminator.apply(h.createContext(), plan);
    expect(h.openQuestionsRepository.get(open.id)?.resolution_version).toBe(
      generation.resolution_version,
    );
    expect(result.changes[0]?.preview).toMatchObject({
      state: "current_record",
      question: {
        resolution_note: longNote,
        resolution_correction_count: 2,
        resolution_corrections: [
          { seq: 2, text: "Second correction" },
          { seq: 1, text: "First correction" },
        ],
      },
    });
    const preview = ruminator.preview(plan).changes[0]?.preview;
    expect(preview).toMatchObject({ state: "historical_plan", note: longNote });
    expect(JSON.stringify(preview)).not.toContain("resolution_correction");
  });
});
