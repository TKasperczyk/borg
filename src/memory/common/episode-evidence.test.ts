import { afterEach, describe, expect, it } from "vitest";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  type OfflineTestHarness,
} from "../../offline/test-support.js";
import {
  createRetiredEvidenceFixture,
  recordTestAnnotation,
} from "../../offline/evidence-test-support.js";
import { createEntityId, parseEpisodeId } from "../../util/ids.js";
import { RecordCorrectionService } from "../lifecycle-ops/correct-record.js";
import {
  annotationPromptFields,
  renderEvidencePromptJson,
  renderEvidencePromptLines,
  EVIDENCE_GUIDANCE,
  MAX_EVIDENCE_AUTHORITY_CHARS,
  MAX_EVIDENCE_PROMPT_CONTEXT_CHARS,
  MAX_EVIDENCE_RECORD_CONTEXT_CHARS,
  MAX_EVIDENCE_LINEAGE_EXCERPT_CHARS,
  episodeEvidencePromptRow,
  withEpisodeEvidenceContext,
  MAX_EVIDENCE_ANNOTATIONS_PER_RECORD,
  MAX_EVIDENCE_ANNOTATION_CHARS,
  MAX_EVIDENCE_LINEAGE_EPISODES,
  MEMORY_EVIDENCE_GUIDANCE,
  SELF_CORRECTION_GUIDANCE,
} from "./episode-evidence.js";

describe("episode evidence envelope", () => {
  let harness: OfflineTestHarness | undefined;
  afterEach(async () => {
    await harness?.cleanup();
    harness = undefined;
  });

  it("bounds annotations, reports omissions, and preserves multilingual text and UTF-16 boundaries", () => {
    const text = "Zażółć. 中文。 العربية. 🧭 e\u0301\n";
    const annotations = Array.from({ length: 5 }, (_, index) => ({
      stream_entry_id: `synthetic-${index}`,
      text,
      written_at: 10 - index,
      authority: null,
    }));
    annotations[1]!.text = "x".repeat(1198) + "🧭" + "y".repeat(100);
    annotations[2]!.text = "z".repeat(1200);
    const fields = annotationPromptFields(annotations);
    expect(fields.annotations).toHaveLength(MAX_EVIDENCE_ANNOTATIONS_PER_RECORD);
    expect(fields.annotations_omitted).toBe(2);
    expect(Buffer.from(fields.annotations![0]!.text)).toEqual(Buffer.from(text));
    expect(fields.annotations![1]!.text).toBe("x".repeat(1198) + "…");
    expect(fields.annotations![2]!.text).toBe(annotations[2]!.text);
    expect(
      fields.annotations!.every(
        (annotation) => annotation.text.length <= MAX_EVIDENCE_ANNOTATION_CHARS,
      ),
    ).toBe(true);
    expect(fields.annotations!.map((annotation) => annotation.written_at)).toEqual(
      annotations.slice(0, 3).map((annotation) => new Date(annotation.written_at).toISOString()),
    );
    expect(annotationPromptFields([])).toEqual({});
  });

  it("keeps the retired citation, adds only visible bounded lineage, and never inherits annotations", async () => {
    harness = await createOfflineTestHarness();
    const { source, lineage } = await createRetiredEvidenceFixture(harness);
    const extra = [];
    for (let i = 0; i < 5; i++)
      extra.push(
        await harness.episodicRepository.createEpisode(
          createEpisodeFixture({ title: `Synthetic raw ${i}` }),
        ),
      );
    const archivedRaw = extra[0]!;
    harness.db
      .prepare("UPDATE episode_stats SET archived = 1 WHERE episode_id = ?")
      .run(archivedRaw.id);
    const sourceWithLineage = {
      ...source,
      lineage: {
        ...source.lineage,
        derived_from: [lineage.id, ...extra.map((episode) => episode.id)],
      },
    };
    const decorated = await withEpisodeEvidenceContext(harness, [sourceWithLineage], {
      includeVisibleLineage: true,
    });
    const row = JSON.parse(renderEvidencePromptJson(episodeEvidencePromptRow(decorated[0]!)));
    expect(row).toMatchObject({
      id: source.id,
      narrative: source.narrative,
      lifecycle: "archived",
      lineage_omitted: 1,
    });
    const lineageRows = row.visible_lineage as Record<string, unknown>[];
    expect(lineageRows).toHaveLength(MAX_EVIDENCE_LINEAGE_EPISODES);
    expect(lineageRows.map((entry) => entry.id)).toEqual([
      lineage.id,
      ...extra.slice(1, 4).map((episode) => episode.id),
    ]);
    expect(lineageRows[0]).toMatchObject({
      excerpt: lineage.narrative,
      annotations: [{ text: "原始记录的独立更正。 تصحيح مستقل." }],
    });
    expect(row.annotations).toEqual([
      expect.objectContaining({
        text: "Synthetic audited correction: the observation was provisional.",
      }),
    ]);
    expect(sourceWithLineage).not.toHaveProperty("evidence_context");
    expect((await harness.episodicRepository.getMany([source.id]))[0]!.narrative).toBe(
      source.narrative,
    );
    expect(
      harness.episodicRepository.getLifecycleMany([
        source.id,
        archivedRaw.id,
        lineage.id,
        parseEpisodeId("ep_syntheticmissing"),
      ]),
    ).toEqual(
      new Map([
        [source.id, "archived"],
        [archivedRaw.id, "archived"],
        [lineage.id, "active"],
        [parseEpisodeId("ep_syntheticmissing"), "unknown"],
      ]),
    );
  });

  it("leaves ordinary lifecycle absent and deferred renderers unchanged", async () => {
    harness = await createOfflineTestHarness();
    const episode = await harness.episodicRepository.createEpisode(
      createEpisodeFixture({ narrative: "Unchanged original.\n原文。" }),
    );
    for (let i = 0; i < 5; i++)
      recordTestAnnotation(harness.identityEventRepository, episode.id, `Synthetic ${i}`, {
        ts: i,
      });
    const originalRow = episodeEvidencePromptRow(episode);
    const decorated = (await withEpisodeEvidenceContext(harness, [episode]))[0]!;
    const row = JSON.parse(renderEvidencePromptJson(episodeEvidencePromptRow(decorated)));
    expect(row).not.toHaveProperty("lifecycle");
    expect(row.annotations).toEqual(
      [4, 3, 2].map((i) => ({
        text: `Synthetic ${i}`,
        written_at: new Date(i).toISOString(),
        authority: "Synthetic operator",
      })),
    );
    expect(row.annotations_omitted).toBe(2);
    expect(row.narrative).toBe(episode.narrative);
    expect(originalRow).not.toHaveProperty("annotations");
    expect(episodeEvidencePromptRow(episode)).toEqual(originalRow);
  });

  it("attributes a self-written correction retained only on a visible lineage record", async () => {
    harness = await createOfflineTestHarness();
    const { source, lineage } = await createRetiredEvidenceFixture(harness);
    const [operatorOnly] = await withEpisodeEvidenceContext(harness, [source], {
      includeVisibleLineage: true,
    });
    expect(
      renderEvidencePromptJson({
        evidence_guidance: EVIDENCE_GUIDANCE,
        episode: episodeEvidencePromptRow(operatorOnly!),
      }),
    ).toContain(MEMORY_EVIDENCE_GUIDANCE);
    const text = "Independent observation was provisional. 更正。";
    await new RecordCorrectionService({
      ...harness,
      entryIndex: harness.createContext().entryIndex!,
    }).correct(
      {
        target_id: lineage.id,
        correction_note: text,
        evidence: [{ id: lineage.id, basis: "own_read" }],
      },
      {
        turnId: "synthetic-turn",
        turnOrigin: "user",
        path: "reflector",
        streamWriter: harness.streamWriter,
      },
    );
    const [decorated] = await withEpisodeEvidenceContext(harness, [source], {
      includeVisibleLineage: true,
    });
    const prompt = renderEvidencePromptJson({
      evidence_guidance: EVIDENCE_GUIDANCE,
      episode: episodeEvidencePromptRow(decorated!),
    });
    expect(prompt).not.toContain(MEMORY_EVIDENCE_GUIDANCE);
    expect(JSON.parse(prompt)).toMatchObject({
      evidence_guidance: `${SELF_CORRECTION_GUIDANCE} lifecycle=archived marks a record retired from recall.`,
      episode: {
        id: source.id,
        lifecycle: "archived",
        visible_lineage: [
          {
            id: lineage.id,
            annotations: expect.arrayContaining([
              expect.objectContaining({
                text,
                author: "borg",
                original_record: { record_type: "episode", record_id: lineage.id },
              }),
            ]),
          },
        ],
      },
    });
    expect(JSON.parse(prompt).episode).not.toHaveProperty("self_correction_count");
  });

  it("keeps a lineage excerpt, its counts, and a short audit when a self correction exceeds the envelope", async () => {
    const h = await createOfflineTestHarness();
    harness = h;
    const { source, lineage } = await createRetiredEvidenceFixture(h);
    const restricted = await h.episodicRepository.createEpisode(
      createEpisodeFixture({
        origin_audience_entity_ids: Array.from({ length: 220 }, () => createEntityId()),
        shared: false,
      }),
    );
    await new RecordCorrectionService({ ...h, entryIndex: h.createContext().entryIndex! }).correct(
      {
        target_id: lineage.id,
        correction_note: "界".repeat(600),
        evidence: [{ id: restricted.id, basis: "own_read" }],
      },
      {
        turnId: "synthetic-turn",
        turnOrigin: "user",
        path: "reflector",
        streamWriter: h.streamWriter,
      },
    );
    const [decorated] = await withEpisodeEvidenceContext(h, [source], {
      includeVisibleLineage: true,
    });
    const member = decorated!.evidence_context!.visible_lineage![0]!;
    expect(member.annotations).toHaveLength(2);
    expect(JSON.stringify(member).length).toBeGreaterThan(MAX_EVIDENCE_RECORD_CONTEXT_CHARS);

    const row = episodeEvidencePromptRow(decorated!);
    const json = JSON.parse(
      renderEvidencePromptJson({ evidence_guidance: EVIDENCE_GUIDANCE, episode: row }),
    ).episode;
    const lines = renderEvidencePromptLines([EVIDENCE_GUIDANCE, row]);
    for (const payload of [json, JSON.parse(lines.split("\n").at(-1)!)]) {
      expect(payload).toMatchObject({
        visible_lineage: [
          {
            id: lineage.id,
            excerpt: lineage.narrative,
            self_correction_count: 1,
            annotations_omitted: 1,
            annotations: [{ text: member.annotations![1]!.text, authority: "Synthetic operator" }],
          },
        ],
      });
      expect(payload).not.toHaveProperty("lineage_omitted");
    }
    expect(member.annotations).toHaveLength(2);
  });
  it("bounds authority and omits unformattable annotation timestamps", () => {
    const fields = annotationPromptFields([
      {
        stream_entry_id: "synthetic",
        text: "Unchanged 更正",
        authority: "a".repeat(1000),
        written_at: 1e100,
      },
    ]);
    expect(fields.annotations![0]!.authority).toHaveLength(MAX_EVIDENCE_AUTHORITY_CHARS);
    expect(fields.annotations![0]!.text).toBe("Unchanged 更正");
    expect(fields.annotations![0]).not.toHaveProperty("written_at");
  });

  it("caps twelve cited sources before serialization without reducing any source narrative", async () => {
    harness = await createOfflineTestHarness();
    const raws = [];
    for (let i = 0; i < 4; i++)
      raws.push(
        await harness.episodicRepository.createEpisode(
          createEpisodeFixture({ narrative: "界".repeat(15_000) }),
        ),
      );
    const roots = [];
    for (let i = 0; i < 12; i++) {
      const { source } = await createRetiredEvidenceFixture(harness);
      const root = {
        ...source,
        narrative: `Original source ${i}: ` + "r".repeat(5000),
        lineage: { ...source.lineage, derived_from: raws.map((raw) => raw.id) },
      };
      roots.push(root);
      for (let n = 0; n < 4; n++)
        recordTestAnnotation(harness.identityEventRepository, root.id, "更".repeat(1500), {
          ts: 2000 + n,
        });
    }
    const ordinaryRows = roots.map((root) => episodeEvidencePromptRow(root));
    const ordinary = renderEvidencePromptJson({
      evidence_guidance: EVIDENCE_GUIDANCE,
      episodes: ordinaryRows,
    });
    const decorated = await withEpisodeEvidenceContext(harness, roots, {
      includeVisibleLineage: true,
    });
    const prompt = renderEvidencePromptJson({
      evidence_guidance: EVIDENCE_GUIDANCE,
      episodes: decorated.map((episode) => episodeEvidencePromptRow(episode)),
    });
    expect(prompt.length - ordinary.length).toBeLessThanOrEqual(MAX_EVIDENCE_PROMPT_CONTEXT_CHARS);
    expect(prompt.length - ordinary.length).toBeGreaterThan(20_000);
    const rows = JSON.parse(prompt).episodes;
    for (const [index, row] of rows.entries()) {
      expect(row.narrative).toBe(roots[index]!.narrative);
      expect(row.lifecycle).toBe("archived");
      expect((row.annotations?.length ?? 0) + (row.annotations_omitted ?? 0)).toBe(5);
      expect((row.visible_lineage?.length ?? 0) + (row.lineage_omitted ?? 0)).toBe(4);
      expect(
        JSON.stringify(row, null, 2).length - JSON.stringify(ordinaryRows[index], null, 2).length,
      ).toBeLessThanOrEqual(MAX_EVIDENCE_RECORD_CONTEXT_CHARS);
      for (const lineage of row.visible_lineage ?? []) {
        expect(lineage.excerpt.length).toBeLessThanOrEqual(MAX_EVIDENCE_LINEAGE_EXCERPT_CHARS);
        expect(lineage).toMatchObject({
          kind: "raw",
          excerpt_truncated: true,
          span: { start_time: expect.any(Number), end_time: expect.any(Number) },
        });
        expect(lineage).not.toHaveProperty("narrative");
      }
    }
    expect(rows[0].annotations).toHaveLength(3);
    expect(rows.at(-1).annotations_omitted).toBe(5);
    expect(rows.at(-1)).not.toHaveProperty("annotations");
    expect(prompt).not.toContain("__borg_evidence_context");
  }, 30_000);
});
