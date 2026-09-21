import { describe, expect, it } from "vitest";
import { createSemanticNodeId } from "../src/util/ids.js";
import { fingerprint } from "./audited-data-repair.js";
import {
  assertCompleteVerdicts,
  operatorVerdictsSchema,
  parseVerdictFile,
} from "./judge-self-label-nodes-dossier.js";

const nodeId = createSemanticNodeId();
const targetId = createSemanticNodeId();
const input = {
  target_node_id: targetId,
  judged_by: {
    entity_id: "existing entity reference",
    evidence: { stream_entry_id: "opaque source", session_id: "opaque session", timestamp: 1234 },
  },
  recorded_by: { entity_id: "transcribing entity reference" },
  verdicts: [{ node_id: nodeId, verdict: "unsure", reason: "  理由\nMotif.  ", confidence: null }],
};

const dossier = {
  version: 2,
  narrative_voice: { every_source_episode_is_self_first_person: true, narrator: "self entity" },
  node: {
    id: nodeId,
    kind: "entity",
    label: "対象",
    description: { text: "説明", truncated: false },
    domain: null,
    aliases: [],
    confidence: 0.8,
    created_at: 1000,
    source_episode_ids: [],
    snapshot_hash: fingerprint("node snapshot"),
  },
  self: {
    entity_id: "existing entity reference",
    canonical_name: "対象",
    aliases: [],
    node_id: targetId,
    description: { text: "説明", truncated: false },
    snapshot_hash: fingerprint("target snapshot"),
  },
  edges: [],
  episodes: [],
  episodes_omitted: 0,
};
const operatorFile = {
  version: 5,
  kind: "judge-self-label-nodes",
  data_dir: "/synthetic-fixture-path",
  started_at: 2000,
  completed_at: 2001,
  judge: { kind: "operator", judged_by: input.judged_by, recorded_by: input.recorded_by },
  verdicts: [
    { ...input.verdicts[0], dossier, dossier_hash: fingerprint(dossier), judged_at: 1234 },
  ],
  failures: [],
  candidate_node_ids: [nodeId],
  target_node: { id: targetId, kind: "self", label: "対象" },
  matching_entities_report_only: [],
};

describe("operator verdict schemas", () => {
  it("preserves reasons and opaque references exactly and accepts explicit null confidence", () => {
    expect(operatorVerdictsSchema.parse(input)).toEqual(input);
    expect(
      operatorVerdictsSchema.parse({
        ...input,
        recorded_by: undefined,
        judged_by: {
          entity_id: input.judged_by.entity_id,
          evidence: { stream_entry_id: "opaque" },
        },
      }).judged_by.evidence,
    ).toEqual({ stream_entry_id: "opaque" });
  });

  it.each([0, 0.5, 1, null])("accepts supplied confidence %s unchanged", (confidence) => {
    expect(
      operatorVerdictsSchema.parse({
        ...input,
        verdicts: [{ ...input.verdicts[0], confidence }],
      }).verdicts[0]!.confidence,
    ).toBe(confidence);
  });

  it.each([undefined, -0.1, 1.1, "0.5", NaN, Infinity])(
    "rejects invalid/missing confidence %s",
    (confidence) => {
      expect(() =>
        operatorVerdictsSchema.parse({
          ...input,
          verdicts: [{ ...input.verdicts[0], confidence }],
        }),
      ).toThrow();
    },
  );

  it.each([
    { ...input, target_node_id: undefined },
    { ...input, target_node_id: "not-a-node-id" },
    { ...input, judged_by: undefined },
    { ...input, judged_by: { ...input.judged_by, entity_id: "" } },
    { ...input, judged_by: { ...input.judged_by, evidence: {} } },
    { ...input, judged_by: { ...input.judged_by, evidence: { stream_entry_id: "" } } },
    { ...input, recorded_by: { entity_id: "" } },
    { ...input, verdicts: [{ ...input.verdicts[0], node_id: "not-a-node-id" }] },
    { ...input, verdicts: [{ ...input.verdicts[0], verdict: "guess" }] },
    { ...input, verdicts: [{ ...input.verdicts[0], reason: "" }] },
    { ...input, verdicts: [{ ...input.verdicts[0], reason: undefined }] },
  ])("rejects malformed required structure %#", (value) => {
    expect(() => operatorVerdictsSchema.parse(value)).toThrow();
  });

  it.each([-1, 1.5, "1234", Infinity])(
    "rejects a non-epoch-millisecond timestamp %s",
    (timestamp) => {
      expect(() =>
        operatorVerdictsSchema.parse({
          ...input,
          judged_by: { ...input.judged_by, evidence: { ...input.judged_by.evidence, timestamp } },
        }),
      ).toThrow();
    },
  );

  it.each([
    { ...input, model: "fabricated" },
    { ...input, judged_by: { ...input.judged_by, kind: "self" } },
    {
      ...input,
      judged_by: { ...input.judged_by, evidence: { ...input.judged_by.evidence, text: "source" } },
    },
    { ...input, recorded_by: { ...input.recorded_by, name: "extra" } },
    { ...input, verdicts: [{ ...input.verdicts[0], model: "fabricated" }] },
  ])("rejects unknown fields at every input object level %#", (value) => {
    expect(() => operatorVerdictsSchema.parse(value)).toThrow();
  });

  it("round-trips version 5 without model attribution or a fabricated confidence", () => {
    expect(parseVerdictFile(operatorFile)).toEqual(operatorFile);
    expect(fingerprint(parseVerdictFile(operatorFile))).toBe(fingerprint(operatorFile));
  });

  it("directs incomplete operator files to record with completed input and a new output", () => {
    const file = parseVerdictFile({ ...operatorFile, verdicts: [] });
    expect(() => assertCompleteVerdicts(file, [nodeId])).toThrow(
      `Failed or missing judgments: ${nodeId} (missing); complete the operator input and run record with a new --out path`,
    );
  });

  it.each([
    { ...operatorFile, model: "fabricated" },
    { ...operatorFile, model_role: "extraction" },
    { ...operatorFile, judge: { ...operatorFile.judge, model: "fabricated" } },
    { ...operatorFile, judge: { ...operatorFile.judge, model_role: "extraction" } },
    { ...operatorFile, judge: { ...operatorFile.judge, kind: "model" } },
    { ...operatorFile, verdicts: [{ ...operatorFile.verdicts[0], model: "fabricated" }] },
    { ...operatorFile, verdicts: [{ ...operatorFile.verdicts[0], model_role: "extraction" }] },
    { ...operatorFile, candidate_node_ids: undefined },
    { ...operatorFile, target_node: undefined },
    {
      ...operatorFile,
      verdicts: [{ ...operatorFile.verdicts[0], dossier_hash: fingerprint("changed") }],
    },
  ])("rejects malformed operator files or model metadata %#", (value) => {
    expect(() => parseVerdictFile(value)).toThrow();
  });
});

describe("legacy verdict-file compatibility", () => {
  it.each([1, 2, 3, 4])(
    "preserves version %s envelopes, stored version-1 dossier hashes and incomplete-file advice",
    (version) => {
      const { narrative_voice: _voice, ...withoutVoice } = dossier;
      const legacyDossier = { ...withoutVoice, version: 1 };
      const file = {
        version,
        kind: "judge-self-label-nodes",
        data_dir: "/synthetic-fixture-path",
        started_at: 2000,
        completed_at: 2001,
        model: "original-model",
        model_role: "extraction",
        verdicts: [
          {
            ...input.verdicts[0],
            confidence: 0.5,
            model: "original-model",
            judged_at: 1234,
            dossier: legacyDossier,
            dossier_hash: fingerprint(legacyDossier),
          },
        ],
        failures: [],
        matching_entities_report_only: [],
        ...(version >= 2 ? { candidate_node_ids: [nodeId] } : {}),
        ...(version === 3 ? { self_node: { id: targetId, label: "対象" } } : {}),
        ...(version === 4 ? { target_node: operatorFile.target_node } : {}),
      };
      expect(parseVerdictFile(file)).toEqual(file);
      expect(fingerprint(parseVerdictFile(file))).toBe(fingerprint(file));
      expect(parseVerdictFile(file).verdicts[0]!.dossier_hash).toBe(fingerprint(legacyDossier));
      expect(() =>
        assertCompleteVerdicts(parseVerdictFile({ ...file, verdicts: [] }), [nodeId]),
      ).toThrow(
        `Failed or missing judgments: ${nodeId} (missing); rerun judge on the existing --out file`,
      );
    },
  );
});
