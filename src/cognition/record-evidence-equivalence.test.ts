import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { beforeAll, describe, expect, it } from "vitest";
import fixture from "./__fixtures__/record-evidence-dev.json";
import { renderOrdinaryRecordSurfaces } from "./record-evidence-equivalence-test-support.js";
import type { createRecordEvidenceFixture } from "./record-evidence-test-support.js";
import {
  annotationPromptFields,
  EVIDENCE_GUIDANCE,
  evidencePromptFields,
  renderEvidencePromptJson,
  renderEvidencePromptLines,
} from "../memory/common/episode-evidence.js";

// Frozen complete outputs were rendered with 87254783 source, including its compactor.
// The inputs were generated synthetically. Neither side is stripped or normalized.
describe("ordinary live surfaces versus dev 87254783", () => {
  let actual: Record<string, string>;
  beforeAll(async () => {
    actual = await renderOrdinaryRecordSurfaces(
      fileURLToPath(new URL("../../", import.meta.url)).replace(/\/$/, ""),
      fixture.input as unknown as ReturnType<typeof createRecordEvidenceFixture>,
    );
  });
  for (const [name, expected] of Object.entries(fixture.outputs))
    it(`${name} is byte-identical to dev`, () => {
      expect(Buffer.from(actual[name]!)).toEqual(Buffer.from(expected));
    });
});

it("preserves operator-only record payload bytes from eaf0c3e8", async () => {
  const input = fixture.input as unknown as ReturnType<typeof createRecordEvidenceFixture>;
  const annotations = [
    {
      stream_entry_id: "synthetic-operator-1",
      text: "暫定の観察。 🧭",
      written_at: 1000,
      authority: "Synthetic operator",
    },
    {
      stream_entry_id: "synthetic-operator-2",
      text: "界".repeat(1201),
      written_at: 900,
      authority: null,
    },
  ];
  const live = await renderOrdinaryRecordSurfaces(
    fileURLToPath(new URL("../../", import.meta.url)).replace(/\/$/, ""),
    input,
    (target) =>
      [input.episodes[0]!.episode.id, input.node.id].some((id) => id === target.record_id)
        ? annotations
        : [],
  );
  // Hash the complete dev outputs, including the legacy annotation clipping and counts.
  expect(Object.keys(live)).toHaveLength(19);
  expect(createHash("sha256").update(JSON.stringify(live)).digest("hex")).toBe(
    "7a9e93aa83743c44659f6901c4bb0707d65067827928d233bab27c0b44236e89",
  );
  const row = {
    id: input.node.id,
    description: "Synthetic description.",
    ...evidencePromptFields(annotationPromptFields(annotations)),
  };
  const packet = { guidance: EVIDENCE_GUIDANCE, rows: [row] };
  const envelopes = [
    renderEvidencePromptJson(packet),
    renderEvidencePromptJson(packet, { compact: true }),
    renderEvidencePromptLines([EVIDENCE_GUIDANCE, row]),
  ];
  expect(createHash("sha256").update(JSON.stringify(envelopes)).digest("hex")).toBe(
    "06ef82e0175ee9789a5dcb34009c1e7e1baa1b911036ec5decb1ee864b18ae97",
  );
});
