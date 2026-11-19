import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import fixture from "./__fixtures__/record-evidence-dev.json";
import { renderOrdinaryRecordSurfaces } from "./record-evidence-equivalence-test-support.js";
import type { createRecordEvidenceFixture } from "./record-evidence-test-support.js";

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
