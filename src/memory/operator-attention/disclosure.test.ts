import { describe, expect, it } from "vitest";

import { createEntityId } from "../../util/ids.js";
import {
  memoryDisclosureLabelMetadata,
  publicMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
} from "../common/disclosure-label.js";
import { operatorAttentionPromptRow } from "./disclosure.js";

const record = {
  record_key: "cclink:disclosure",
  filed_at: 1_000,
  filer_entity_id: createEntityId(),
  subject: "Filing subject",
};

describe("operator attention disclosure rows", () => {
  it("labels metadata structurally without inferring operator recipients from the filer", () => {
    expect(operatorAttentionPromptRow(record)).toEqual({
      ...record,
      disclosure_label: {
        disclosureClass: "operator_private",
        originAudienceEntityIds: [record.filer_entity_id],
        privateToEntityIds: [],
        publicToEntityIds: [],
      },
    });
  });

  it.each([
    unknownMemoryDisclosureLabel(),
    memoryDisclosureLabelMetadata(unknownMemoryDisclosureLabel()),
    null,
    { disclosureClass: "public" } as never,
  ])("fails closed when a carried label is unknown or malformed: %j", (disclosureLabel) => {
    const row = operatorAttentionPromptRow({ ...record, disclosure_label: disclosureLabel });
    expect(row.disclosure_label.disclosureClass).toBe("unknown");
    expect(row.disclosure_label.originAudienceEntityIds).toEqual([record.filer_entity_id]);
  });

  it("does not demote private filing metadata to public and clones carried label arrays", () => {
    const row = operatorAttentionPromptRow({
      ...record,
      disclosure_label: publicMemoryDisclosureLabel(),
    });
    expect(row.disclosure_label.disclosureClass).toBe("operator_private");
    const captured = operatorAttentionPromptRow(row);
    expect(captured).toEqual(row);
    expect(captured.disclosure_label).not.toBe(row.disclosure_label);
    expect(captured.disclosure_label.originAudienceEntityIds).not.toBe(
      row.disclosure_label.originAudienceEntityIds,
    );
  });
});
