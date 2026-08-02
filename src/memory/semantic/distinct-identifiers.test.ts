import { describe, expect, it } from "vitest";

import {
  disjointDistinctIdentifiers,
  distinctIdentifiersFromLabel,
} from "./distinct-identifiers.js";

describe("distinct semantic label identifiers", () => {
  it("extracts and normalizes ticket keys", () => {
    expect(distinctIdentifiersFromLabel("Ticket aininjas-1110 replaced AININJAS-1110")).toEqual([
      "ticket:AININJAS-1110",
    ]);
    expect(disjointDistinctIdentifiers("AININJAS-1110", "Incident AININJAS-1111")).toEqual({
      left: ["ticket:AININJAS-1110"],
      right: ["ticket:AININJAS-1111"],
    });
  });

  it("extracts exact 32-hex run identifiers", () => {
    const first = "Autonomous Run f0bec94550ab5cd0e0d4408f710727fe";
    const second = "Autonomous Run 8c16745d9ce140ed90c92f6ef06fb921";

    expect(distinctIdentifiersFromLabel(first)).toEqual(["run:F0BEC94550AB5CD0E0D4408F710727FE"]);
    expect(disjointDistinctIdentifiers(first, second)).not.toBeNull();
    expect(distinctIdentifiersFromLabel("run 0f0bec94550ab5cd0e0d4408f710727fe0")).toEqual([]);
  });

  it("uses case-preserving URL paths and ignores hosts, query strings, and trailing slashes", () => {
    const first = "http://pbatman1.p4.int/ai/ai-summary/-/merge_requests/34?view=changes";
    const samePath = "https://gitlab.example/ai/ai-summary/-/merge_requests/34/";
    const otherPath = "http://pbatman1.p4.int/ai/ai-summary/-/merge_requests/33";

    expect(distinctIdentifiersFromLabel(first)).toEqual([
      "url_path:/ai/ai-summary/-/merge_requests/34",
    ]);
    expect(disjointDistinctIdentifiers(first, samePath)).toBeNull();
    expect(disjointDistinctIdentifiers(first, otherPath)).not.toBeNull();
  });

  it("strips trailing prose punctuation without folding case-sensitive URL paths", () => {
    const punctuated = "See (https://gitlab.example/Team/Project/-/merge_requests/34).";
    const samePath = "https://other.example/Team/Project/-/merge_requests/34";
    const differentlyCasedPath = "https://other.example/team/project/-/merge_requests/34";

    expect(distinctIdentifiersFromLabel(punctuated)).toEqual([
      "url_path:/Team/Project/-/merge_requests/34",
    ]);
    expect(disjointDistinctIdentifiers(punctuated, samePath)).toBeNull();
    expect(disjointDistinctIdentifiers(punctuated, differentlyCasedPath)).not.toBeNull();
  });

  it("extracts independent long digit runs alongside other identifier classes", () => {
    expect(distinctIdentifiersFromLabel("Call record 48123456789")).toEqual(["digits:48123456789"]);
    expect(
      distinctIdentifiersFromLabel("AININJAS-123456789 at https://jira.example/browse/123456789"),
    ).toEqual(["ticket:AININJAS-123456789", "url_path:/browse/123456789"]);
    expect(distinctIdentifiersFromLabel("ABC-1 batch 123456789")).toEqual([
      "ticket:ABC-1",
      "digits:123456789",
    ]);
    expect(
      disjointDistinctIdentifiers("ABC-1 batch 123456789", "ABC-2 batch 123456789"),
    ).toBeNull();
    expect(disjointDistinctIdentifiers("Call 48123456789", "Call 48123456780")).not.toBeNull();
  });

  it("falls through when either label has no identifiers", () => {
    expect(disjointDistinctIdentifiers("AININJAS-1110", "Ticket creation outcome")).toBeNull();
    expect(disjointDistinctIdentifiers("Atlas platform", "Deployment platform")).toBeNull();
  });

  it("falls through when the identifier sets overlap even if they are not equal", () => {
    expect(
      disjointDistinctIdentifiers(
        "AININJAS-1110 compared with AININJAS-1111",
        "Ticket AININJAS-1110",
      ),
    ).toBeNull();
  });
});
