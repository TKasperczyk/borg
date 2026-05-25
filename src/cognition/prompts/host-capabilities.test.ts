import { describe, expect, it } from "vitest";

import { BORG_HOST_CAPABILITY_BOUNDARY_PROMPT } from "./host-capabilities.js";

describe("BORG_HOST_CAPABILITY_BOUNDARY_PROMPT", () => {
  it("distinguishes internal conversation memory from external documents", () => {
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain(
      'Conversation memory is internal shared state: if someone says "the log" here',
    );
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain(
      "Do not promise an external shareable link, exportable document, or editable log",
    );
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain("log/doc link");
  });

  it("prefers reactive surfacing language over proactive prompt language", () => {
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain(
      "When you next bring this back here, I'll surface X",
    );
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain(
      "When someone asks about X in this channel again, I'll mention Y",
    );
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain('Avoid unqualified "I\'ll prompt you"');
  });

  it("includes relationship-label grounding guidance", () => {
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain("Relationship label grounding");
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain("sibling, spouse, parent");
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain("partner, manager, owner");
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain("direct evidence supports");
  });
});
