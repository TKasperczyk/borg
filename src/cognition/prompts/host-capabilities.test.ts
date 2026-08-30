import { describe, expect, it } from "vitest";

import {
  BORG_HOST_CAPABILITY_BOUNDARY_PROMPT,
  buildHostCapabilitiesSection,
  withDerivedOutboundCapabilities,
} from "./host-capabilities.js";

describe("BORG_HOST_CAPABILITY_BOUNDARY_PROMPT", () => {
  it("distinguishes internal conversation memory from external documents", () => {
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain(
      'Conversation memory is my internal shared state: if someone says "the log" here',
    );
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain(
      "I do not promise an external shareable link, exportable document, or editable log",
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
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain(
      'I avoid unqualified "I\'ll prompt you"',
    );
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain(
      "A self-scheduled wake is private internal reflection",
    );
  });

  it("includes relationship-claim grounding guidance", () => {
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain(
      "Sensitive relationship claim grounding",
    );
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain("relationship_claim");
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain("label_family");
    expect(BORG_HOST_CAPABILITY_BOUNDARY_PROMPT).toContain("evidence_stream_entry_ids");
  });
});

describe("buildHostCapabilitiesSection", () => {
  it("renders wired proactive outbound source types as available", () => {
    const section = buildHostCapabilitiesSection({
      outboundSourceTypes: ["demo"],
    });

    expect(section).toContain(
      "Proactive outbound messaging via wired source_type connector(s): demo",
    );
    expect(section).toContain("tool.outbound.post");
    expect(section).toContain("Host-wired outbound capabilities available now:");
    expect(section).toContain("Targets without a wired connector are not transportable");
    expect(section).toContain(
      "A wired connector means the host can attempt the post, not that the far side accepts it",
    );
    expect(section).toContain("is not evidence it was carried");
    expect(section).not.toContain(
      "Capabilities NOT available unless the host has declared them otherwise:\n- Proactive outbound messaging via wired source_type connector(s): demo",
    );
  });

  it("keeps proactive outbound unavailable without wired connectors", () => {
    const section = buildHostCapabilitiesSection();

    expect(section).toContain("EmitContinueThought");
    expect(section).toContain("I can pursue an open question privately");
    expect(section).toContain("I can schedule a future self-wake");
    expect(section).toContain(
      "Proactive outbound messaging (I cannot reach out to participants later on my own initiative)",
    );
  });

  it("appends derived outbound status to custom host capability text", () => {
    const section = withDerivedOutboundCapabilities({
      hostCapabilities: "Custom host capabilities.",
      outboundSourceTypes: ["demo"],
    });

    expect(section).toContain("Custom host capabilities.");
    expect(section).toContain("Host-wired outbound capability status:");
    expect(section).toContain(
      "Proactive outbound messaging via wired source_type connector(s): demo",
    );
  });
});
