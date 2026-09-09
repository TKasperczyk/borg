import { describe, expect, it } from "vitest";
import { optionAProbeConfigFromEnv, selectSidecarInboxRunner } from "./option-a-probe.js";

describe("option A runner selection", () => {
  const runner = { async run() {} };
  it.each([undefined, "", "0", "false"])("keeps the exact override for flag %s", (flag) => {
    const probe = optionAProbeConfigFromEnv({ BORG_OPTION_A_PROBE: flag });
    expect(probe).toEqual({ enabled: false });
    expect(selectSidecarInboxRunner(probe, "team-agent-ai", runner)).toEqual({ runner });
    expect(selectSidecarInboxRunner(probe, "team-agent-ai", runner).runner).toBe(runner);
  });
  it.each(["1", "true"])("selects native without tools only for the target tenant (%s)", (flag) => {
    const probe = optionAProbeConfigFromEnv({ BORG_OPTION_A_PROBE: flag });
    expect(selectSidecarInboxRunner(probe, "team-agent-ai", runner)).toEqual({
      native: { tools: "none", agentDeliveries: true },
    });
    expect(selectSidecarInboxRunner(probe, "another-tenant", runner)).toEqual({ runner });
  });
  it("accepts an explicit tenant and rejects invalid tenant ids", () => {
    expect(
      optionAProbeConfigFromEnv({
        BORG_OPTION_A_PROBE: "1",
        BORG_OPTION_A_PROBE_TENANT: "probe-copy",
      }),
    ).toEqual({ enabled: true, tenant: "probe-copy" });
    expect(() =>
      optionAProbeConfigFromEnv({
        BORG_OPTION_A_PROBE: "1",
        BORG_OPTION_A_PROBE_TENANT: "../other",
      }),
    ).toThrow("BORG_OPTION_A_PROBE_TENANT");
  });
});
