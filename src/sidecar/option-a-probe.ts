import type { BorgOpenOptions } from "../borg/types.js";
import { DEFAULT_TENANT_ID_PATTERN } from "../borg/tenant-directories.js";
import { ConfigError } from "../util/errors.js";
import { optionAProbeEnabledFromEnv } from "../config/index.js";

export type OptionAProbeConfig = { enabled: false } | { enabled: true; tenant: string };

export function optionAProbeConfigFromEnv(env: NodeJS.ProcessEnv): OptionAProbeConfig {
  if (!optionAProbeEnabledFromEnv(env)) return { enabled: false };
  const tenant = env.BORG_OPTION_A_PROBE_TENANT?.trim() ?? "team-agent-ai";
  if (!DEFAULT_TENANT_ID_PATTERN.test(tenant)) {
    throw new ConfigError("Invalid BORG_OPTION_A_PROBE_TENANT");
  }
  return { enabled: true, tenant };
}

/** Omit the override to select Borg.open's native runner, preserving the default verbatim. */
export function selectSidecarInboxRunner(
  probe: OptionAProbeConfig,
  tenant: string,
  runner: NonNullable<NonNullable<BorgOpenOptions["inbox"]>["runner"]>,
): Pick<NonNullable<BorgOpenOptions["inbox"]>, "runner" | "native"> {
  return probe.enabled && probe.tenant === tenant
    ? { native: { tools: "none", agentDeliveries: true } }
    : { runner };
}
