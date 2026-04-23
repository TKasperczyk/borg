// Borg lifecycle helper for commands that need a temporary library instance.
import { Borg, type BorgOpenOptions } from "../../index.js";
import type { RunCliOptions } from "../types.js";

export async function withBorg<T>(
  options: RunCliOptions,
  fn: (borg: Borg) => Promise<T>,
): Promise<T> {
  const openOptions: BorgOpenOptions = {
    env: options.env,
    dataDir: options.dataDir,
  };
  const borg = await (options.openBorg?.(openOptions) ?? Borg.open(openOptions));

  try {
    return await fn(borg);
  } finally {
    await borg.close();
  }
}
