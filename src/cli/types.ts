// Shared CLI types for command registration, command output, and Borg bootstrap options.
import type { CAC } from "cac";

import type { Borg, BorgOpenOptions } from "../index.js";

export type Output = Pick<NodeJS.WriteStream, "write">;

export type RunCliOptions = {
  stdout?: Output;
  stderr?: Output;
  env?: NodeJS.ProcessEnv;
  dataDir?: string;
  openBorg?: (options: BorgOpenOptions) => Promise<Borg>;
};

export type CliCommandDeps = {
  stdout: Output;
  stderr: Output;
  env: NodeJS.ProcessEnv;
  options: RunCliOptions;
};

export type CommandOptions = Record<string, unknown>;

export type RegisterCliCommands = (cli: CAC, deps: CliCommandDeps) => void;
