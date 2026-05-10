import type { OfflineProcessError, OfflineProcessName } from "./types.js";

type OfflineProcessErrorOptions = {
  code?: string;
  includeErrorCode?: boolean;
  target_type?: OfflineProcessError["target_type"];
  target_id?: string;
};

function errorCode(error: unknown): string | undefined {
  return error instanceof Error && "code" in error ? String(error.code) : undefined;
}

export function offlineProcessError<ProcessName extends OfflineProcessName>(
  process: ProcessName,
  error: unknown,
  options: OfflineProcessErrorOptions = {},
): OfflineProcessError & { process: ProcessName } {
  return {
    process,
    message: error instanceof Error ? error.message : String(error),
    code: options.code ?? (options.includeErrorCode === false ? undefined : errorCode(error)),
    ...(options.target_type === undefined ? {} : { target_type: options.target_type }),
    ...(options.target_id === undefined ? {} : { target_id: options.target_id }),
  };
}
