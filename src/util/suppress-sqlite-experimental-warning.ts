const suppressionMarker = "__borgSqliteExperimentalWarningSuppressed";
const globalWithSuppressionMarker = globalThis as Record<string, unknown>;

function warningName(warning: string | Error, args: readonly unknown[]): string {
  if (warning instanceof Error && warning.name.length > 0) {
    return warning.name;
  }

  const firstArg = args[0];

  if (typeof firstArg === "string") {
    return firstArg;
  }

  if (
    firstArg !== null &&
    typeof firstArg === "object" &&
    "type" in firstArg &&
    typeof firstArg.type === "string"
  ) {
    return firstArg.type;
  }

  return "Warning";
}

function warningMessage(warning: string | Error): string {
  return typeof warning === "string" ? warning : warning.message;
}

function isSqliteExperimentalWarning(warning: string | Error, args: readonly unknown[]): boolean {
  return (
    warningName(warning, args) === "ExperimentalWarning" &&
    warningMessage(warning).includes("SQLite")
  );
}

if (globalWithSuppressionMarker[suppressionMarker] !== true) {
  globalWithSuppressionMarker[suppressionMarker] = true;
  const originalEmitWarning = process.emitWarning;

  process.emitWarning = function emitWarningWithoutSqliteExperimentalNoise(
    warning: string | Error,
    ...args: unknown[]
  ): void {
    if (isSqliteExperimentalWarning(warning, args)) {
      return;
    }

    (originalEmitWarning as (...emitArgs: unknown[]) => void).call(process, warning, ...args);
  } as typeof process.emitWarning;
}
