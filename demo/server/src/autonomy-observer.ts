import { BorgError, type Borg, type TickResult, type TurnTracer } from "borg";

const MAX_ERROR_MESSAGE_LENGTH = 512;
const MAX_SOURCE_NAME_LENGTH = 128;

function traceErrorFields(tracer: TurnTracer, message: string, name: string) {
  const errorName = name.slice(0, MAX_SOURCE_NAME_LENGTH);
  return {
    errorName,
    error: tracer.includePayloads ? message.slice(0, MAX_ERROR_MESSAGE_LENGTH) : errorName,
  };
}

export function wireAutonomySchedulerObserver(
  scheduler: Pick<Borg["autonomy"]["scheduler"], "setObserver">,
  tracer: TurnTracer,
  logger: Pick<Console, "warn" | "error"> = console,
): void {
  scheduler.setObserver({
    onTick: (result: TickResult) => {
      const hasErrors =
        result.errorCount > 0 || result.sourceErrorCount > 0 || result.bookkeepingErrorCount > 0;
      if (result.dueEvents === 0 && result.budgetSkipped === 0 && !hasErrors) return;

      // Count events that passed admission, including failed preparation/turns
      // and busy attempts. Coalesced events can share one fired wake/turn.
      const admittedEvents = result.events.filter(
        ({ status }) =>
          status === "fired" ||
          status === "error" ||
          status === "bookkeeping_error" ||
          status === "busy_skipped",
      ).length;
      const summary = {
        dueEvents: result.dueEvents,
        admittedEvents,
        firedEvents: result.firedEvents,
        budgetSkipped: result.budgetSkipped,
        fleetCooldownSkipped: result.fleetCooldownSkipped,
        errorCircuitSkipped: result.errorCircuitSkipped,
        busySkipped: result.busySkipped,
        errorCount: result.errorCount,
        sourceErrorCount: result.sourceErrorCount,
        bookkeepingErrorCount: result.bookkeepingErrorCount,
      };

      if (hasErrors) {
        logger.warn(`autonomy.tick ${JSON.stringify(summary)}`);
      }

      try {
        tracer.emit("autonomy.tick", {
          turnId: `autonomy_tick_${process.pid}_${result.ts}`,
          ts: result.ts,
          ...summary,
          triggerNames: [...new Set(result.events.map(({ sourceName }) => sourceName))].map(
            (name) => name.slice(0, MAX_SOURCE_NAME_LENGTH),
          ),
          // Payload fields are never copied. Exception text can echo content,
          // so include it only for sinks with payload tracing enabled.
          errors: result.events.flatMap(({ sourceName, error, errorName }) =>
            error === undefined
              ? []
              : [
                  {
                    sourceName: sourceName.slice(0, MAX_SOURCE_NAME_LENGTH),
                    ...traceErrorFields(tracer, error, errorName ?? "AutonomyError"),
                  },
                ],
          ),
          sourceErrors: (result.sourceErrors ?? []).map(({ sourceName, error, errorName }) => ({
            sourceName: sourceName.slice(0, MAX_SOURCE_NAME_LENGTH),
            ...traceErrorFields(tracer, error, errorName ?? "AutonomySourcePreparationError"),
          })),
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        logger.error(
          `Autonomy tick trace failed: ${JSON.stringify(message.slice(0, MAX_ERROR_MESSAGE_LENGTH))}`,
        );
      }
    },
    onError: (error: unknown) => {
      try {
        // These wrappers are also recorded in the completed TickResult.
        if (
          error instanceof BorgError &&
          (error.code === "AUTONOMY_SOURCE_PREPARATION_FAILED" ||
            error.code === "AUTONOMY_BOOKKEEPING_FAILED")
        )
          return;

        const ts = Date.now();
        const errorName = (error instanceof Error ? error.name : "UnknownError").slice(
          0,
          MAX_SOURCE_NAME_LENGTH,
        );
        let traceErrorCount = 0;
        try {
          tracer.emit("autonomy.tick_error", {
            turnId: `autonomy_tick_error_${process.pid}_${ts}`,
            ts,
            errorCount: 1,
            errors: [
              traceErrorFields(
                tracer,
                error instanceof Error ? error.message : String(error),
                errorName,
              ),
            ],
          });
        } catch {
          traceErrorCount = 1;
        }
        logger.warn(
          `autonomy.tick_error ${JSON.stringify({ errorCount: 1, errorName, traceErrorCount })}`,
        );
      } catch {
        // The error observer must not throw even if the journal sink fails.
      }
    },
  });
}
