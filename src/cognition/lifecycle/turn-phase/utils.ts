import type { CorrectivePreferenceTurnService } from "../../commitments/corrective-preference-service.js";
import type { SharedStateRenderOptions } from "../../evidence-ledger/index.js";
import type { AnsweredStreamWindow, StreamIngestionCoordinator } from "../../ingestion/index.js";
import { appendInternalFailureEvent } from "../../../memory/self/index.js";
import type { StreamWriter } from "../../../stream/index.js";
import type { Config } from "../../../config/index.js";
import type { SessionId } from "../../../util/ids.js";

export const ACTIVE_TURN_STATUS = "active";

export type AppendHookFailureEvent = (
  streamWriter: StreamWriter,
  hook: string,
  error: unknown,
  details?: Record<string, unknown>,
) => Promise<void>;

export function sharedStateRenderOptions(config: Config): SharedStateRenderOptions {
  const sharedStateConfig = config.generation.evidenceLedger.decisionArtifact;

  return {
    maxEntries: sharedStateConfig.renderMaxEntries,
    maxTokens: sharedStateConfig.renderMaxTokens,
    reservedSlots: sharedStateConfig.renderReservedSlots,
    lockedMaxEntries: sharedStateConfig.renderLockedCap,
    newestStateChangeReservedSlots: sharedStateConfig.newestStateChangeReservedSlots,
  };
}

export async function appendHookFailureEvent(
  streamWriter: StreamWriter,
  hook: string,
  error: unknown,
  details?: Record<string, unknown>,
): Promise<void> {
  await appendInternalFailureEvent(streamWriter, hook, error, details);
}

export async function persistCorrectiveCommitment(input: {
  service: CorrectivePreferenceTurnService;
  streamWriter: StreamWriter;
  turnId: string;
  sessionId: SessionId;
  commitment: Parameters<CorrectivePreferenceTurnService["persistCommitment"]>[0]["commitment"];
  supersession: Parameters<CorrectivePreferenceTurnService["persistCommitment"]>[0]["supersession"];
  appendHookFailureEvent: AppendHookFailureEvent;
}): Promise<void> {
  await input.service.persistCommitment({
    commitment: input.commitment,
    supersession: input.supersession,
    turnId: input.turnId,
    sessionId: input.sessionId,
    onHookFailure: (hook, error, details) =>
      input.appendHookFailureEvent(input.streamWriter, hook, error, details),
  });
}

export async function catchUpStreamIngestion(input: {
  coordinator: StreamIngestionCoordinator | undefined;
  sessionId: SessionId;
  streamWriter: StreamWriter;
  maxEntries: number;
  clampToChatResponseWatermark?: boolean;
  appendHookFailureEvent: AppendHookFailureEvent;
}): Promise<void> {
  if (input.coordinator === undefined) {
    return;
  }

  try {
    const result = await input.coordinator.catchUp(input.sessionId, {
      maxEntries: input.maxEntries,
      ...(input.clampToChatResponseWatermark === true
        ? { clampToChatResponseWatermark: true }
        : {}),
    });

    if (result.error !== undefined) {
      await input.appendHookFailureEvent(
        input.streamWriter,
        "stream_ingestion_pre_turn_catchup",
        result.error,
        {
          processedEntries: result.processedEntries,
        },
      );
    }
  } catch (error) {
    await input.appendHookFailureEvent(
      input.streamWriter,
      "stream_ingestion_pre_turn_catchup",
      error,
    );
  }
}

export function startLiveIngestion(
  coordinator: StreamIngestionCoordinator | undefined,
  sessionId: SessionId,
  options: { answeredWindow?: AnsweredStreamWindow } = {},
): void {
  if (coordinator !== undefined) {
    const ingestPromise =
      options.answeredWindow === undefined
        ? coordinator.ingest(sessionId)
        : coordinator.ingest(sessionId, { answeredWindow: options.answeredWindow });

    void ingestPromise.catch((error) => {
      console.error("Live stream ingestion failed", error);
    });
  }
}
