import type {
  AgentSuppressedStreamContent,
  FinalizerInvalidToolDiagnostic,
  FinalizerNoOutputCategory,
  FinalizerNoOutputPrimaryReason,
  FinalizerNoOutputStructuralFlag,
  GenerationSuppressionReason,
} from "./generation/types.js";
import type {
  RecentRegenerationEntry,
  RecentSuppressionEntry,
  WorkingMemory,
} from "../memory/working/index.js";
import {
  RECENT_REGENERATIONS_LIMIT,
  RECENT_SUPPRESSIONS_LIMIT,
} from "./generation/discourse-state.js";
import {
  hydrateStreamEntriesById,
  type StreamEntry,
  type StreamEntryIndexRepository,
  type StreamReader,
} from "../stream/index.js";
import type { SessionId, StreamEntryId } from "../util/ids.js";

export type HydratedSuppressionDiagnostic = {
  noOutputCategories?: readonly FinalizerNoOutputCategory[];
  primaryNoOutputReason?: FinalizerNoOutputPrimaryReason;
  structuralNoOutputFlags?: readonly FinalizerNoOutputStructuralFlag[];
  finalizerInvalidTool?: FinalizerInvalidToolDiagnostic;
};

export type HydratedRecentSuppression = {
  turnId: string;
  reason: GenerationSuppressionReason | string;
  ts: number;
  sourceStreamEntryId?: StreamEntryId;
  diagnostic?: HydratedSuppressionDiagnostic;
};

export type HydratedRecentRegeneration = {
  turnId: string;
  mechanism: RecentRegenerationEntry["mechanism"];
  ts: number;
  sourceStreamEntryId?: StreamEntryId;
};

export type TurnMechanismEvidence = {
  recentSuppressions: readonly HydratedRecentSuppression[];
  recentRegenerations: readonly HydratedRecentRegeneration[];
};

export type HydrateTurnMechanismEvidenceInput = {
  dataDir: string;
  sessionId: SessionId;
  workingMemory: WorkingMemory;
  entryIndex?: Pick<StreamEntryIndexRepository, "lookupMany">;
  createStreamReader: (sessionId: SessionId) => StreamReader;
};

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringArray(value: unknown): string[] | undefined {
  return Array.isArray(value) && value.every((entry) => typeof entry === "string")
    ? [...value]
    : undefined;
}

function finalizerInvalidToolDiagnostic(
  value: unknown,
): FinalizerInvalidToolDiagnostic | undefined {
  if (!isObjectRecord(value)) {
    return undefined;
  }

  return typeof value.tool_name === "string" &&
    typeof value.reason === "string" &&
    (value.attempt === "initial" || value.attempt === "regenerate")
    ? {
        tool_name: value.tool_name,
        reason: value.reason,
        attempt: value.attempt,
      }
    : undefined;
}

function suppressionDiagnosticFromEntry(
  entry: StreamEntry | undefined,
): HydratedSuppressionDiagnostic {
  if (entry?.kind !== "agent_suppressed" || !isObjectRecord(entry.content)) {
    return {};
  }

  const content = entry.content as Partial<AgentSuppressedStreamContent>;
  const noOutputCategories = stringArray(content.no_output_categories);
  const structuralNoOutputFlags = stringArray(content.structural_no_output_flags);
  const primaryNoOutputReason =
    typeof content.primary_no_output_reason === "string"
      ? content.primary_no_output_reason
      : undefined;
  const finalizerInvalidTool = finalizerInvalidToolDiagnostic(content.finalizer_invalid_tool);

  return {
    ...(noOutputCategories === undefined
      ? {}
      : { noOutputCategories: noOutputCategories as FinalizerNoOutputCategory[] }),
    ...(primaryNoOutputReason === undefined
      ? {}
      : { primaryNoOutputReason: primaryNoOutputReason as FinalizerNoOutputPrimaryReason }),
    ...(structuralNoOutputFlags === undefined
      ? {}
      : { structuralNoOutputFlags: structuralNoOutputFlags as FinalizerNoOutputStructuralFlag[] }),
    ...(finalizerInvalidTool === undefined ? {} : { finalizerInvalidTool }),
  };
}

function hydratedRecentRegeneration(entry: RecentRegenerationEntry): HydratedRecentRegeneration {
  return {
    turnId: entry.turn_id,
    mechanism: entry.mechanism,
    ts: entry.ts,
    ...(entry.source_stream_entry_id === undefined
      ? {}
      : { sourceStreamEntryId: entry.source_stream_entry_id }),
  };
}

export async function hydrateTurnMechanismEvidence(
  input: HydrateTurnMechanismEvidenceInput,
): Promise<TurnMechanismEvidence> {
  const recentSuppressions = (input.workingMemory.discourse_state?.recent_suppressions ?? []).slice(
    -RECENT_SUPPRESSIONS_LIMIT,
  );
  const recentRegenerations = (
    input.workingMemory.discourse_state?.recent_regenerations ?? []
  ).slice(-RECENT_REGENERATIONS_LIMIT);
  const sourceStreamEntryIds = recentSuppressions.flatMap((entry) =>
    entry.source_stream_entry_id === undefined ? [] : [entry.source_stream_entry_id],
  );
  const entriesById = await hydrateStreamEntriesById({
    dataDir: input.dataDir,
    sessionId: input.sessionId,
    streamEntryIds: sourceStreamEntryIds,
    entryIndex: input.entryIndex,
    createStreamReader: input.createStreamReader,
  });

  return {
    recentSuppressions: recentSuppressions.map((entry: RecentSuppressionEntry) => ({
      turnId: entry.turn_id,
      reason: entry.reason,
      ts: entry.ts,
      ...(entry.source_stream_entry_id === undefined
        ? {}
        : {
            sourceStreamEntryId: entry.source_stream_entry_id,
            diagnostic: suppressionDiagnosticFromEntry(
              entriesById.get(entry.source_stream_entry_id),
            ),
          }),
    })),
    recentRegenerations: recentRegenerations.map(hydratedRecentRegeneration),
  };
}
