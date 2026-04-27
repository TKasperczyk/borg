import type { StreamEntry, StreamWriter } from "../../stream/index.js";
import type {
  PendingSocialAttribution,
  PendingTraitAttribution,
  WorkingMemory,
  WorkingMemoryStore,
} from "../../memory/working/index.js";
import type { SuppressionSet } from "../attention/index.js";
import type { PerceptionResult } from "../types.js";

export type TurnOpeningPersistenceOptions = {
  workingMemoryStore: Pick<WorkingMemoryStore, "save">;
};

export type TurnOpeningPersistenceInput = {
  streamWriter: Pick<StreamWriter, "append">;
  userMessage: string;
  audience?: string;
  workingMemory: WorkingMemory;
  pendingSocialAttribution: PendingSocialAttribution | null;
  pendingTraitAttribution: PendingTraitAttribution | null;
  suppressionSet: SuppressionSet;
  perception: PerceptionResult;
  now: () => number;
};

export type TurnOpeningPersistenceResult = {
  persistedUserEntry: StreamEntry;
  workingMemory: WorkingMemory;
};

export class TurnOpeningPersistence {
  constructor(private readonly options: TurnOpeningPersistenceOptions) {}

  async persist(input: TurnOpeningPersistenceInput): Promise<TurnOpeningPersistenceResult> {
    const userEntry = {
      kind: "user_msg",
      content: input.userMessage,
      ...(input.audience === undefined ? {} : { audience: input.audience }),
    } satisfies Parameters<StreamWriter["append"]>[0];

    const persistedUserEntry = await input.streamWriter.append(userEntry);

    const workingMemory = this.options.workingMemoryStore.save({
      ...input.workingMemory,
      pending_social_attribution: input.pendingSocialAttribution,
      pending_trait_attribution: input.pendingTraitAttribution,
      suppressed: input.suppressionSet.snapshot(),
      updated_at: input.now(),
    });

    await input.streamWriter.append({
      kind: "perception",
      content: {
        mode: input.perception.mode,
        entities: input.perception.entities,
        temporalCue: input.perception.temporalCue,
        affectiveSignal: input.perception.affectiveSignal,
      },
      ...(input.audience === undefined ? {} : { audience: input.audience }),
    });

    return {
      persistedUserEntry,
      workingMemory,
    };
  }
}
