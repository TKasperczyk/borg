import type { StreamEntry } from "../../stream/index.js";
import type { EvidenceLedgerBuildInput, EvidenceLedgerBuilderOptions } from "./builder-types.js";
import type { SectionBuckets } from "./section-buckets.js";
import type { ScopeResolver } from "./scope-resolver.js";
import type { TranscriptCompactionResult } from "./transcript-compaction.js";

export type BuilderSectionContext = {
  input: EvidenceLedgerBuildInput;
  resolver: ScopeResolver;
  buckets: SectionBuckets;
  options: Pick<
    EvidenceLedgerBuilderOptions,
    "actionThreadRenderLimit" | "actionThreadSimilarityThreshold" | "actionThreadSourceRecordLimit"
  >;
  transcript: TranscriptCompactionResult;
  streamEntries: readonly StreamEntry[];
  repos: {
    relationalSlots: EvidenceLedgerBuilderOptions["relationalSlotRepository"];
    actions: EvidenceLedgerBuilderOptions["actionRepository"];
    commitments: EvidenceLedgerBuilderOptions["commitmentRepository"];
    goals: EvidenceLedgerBuilderOptions["goalsRepository"];
    openQuestions: EvidenceLedgerBuilderOptions["openQuestionsRepository"];
    entities: EvidenceLedgerBuilderOptions["entityRepository"];
  };
};
