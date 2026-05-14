import {
  activeSessionTranscriptEntries,
  loadSessionStreamEntries,
  type StreamEntry,
} from "../../stream/index.js";
import type { BuilderSectionContext } from "./builder-context.js";
import type {
  EvidenceLedgerBuildInput,
  EvidenceLedgerBuilderOptions,
} from "./builder-types.js";
import {
  createSectionBuckets,
  finalSections,
} from "./section-buckets.js";
import {
  buildEpisodeScopeMap,
  buildEpisodeSourceStreamIdMap,
  type ScopeResolver,
} from "./scope-resolver.js";
import { addActionStatesSection } from "./sections/action-states.js";
import { addCommitmentsAndConstraintsSection } from "./sections/constraints.js";
import {
  addCurrentSessionTranscriptSection,
  addCurrentUserMessageSection,
} from "./sections/current-session.js";
import { addDiscourseStateSection } from "./sections/discourse-state.js";
import { addEpisodesSection } from "./sections/episodes.js";
import { addGroupChannelMemorySection } from "./sections/group-channel-memory.js";
import { addOpenQuestionsSection } from "./sections/open-questions.js";
import { addContradictionsAndQuarantinesSection } from "./sections/quarantines.js";
import { addRelationalSlotsSection } from "./sections/relational-slots.js";
import {
  addRetrievedRawStreamEvidenceSection,
  addRetrievedStructuredEvidenceSection,
} from "./sections/retrieved-evidence.js";
import { addSemanticGraphSection } from "./sections/semantic-graph.js";
import { estimateLedgerTokens } from "./trace-summary.js";
import { compactTranscriptEntries } from "./transcript-compaction.js";
import type { EvidenceLedger } from "./types.js";

export type {
  EvidenceLedgerBuildInput,
  EvidenceLedgerBuilderOptions,
} from "./builder-types.js";
export { summarizeEvidenceLedgerTrace } from "./trace-summary.js";

export class EvidenceLedgerBuilder {
  constructor(private readonly options: EvidenceLedgerBuilderOptions) {}

  async build(input: EvidenceLedgerBuildInput): Promise<EvidenceLedger> {
    const streamEntries = await loadSessionStreamEntries(
      this.options.createStreamReader(input.sessionId),
    );
    const streamEntriesById = new Map<string, StreamEntry>();
    const streamOrderById = new Map<string, number>();

    for (const [index, entry] of streamEntries.entries()) {
      streamOrderById.set(entry.id, index);
    }

    if (
      input.currentUserEntry !== undefined &&
      input.currentUserEntry.session_id === input.sessionId &&
      !streamOrderById.has(input.currentUserEntry.id)
    ) {
      streamOrderById.set(input.currentUserEntry.id, streamOrderById.size);
    }

    for (const entry of [
      ...streamEntries,
      ...input.retrievedEpisodes.flatMap((result) => result.citationChain),
    ]) {
      streamEntriesById.set(entry.id, entry);
    }

    const resolverBase = {
      currentSessionId: input.sessionId,
      streamEntriesById,
      streamOrderById,
    };
    const episodeScopesById = buildEpisodeScopeMap(input.retrievedEpisodes, resolverBase);
    const episodeSourceStreamIdsById = buildEpisodeSourceStreamIdMap(input.retrievedEpisodes);
    const resolver: ScopeResolver = {
      ...resolverBase,
      episodeScopesById,
      episodeSourceStreamIdsById,
    };
    const transcriptEntries = activeSessionTranscriptEntries(streamEntries);
    const transcript = compactTranscriptEntries({
      entries: transcriptEntries,
      budget: this.options.currentSessionTranscriptTokenBudget,
      currentUserEntryId: input.currentUserEntry?.id,
      resolver,
      entityRepository: this.options.entityRepository,
    });
    const sections = createSectionBuckets();
    const context: BuilderSectionContext = {
      input,
      resolver,
      buckets: sections,
      options: {
        actionThreadRenderLimit: this.options.actionThreadRenderLimit,
        actionThreadSimilarityThreshold: this.options.actionThreadSimilarityThreshold,
        actionThreadSourceRecordLimit: this.options.actionThreadSourceRecordLimit,
      },
      transcript,
      streamEntries,
      repos: {
        relationalSlots: this.options.relationalSlotRepository,
        actions: this.options.actionRepository,
        commitments: this.options.commitmentRepository,
        goals: this.options.goalsRepository,
        openQuestions: this.options.openQuestionsRepository,
        entities: this.options.entityRepository,
      },
    };

    addCurrentUserMessageSection(context);
    addCurrentSessionTranscriptSection(context);
    addCommitmentsAndConstraintsSection(context);
    addDiscourseStateSection(context);
    addContradictionsAndQuarantinesSection(context);
    await addActionStatesSection(context);
    addGroupChannelMemorySection(context);
    addRelationalSlotsSection(context);
    // Sprint 8d.6.3: stream IDs covered by the current_session_transcript
    // section don't need to be re-rendered as retrieved_raw_stream_evidence.
    // The same underlying entry's text was duplicated across both sections
    // (~25k tokens on heavy v37 turns) because dedupe only matched on
    // rendered ledger entry IDs, not provenance stream IDs.
    addRetrievedRawStreamEvidenceSection(context);
    addRetrievedStructuredEvidenceSection(context);
    addEpisodesSection(context);
    addSemanticGraphSection(context);
    addOpenQuestionsSection(context);

    const orderedSections = finalSections(sections);

    return {
      sections: orderedSections,
      transcriptIncluded: true,
      transcriptCompacted: transcript.compacted,
      originalTranscriptTokenEstimate: transcript.originalTokenEstimate,
      compactedTranscriptEntryCount: transcript.compactedEntryCount,
      rawPreservedUserTranscriptEntryCount: transcript.rawPreservedUserEntryCount,
      estimatedTokens: estimateLedgerTokens(orderedSections),
    };
  }
}
