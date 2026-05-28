import {
  activeSessionTranscriptEntries,
  type StreamEntry,
  type StreamReader,
  type StreamReverseScanResult,
} from "../../stream/index.js";
import type { BuilderSectionContext } from "./builder-context.js";
import type { EvidenceLedgerBuildInput, EvidenceLedgerBuilderOptions } from "./builder-types.js";
import { createSectionBuckets, finalSections } from "./section-buckets.js";
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
import { addCrossSessionSelfActivitySection } from "./sections/cross-session-activity.js";
import {
  addAttributionMatrixSection,
  addCurrentSessionAttributionSidebarSection,
} from "./sections/attribution.js";
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
import type { EvidenceItem } from "../../retrieval/index.js";
import type { AttachmentId } from "../../util/ids.js";
import { validateImageForFinalizerRender } from "../../attachments/index.js";
import type { EvidenceLedger, EvidenceLedgerImageAttachment } from "./types.js";

export type { EvidenceLedgerBuildInput, EvidenceLedgerBuilderOptions } from "./builder-types.js";
export { summarizeEvidenceLedgerTrace } from "./trace-summary.js";

const EVIDENCE_LEDGER_SESSION_SCAN_MAX_ENTRIES = 1_024;
const EVIDENCE_LEDGER_SESSION_SCAN_MAX_BYTES = 8 * 1024 * 1024;
const DEFAULT_MAX_IMAGES_PER_LEDGER = 4;
const DEFAULT_MAX_LEDGER_IMAGE_BYTES = 8 * 1024 * 1024;
const DEFAULT_IMAGE_RENDER_MAX_DIMENSION = 8192;

function scanRecentSessionStreamEntries(reader: StreamReader): StreamReverseScanResult {
  return reader.scanReverse({
    maxEntries: EVIDENCE_LEDGER_SESSION_SCAN_MAX_ENTRIES,
    maxBytes: EVIDENCE_LEDGER_SESSION_SCAN_MAX_BYTES,
    budgetFilter: (entry) => entry.kind !== "user_image_attachment",
  });
}

export class EvidenceLedgerBuilder {
  constructor(private readonly options: EvidenceLedgerBuilderOptions) {}

  async build(input: EvidenceLedgerBuildInput): Promise<EvidenceLedger> {
    const streamScan = scanRecentSessionStreamEntries(
      this.options.createStreamReader(input.sessionId),
    );
    const streamEntries = streamScan.entries;
    if (this.options.tracer?.enabled === true && input.turnId !== undefined) {
      this.options.tracer.emit("evidence_ledger.reverse_scan", {
        turnId: input.turnId,
        ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
        ledger_reverse_scan_entries: streamEntries.length,
        ledger_reverse_scan_bytes: streamScan.scannedBytes,
        ledger_reverse_scan_entry_cap_hit: streamScan.capReached === "entries",
        ledger_reverse_scan_byte_cap_hit: streamScan.capReached === "bytes",
      });
    }

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
    const imageBudget = collectLedgerImageAttachments(input.retrievedEvidence, {
      attachmentRepository: this.options.attachmentRepository,
      maxImagesPerLedger: this.options.maxImagesPerLedger ?? DEFAULT_MAX_IMAGES_PER_LEDGER,
      maxLedgerImageBytes: this.options.maxLedgerImageBytes ?? DEFAULT_MAX_LEDGER_IMAGE_BYTES,
      imageRenderMaxDimension:
        this.options.imageRenderMaxDimension ?? DEFAULT_IMAGE_RENDER_MAX_DIMENSION,
      tracer: this.options.tracer,
      turnId: input.turnId,
      sessionId: input.sessionId,
    });
    const retrievedEvidence = applyImageBudgetAnnotations(input.retrievedEvidence, imageBudget);
    const sections = createSectionBuckets();
    const context: BuilderSectionContext = {
      input: {
        ...input,
        retrievedEvidence,
      },
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
    addCrossSessionSelfActivitySection(context);
    addCurrentSessionAttributionSidebarSection(context);
    addAttributionMatrixSection(context);
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
      ...(imageBudget.attachments.length === 0
        ? {}
        : { imageAttachments: imageBudget.attachments }),
    };
  }
}

type ImageBudgetResult = {
  attachments: EvidenceLedgerImageAttachment[];
  omittedBudget: Set<string>;
  omittedInactive: Set<string>;
};

function collectLedgerImageAttachments(
  evidence: readonly EvidenceItem[],
  options: {
    attachmentRepository?: {
      get(attachmentId: AttachmentId): {
        attachment_id: AttachmentId;
        active: boolean;
        byte_size: number;
        width: number;
        height: number;
        created_turn_global: number | null;
      } | null;
    };
    maxImagesPerLedger: number;
    maxLedgerImageBytes: number;
    imageRenderMaxDimension: number;
    tracer?: EvidenceLedgerBuilderOptions["tracer"];
    turnId?: string;
    sessionId?: EvidenceLedgerBuildInput["sessionId"];
  },
): ImageBudgetResult {
  const labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const byAttachment = new Map<
    string,
    {
      attachmentId: AttachmentId;
      label: string;
      score: number;
      byteSize: number;
      createdTurnGlobal: number;
      active: boolean;
    }
  >();
  const omittedInactive = new Set<string>();

  for (const item of evidence) {
    if (item.imageAttachmentId === undefined) {
      continue;
    }

    if (!byAttachment.has(item.imageAttachmentId)) {
      const record = options.attachmentRepository?.get(item.imageAttachmentId);
      const active = record?.active ?? true;
      if (!active) {
        omittedInactive.add(item.imageAttachmentId);
        continue;
      }
      if (record !== null && record !== undefined) {
        validateImageForFinalizerRender(record, {
          maxDimension: options.imageRenderMaxDimension,
        });
      }
      // Labels are scoped to the rendered turn. Stable cross-turn labels would
      // need durable UI state and are intentionally deferred.
      const letter = labels[byAttachment.size] ?? String(byAttachment.size + 1);
      byAttachment.set(item.imageAttachmentId, {
        attachmentId: item.imageAttachmentId,
        label: (item.imageLabel ?? "Image").replace(/^Image:/, `Image ${letter}:`),
        score: item.score,
        byteSize: record?.byte_size ?? 0,
        createdTurnGlobal: record?.created_turn_global ?? -1,
        active,
      });
    } else {
      const current = byAttachment.get(item.imageAttachmentId);
      if (current !== undefined && item.score > current.score) {
        current.score = item.score;
      }
    }
  }

  const ranked = [...byAttachment.values()]
    .filter((item) => item.active)
    .sort(
      (left, right) =>
        right.score - left.score ||
        right.createdTurnGlobal - left.createdTurnGlobal ||
        left.attachmentId.localeCompare(right.attachmentId),
    );
  const entryCapped = ranked.slice(0, options.maxImagesPerLedger);
  const omittedBudget = new Set(
    ranked.slice(options.maxImagesPerLedger).map((item) => item.attachmentId),
  );
  const byteCapped: typeof entryCapped = [];
  let bytesAttached = 0;

  for (const item of entryCapped) {
    if (bytesAttached + item.byteSize > options.maxLedgerImageBytes) {
      omittedBudget.add(item.attachmentId);
      continue;
    }
    byteCapped.push(item);
    bytesAttached += item.byteSize;
  }

  const attachments = byteCapped
    .sort((left, right) => ranked.indexOf(left) - ranked.indexOf(right))
    .map((item) => ({
      attachment_id: item.attachmentId,
      label: item.label,
      byte_size: item.byteSize,
      citation_type: "original_image" as const,
    }));

  const consideredCount = byAttachment.size + omittedInactive.size;
  if (consideredCount > 0 && options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("evidence_ledger.image_attach", {
      turnId: options.turnId,
      ...(options.sessionId !== undefined ? { session_id: options.sessionId } : {}),
      considered_count: consideredCount,
      attached_count: attachments.length,
      omitted_count: omittedBudget.size + omittedInactive.size,
      omitted_budget_count: omittedBudget.size,
      omitted_inactive_count: omittedInactive.size,
      bytes_attached: bytesAttached,
      cap_hit_entry: ranked.length > options.maxImagesPerLedger,
      cap_hit_bytes: omittedBudget.size > Math.max(0, ranked.length - options.maxImagesPerLedger),
      ledger_image_refs_considered_total: consideredCount,
      ledger_image_refs_attached_total: attachments.length,
      ledger_image_refs_omitted_budget_total: omittedBudget.size,
      ledger_image_refs_omitted_inactive_total: omittedInactive.size,
      ledger_image_bytes_attached_total: bytesAttached,
    });
    for (const attachmentId of omittedInactive) {
      options.tracer.emit("citation.image_filtered", {
        turnId: options.turnId,
        ...(options.sessionId !== undefined ? { session_id: options.sessionId } : {}),
        attachment_id: attachmentId,
        reason: "inactive",
      });
    }
  }

  return { attachments, omittedBudget, omittedInactive };
}

function applyImageBudgetAnnotations(
  evidence: readonly EvidenceItem[],
  budget: ImageBudgetResult,
): EvidenceItem[] {
  return evidence.flatMap((item) => {
    if (item.imageAttachmentId === undefined) {
      return [item];
    }

    if (budget.omittedInactive.has(item.imageAttachmentId)) {
      return [];
    }

    if (budget.omittedBudget.has(item.imageAttachmentId)) {
      return [
        {
          ...item,
          text: `${item.text}\nImage attachment unavailable this turn: ledger image budget. Use this perception text only as generated_perception_text evidence.`,
          imageUnavailableReason: "budget",
          citationType: "generated_perception_text" as const,
          imageAttachmentId: undefined,
        },
      ];
    }

    return [
      {
        ...item,
        citationType: "original_image" as const,
      },
    ];
  });
}
