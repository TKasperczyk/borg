import type { TurnOrigin } from "../../contracts/cognitive-contracts.js";
import type { StreamEntryIndexRepository, StreamWriter } from "../../stream/index.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import { StorageError, ToolError } from "../../util/errors.js";
import { episodeIdHelpers, type SessionId } from "../../util/ids.js";
import {
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromEpisodeAccess,
  renderMemoryDisclosureLabelForModel,
  resolveDisclosureLabelsByEpisodeId,
} from "../common/disclosure-label.js";
import { semanticNodeMemoryDisclosureLabel } from "../common/disclosure-serializers.js";
import { annotationPromptFields } from "../common/episode-evidence.js";
import type { EpisodicRepository } from "../episodic/repository.js";
import {
  LEDGER_ANNOTATION_OPTIONS,
  RECORD_CORRECTION_TOOL_NAME,
  annotationTargetKey,
  recordCorrectionInputSchema,
  recordCorrectionMetadataSchema,
  recordCorrectionResultSchema,
  recordCorrectionTextHash,
  type RecordCorrectionInput,
  type RecordCorrectionResult,
} from "../identity/annotations.js";
import type { IdentityEventRepository } from "../identity/repository.js";
import type { SemanticNodeRepository } from "../semantic/repository.js";
import {
  correctionEvidenceKey,
  MAX_RESOLUTION_CORRECTIONS,
} from "../self/resolution-corrections.js";
import { prepareCorrectionEvidence } from "./correct-resolution.js";

type RecordCorrectionContext = {
  turnId?: string;
  sessionId?: SessionId;
  turnOrigin: TurnOrigin;
  path: "tool" | "reflector";
  streamWriter: Pick<StreamWriter, "append">;
};

export const RECORD_CORRECTION_IDENTITY_FAILED = "RECORD_CORRECTION_IDENTITY_FAILED";

export class RecordCorrectionService {
  private readonly pending = new Map<string, Promise<void>>();

  constructor(
    private readonly options: {
      episodicRepository: Pick<EpisodicRepository, "get" | "getMany">;
      semanticNodeRepository: Pick<SemanticNodeRepository, "get">;
      entryIndex: Pick<StreamEntryIndexRepository, "lookupEntriesById">;
      identityEventRepository: IdentityEventRepository;
      tracer?: TurnTracer;
    },
  ) {}

  async correct(
    input: RecordCorrectionInput,
    context: RecordCorrectionContext,
  ): Promise<RecordCorrectionResult> {
    const parsed = recordCorrectionInputSchema.parse(input);
    const key = parsed.target_id;
    const prior = this.pending.get(key);
    let release!: () => void;
    const done = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.pending.set(key, done);
    try {
      await prior;
      return await this.appendCorrection(parsed, context);
    } finally {
      release();
      if (this.pending.get(key) === done) this.pending.delete(key);
    }
  }

  private async appendCorrection(input: RecordCorrectionInput, context: RecordCorrectionContext) {
    if (!context.turnId) {
      throw new ToolError("A record correction requires a turn id", {
        code: "RECORD_CORRECTION_TURN_REQUIRED",
      });
    }
    const { episodicRepository, semanticNodeRepository, identityEventRepository } = this.options;
    const { writer, disclosureLabel } = await prepareCorrectionEvidence({
      ...context,
      episodicRepository,
      entryIndex: this.options.entryIndex,
      evidence: input.evidence,
      process: context.path === "tool" ? RECORD_CORRECTION_TOOL_NAME : "reflector",
      missingCode: "RECORD_CORRECTION_EVIDENCE_NOT_FOUND",
    });
    const target = episodeIdHelpers.is(input.target_id)
      ? { record_type: "episode" as const, record_id: input.target_id }
      : { record_type: "semantic_node" as const, record_id: input.target_id };
    const nodeForSources =
      target.record_type === "semantic_node"
        ? await semanticNodeRepository.get(target.record_id)
        : null;
    const sourceLabels = await resolveDisclosureLabelsByEpisodeId(
      nodeForSources?.source_episode_ids ?? [],
      (ids) => episodicRepository.getMany(ids),
    );
    const history = identityEventRepository.listAnnotationsForTargets([target], {
      includeRetracted: true,
    })[0]!.annotations;
    const corrections = history.flatMap((row) =>
      row.self_correction === undefined ? [] : [{ row, details: row.self_correction }],
    );
    const targetSeq = input.targets_correction_seq ?? null;
    const earlier =
      targetSeq === null ? undefined : corrections.find(({ details }) => details.seq === targetSeq);
    if (targetSeq !== null && earlier === undefined) {
      throw new ToolError("The targeted correction sequence does not exist on this record", {
        code: "RECORD_CORRECTION_SEQUENCE_NOT_FOUND",
      });
    }
    const evidenceKey = correctionEvidenceKey(input.evidence);
    const evidence = [
      ...new Map(input.evidence.map((item) => [JSON.stringify(item), item])).values(),
    ];
    // Bind to a fresh read after evidence and disclosure lookup. New source episodes whose
    // disclosure could not be resolved from the preliminary read are labeled unknown.
    const record =
      target.record_type === "episode"
        ? await episodicRepository.get(target.record_id, { includeArchived: true })
        : await semanticNodeRepository.get(target.record_id);
    if (record === null) {
      throw new ToolError(`No stored record at ${input.target_id}`, {
        code: "RECORD_CORRECTION_TARGET_NOT_FOUND",
      });
    }
    const currentText = "narrative" in record ? record.narrative : record.description;
    const quotedText = earlier?.row.text ?? currentText;
    // Literal quotation integrity only. Keep both the correction and quote verbatim.
    if (input.corrects_span !== undefined && !quotedText.includes(input.corrects_span)) {
      throw new ToolError("The quoted span does not occur verbatim in the target text", {
        code: "RECORD_CORRECTION_SPAN_NOT_FOUND",
      });
    }

    const targetTextHash = recordCorrectionTextHash(currentText);
    const repeated = corrections.find(
      ({ row, details }) =>
        row.text === input.correction_note &&
        details.target_text_hash === targetTextHash &&
        details.corrects_span === input.corrects_span &&
        details.targets_correction_seq === targetSeq &&
        correctionEvidenceKey(details.evidence) === evidenceKey,
    );
    let streamEntryId = repeated?.row.stream_entry_id;
    let createdAt = repeated?.row.written_at;
    let metadata = repeated?.details;

    if (repeated === undefined) {
      if (corrections.length >= MAX_RESOLUTION_CORRECTIONS) {
        throw new ToolError(`This record already has ${MAX_RESOLUTION_CORRECTIONS} corrections`, {
          code: "RECORD_CORRECTION_LIMIT",
        });
      }
      const seq = Math.max(0, ...corrections.map(({ details }) => details.seq)) + 1;
      const targetLabel =
        "narrative" in record
          ? memoryDisclosureLabelFromEpisodeAccess(record)
          : semanticNodeMemoryDisclosureLabel(sourceLabels, record);
      const { seq: _seq, ...details } = recordCorrectionMetadataSchema.parse({
        seq,
        writer,
        target_text_hash: targetTextHash,
        ...(earlier === undefined || input.corrects_span === undefined
          ? {}
          : { quoted_text_hash: recordCorrectionTextHash(quotedText) }),
        targets_correction_seq: targetSeq,
        ...(input.corrects_span === undefined ? {} : { corrects_span: input.corrects_span }),
        evidence,
        disclosure_label: combineMemoryDisclosureLabels([
          targetLabel,
          disclosureLabel,
          ...(earlier === undefined ? [] : [earlier.details.disclosure_label]),
        ]),
      });
      const annotation = {
        event: "episode_annotation" as const,
        target_ids: [target.record_id],
        correction: input.correction_note,
      };
      // Only committed identity rows claim a sequence. An orphaned stream entry has none.
      const entry = await context.streamWriter.append({
        kind: "internal_event",
        turn_id: context.turnId,
        content: { ...annotation, self_correction: details },
      });
      const committed = { ...details, seq };
      try {
        identityEventRepository.runInTransaction(() =>
          identityEventRepository.record({
            ...target,
            action: "record_self_correction",
            old_value: null,
            new_value: {
              stream_entry_id: entry.id,
              annotation: { ...annotation, self_correction: committed },
            },
            provenance: writer.provenance,
            ts: entry.timestamp,
          }),
        );
      } catch (cause) {
        if (this.options.tracer?.enabled === true) {
          this.options.tracer.emit("memory.annotations.write_failed", {
            turnId: context.turnId,
            ...(context.sessionId === undefined ? {} : { session_id: context.sessionId }),
            target: annotationTargetKey(target),
            stream_entry_id: entry.id,
            reason: "identity_append_failed",
          });
        }
        throw new StorageError(
          "Correction stream entry was appended but its identity event could not be recorded",
          {
            code: RECORD_CORRECTION_IDENTITY_FAILED,
            cause,
          },
        );
      }
      streamEntryId = entry.id;
      createdAt = entry.timestamp;
      metadata = { ...committed, original_record: target };
    }

    const page = annotationPromptFields(
      identityEventRepository.listAnnotationsForTargets([target])[0]!.annotations,
      {
        ...LEDGER_ANNOTATION_OPTIONS,
        currentText,
      },
    );
    return recordCorrectionResultSchema.parse({
      correction: {
        ...metadata!,
        text: input.correction_note,
        stream_entry_id: streamEntryId,
        created_at: createdAt,
        created_at_iso: new Date(createdAt!).toISOString(),
        disclosure: renderMemoryDisclosureLabelForModel(metadata!.disclosure_label),
      },
      seq: metadata!.seq,
      total_count: page.self_correction_count ?? 0,
      shown_count: page.annotations?.filter((row) => row.author === "borg").length ?? 0,
    });
  }
}
