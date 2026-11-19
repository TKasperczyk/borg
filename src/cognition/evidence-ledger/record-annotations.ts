import type { AnnotationTarget } from "../../memory/identity/annotations.js";
import {
  readRecordEvidenceContexts,
  recordEvidenceContext,
  RECORD_ANNOTATION_GUIDANCE,
  type RecordEvidenceReader,
} from "../../memory/common/record-evidence.js";
import { retrievedRecordTargets } from "../record-evidence.js";
import type { BuilderSectionContext } from "./builder-context.js";
import type { EvidenceLedgerBuildInput } from "./builder-types.js";
import type { EvidenceLedgerEntry, EvidenceLedgerSection } from "./types.js";

export const MAX_LEDGER_ANNOTATIONS_PER_RECORD = 2;
export const MAX_LEDGER_ANNOTATION_CHARS = 600;

export function readLedgerRecordEvidence(
  reader: RecordEvidenceReader | undefined,
  input: EvidenceLedgerBuildInput,
) {
  return readRecordEvidenceContexts(
    reader,
    [
      ...retrievedRecordTargets({
        episodes: input.retrievedEpisodes,
        semantic: input.retrievedSemantic,
        evidence: input.retrievedEvidence,
      }),
      ...(input.autobiographicalRecall?.evidence ?? []).flatMap((item) =>
        item.kind === "episode" && typeof item.metadata.episode_id === "string"
          ? [{ record_type: "episode" as const, record_id: item.metadata.episode_id }]
          : [],
      ),
    ],
    {
      maxAnnotations: MAX_LEDGER_ANNOTATIONS_PER_RECORD,
      maxChars: MAX_LEDGER_ANNOTATION_CHARS,
      wholeTextOnly: true,
    },
  );
}

export function ledgerRecordEvidenceFields(
  context: BuilderSectionContext,
  target: AnnotationTarget | undefined,
): Pick<EvidenceLedgerEntry, "evidence_context"> {
  const envelope = recordEvidenceContext(context.recordEvidence, target);
  return envelope === undefined ? {} : { evidence_context: envelope };
}

export function ledgerAnnotationCount(entry: EvidenceLedgerEntry): number {
  return (
    (entry.evidence_context?.annotations?.length ?? 0) +
    (entry.evidence_context?.annotations_omitted ?? 0)
  );
}
export function ledgerAnnotations(entry: EvidenceLedgerEntry) {
  return entry.evidence_context?.annotations ?? [];
}
export function withLedgerAnnotations(
  entry: EvidenceLedgerEntry,
  annotations: ReturnType<typeof ledgerAnnotations>,
): EvidenceLedgerEntry {
  const total = ledgerAnnotationCount(entry);
  if (total === 0) return entry;
  const { annotations: _old, annotations_omitted: _omitted, ...context } = entry.evidence_context!;
  return {
    ...entry,
    evidence_context: {
      ...context,
      ...(annotations.length === 0 ? {} : { annotations }),
      ...(annotations.length === total ? {} : { annotations_omitted: total - annotations.length }),
    },
  };
}

export function ledgerAnnotationGuidance(
  sections: readonly EvidenceLedgerSection[],
): string | null {
  return sections.some(
    (section) =>
      !section.omitted_from_prompt &&
      section.entries.some((entry) => entry.evidence_context !== undefined),
  )
    ? RECORD_ANNOTATION_GUIDANCE
    : null;
}
