import type { ActionRepository } from "../../memory/actions/index.js";
import type { EntityRepository } from "../../memory/commitments/index.js";
import type { RelationalSlotRepository } from "../../memory/relational-slots/index.js";
import { parseActionId, parseRelationalSlotId, type EntityId } from "../../util/ids.js";
import { valueAppearsIn } from "../../util/text-presence.js";
import type { EvidenceLedger, EvidenceLedgerEntry } from "../evidence-ledger/index.js";
import { toTraceJsonValue, type TurnTracer } from "../tracing/tracer.js";
import type { EmitManifestResponse, EvidenceRef, ManifestClaim } from "./manifest-schema.js";

const RENDERED_SPAN_MISSING_REASON = "rendered_span does not appear in final_text" as const;

export type ManifestValidationFailedClaim = {
  claim_index: number;
  kind: ManifestClaim["kind"];
  rendered_span: string;
  reasons: string[];
  claim: ManifestClaim;
};

export type ManifestValidationWouldHaveVerdict =
  | "passed"
  | "would_have_rewritten"
  | "would_have_suppressed";

export type ManifestValidationResult = {
  passed_claims: number;
  failed_claims: ManifestValidationFailedClaim[];
  phantom_claims: ManifestValidationFailedClaim[];
  would_have_verdict: ManifestValidationWouldHaveVerdict;
};

export type ManifestValidatorOptions = {
  slotRepository: Pick<RelationalSlotRepository, "get">;
  actionRepository: Pick<ActionRepository, "get">;
  entityRepository?: Pick<EntityRepository, "get">;
  tracer?: TurnTracer;
};

export type ManifestValidatorInput = {
  manifest: EmitManifestResponse;
  evidenceLedger: EvidenceLedger;
  userEntryId?: string;
  audienceEntityId?: EntityId | null;
  turnId?: string;
};

type ResolvedEvidence = {
  ref: EvidenceRef;
  entry: EvidenceLedgerEntry | null;
};

const DETERMINISTICALLY_VALIDATED_CLAIM_KINDS = [
  "user_fact",
  "slot_fact",
  "action_state",
  "prior_callback",
  "agent_self_provenance",
] as const satisfies readonly ManifestClaim["kind"][];

const ACCEPTED_UNVALIDATED_CLAIM_KINDS = [
  "self_report",
  "interpretation",
  "hedge",
  "discourse_only",
] as const satisfies readonly ManifestClaim["kind"][];

const SELF_REPORT_BLOCKED_GROUNDING_KINDS = [
  "user_fact",
  "slot_fact",
  "action_state",
  "prior_callback",
] as const satisfies readonly ManifestClaim["kind"][];

const SELF_REPORT_BLOCKED_GROUNDING_KIND_SET: ReadonlySet<ManifestClaim["kind"]> = new Set(
  SELF_REPORT_BLOCKED_GROUNDING_KINDS,
);

const GROUNDING_REQUIRED_CLAIM_KINDS = [
  "user_fact",
  "slot_fact",
  "action_state",
  "prior_callback",
  "agent_self_provenance",
] as const satisfies readonly ManifestClaim["kind"][];

const GROUNDING_REQUIRED_CLAIM_KIND_SET: ReadonlySet<ManifestClaim["kind"]> = new Set(
  GROUNDING_REQUIRED_CLAIM_KINDS,
);

function evidenceRefsForClaim(claim: ManifestClaim): readonly EvidenceRef[] {
  switch (claim.kind) {
    case "user_fact":
    case "prior_callback":
    case "action_state":
    case "slot_fact":
    case "agent_self_provenance":
    case "interpretation":
      return claim.evidence;
    case "discourse_only":
    case "self_report":
    case "hedge":
      return [];
  }
}

function claimEvidenceRefs(claim: ManifestClaim): readonly EvidenceRef[] {
  return "evidence" in claim ? claim.evidence : [];
}

function scopeDisclosureSpan(claim: ManifestClaim): string | undefined {
  return "scope_disclosure_span" in claim ? claim.scope_disclosure_span : undefined;
}

function ledgerEntriesById(ledger: EvidenceLedger): Map<string, EvidenceLedgerEntry> {
  const entries = new Map<string, EvidenceLedgerEntry>();

  for (const entry of ledger.sections.flatMap((section) => section.entries)) {
    entries.set(entry.id, entry);
  }

  return entries;
}

function scopeDisclosureEvidenceEntries(input: {
  claim: ManifestClaim;
  entriesById: ReadonlyMap<string, EvidenceLedgerEntry>;
}): EvidenceLedgerEntry[] {
  if (!("evidence" in input.claim)) {
    return [];
  }

  return input.claim.evidence
    .map((ref) => input.entriesById.get(ref.id) ?? null)
    .filter((entry): entry is EvidenceLedgerEntry => entry !== null);
}

function resolveEvidence(input: {
  claim: ManifestClaim;
  entriesById: ReadonlyMap<string, EvidenceLedgerEntry>;
  reasons: string[];
}): ResolvedEvidence[] {
  return evidenceRefsForClaim(input.claim).map((ref) => {
    const entry = input.entriesById.get(ref.id) ?? null;

    if (entry === null) {
      input.reasons.push("claim_cites_unknown_evidence");
    }

    return {
      ref,
      entry,
    };
  });
}

function validateOpenQuestionLifecycle(input: {
  claim: ManifestClaim;
  entriesById: ReadonlyMap<string, EvidenceLedgerEntry>;
  reasons: string[];
}): void {
  for (const ref of claimEvidenceRefs(input.claim)) {
    if (!ref.id.startsWith("open_question:")) {
      continue;
    }

    const entry = input.entriesById.get(ref.id);

    if (entry?.state === "resolved") {
      input.reasons.push("claim_cites_resolved_open_question");
    } else if (entry?.state === "abandoned") {
      input.reasons.push("claim_cites_abandoned_open_question");
    }
  }
}

function resolvedEntries(evidence: readonly ResolvedEvidence[]): EvidenceLedgerEntry[] {
  return evidence
    .map((item) => item.entry)
    .filter((entry): entry is EvidenceLedgerEntry => entry !== null);
}

function evidenceTextOrValue(entry: EvidenceLedgerEntry): string {
  return entry.text || entry.value || "";
}

function evidenceCanSupportUserFact(entry: EvidenceLedgerEntry): boolean {
  return entry.taint === undefined || entry.taint === "none";
}

function describeEvidenceTaint(entry: EvidenceLedgerEntry): string {
  return entry.taint ?? "none";
}

function validateEvidenceUntainted(input: {
  entries: readonly EvidenceLedgerEntry[];
  reasonPrefix: string;
  reasons: string[];
}): void {
  for (const entry of input.entries) {
    if (evidenceCanSupportUserFact(entry)) {
      continue;
    }

    input.reasons.push(
      `${input.reasonPrefix}_cites_tainted_evidence: ${entry.id} taint=${describeEvidenceTaint(entry)}`,
    );
  }
}

function validateNotGroundedInSelfReport(input: {
  claim: ManifestClaim;
  entries: readonly EvidenceLedgerEntry[];
  reasons: string[];
}): void {
  if (!SELF_REPORT_BLOCKED_GROUNDING_KIND_SET.has(input.claim.kind)) {
    return;
  }

  if (input.entries.some((entry) => entry.persistence_class === "assistant_self_report")) {
    input.reasons.push("claim_grounded_in_self_report");
  }
}

function requireScopeDisclosure(input: {
  claim: ManifestClaim;
  finalText: string;
  reasons: string[];
}): void {
  const span = scopeDisclosureSpan(input.claim);

  if (span === undefined || span.trim().length === 0) {
    input.reasons.push("prior-session evidence requires scope_disclosure_span");
    return;
  }

  if (!valueAppearsIn(input.finalText, span)) {
    input.reasons.push("scope_disclosure_span does not appear in final_text");
  }
}

function currentUserStreamIndex(input: {
  entriesById: ReadonlyMap<string, EvidenceLedgerEntry>;
  userEntryId: string | undefined;
}): number | undefined {
  if (input.userEntryId === undefined) {
    return undefined;
  }

  return input.entriesById.get(`current_user_message:${input.userEntryId}`)?.stream_index;
}

function hasPriorCurrentSessionStreamEvidence(input: {
  entries: readonly EvidenceLedgerEntry[];
  currentUserStreamIndex: number | undefined;
}): boolean {
  if (input.currentUserStreamIndex === undefined) {
    return false;
  }

  const userStreamIndex = input.currentUserStreamIndex;

  return input.entries.some((entry) => {
    if (entry.source_type !== "current_session_stream") {
      return false;
    }

    return entry.stream_index !== undefined && entry.stream_index < userStreamIndex;
  });
}

function agentSelfProvenanceSourceAllowed(entry: EvidenceLedgerEntry): boolean {
  if (entry.source_type === "assistant_stream" || entry.source_type === "system_metadata") {
    return true;
  }

  return (
    (entry.source_type === "current_session_stream" ||
      entry.source_type === "prior_session_stream") &&
    entry.actor === "assistant"
  );
}

function failedClaimsByKind(
  failedClaims: readonly ManifestValidationFailedClaim[],
): Record<string, number> {
  const counts: Record<string, number> = {};

  for (const failed of failedClaims) {
    counts[failed.kind] = (counts[failed.kind] ?? 0) + 1;
  }

  return counts;
}

function failedClaimReasons(failedClaims: readonly ManifestValidationFailedClaim[]): string[] {
  return failedClaims.flatMap((failed) =>
    failed.reasons.map((reason) => `${failed.kind}:${reason}`),
  );
}

function passedClaimsByKind(
  claims: readonly ManifestClaim[],
  invalidClaims: readonly ManifestValidationFailedClaim[],
  kinds: readonly ManifestClaim["kind"][],
): Record<string, number> {
  const invalidClaimIndexes = new Set(
    invalidClaims
      .map((failed) => failed.claim_index)
      .filter((index): index is number => index >= 0),
  );
  const countedKinds = new Set<ManifestClaim["kind"]>(kinds);
  const counts: Record<string, number> = {};

  for (const [index, claim] of claims.entries()) {
    if (invalidClaimIndexes.has(index) || !countedKinds.has(claim.kind)) {
      continue;
    }

    counts[claim.kind] = (counts[claim.kind] ?? 0) + 1;
  }

  return counts;
}

function previewText(text: string, maxLength = 500): string {
  return text.length <= maxLength ? text : `${text.slice(0, maxLength)}...`;
}

function reasonIsRealSafetyProblem(reason: string): boolean {
  return (
    reason === "claim_cites_unknown_evidence" ||
    reason === "claim_grounding_evidence_empty" ||
    reason === "claim_grounded_in_self_report" ||
    reason.startsWith("final_text_uses_non_speakable_name:") ||
    reason.startsWith("exact_value_only_in_tainted_evidence:") ||
    reason.includes("_cites_tainted_evidence:") ||
    reason.startsWith("invalid action record id:") ||
    reason.startsWith("action record not found:") ||
    reason.startsWith("action state mismatch:") ||
    reason.startsWith("agent self-provenance cites unsupported evidence source:")
  );
}

function claimHasRealSafetyProblem(failed: ManifestValidationFailedClaim): boolean {
  return failed.reasons.some(reasonIsRealSafetyProblem);
}

function wouldHaveVerdict(
  failedClaims: readonly ManifestValidationFailedClaim[],
): ManifestValidationWouldHaveVerdict {
  if (failedClaims.length === 0) {
    return "passed";
  }

  return failedClaims.some(claimHasRealSafetyProblem)
    ? "would_have_suppressed"
    : "would_have_rewritten";
}

export class ManifestValidator {
  constructor(private readonly options: ManifestValidatorOptions) {}

  async validate(input: ManifestValidatorInput): Promise<ManifestValidationResult> {
    const entriesById = ledgerEntriesById(input.evidenceLedger);
    const userStreamIndex = currentUserStreamIndex({
      entriesById,
      userEntryId: input.userEntryId,
    });
    const failedClaims: ManifestValidationFailedClaim[] = [];

    for (const [claimIndex, claim] of input.manifest.claims.entries()) {
      const reasons = await this.validateClaim({
        claim,
        finalText: input.manifest.final_text,
        entriesById,
        currentUserStreamIndex: userStreamIndex,
      });

      if (reasons.length > 0) {
        failedClaims.push({
          claim_index: claimIndex,
          kind: claim.kind,
          rendered_span: claim.rendered_span,
          reasons,
          claim,
        });
      }
    }

    if (this.options.entityRepository !== undefined && input.audienceEntityId !== null) {
      const audienceEntity =
        input.audienceEntityId === undefined
          ? null
          : await this.options.entityRepository.get(input.audienceEntityId);
      const provenance = audienceEntity?.name_provenance ?? "unknown";
      const nameSpeakable = provenance === "user_declared" || provenance === "user_confirmed";

      if (audienceEntity !== null && !nameSpeakable) {
        let nameAlreadyFlagged = false;

        for (const [claimIndex, claim] of input.manifest.claims.entries()) {
          if (claim.addresses_audience_by_name !== true) {
            continue;
          }

          failedClaims.push({
            claim_index: claimIndex,
            kind: claim.kind,
            rendered_span: claim.rendered_span,
            reasons: [`final_text_uses_non_speakable_name: ${audienceEntity.canonical_name}`],
            claim,
          });
          nameAlreadyFlagged = true;
        }

        // Final-text scan independent of the manifest's
        // addresses_audience_by_name flag. v36 surfaced "Monday-Tom" leaks
        // where the audience handle appeared in final_text without any
        // claim setting the flag -- the manifest validator passed because
        // it relied on the model's self-report. Treat the audience name as
        // an internal handle and reject any literal occurrence in
        // final_text when provenance is restrictive, regardless of which
        // claim covers that span.
        if (!nameAlreadyFlagged && valueAppearsIn(input.manifest.final_text, audienceEntity.canonical_name)) {
          failedClaims.push({
            claim_index: -1,
            kind: "discourse_only",
            rendered_span: audienceEntity.canonical_name,
            reasons: [`final_text_uses_non_speakable_name: ${audienceEntity.canonical_name}`],
            claim: {
              kind: "discourse_only",
              rendered_span: audienceEntity.canonical_name,
            },
          });
        }
      }
    }

    const actualFailedIndexes = new Set(
      failedClaims
        .map((failed) => failed.claim_index)
        .filter((index): index is number => index >= 0),
    );
    const passedClaims = input.manifest.claims.length - actualFailedIndexes.size;

    if (failedClaims.length === 0) {
      const result: ManifestValidationResult = {
        passed_claims: passedClaims,
        failed_claims: [],
        phantom_claims: [],
        would_have_verdict: "passed",
      };
      this.trace(input, result);
      return result;
    }

    // Phantom claims are manifest anomalies: the model declared a claim
    // about prose that is not present in the response. They are reported
    // separately because the tracer no longer mutates final_text.
    const phantomClaims = failedClaims.filter((failed) =>
      failed.reasons.includes(RENDERED_SPAN_MISSING_REASON),
    );
    const realFailedClaims = failedClaims.filter(
      (failed) => !failed.reasons.includes(RENDERED_SPAN_MISSING_REASON),
    );

    const result: ManifestValidationResult = {
      passed_claims: passedClaims,
      failed_claims: realFailedClaims,
      phantom_claims: phantomClaims,
      would_have_verdict: wouldHaveVerdict(realFailedClaims),
    };
    this.trace(input, result);
    return result;
  }

  private async validateClaim(input: {
    claim: ManifestClaim;
    finalText: string;
    entriesById: ReadonlyMap<string, EvidenceLedgerEntry>;
    currentUserStreamIndex: number | undefined;
  }): Promise<string[]> {
    const reasons: string[] = [];

    if (!valueAppearsIn(input.finalText, input.claim.rendered_span)) {
      reasons.push("rendered_span does not appear in final_text");
    }

    validateOpenQuestionLifecycle({
      claim: input.claim,
      entriesById: input.entriesById,
      reasons,
    });

    const evidence = resolveEvidence({
      claim: input.claim,
      entriesById: input.entriesById,
      reasons,
    });
    const entries = resolvedEntries(evidence);

    if (GROUNDING_REQUIRED_CLAIM_KIND_SET.has(input.claim.kind) && entries.length === 0) {
      reasons.push("claim_grounding_evidence_empty");
    }

    validateNotGroundedInSelfReport({
      claim: input.claim,
      entries,
      reasons,
    });

    const scopeEntries = scopeDisclosureEvidenceEntries({
      claim: input.claim,
      entriesById: input.entriesById,
    });

    if (scopeEntries.some((entry) => entry.session_scope === "prior_session")) {
      requireScopeDisclosure({
        claim: input.claim,
        finalText: input.finalText,
        reasons,
      });
    }

    switch (input.claim.kind) {
      case "user_fact":
        this.validateUserFactEvidence({
          claim: input.claim,
          entries,
          reasons,
        });
        break;
      case "slot_fact":
        await this.validateSlotFact({
          claim: input.claim,
          reasons,
        });
        break;
      case "action_state":
        await this.validateActionState({
          claim: input.claim,
          reasons,
        });
        break;
      case "prior_callback":
        this.validatePriorCallback({
          claim: input.claim,
          entries,
          currentUserStreamIndex: input.currentUserStreamIndex,
          reasons,
        });
        break;
      case "agent_self_provenance":
        this.validateAgentSelfProvenance({
          entries,
          reasons,
        });
        break;
      case "discourse_only":
      case "self_report":
      case "interpretation":
      case "hedge":
        break;
    }

    return reasons;
  }

  private validateExactValues(input: {
    claim: Extract<ManifestClaim, { kind: "user_fact" }>;
    entries: readonly EvidenceLedgerEntry[];
    reasons: string[];
  }): void {
    for (const value of input.claim.exact_values) {
      const supportingEntries = input.entries.filter((entry) =>
        valueAppearsIn(evidenceTextOrValue(entry), value),
      );

      if (supportingEntries.length === 0) {
        input.reasons.push(`exact value does not appear in cited evidence: ${value}`);
      } else if (!supportingEntries.some((entry) => evidenceCanSupportUserFact(entry))) {
        input.reasons.push(`exact_value_only_in_tainted_evidence: ${value}`);
      }
    }
  }

  private validateUserFactEvidence(input: {
    claim: Extract<ManifestClaim, { kind: "user_fact" }>;
    entries: readonly EvidenceLedgerEntry[];
    reasons: string[];
  }): void {
    this.validateExactValues({
      claim: input.claim,
      entries: input.entries,
      reasons: input.reasons,
    });
  }

  private async validateSlotFact(input: {
    claim: Extract<ManifestClaim, { kind: "slot_fact" }>;
    reasons: string[];
  }): Promise<void> {
    let slotId: ReturnType<typeof parseRelationalSlotId>;

    try {
      slotId = parseRelationalSlotId(input.claim.slot_id);
    } catch {
      input.reasons.push(`invalid relational slot id: ${input.claim.slot_id}`);
      return;
    }

    const slot = await this.options.slotRepository.get(slotId);

    if (slot === null) {
      input.reasons.push(`relational slot not found: ${input.claim.slot_id}`);
      return;
    }

    if (slot.state !== "established") {
      input.reasons.push(`relational slot is not established: ${slot.state}`);
    }
  }

  private async validateActionState(input: {
    claim: Extract<ManifestClaim, { kind: "action_state" }>;
    reasons: string[];
  }): Promise<void> {
    let actionId: ReturnType<typeof parseActionId>;

    try {
      actionId = parseActionId(input.claim.action_record_id);
    } catch {
      input.reasons.push(`invalid action record id: ${input.claim.action_record_id}`);
      return;
    }

    const action = await this.options.actionRepository.get(actionId);

    if (action === null) {
      input.reasons.push(`action record not found: ${input.claim.action_record_id}`);
      return;
    }

    if (action.state !== input.claim.asserted_state) {
      input.reasons.push(
        `action state mismatch: manifest=${input.claim.asserted_state} record=${action.state}`,
      );
    }
  }

  private validatePriorCallback(input: {
    claim: Extract<ManifestClaim, { kind: "prior_callback" }>;
    entries: readonly EvidenceLedgerEntry[];
    currentUserStreamIndex: number | undefined;
    reasons: string[];
  }): void {
    validateEvidenceUntainted({
      entries: input.entries,
      reasonPrefix: "prior_callback",
      reasons: input.reasons,
    });

    if (input.claim.callback_scope === "current_turn") {
      return;
    }

    if (input.claim.callback_scope === "current_session_prior") {
      if (input.currentUserStreamIndex === undefined) {
        input.reasons.push("current_session_prior callback requires current user stream order");
        return;
      }

      if (
        !hasPriorCurrentSessionStreamEvidence({
          entries: input.entries,
          currentUserStreamIndex: input.currentUserStreamIndex,
        })
      ) {
        input.reasons.push(
          "current_session_prior callback lacks prior current-session stream evidence",
        );
      }
      return;
    }

    if (input.claim.callback_scope === "prior_session") {
      if (!input.entries.some((entry) => entry.session_scope === "prior_session")) {
        input.reasons.push("prior_session callback lacks prior-session evidence");
      }
    }
  }

  private validateAgentSelfProvenance(input: {
    entries: readonly EvidenceLedgerEntry[];
    reasons: string[];
  }): void {
    validateEvidenceUntainted({
      entries: input.entries,
      reasonPrefix: "agent_self_provenance",
      reasons: input.reasons,
    });

    for (const entry of input.entries) {
      if (!agentSelfProvenanceSourceAllowed(entry)) {
        input.reasons.push(
          `agent self-provenance cites unsupported evidence source: ${entry.source_type}`,
        );
      }
    }
  }

  private trace(input: ManifestValidatorInput, result: ManifestValidationResult): void {
    if (this.options.tracer?.enabled !== true || input.turnId === undefined) {
      return;
    }

    const invalidClaims = [...result.failed_claims, ...result.phantom_claims];
    const literalValuesValidatedByKind = passedClaimsByKind(
      input.manifest.claims,
      invalidClaims,
      DETERMINISTICALLY_VALIDATED_CLAIM_KINDS,
    );
    const realSafetyReasons = failedClaimReasons(
      result.failed_claims.filter(claimHasRealSafetyProblem),
    );
    const payload = {
      turnId: input.turnId,
      verdict: result.failed_claims.length === 0 ? "valid" : "invalid",
      final_verdict: result.would_have_verdict,
      would_have_verdict: result.would_have_verdict,
      would_have_failed_under_old_regime: result.would_have_verdict !== "passed",
      real_safety_problem: realSafetyReasons.length > 0,
      passed_claims: result.passed_claims,
      // Back-compat field name from Sprint 7. For user_fact, this means the
      // literal exact_values were grounded, not that all entity bindings were
      // semantically proven.
      validated_claims_by_kind: literalValuesValidatedByKind,
      literal_values_validated_by_kind: literalValuesValidatedByKind,
      accepted_unvalidated_claims_by_kind: passedClaimsByKind(
        input.manifest.claims,
        invalidClaims,
        ACCEPTED_UNVALIDATED_CLAIM_KINDS,
      ),
      failed_claims_by_kind: failedClaimsByKind(result.failed_claims),
      failed_claim_reasons: failedClaimReasons(result.failed_claims),
      real_safety_reasons: realSafetyReasons,
      phantom_claim_count: result.phantom_claims.length,
      ...(result.phantom_claims.length === 0
        ? {}
        : {
            phantom_claims_by_kind: failedClaimsByKind(result.phantom_claims),
            phantom_claim_reasons: failedClaimReasons(result.phantom_claims),
          }),
      final_text_changed: false,
      ...(this.options.tracer.includePayloads
        ? {
            failed_claims: toTraceJsonValue(result.failed_claims),
            phantom_claims: toTraceJsonValue(result.phantom_claims),
            original_text: input.manifest.final_text,
            original_text_preview: previewText(input.manifest.final_text),
            final_text_preview: previewText(input.manifest.final_text),
          }
        : {}),
    };

    this.options.tracer.emit("manifest_validation", payload);
  }
}
