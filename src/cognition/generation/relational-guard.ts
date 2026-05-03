import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { CommitmentRecord } from "../../memory/commitments/index.js";
import type { RetrievedEpisode } from "../../retrieval/index.js";
import type { StreamEntry, StreamEntryKind } from "../../stream/index.js";
import type { CommitmentId, EpisodeId, StreamEntryId } from "../../util/ids.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { PendingTurnEmission } from "./types.js";

export const RELATIONAL_CLAIM_KINDS = [
  "relational_identity",
  "callback",
  "session_scoped",
  "action_completion",
  "self_correction",
] as const;

export type RelationalClaimKind = (typeof RELATIONAL_CLAIM_KINDS)[number];

const CLAIM_AUDIT_TOOL_NAME = "EmitClaimAudit";

const relationalClaimSchema = z
  .object({
    kind: z.enum(RELATIONAL_CLAIM_KINDS),
    asserted: z.string().min(1),
    cited_stream_entry_ids: z.array(z.string().min(1)),
    cited_episode_ids: z.array(z.string().min(1)),
    cited_commitment_ids: z.array(z.string().min(1)),
    quoted_evidence_text: z.string().min(1).nullable(),
  })
  .strict();

const claimAuditSchema = z
  .object({
    claims: z.array(relationalClaimSchema),
  })
  .strict();

const CLAIM_AUDIT_TOOL = {
  name: CLAIM_AUDIT_TOOL_NAME,
  description:
    "Enumerate relational, callback, session-scoped, action-completion, and self-correction claims in the response.",
  inputSchema: toToolInputSchema(claimAuditSchema),
} satisfies LLMToolDefinition;

const CLAIM_AUDIT_SYSTEM_PROMPT = [
  "You audit a just-generated assistant response for specific claim types.",
  "Extract every claim that falls into one of these categories:",
  "- relational_identity: assertions about a person's relationship or identity, such as a partner name.",
  "- callback: assertions that the user said, mentioned, corrected, asked, or established something earlier.",
  "- session_scoped: assertions scoped to this conversation, thread, just now, or earlier in this conversation.",
  "- action_completion: assertions that an action has already been completed, booked, sent, filed, done, or carried out.",
  "- self_correction: assertions that Borg or the assistant corrected itself, was corrected by the user, or already fixed a prior mistake.",
  "Return an empty claims array when the response contains none of these claims.",
  "For every extracted claim, cite only evidence IDs present in the supplied evidence manifest.",
  "Do not infer support yourself beyond selecting cited handles. The validator will check handles deterministically.",
].join("\n");

const RELATIONAL_GUARD_REWRITE_SYSTEM_PROMPT =
  "Remove or neutralize the following specific phrases from your response. Do not change anything else. Do not introduce new information. If a sentence cannot survive removal, delete the sentence.";

export type RelationalGuardStreamEvidence = {
  entry_id: StreamEntryId;
  role: "user" | "assistant";
  kind: Extract<StreamEntryKind, "user_msg" | "agent_msg" | "agent_suppressed">;
  session_id: string;
  ts: number;
  snippet: string;
  content: string;
  compressed: boolean;
};

export type RelationalGuardCurrentUserMessage = {
  text: string;
  stream_entry_id: StreamEntryId | null;
  ts: number;
};

export type RelationalGuardEpisodeEvidence = {
  episode_id: EpisodeId;
  asOf: number;
  snippet: string;
};

export type RelationalGuardCommitmentEvidence = {
  commitment_id: CommitmentId;
  source_entry_id: StreamEntryId | null;
  summary: string;
};

export type RelationalGuardCorrectivePreferenceEvidence = {
  commitment_id: CommitmentId;
  source_entry_id: StreamEntryId;
};

export type RelationalGuardEvidenceManifest = {
  current_user_message: RelationalGuardCurrentUserMessage | null;
  current_session_stream_entries: readonly RelationalGuardStreamEvidence[];
  retrieved_episodes: readonly RelationalGuardEpisodeEvidence[];
  active_commitments: readonly RelationalGuardCommitmentEvidence[];
  corrective_preferences: readonly RelationalGuardCorrectivePreferenceEvidence[];
};

export type RelationalClaimAuditClaim = z.infer<typeof relationalClaimSchema>;

export type RelationalClaimValidation = {
  claim_id: string;
  claim: RelationalClaimAuditClaim;
  status: "valid" | "unsupported";
  reason: string;
};

export type RelationalClaimValidationSummary = {
  validations: RelationalClaimValidation[];
  unsupported: RelationalClaimValidation[];
};

export type RelationalClaimGuardResult = {
  emission: PendingTurnEmission;
  claims: RelationalClaimAuditClaim[];
  validations: RelationalClaimValidation[];
  verdict: "passed" | "rewritten" | "suppressed";
  suppressionReason?: Extract<PendingTurnEmission, { kind: "suppressed" }>["reason"];
};

export type RelationalClaimGuardOptions = {
  llmClient: LLMClient;
  auditModel: string;
  rewriteModel: string;
  tracer?: TurnTracer;
  hasCorrectivePreferenceEvidence: (entryId: StreamEntryId) => boolean;
};

export type RelationalClaimGuardInput = {
  turnId: string;
  response: string;
  evidence: RelationalGuardEvidenceManifest;
  currentSessionId: string;
  currentTurnTs: number;
};

function streamEntryContentToString(content: StreamEntry["content"]): string {
  if (typeof content === "string") {
    return content;
  }

  try {
    return JSON.stringify(content);
  } catch {
    return String(content);
  }
}

function snippet(text: string, maxLength = 700): string {
  const normalized = text.replace(/\s+/g, " ").trim();

  if (normalized.length <= maxLength) {
    return normalized;
  }

  return normalized.slice(0, maxLength);
}

export function streamEntryToRelationalGuardEvidence(
  entry: StreamEntry,
): RelationalGuardStreamEvidence | null {
  if (
    entry.kind !== "user_msg" &&
    entry.kind !== "agent_msg" &&
    entry.kind !== "agent_suppressed"
  ) {
    return null;
  }

  const content = streamEntryContentToString(entry.content);

  return {
    entry_id: entry.id,
    role: entry.kind === "user_msg" ? "user" : "assistant",
    kind: entry.kind,
    session_id: entry.session_id,
    ts: entry.timestamp,
    snippet: snippet(content),
    content,
    compressed: entry.compressed,
  };
}

export function retrievedEpisodeToRelationalGuardEvidence(
  episode: RetrievedEpisode,
): RelationalGuardEpisodeEvidence {
  return {
    episode_id: episode.episode.id,
    asOf: episode.episode.updated_at,
    snippet: snippet(episode.episode.narrative),
  };
}

export function commitmentToRelationalGuardEvidence(
  commitment: CommitmentRecord,
): RelationalGuardCommitmentEvidence {
  return {
    commitment_id: commitment.id,
    source_entry_id: commitment.source_stream_entry_ids?.[0] ?? null,
    summary: commitment.directive,
  };
}

export function correctivePreferencesFromCommitments(
  commitments: readonly CommitmentRecord[],
): RelationalGuardCorrectivePreferenceEvidence[] {
  const preferences: RelationalGuardCorrectivePreferenceEvidence[] = [];

  for (const commitment of commitments) {
    if (commitment.provenance.kind !== "online") {
      continue;
    }

    if (commitment.provenance.process !== "corrective-preference-extractor") {
      continue;
    }

    for (const sourceEntryId of commitment.source_stream_entry_ids ?? []) {
      preferences.push({
        commitment_id: commitment.id,
        source_entry_id: sourceEntryId,
      });
    }
  }

  return preferences;
}

function manifestForPrompt(evidence: RelationalGuardEvidenceManifest): unknown {
  return {
    current_user_message:
      evidence.current_user_message === null
        ? null
        : {
            text: evidence.current_user_message.text,
            stream_entry_id: evidence.current_user_message.stream_entry_id,
          },
    current_session_stream_entries: evidence.current_session_stream_entries.map((entry) => ({
      entry_id: entry.entry_id,
      role: entry.role,
      kind: entry.kind,
      ts: entry.ts,
      snippet: entry.snippet,
    })),
    retrieved_episodes: evidence.retrieved_episodes,
    active_commitments: evidence.active_commitments,
    corrective_preferences: evidence.corrective_preferences,
  };
}

function buildAuditMessages(input: {
  response: string;
  evidence: RelationalGuardEvidenceManifest;
}): LLMMessage[] {
  return [
    {
      role: "user",
      content: JSON.stringify({
        response: input.response,
        evidence_manifest: manifestForPrompt(input.evidence),
      }),
    },
  ];
}

function parseAuditResponse(result: LLMCompleteResult): RelationalClaimAuditClaim[] {
  const call = result.tool_calls.find((toolCall) => toolCall.name === CLAIM_AUDIT_TOOL_NAME);

  if (call === undefined) {
    throw new Error(`Relational claim auditor did not emit ${CLAIM_AUDIT_TOOL_NAME}`);
  }

  const parsed = claimAuditSchema.safeParse(call.input);

  if (!parsed.success) {
    throw parsed.error;
  }

  return parsed.data.claims;
}

function buildStreamEvidenceIndex(
  evidence: RelationalGuardEvidenceManifest,
): Map<string, RelationalGuardStreamEvidence> {
  const entries = new Map<string, RelationalGuardStreamEvidence>();

  for (const entry of evidence.current_session_stream_entries) {
    entries.set(entry.entry_id, entry);
  }

  const current = evidence.current_user_message;

  if (
    current !== null &&
    current.stream_entry_id !== null &&
    !entries.has(current.stream_entry_id)
  ) {
    entries.set(current.stream_entry_id, {
      entry_id: current.stream_entry_id,
      role: "user",
      kind: "user_msg",
      session_id: "",
      ts: current.ts,
      snippet: snippet(current.text),
      content: current.text,
      compressed: false,
    });
  }

  return entries;
}

function buildEpisodeEvidenceIndex(
  evidence: RelationalGuardEvidenceManifest,
): Set<string> {
  return new Set(evidence.retrieved_episodes.map((episode) => episode.episode_id));
}

function buildCommitmentEvidenceIndex(
  evidence: RelationalGuardEvidenceManifest,
): Set<string> {
  return new Set(evidence.active_commitments.map((commitment) => commitment.commitment_id));
}

function streamIdsExistInCurrentSession(input: {
  claim: RelationalClaimAuditClaim;
  streamEntries: ReadonlyMap<string, RelationalGuardStreamEvidence>;
  currentSessionId: string;
}): RelationalGuardStreamEvidence[] | string {
  const entries: RelationalGuardStreamEvidence[] = [];

  for (const entryId of input.claim.cited_stream_entry_ids) {
    const entry = input.streamEntries.get(entryId);

    if (entry === undefined) {
      return `cited stream entry ${entryId} was not found in the current session evidence`;
    }

    if (entry.session_id.length > 0 && entry.session_id !== input.currentSessionId) {
      return `cited stream entry ${entryId} is outside the current session`;
    }

    entries.push(entry);
  }

  return entries;
}

function validateQuotedEvidence(
  claim: RelationalClaimAuditClaim,
  entries: readonly RelationalGuardStreamEvidence[],
): string | null {
  if (claim.quoted_evidence_text === null || entries.length === 0) {
    return null;
  }

  if (entries.some((entry) => entry.compressed)) {
    return null;
  }

  const found = entries.some((entry) => entry.content.includes(claim.quoted_evidence_text ?? ""));

  return found ? null : "quoted evidence text does not appear verbatim in the cited stream entry";
}

function validateCallbackScope(input: {
  claim: RelationalClaimAuditClaim;
  streamEntries: ReadonlyMap<string, RelationalGuardStreamEvidence>;
  currentSessionId: string;
  currentTurnTs: number;
}): { entries: RelationalGuardStreamEvidence[]; reason: string | null } {
  if (input.claim.cited_stream_entry_ids.length === 0) {
    return {
      entries: [],
      reason: "claim has no cited current-session stream entry",
    };
  }

  const resolved = streamIdsExistInCurrentSession({
    claim: input.claim,
    streamEntries: input.streamEntries,
    currentSessionId: input.currentSessionId,
  });

  if (typeof resolved === "string") {
    return {
      entries: [],
      reason: resolved,
    };
  }

  for (const entry of resolved) {
    if (entry.kind === "agent_suppressed") {
      return {
        entries: resolved,
        reason: `cited stream entry ${entry.entry_id} is an agent_suppressed marker`,
      };
    }

    if (entry.ts >= input.currentTurnTs) {
      return {
        entries: resolved,
        reason: `cited stream entry ${entry.entry_id} is not before the current turn`,
      };
    }
  }

  return {
    entries: resolved,
    reason: validateQuotedEvidence(input.claim, resolved),
  };
}

function validateRelationalIdentityClaim(input: {
  claim: RelationalClaimAuditClaim;
  streamEntries: ReadonlyMap<string, RelationalGuardStreamEvidence>;
  currentSessionId: string;
}): string | null {
  if (input.claim.cited_stream_entry_ids.length === 0) {
    return "relational identity claim has no user stream citation";
  }

  const resolved = streamIdsExistInCurrentSession({
    claim: input.claim,
    streamEntries: input.streamEntries,
    currentSessionId: input.currentSessionId,
  });

  if (typeof resolved === "string") {
    return resolved;
  }

  if (resolved.some((entry) => entry.kind !== "user_msg")) {
    return "relational identity claim cites non-user evidence";
  }

  return null;
}

function validateActionCompletionClaim(input: {
  claim: RelationalClaimAuditClaim;
  streamEntries: ReadonlyMap<string, RelationalGuardStreamEvidence>;
  episodes: ReadonlySet<string>;
  commitments: ReadonlySet<string>;
}): string | null {
  const citedStreamEntries = input.claim.cited_stream_entry_ids
    .map((entryId) => input.streamEntries.get(entryId))
    .filter((entry): entry is RelationalGuardStreamEvidence => entry !== undefined);
  const hasStreamEvidence = citedStreamEntries.length > 0;
  const hasEpisodeEvidence = input.claim.cited_episode_ids.some((episodeId) =>
    input.episodes.has(episodeId),
  );
  const hasCommitmentEvidence = input.claim.cited_commitment_ids.some((commitmentId) =>
    input.commitments.has(commitmentId),
  );

  // TODO(6c-4): require a completed ActionRecord for the asserted action once ActionRecord lands.
  if (!hasStreamEvidence && !hasEpisodeEvidence && !hasCommitmentEvidence) {
    return "action completion claim has no existing cited evidence handle";
  }

  return validateQuotedEvidence(input.claim, citedStreamEntries);
}

function validateSelfCorrectionClaim(input: {
  claim: RelationalClaimAuditClaim;
  streamEntries: ReadonlyMap<string, RelationalGuardStreamEvidence>;
  currentSessionId: string;
  currentTurnTs: number;
  correctivePreferences: ReadonlySet<string>;
  hasCorrectivePreferenceEvidence: (entryId: StreamEntryId) => boolean;
}): string | null {
  const callback = validateCallbackScope({
    claim: input.claim,
    streamEntries: input.streamEntries,
    currentSessionId: input.currentSessionId,
    currentTurnTs: input.currentTurnTs,
  });

  if (callback.reason !== null) {
    return callback.reason;
  }

  const hasCorrectiveEvidence = callback.entries.some(
    (entry) =>
      input.correctivePreferences.has(entry.entry_id) ||
      input.hasCorrectivePreferenceEvidence(entry.entry_id),
  );

  return hasCorrectiveEvidence
    ? null
    : "self-correction claim has no corrective preference evidence";
}

export function validateRelationalClaims(input: {
  claims: readonly RelationalClaimAuditClaim[];
  evidence: RelationalGuardEvidenceManifest;
  currentSessionId: string;
  currentTurnTs: number;
  hasCorrectivePreferenceEvidence: (entryId: StreamEntryId) => boolean;
}): RelationalClaimValidationSummary {
  const streamEntries = buildStreamEvidenceIndex(input.evidence);
  const episodes = buildEpisodeEvidenceIndex(input.evidence);
  const commitments = buildCommitmentEvidenceIndex(input.evidence);
  const correctivePreferences = new Set(
    input.evidence.corrective_preferences.map((preference) => preference.source_entry_id),
  );
  const validations = input.claims.map((claim, index): RelationalClaimValidation => {
    let reason: string | null;

    switch (claim.kind) {
      case "relational_identity":
        reason = validateRelationalIdentityClaim({
          claim,
          streamEntries,
          currentSessionId: input.currentSessionId,
        });
        break;
      case "callback":
      case "session_scoped":
        reason = validateCallbackScope({
          claim,
          streamEntries,
          currentSessionId: input.currentSessionId,
          currentTurnTs: input.currentTurnTs,
        }).reason;
        break;
      case "action_completion":
        reason = validateActionCompletionClaim({
          claim,
          streamEntries,
          episodes,
          commitments,
        });
        break;
      case "self_correction":
        reason = validateSelfCorrectionClaim({
          claim,
          streamEntries,
          currentSessionId: input.currentSessionId,
          currentTurnTs: input.currentTurnTs,
          correctivePreferences,
          hasCorrectivePreferenceEvidence: input.hasCorrectivePreferenceEvidence,
        });
        break;
    }

    return {
      claim_id: `claim_${index}`,
      claim,
      status: reason === null ? "valid" : "unsupported",
      reason: reason ?? "supported by cited evidence",
    };
  });
  const unsupported = validations.filter((validation) => validation.status === "unsupported");

  return {
    validations,
    unsupported,
  };
}

function buildRewriteMessages(input: {
  response: string;
  unsupportedClaims: readonly RelationalClaimValidation[];
}): LLMMessage[] {
  return [
    {
      role: "user",
      content: JSON.stringify({
        response: input.response,
        phrases_to_remove_or_neutralize: input.unsupportedClaims.map(
          (validation) => validation.claim.asserted,
        ),
      }),
    },
  ];
}

function emitTrace(input: {
  tracer?: TurnTracer;
  turnId: string;
  claims: readonly RelationalClaimAuditClaim[];
  validations: readonly RelationalClaimValidation[];
  unsupported: readonly RelationalClaimValidation[];
  verdict: "passed" | "rewritten" | "suppressed";
  suppressionReason?: string;
}): void {
  if (input.tracer?.enabled !== true) {
    return;
  }

  input.tracer.emit("relational_claim_guard", {
    turnId: input.turnId,
    claimsExtracted: input.claims.length,
    claimsValid: input.validations.length - input.unsupported.length,
    claimsUnsupported: input.unsupported.length,
    unsupportedClaims: input.unsupported.map((validation) => ({
      kind: validation.claim.kind,
      reason: validation.reason,
    })),
    verdict: input.verdict,
    ...(input.suppressionReason === undefined
      ? {}
      : { suppressionReason: input.suppressionReason }),
  });
}

export class RelationalClaimGuard {
  constructor(private readonly options: RelationalClaimGuardOptions) {}

  private async audit(input: {
    response: string;
    evidence: RelationalGuardEvidenceManifest;
  }): Promise<RelationalClaimAuditClaim[]> {
    const result = await this.options.llmClient.complete({
      model: this.options.auditModel,
      system: CLAIM_AUDIT_SYSTEM_PROMPT,
      messages: buildAuditMessages(input),
      tools: [CLAIM_AUDIT_TOOL],
      tool_choice: { type: "tool", name: CLAIM_AUDIT_TOOL_NAME },
      max_tokens: 1024,
      budget: "relational-claim-auditor",
    });

    return parseAuditResponse(result);
  }

  private async rewrite(input: {
    response: string;
    unsupportedClaims: readonly RelationalClaimValidation[];
  }): Promise<string> {
    const result = await this.options.llmClient.complete({
      model: this.options.rewriteModel,
      system: RELATIONAL_GUARD_REWRITE_SYSTEM_PROMPT,
      messages: buildRewriteMessages(input),
      max_tokens: 1024,
      temperature: 0,
      budget: "relational-guard-rewrite",
    });

    return result.text.trim();
  }

  async run(input: RelationalClaimGuardInput): Promise<RelationalClaimGuardResult> {
    let claims: RelationalClaimAuditClaim[];

    try {
      claims = await this.audit({
        response: input.response,
        evidence: input.evidence,
      });
    } catch {
      const suppressionReason = "relational_guard_rewrite_failed";

      emitTrace({
        tracer: this.options.tracer,
        turnId: input.turnId,
        claims: [],
        validations: [],
        unsupported: [],
        verdict: "suppressed",
        suppressionReason,
      });

      return {
        emission: {
          kind: "suppressed",
          reason: suppressionReason,
        },
        claims: [],
        validations: [],
        verdict: "suppressed",
        suppressionReason,
      };
    }

    const firstValidation = validateRelationalClaims({
      claims,
      evidence: input.evidence,
      currentSessionId: input.currentSessionId,
      currentTurnTs: input.currentTurnTs,
      hasCorrectivePreferenceEvidence: this.options.hasCorrectivePreferenceEvidence,
    });

    if (firstValidation.unsupported.length === 0) {
      emitTrace({
        tracer: this.options.tracer,
        turnId: input.turnId,
        claims,
        validations: firstValidation.validations,
        unsupported: [],
        verdict: "passed",
      });

      return {
        emission: {
          kind: "message",
          content: input.response,
        },
        claims,
        validations: firstValidation.validations,
        verdict: "passed",
      };
    }

    if (
      firstValidation.unsupported.some(
        (validation) => validation.claim.kind === "self_correction",
      )
    ) {
      const suppressionReason = "relational_guard_self_correction";

      emitTrace({
        tracer: this.options.tracer,
        turnId: input.turnId,
        claims,
        validations: firstValidation.validations,
        unsupported: firstValidation.unsupported,
        verdict: "suppressed",
        suppressionReason,
      });

      return {
        emission: {
          kind: "suppressed",
          reason: suppressionReason,
        },
        claims,
        validations: firstValidation.validations,
        verdict: "suppressed",
        suppressionReason,
      };
    }

    let rewritten: string;

    try {
      rewritten = await this.rewrite({
        response: input.response,
        unsupportedClaims: firstValidation.unsupported,
      });
    } catch {
      const suppressionReason = "relational_guard_rewrite_failed";

      emitTrace({
        tracer: this.options.tracer,
        turnId: input.turnId,
        claims,
        validations: firstValidation.validations,
        unsupported: firstValidation.unsupported,
        verdict: "suppressed",
        suppressionReason,
      });

      return {
        emission: {
          kind: "suppressed",
          reason: suppressionReason,
        },
        claims,
        validations: firstValidation.validations,
        verdict: "suppressed",
        suppressionReason,
      };
    }

    if (rewritten.length === 0) {
      const suppressionReason = "relational_guard_rewrite_failed";

      emitTrace({
        tracer: this.options.tracer,
        turnId: input.turnId,
        claims,
        validations: firstValidation.validations,
        unsupported: firstValidation.unsupported,
        verdict: "suppressed",
        suppressionReason,
      });

      return {
        emission: {
          kind: "suppressed",
          reason: suppressionReason,
        },
        claims,
        validations: firstValidation.validations,
        verdict: "suppressed",
        suppressionReason,
      };
    }

    let secondClaims: RelationalClaimAuditClaim[];

    try {
      secondClaims = await this.audit({
        response: rewritten,
        evidence: input.evidence,
      });
    } catch {
      const suppressionReason = "relational_guard_rewrite_failed";

      emitTrace({
        tracer: this.options.tracer,
        turnId: input.turnId,
        claims,
        validations: firstValidation.validations,
        unsupported: firstValidation.unsupported,
        verdict: "suppressed",
        suppressionReason,
      });

      return {
        emission: {
          kind: "suppressed",
          reason: suppressionReason,
        },
        claims,
        validations: firstValidation.validations,
        verdict: "suppressed",
        suppressionReason,
      };
    }

    const secondValidation = validateRelationalClaims({
      claims: secondClaims,
      evidence: input.evidence,
      currentSessionId: input.currentSessionId,
      currentTurnTs: input.currentTurnTs,
      hasCorrectivePreferenceEvidence: this.options.hasCorrectivePreferenceEvidence,
    });

    if (secondValidation.unsupported.length > 0) {
      const suppressionReason = "relational_guard_rewrite_failed";

      emitTrace({
        tracer: this.options.tracer,
        turnId: input.turnId,
        claims: secondClaims,
        validations: secondValidation.validations,
        unsupported: secondValidation.unsupported,
        verdict: "suppressed",
        suppressionReason,
      });

      return {
        emission: {
          kind: "suppressed",
          reason: suppressionReason,
        },
        claims: secondClaims,
        validations: secondValidation.validations,
        verdict: "suppressed",
        suppressionReason,
      };
    }

    emitTrace({
      tracer: this.options.tracer,
      turnId: input.turnId,
      claims,
      validations: firstValidation.validations,
      unsupported: firstValidation.unsupported,
      verdict: "rewritten",
    });

    return {
      emission: {
        kind: "message",
        content: rewritten,
      },
      claims: secondClaims,
      validations: secondValidation.validations,
      verdict: "rewritten",
    };
  }
}
