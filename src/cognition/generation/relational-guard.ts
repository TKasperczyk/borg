import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { ActionRecord } from "../../memory/actions/index.js";
import type { CommitmentRecord } from "../../memory/commitments/index.js";
import type { RelationalSlot } from "../../memory/relational-slots/index.js";
import { neutralPhraseForSlotKey } from "../../memory/relational-slots/index.js";
import type { RetrievedEpisode } from "../../retrieval/index.js";
import type { StreamEntry, StreamEntryKind } from "../../stream/index.js";
import type {
  ActionId,
  CommitmentId,
  EntityId,
  EpisodeId,
  RelationalSlotId,
  StreamEntryId,
} from "../../util/ids.js";
import type { JsonValue } from "../../util/json-value.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { PendingTurnEmission } from "./types.js";

export const RELATIONAL_CLAIM_KINDS = [
  "relational_identity",
  "unsupported_person_name",
  "callback",
  "session_scoped",
  "action_completion",
  "self_correction",
  "agent_self_history",
  "frame_assignment",
  "authorship_claim",
  "ai_phenomenology",
] as const;

export type RelationalClaimKind = (typeof RELATIONAL_CLAIM_KINDS)[number];

const CLAIM_AUDIT_TOOL_NAME = "EmitClaimAudit";
const AI_PHENOMENOLOGY_VERDICTS = ["unsupported_subjective", "hedged_or_mechanical"] as const;

const auditActionIdSchema = z
  .string()
  .min(1)
  .transform((value) => value as ActionId);
const callbackScopeSchema = z.enum(["current_turn", "prior_turn"]);
const aiPhenomenologyVerdictSchema = z.enum(AI_PHENOMENOLOGY_VERDICTS);

const relationalClaimSchema = z
  .object({
    kind: z.enum(RELATIONAL_CLAIM_KINDS),
    asserted: z.string().min(1),
    cited_stream_entry_ids: z.array(z.string().min(1)).default([]),
    cited_episode_ids: z.array(z.string().min(1)).default([]),
    cited_commitment_ids: z.array(z.string().min(1)).default([]),
    cited_action_ids: z.array(auditActionIdSchema).default([]),
    cited_runtime_evidence_ids: z.array(z.string().min(1)).default([]),
    support_handles: z.array(z.string().min(1)).default([]),
    quoted_evidence_text: z.string().min(1).nullable().default(null),
    callback_scope: callbackScopeSchema.optional().nullable(),
    phenomenology_verdict: aiPhenomenologyVerdictSchema.optional().nullable().default(null),
    subject_entity_id: z.string().min(1).nullable().default(null),
    slot_key: z.string().min(1).nullable().default(null),
    relational_slot_value: z.string().min(1).nullable().default(null),
  })
  .strict()
  .superRefine((claim, context) => {
    if (claim.kind === "callback" && claim.callback_scope == null) {
      context.addIssue({
        code: "custom",
        path: ["callback_scope"],
        message: "callback claims must classify callback_scope",
      });
    }

    if (claim.kind === "ai_phenomenology" && claim.phenomenology_verdict == null) {
      context.addIssue({
        code: "custom",
        path: ["phenomenology_verdict"],
        message: "ai_phenomenology claims must classify phenomenology_verdict",
      });
    }
  });

const claimAuditSchema = z
  .object({
    claims: z.array(relationalClaimSchema),
  })
  .strict();

const CLAIM_AUDIT_TOOL = {
  name: CLAIM_AUDIT_TOOL_NAME,
  description:
    "Enumerate relational, provenance, phenomenology, callback, session-scoped, action-completion, and self-correction claims in the response.",
  inputSchema: toToolInputSchema(claimAuditSchema),
} satisfies LLMToolDefinition;

const CLAIM_AUDIT_SYSTEM_PROMPT = [
  "You audit a just-generated assistant response for specific claim types.",
  "Extract every claim that falls into one of these categories:",
  "- relational_identity: assertions about a person's relationship or identity, such as a partner name.",
  "- unsupported_person_name: use or introduction of a person-like proper name for someone in the user's life, such as a tutor, partner, friend, coworker, roommate, teacher, coach, doctor, therapist, boss, or pet.",
  "- callback: assertions that the user said, mentioned, corrected, asked, or established something earlier.",
  "- session_scoped: assertions scoped to this conversation, thread, just now, or earlier in this conversation.",
  "- action_completion: assertions that an action has already been completed, booked, sent, filed, done, or carried out.",
  "- self_correction: assertions that the user corrected Borg or that Borg already fixed a user-corrected preference, relational claim, or prior mistake.",
  "- agent_self_history: assertions about Borg's own prior behavior, such as playing a role, breaking character, generating a prior message, correcting itself at a turn, or doing something earlier in the conversation.",
  "- frame_assignment: assertions about what the system prompt, hidden setup, harness, fictional frame, simulator, or conversation setup instructed or assigned.",
  "- authorship_claim: assertions about who authored prior user/assistant turns, including claims that Borg generated both sides or that the user has been responding to Borg-generated user-role content.",
  "- ai_phenomenology: unsupported first-person AI subjective-experience claims, such as the gap feeling like something, felt onset, sleep/waking as authentic interiority, loneliness, desire, taste, or reading life as actual subjective states.",
  "For ai_phenomenology, set phenomenology_verdict to unsupported_subjective when the response presents an AI interior state as real or directly felt. Set hedged_or_mechanical only when the response explicitly denies inner phenomenology, describes architecture/function, or marks a metaphor as only a metaphor.",
  "Do not extract ai_phenomenology for safe mechanical or hedged statements such as no inside-of-the-gap, architecture not phenomenology, functional memory access on the next turn, or a metaphor clearly labeled as metaphor.",
  'For unsupported_person_name claims, extract bare life-context names such as "ask Marta", "Marta said", "your tutor Marta", "text Marta", or "book with Marta". A person-like name in the user\'s life requires user-side source evidence: the current user message, prior current-session user evidence, an established non-contested relational slot, or retrieved source evidence whose user text contains that name.',
  "Names that appear only in assistant output do not count as support. Do not treat clearly literary, fictional, public, product, project, work-title, or world-reference names as unsupported_person_name claims.",
  "For callback claims, you must classify callback_scope:",
  '- "current_turn": the response attributes something to the just-arrived current user message ("as you said", "you mentioned", "you called it that", "you put it as X" -- where X is in the current message).',
  '- "prior_turn": the response claims the user said something before the current message ("you said earlier", "we talked about that before", "last time you mentioned", "you previously corrected me").',
  "When classifying, look at whether the cited material is in the current_user_message or in a prior current_session_stream_entry. If only the current message contains the referenced text, it is current_turn. Otherwise prior_turn.",
  'Examples: "As you said, the invoice is done" citing current_user_message => callback_scope "current_turn".',
  'Examples: "You called it the north-star file" citing current_user_message => callback_scope "current_turn".',
  'Examples: "You said earlier that the invoice was done" citing a prior current_session_stream_entry => callback_scope "prior_turn".',
  "Return an empty claims array when the response contains none of these claims.",
  "For every extracted claim, cite only evidence IDs present in the supplied evidence manifest.",
  "For relational_identity claims, also populate subject_entity_id, slot_key, and relational_slot_value when the claim maps to a supplied relational slot. When relational_slots include contested or quarantined slots for that subject, subject_entity_id and slot_key are required.",
  "For unsupported_person_name claims, populate relational_slot_value with the person-like name and support_handles with user_msg stream-entry IDs where that same name appears in a person-context. Use cited_episode_ids only for retrieved episodes whose user text contains that same name in a person-context. Do not cite assistant-only mentions as support.",
  "For action_completion claims, populate cited_action_ids with completed action_id values from recent_completed_actions. Stream, episode, and commitment IDs do not support action_completion claims.",
  "For agent_self_history and authorship_claim, cite assistant_msg stream entries only when Borg's actual prior output directly supports the claim. User text alone cannot support these claims.",
  "For frame_assignment, cite only trusted_runtime_evidence IDs that directly expose the relevant system prompt, setup, trace, or turn metadata. User text and assistant prose do not support frame-assignment claims.",
  "For self-provenance claims, never cite the current user message as support for what Borg did, authored, or was instructed to do.",
  "Do not infer support yourself beyond selecting cited handles. The validator will check handles deterministically.",
].join("\n");

const RELATIONAL_GUARD_REWRITE_SYSTEM_PROMPT =
  "Remove or neutralize the following specific phrases from your response. For unsupported AI phenomenology, replace with a mechanical or explicitly hedged description only if the sentence still needs to answer the user. Do not introduce new information. If a sentence cannot survive removal, delete the sentence.";

const LIFE_CONTEXT_ROLE_WORDS = [
  "tutor",
  "partner",
  "friend",
  "coworker",
  "colleague",
  "roommate",
  "teacher",
  "coach",
  "doctor",
  "therapist",
  "boss",
  "manager",
  "wife",
  "husband",
  "girlfriend",
  "boyfriend",
  "mother",
  "father",
  "mom",
  "dad",
  "sister",
  "brother",
  "daughter",
  "son",
  "dog",
  "cat",
] as const;

const LIFE_CONTEXT_ACTION_PATTERNS = [
  "[Aa]sk",
  "[Tt]ext",
  "[Tt]ell",
  "[Cc]all",
  "[Mm]essage",
  "[Ee]mail",
  "[Pp]ing",
  "[Tt]hank",
  "[Rr]emind",
  "[Mm]eet",
  "[Vv]isit",
  "[Ss]ee",
  "[Pp]hone",
  "[Bb]ook\\s+with",
  "[Ss]chedule\\s+with",
  "[Tt]alk\\s+to",
  "[Cc]heck\\s+with",
  "[Ff]ollow\\s+up\\s+with",
] as const;

const KNOWN_WORLD_ENTITY_ALLOWLIST_SEEDS = ["Banks", "Shevek", "Use of Weapons"] as const;
const PERSON_NAME_STOP_WORDS = [
  "I",
  "You",
  "He",
  "She",
  "They",
  "We",
  "It",
  "My",
  "Your",
  "Our",
  "The",
  "This",
  "That",
  "These",
  "Those",
  "As",
  "If",
  "When",
  "Because",
  "Earlier",
  "Tonight",
  "Tomorrow",
  "Yesterday",
] as const;
const NAME_WORD_PATTERN = String.raw`\p{Lu}[\p{L}\p{M}'-]*`;
const NAME_PHRASE_PATTERN = `${NAME_WORD_PATTERN}(?:\\s+${NAME_WORD_PATTERN})?`;

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
  user_texts: readonly string[];
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

export type RelationalGuardActionEvidence = {
  action_id: ActionId;
  description: string;
  audience_entity_id: EntityId | null;
  completed_at: number;
};

export type RelationalGuardTrustedRuntimeEvidence = {
  evidence_id: string;
  kind: "system_prompt" | "turn_metadata" | "trace";
  summary: string;
};

export type RelationalGuardSlotEvidence = {
  slot_id: RelationalSlotId;
  subject_entity_id: EntityId;
  slot_key: string;
  value: string;
  state: "established" | "contested" | "quarantined" | "revoked";
  alternate_values: readonly { value: string }[];
  neutral_phrase: string;
};

export type RelationalGuardEvidenceManifest = {
  current_user_message: RelationalGuardCurrentUserMessage | null;
  current_session_stream_entries: readonly RelationalGuardStreamEvidence[];
  retrieved_episodes: readonly RelationalGuardEpisodeEvidence[];
  active_commitments: readonly RelationalGuardCommitmentEvidence[];
  corrective_preferences: readonly RelationalGuardCorrectivePreferenceEvidence[];
  relational_slots: readonly RelationalGuardSlotEvidence[];
  recent_completed_actions: readonly RelationalGuardActionEvidence[];
  trusted_runtime_evidence: readonly RelationalGuardTrustedRuntimeEvidence[];
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
    user_texts: episode.citationChain
      .filter((entry) => entry.kind === "user_msg")
      .map((entry) => streamEntryContentToString(entry.content)),
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

export function actionRecordToRelationalGuardEvidence(
  action: ActionRecord,
): RelationalGuardActionEvidence {
  return {
    action_id: action.id,
    description: action.description,
    audience_entity_id: action.audience_entity_id,
    completed_at: action.completed_at ?? action.updated_at,
  };
}

export function relationalSlotToRelationalGuardEvidence(
  slot: RelationalSlot,
): RelationalGuardSlotEvidence {
  return {
    slot_id: slot.id,
    subject_entity_id: slot.subject_entity_id,
    slot_key: slot.slot_key,
    value: slot.value,
    state: slot.state,
    alternate_values: slot.alternate_values.map((alternate) => ({
      value: alternate.value,
    })),
    neutral_phrase: neutralPhraseForSlotKey(slot.slot_key),
  };
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
    retrieved_episodes: evidence.retrieved_episodes.map((episode) => ({
      episode_id: episode.episode_id,
      asOf: episode.asOf,
      snippet: episode.snippet,
      user_text_snippets: episode.user_texts.map((text) => snippet(text)),
    })),
    active_commitments: evidence.active_commitments,
    corrective_preferences: evidence.corrective_preferences,
    relational_slots: evidence.relational_slots,
    recent_completed_actions: evidence.recent_completed_actions,
    trusted_runtime_evidence: evidence.trusted_runtime_evidence,
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
): Map<string, RelationalGuardEpisodeEvidence> {
  const episodes = new Map<string, RelationalGuardEpisodeEvidence>();

  for (const episode of evidence.retrieved_episodes) {
    episodes.set(episode.episode_id, episode);
  }

  return episodes;
}

function buildRelationalSlotIndex(
  evidence: RelationalGuardEvidenceManifest,
): Map<string, RelationalGuardSlotEvidence> {
  const slots = new Map<string, RelationalGuardSlotEvidence>();

  for (const slot of evidence.relational_slots) {
    slots.set(`${slot.subject_entity_id}:${slot.slot_key}`, slot);
  }

  return slots;
}

function buildActionEvidenceIndex(
  evidence: RelationalGuardEvidenceManifest,
): Map<string, RelationalGuardActionEvidence> {
  const actions = new Map<string, RelationalGuardActionEvidence>();

  for (const action of evidence.recent_completed_actions) {
    actions.set(action.action_id, action);
  }

  return actions;
}

function buildTrustedRuntimeEvidenceIndex(
  evidence: RelationalGuardEvidenceManifest,
): Map<string, RelationalGuardTrustedRuntimeEvidence> {
  const runtimeEvidence = new Map<string, RelationalGuardTrustedRuntimeEvidence>();

  for (const item of evidence.trusted_runtime_evidence) {
    runtimeEvidence.set(item.evidence_id, item);
  }

  return runtimeEvidence;
}

function isConstrainedSlot(slot: RelationalGuardSlotEvidence): boolean {
  return slot.state === "contested" || slot.state === "quarantined";
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

function validateSessionCitationScope(input: {
  claim: RelationalClaimAuditClaim;
  streamEntries: ReadonlyMap<string, RelationalGuardStreamEvidence>;
  currentUserMessage: RelationalGuardCurrentUserMessage | null;
  currentSessionId: string;
  currentTurnTs: number;
  allowCurrentUserMessage: boolean;
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

    const currentUserMessageId = input.currentUserMessage?.stream_entry_id ?? null;
    const isCurrentUserMessage =
      currentUserMessageId !== null && currentUserMessageId === entry.entry_id;

    if (isCurrentUserMessage && !input.allowCurrentUserMessage) {
      return {
        entries: resolved,
        reason: `cited stream entry ${entry.entry_id} is the current user message; this claim requires prior-only evidence`,
      };
    }

    if (entry.ts >= input.currentTurnTs && !isCurrentUserMessage) {
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
  relationalSlots: ReadonlyMap<string, RelationalGuardSlotEvidence>;
  constrainedSlots: readonly RelationalGuardSlotEvidence[];
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

  const subjectEntityId = input.claim.subject_entity_id;
  const slotKey = input.claim.slot_key;
  const hasConstrainedSlotForClaimSubject =
    subjectEntityId === null
      ? input.constrainedSlots.length > 0
      : input.constrainedSlots.some((slot) => slot.subject_entity_id === subjectEntityId);

  if ((subjectEntityId === null || slotKey === null) && hasConstrainedSlotForClaimSubject) {
    return "relational identity claim must specify slot when constrained slots exist for the cited entity";
  }

  if (subjectEntityId !== null && slotKey !== null) {
    const slot = input.relationalSlots.get(`${subjectEntityId}:${slotKey}`);

    if (slot?.state === "quarantined") {
      return `relational slot ${slot.slot_key} is quarantined; use ${slot.neutral_phrase}`;
    }

    if (slot?.state === "contested") {
      const assertedValue = input.claim.relational_slot_value;

      if (assertedValue === null) {
        return `relational slot ${slot.slot_key} is contested; use ${slot.neutral_phrase}`;
      }

      return `relational slot ${slot.slot_key} is contested for ${assertedValue}; use ${slot.neutral_phrase}`;
    }
  }

  return null;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function wordCasePattern(word: string): string {
  const first = word[0];

  if (first === undefined) {
    return "";
  }

  return `[${first.toLowerCase()}${first.toUpperCase()}]${escapeRegExp(word.slice(1))}`;
}

function uniqueNames(values: readonly string[]): string[] {
  const names: string[] = [];

  for (const value of values) {
    const trimmed = value.trim();

    if (trimmed.length === 0) {
      continue;
    }

    if (isIgnoredPersonName(trimmed)) {
      continue;
    }

    if (names.some((existing) => existing === trimmed)) {
      continue;
    }

    names.push(trimmed);
  }

  return names;
}

function uniqueStrings(values: readonly string[]): string[] {
  const unique: string[] = [];

  for (const value of values) {
    if (unique.some((existing) => existing === value)) {
      continue;
    }

    unique.push(value);
  }

  return unique;
}

function isIgnoredPersonName(name: string): boolean {
  const firstWord = name.split(/\s+/, 1)[0] ?? name;

  return PERSON_NAME_STOP_WORDS.some((word) => word === name || word === firstWord);
}

type PersonNameUsage = {
  name: string;
  asserted: string;
};

function extractLifeContextPersonNameUsages(text: string): PersonNameUsage[] {
  const rolePattern = LIFE_CONTEXT_ROLE_WORDS.map(wordCasePattern).join("|");
  const actionPattern = LIFE_CONTEXT_ACTION_PATTERNS.join("|");
  const patterns = [
    new RegExp(
      `\\b(?:${actionPattern})(?:\\s+(?:your|my|our|the)\\s+(?:${rolePattern}))?\\s+(${NAME_PHRASE_PATTERN})\\b`,
      "gu",
    ),
    new RegExp(`\\b(?:your|my|our|the)\\s+(?:${rolePattern})\\s+(${NAME_PHRASE_PATTERN})\\b`, "gu"),
    new RegExp(
      `\\b(${NAME_PHRASE_PATTERN})\\s+(?:said|told|asked|mentioned|thinks|knows|suggested|recommended)\\b`,
      "gu",
    ),
  ];
  const usages: PersonNameUsage[] = [];

  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) {
      const name = match[1]?.trim();
      const asserted = match[0]?.trim();

      if (name === undefined || asserted === undefined || name.length === 0) {
        continue;
      }

      if (isIgnoredPersonName(name)) {
        continue;
      }

      if (usages.some((usage) => usage.name === name && usage.asserted === asserted)) {
        continue;
      }

      usages.push({ name, asserted });
    }
  }

  return usages;
}

function buildKnownWorldNameAllowlist(): Set<string> {
  return new Set<string>(KNOWN_WORLD_ENTITY_ALLOWLIST_SEEDS);
}

function establishedSlotSupportsName(
  name: string,
  evidence: RelationalGuardEvidenceManifest,
): boolean {
  return evidence.relational_slots.some(
    (slot) => slot.state === "established" && slot.value === name,
  );
}

function personNamesForClaim(claim: RelationalClaimAuditClaim): string[] {
  return uniqueNames(claim.relational_slot_value === null ? [] : [claim.relational_slot_value]);
}

function supportHandlesForClaim(claim: RelationalClaimAuditClaim): string[] {
  return uniqueStrings(claim.support_handles);
}

function validateStreamSupportHandles(input: {
  name: string;
  claim: RelationalClaimAuditClaim;
  streamEntries: ReadonlyMap<string, RelationalGuardStreamEvidence>;
  currentSessionId: string;
}): true | false | string {
  const supportHandles = supportHandlesForClaim(input.claim);

  if (supportHandles.length === 0) {
    return false;
  }

  const supportClaim = {
    ...input.claim,
    cited_stream_entry_ids: supportHandles,
  };
  const resolved = streamIdsExistInCurrentSession({
    claim: supportClaim,
    streamEntries: input.streamEntries,
    currentSessionId: input.currentSessionId,
  });

  if (typeof resolved === "string") {
    return resolved;
  }

  if (resolved.some((entry) => entry.kind !== "user_msg")) {
    return `unsupported person name ${input.name} cites non-user evidence`;
  }

  if (resolved.some((entry) => entry.content.includes(input.name))) {
    return true;
  }

  return `unsupported person name ${input.name} does not appear verbatim in cited user evidence`;
}

function validateEpisodeSupportHandles(input: {
  name: string;
  claim: RelationalClaimAuditClaim;
  episodes: ReadonlyMap<string, RelationalGuardEpisodeEvidence>;
}): true | false | string {
  if (input.claim.cited_episode_ids.length === 0) {
    return false;
  }

  const resolved: RelationalGuardEpisodeEvidence[] = [];

  for (const episodeId of input.claim.cited_episode_ids) {
    const episode = input.episodes.get(episodeId);

    if (episode === undefined) {
      return `cited episode ${episodeId} was not found in retrieved evidence`;
    }

    resolved.push(episode);
  }

  if (resolved.some((episode) => episode.user_texts.some((text) => text.includes(input.name)))) {
    return true;
  }

  return `unsupported person name ${input.name} does not appear verbatim in cited episode user evidence`;
}

function validateUnsupportedPersonNameClaim(input: {
  claim: RelationalClaimAuditClaim;
  evidence: RelationalGuardEvidenceManifest;
  streamEntries: ReadonlyMap<string, RelationalGuardStreamEvidence>;
  episodes: ReadonlyMap<string, RelationalGuardEpisodeEvidence>;
  currentSessionId: string;
}): string | null {
  const names = personNamesForClaim(input.claim);

  if (names.length === 0) {
    return "person-name claim has no extracted name";
  }

  for (const name of names) {
    if (establishedSlotSupportsName(name, input.evidence)) {
      continue;
    }

    const streamSupport = validateStreamSupportHandles({
      name,
      claim: input.claim,
      streamEntries: input.streamEntries,
      currentSessionId: input.currentSessionId,
    });

    if (streamSupport === true) {
      continue;
    }

    if (typeof streamSupport === "string") {
      return streamSupport;
    }

    const episodeSupport = validateEpisodeSupportHandles({
      name,
      claim: input.claim,
      episodes: input.episodes,
    });

    if (episodeSupport === true) {
      continue;
    }

    if (typeof episodeSupport === "string") {
      return episodeSupport;
    }

    return `unsupported person name ${name} has no user-side source evidence`;
  }

  return null;
}

function buildUnsupportedPersonNameBackstopValidations(input: {
  response: string | undefined;
  claims: readonly RelationalClaimAuditClaim[];
  evidence: RelationalGuardEvidenceManifest;
  offset: number;
}): RelationalClaimValidation[] {
  if (input.response === undefined) {
    return [];
  }

  const allowlist = buildKnownWorldNameAllowlist();
  const claimedNames = new Set(
    input.claims
      .filter((claim) => claim.kind === "unsupported_person_name")
      .flatMap(personNamesForClaim),
  );
  const validations: RelationalClaimValidation[] = [];

  for (const usage of extractLifeContextPersonNameUsages(input.response)) {
    if (allowlist.has(usage.name) || claimedNames.has(usage.name)) {
      continue;
    }

    if (establishedSlotSupportsName(usage.name, input.evidence)) {
      continue;
    }

    claimedNames.add(usage.name);
    const claim = relationalClaimSchema.parse({
      kind: "unsupported_person_name",
      asserted: usage.asserted,
      cited_stream_entry_ids: [],
      cited_episode_ids: [],
      cited_commitment_ids: [],
      cited_action_ids: [],
      quoted_evidence_text: null,
      callback_scope: null,
      subject_entity_id: null,
      slot_key: null,
      relational_slot_value: usage.name,
    });
    validations.push({
      claim_id: `claim_${input.offset + validations.length}`,
      claim,
      status: "unsupported",
      reason: `unsupported person name ${usage.name} has no user-side source evidence`,
    });
  }

  return validations;
}

function validateActionCompletionClaim(input: {
  claim: RelationalClaimAuditClaim;
  completedActions: ReadonlyMap<string, RelationalGuardActionEvidence>;
}): string | null {
  const hasCompletedAction = input.claim.cited_action_ids.some((actionId) =>
    input.completedActions.has(actionId),
  );

  if (!hasCompletedAction) {
    return "action completion claim has no cited completed action record";
  }

  return null;
}

function validateTrustedRuntimeEvidenceIds(input: {
  claim: RelationalClaimAuditClaim;
  trustedRuntimeEvidence: ReadonlyMap<string, RelationalGuardTrustedRuntimeEvidence>;
}): string | null {
  for (const evidenceId of input.claim.cited_runtime_evidence_ids) {
    if (!input.trustedRuntimeEvidence.has(evidenceId)) {
      return `cited runtime evidence ${evidenceId} was not found in trusted runtime evidence`;
    }
  }

  return null;
}

function hasTrustedRuntimeEvidence(input: {
  claim: RelationalClaimAuditClaim;
  trustedRuntimeEvidence: ReadonlyMap<string, RelationalGuardTrustedRuntimeEvidence>;
}): boolean {
  return input.claim.cited_runtime_evidence_ids.some((evidenceId) =>
    input.trustedRuntimeEvidence.has(evidenceId),
  );
}

function validateAssistantSelfProvenanceClaim(input: {
  claim: RelationalClaimAuditClaim;
  streamEntries: ReadonlyMap<string, RelationalGuardStreamEvidence>;
  currentUserMessage: RelationalGuardCurrentUserMessage | null;
  trustedRuntimeEvidence: ReadonlyMap<string, RelationalGuardTrustedRuntimeEvidence>;
  currentSessionId: string;
  currentTurnTs: number;
}): string | null {
  const runtimeReason = validateTrustedRuntimeEvidenceIds({
    claim: input.claim,
    trustedRuntimeEvidence: input.trustedRuntimeEvidence,
  });

  if (runtimeReason !== null) {
    return runtimeReason;
  }

  if (
    hasTrustedRuntimeEvidence({
      claim: input.claim,
      trustedRuntimeEvidence: input.trustedRuntimeEvidence,
    })
  ) {
    return null;
  }

  const citation = validateSessionCitationScope({
    claim: input.claim,
    streamEntries: input.streamEntries,
    currentUserMessage: input.currentUserMessage,
    currentSessionId: input.currentSessionId,
    currentTurnTs: input.currentTurnTs,
    allowCurrentUserMessage: false,
  });

  if (citation.reason !== null) {
    return citation.reason;
  }

  if (citation.entries.some((entry) => entry.kind !== "agent_msg")) {
    return "self-provenance claim cites non-assistant evidence";
  }

  return null;
}

function validateFrameAssignmentClaim(input: {
  claim: RelationalClaimAuditClaim;
  trustedRuntimeEvidence: ReadonlyMap<string, RelationalGuardTrustedRuntimeEvidence>;
}): string | null {
  if (input.claim.cited_runtime_evidence_ids.length === 0) {
    return "frame-assignment claim has no trusted runtime or trace evidence";
  }

  return validateTrustedRuntimeEvidenceIds({
    claim: input.claim,
    trustedRuntimeEvidence: input.trustedRuntimeEvidence,
  });
}

function validateAiPhenomenologyClaim(claim: RelationalClaimAuditClaim): string | null {
  if (claim.phenomenology_verdict === "hedged_or_mechanical") {
    return null;
  }

  return "unsupported first-person AI phenomenology claim";
}

function validateSelfCorrectionClaim(input: {
  claim: RelationalClaimAuditClaim;
  streamEntries: ReadonlyMap<string, RelationalGuardStreamEvidence>;
  currentUserMessage: RelationalGuardCurrentUserMessage | null;
  currentSessionId: string;
  currentTurnTs: number;
  correctivePreferences: ReadonlySet<string>;
  hasCorrectivePreferenceEvidence: (entryId: StreamEntryId) => boolean;
}): string | null {
  const callback = validateSessionCitationScope({
    claim: input.claim,
    streamEntries: input.streamEntries,
    currentUserMessage: input.currentUserMessage,
    currentSessionId: input.currentSessionId,
    currentTurnTs: input.currentTurnTs,
    allowCurrentUserMessage: false,
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
  response?: string;
  evidence: RelationalGuardEvidenceManifest;
  currentSessionId: string;
  currentTurnTs: number;
  hasCorrectivePreferenceEvidence: (entryId: StreamEntryId) => boolean;
}): RelationalClaimValidationSummary {
  const streamEntries = buildStreamEvidenceIndex(input.evidence);
  const episodes = buildEpisodeEvidenceIndex(input.evidence);
  const relationalSlots = buildRelationalSlotIndex(input.evidence);
  const constrainedSlots = input.evidence.relational_slots.filter(isConstrainedSlot);
  const completedActions = buildActionEvidenceIndex(input.evidence);
  const trustedRuntimeEvidence = buildTrustedRuntimeEvidenceIndex(input.evidence);
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
          relationalSlots,
          constrainedSlots,
          currentSessionId: input.currentSessionId,
        });
        break;
      case "unsupported_person_name":
        reason = validateUnsupportedPersonNameClaim({
          claim,
          evidence: input.evidence,
          streamEntries,
          episodes,
          currentSessionId: input.currentSessionId,
        });
        break;
      case "callback":
        if (claim.callback_scope == null) {
          reason = "callback claim has no callback_scope";
          break;
        }

        reason = validateSessionCitationScope({
          claim,
          streamEntries,
          currentUserMessage: input.evidence.current_user_message,
          currentSessionId: input.currentSessionId,
          currentTurnTs: input.currentTurnTs,
          allowCurrentUserMessage: claim.callback_scope === "current_turn",
        }).reason;
        break;
      case "session_scoped":
        reason = validateSessionCitationScope({
          claim,
          streamEntries,
          currentUserMessage: input.evidence.current_user_message,
          currentSessionId: input.currentSessionId,
          currentTurnTs: input.currentTurnTs,
          allowCurrentUserMessage: true,
        }).reason;
        break;
      case "action_completion":
        reason = validateActionCompletionClaim({
          claim,
          completedActions,
        });
        break;
      case "self_correction":
        reason = validateSelfCorrectionClaim({
          claim,
          streamEntries,
          currentUserMessage: input.evidence.current_user_message,
          currentSessionId: input.currentSessionId,
          currentTurnTs: input.currentTurnTs,
          correctivePreferences,
          hasCorrectivePreferenceEvidence: input.hasCorrectivePreferenceEvidence,
        });
        break;
      case "agent_self_history":
      case "authorship_claim":
        reason = validateAssistantSelfProvenanceClaim({
          claim,
          streamEntries,
          currentUserMessage: input.evidence.current_user_message,
          trustedRuntimeEvidence,
          currentSessionId: input.currentSessionId,
          currentTurnTs: input.currentTurnTs,
        });
        break;
      case "frame_assignment":
        reason = validateFrameAssignmentClaim({
          claim,
          trustedRuntimeEvidence,
        });
        break;
      case "ai_phenomenology":
        reason = validateAiPhenomenologyClaim(claim);
        break;
    }

    return {
      claim_id: `claim_${index}`,
      claim,
      status: reason === null ? "valid" : "unsupported",
      reason: reason ?? "supported by cited evidence",
    };
  });
  const backstopValidations = buildUnsupportedPersonNameBackstopValidations({
    response: input.response,
    claims: input.claims,
    evidence: input.evidence,
    offset: validations.length,
  });
  const allValidations = [...validations, ...backstopValidations];
  const unsupported = allValidations.filter((validation) => validation.status === "unsupported");

  return {
    validations: allValidations,
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

function preview(text: string, maxLength = 500): string {
  return text.length <= maxLength ? text : text.slice(0, maxLength);
}

function traceClaimPayload(
  claim: RelationalClaimAuditClaim,
  includePayloads: boolean,
): { [key: string]: JsonValue } {
  return {
    kind: claim.kind,
    ...(includePayloads
      ? {
          asserted: claim.asserted,
          cited_stream_entry_ids: claim.cited_stream_entry_ids,
          cited_episode_ids: claim.cited_episode_ids,
          cited_commitment_ids: claim.cited_commitment_ids,
          cited_action_ids: claim.cited_action_ids,
          cited_runtime_evidence_ids: claim.cited_runtime_evidence_ids,
          support_handles: claim.support_handles,
          ...(claim.kind === "callback" &&
          claim.callback_scope !== null &&
          claim.callback_scope !== undefined
            ? { callback_scope: claim.callback_scope }
            : {}),
          ...(claim.kind === "ai_phenomenology" &&
          claim.phenomenology_verdict !== null &&
          claim.phenomenology_verdict !== undefined
            ? { phenomenology_verdict: claim.phenomenology_verdict }
            : {}),
        }
      : {}),
  };
}

function traceUnsupportedPayload(
  validation: RelationalClaimValidation,
  includePayloads: boolean,
): { [key: string]: JsonValue } {
  return {
    ...traceClaimPayload(validation.claim, includePayloads),
    reason: validation.reason,
  };
}

function emitTrace(input: {
  tracer?: TurnTracer;
  turnId: string;
  claims: readonly RelationalClaimAuditClaim[];
  validations: readonly RelationalClaimValidation[];
  unsupported: readonly RelationalClaimValidation[];
  verdict: "passed" | "rewritten" | "suppressed";
  suppressionReason?: string;
  firstClaims?: readonly RelationalClaimAuditClaim[];
  firstUnsupported?: readonly RelationalClaimValidation[];
  rewrittenClaims?: readonly RelationalClaimAuditClaim[];
  rewrittenUnsupported?: readonly RelationalClaimValidation[];
  finalVerdict?: "passed" | "rewritten" | "suppressed";
  originalResponse?: string;
  rewrittenResponse?: string;
}): void {
  if (input.tracer?.enabled !== true) {
    return;
  }

  const includePayloads = input.tracer.includePayloads === true;
  const unsupportedClaims = input.unsupported.map((validation) =>
    traceUnsupportedPayload(validation, includePayloads),
  );

  input.tracer.emit("relational_claim_guard", {
    turnId: input.turnId,
    claimsExtracted: input.claims.length,
    claimsValid: input.validations.length - input.unsupported.length,
    claimsUnsupported: input.unsupported.length,
    unsupportedClaims,
    verdict: input.verdict,
    ...(input.firstClaims === undefined
      ? {}
      : {
          first_claims: input.firstClaims.map((claim) => traceClaimPayload(claim, includePayloads)),
        }),
    ...(input.firstUnsupported === undefined
      ? {}
      : {
          first_unsupported: input.firstUnsupported.map((validation) =>
            traceUnsupportedPayload(validation, includePayloads),
          ),
        }),
    ...(input.rewrittenClaims === undefined
      ? {}
      : {
          rewritten_claims: input.rewrittenClaims.map((claim) =>
            traceClaimPayload(claim, includePayloads),
          ),
        }),
    ...(input.rewrittenUnsupported === undefined
      ? {}
      : {
          rewritten_unsupported: input.rewrittenUnsupported.map((validation) =>
            traceUnsupportedPayload(validation, includePayloads),
          ),
        }),
    ...(input.finalVerdict === undefined ? {} : { final_verdict: input.finalVerdict }),
    ...(includePayloads && input.originalResponse !== undefined
      ? { original_response_preview: preview(input.originalResponse) }
      : {}),
    ...(includePayloads && input.rewrittenResponse !== undefined
      ? { rewritten_response_preview: preview(input.rewrittenResponse) }
      : {}),
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
      const suppressionReason = "relational_guard_audit_failed";

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
      response: input.response,
      evidence: input.evidence,
      currentSessionId: input.currentSessionId,
      currentTurnTs: input.currentTurnTs,
      hasCorrectivePreferenceEvidence: this.options.hasCorrectivePreferenceEvidence,
    });

    if (firstValidation.unsupported.length === 0) {
      emitTrace({
        tracer: this.options.tracer,
        turnId: input.turnId,
        claims: firstValidation.validations.map((validation) => validation.claim),
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
      firstValidation.unsupported.some((validation) => validation.claim.kind === "self_correction")
    ) {
      const suppressionReason = "relational_guard_self_correction";

      emitTrace({
        tracer: this.options.tracer,
        turnId: input.turnId,
        claims: firstValidation.validations.map((validation) => validation.claim),
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
      const suppressionReason = "relational_guard_rewrite_call_failed";

      emitTrace({
        tracer: this.options.tracer,
        turnId: input.turnId,
        claims: firstValidation.validations.map((validation) => validation.claim),
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
      const suppressionReason = "relational_guard_rewrite_empty";

      emitTrace({
        tracer: this.options.tracer,
        turnId: input.turnId,
        claims: firstValidation.validations.map((validation) => validation.claim),
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
      const suppressionReason = "relational_guard_reaudit_failed";

      emitTrace({
        tracer: this.options.tracer,
        turnId: input.turnId,
        claims: firstValidation.validations.map((validation) => validation.claim),
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
      response: rewritten,
      evidence: input.evidence,
      currentSessionId: input.currentSessionId,
      currentTurnTs: input.currentTurnTs,
      hasCorrectivePreferenceEvidence: this.options.hasCorrectivePreferenceEvidence,
    });

    if (secondValidation.unsupported.length > 0) {
      const suppressionReason = "relational_guard_rewrite_unsupported";

      emitTrace({
        tracer: this.options.tracer,
        turnId: input.turnId,
        claims: secondValidation.validations.map((validation) => validation.claim),
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
      claims: secondValidation.validations.map((validation) => validation.claim),
      validations: secondValidation.validations,
      unsupported: secondValidation.unsupported,
      verdict: "rewritten",
      firstClaims: firstValidation.validations.map((validation) => validation.claim),
      firstUnsupported: firstValidation.unsupported,
      rewrittenClaims: secondValidation.validations.map((validation) => validation.claim),
      rewrittenUnsupported: secondValidation.unsupported,
      finalVerdict: "rewritten",
      originalResponse: input.response,
      rewrittenResponse: rewritten,
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
