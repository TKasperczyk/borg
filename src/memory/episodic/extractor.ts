import type { EmbeddingClient } from "../../embeddings/index.js";
import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import {
  affectiveSignalSchema,
  emotionalArcSchema,
  type AffectiveSignal,
  type EmotionalArc,
} from "../affective/index.js";
import type { EntityRepository } from "../commitments/index.js";
import {
  StreamReader,
  filterActiveStreamEntries,
  type StreamCursor,
  type StreamEntry,
} from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { LLMError } from "../../util/errors.js";
import { createEpisodeId, DEFAULT_SESSION_ID, type SessionId } from "../../util/ids.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import { normalizeEpisodeAccess } from "./access.js";
import { EpisodicRepository } from "./repository.js";
import { type Episode } from "./types.js";
import type {
  RelationalSlotAssertionConfirmation,
  RelationalSlotRepository,
} from "../relational-slots/index.js";
import type { WorkingMemoryStore } from "../working/index.js";

import { z } from "zod";

const extractorCandidateSchema = z.object({
  title: z.string().min(1),
  narrative: z.string().min(1),
  source_stream_ids: z.array(z.string().min(1)).min(1),
  participants: z.array(z.string().min(1)),
  location: z.string().min(1).nullable(),
  tags: z.array(z.string().min(1)),
  emotional_arc: emotionalArcSchema.nullable().default(null),
  confidence: z.number().min(0).max(1),
  significance: z.number().min(0).max(1),
});

const relationalSlotUpdateCandidateSchema = z
  .object({
    subject_entity_id: z.string().min(1),
    slot_key: z.string().min(1),
    asserted_value: z.string().min(1),
    source_stream_entry_ids: z.array(z.string().min(1)).min(1),
    confirmation_kind: z.enum(["direct", "explicit"]),
  })
  .strict();

const extractorResponseSchema = z.object({
  episodes: z.array(extractorCandidateSchema),
  relational_slot_updates: z.array(relationalSlotUpdateCandidateSchema).default([]),
});

type ExtractorCandidate = z.infer<typeof extractorCandidateSchema>;
type RelationalSlotUpdateCandidate = z.infer<typeof relationalSlotUpdateCandidateSchema>;
type ExtractorResponse = z.infer<typeof extractorResponseSchema>;
type RelationalSlotSubject = {
  entity_id: EntityId;
  label: string;
  source: "default_user" | "audience";
};
const EXTRACT_EPISODES_TOOL_NAME = "EmitEpisodeCandidates";
const EPISODIC_SOURCE_STREAM_KINDS = ["user_msg", "agent_msg"] as const;
const EPISODIC_CONTEXT_STREAM_KINDS = [
  "user_msg",
  "agent_msg",
  "agent_suppressed",
  "perception",
  "internal_event",
] as const;
const perceptionAffectiveContentSchema = z.object({
  affectiveSignal: affectiveSignalSchema,
  affectiveSignalDegraded: z.boolean().optional(),
});
const perceptionContextContentSchema = z.object({
  mode: z.string().min(1),
  entities: z.array(z.string().min(1)),
  temporalCue: z.unknown().nullable().default(null),
  affectiveSignal: affectiveSignalSchema,
  affectiveSignalDegraded: z.boolean().optional(),
});
export const EXTRACT_EPISODES_TOOL = {
  name: EXTRACT_EPISODES_TOOL_NAME,
  description: "Emit grounded episodic memory candidates for the provided stream chunk.",
  inputSchema: toToolInputSchema(extractorResponseSchema),
} satisfies LLMToolDefinition;

export type EpisodicExtractorOptions = {
  dataDir: string;
  episodicRepository: EpisodicRepository;
  embeddingClient: EmbeddingClient;
  llmClient: LLMClient;
  model: string;
  entityRepository: EntityRepository;
  relationalSlotRepository?: RelationalSlotRepository;
  workingMemoryStore?: Pick<WorkingMemoryStore, "sanitizePendingActionsForRelationalSlot">;
  defaultUser?: string;
  clock?: Clock;
  chunkTokenLimit?: number;
  maxTokens?: number;
};

export type ExtractFromStreamOptions = {
  session?: SessionId;
  sinceTs?: number;
  sinceCursor?: StreamCursor;
  untilTs?: number;
  untilCursor?: StreamCursor;
};

export type ExtractFromStreamResult = {
  inserted: number;
  updated: number;
  skipped: number;
};

function estimateTokens(entry: StreamEntry): number {
  if (entry.token_estimate !== undefined) {
    return entry.token_estimate;
  }

  const content =
    typeof entry.content === "string" ? entry.content : JSON.stringify(entry.content ?? null);
  return Math.max(1, Math.ceil(content.length / 4));
}

function chunkEntries(entries: readonly StreamEntry[], maxTokens: number): StreamEntry[][] {
  if (entries.length === 0) {
    return [];
  }

  const chunks: StreamEntry[][] = [];
  let currentChunk: StreamEntry[] = [];
  let currentTokens = 0;

  for (const entry of entries) {
    const entryTokens = estimateTokens(entry);

    if (currentChunk.length > 0 && currentTokens + entryTokens > maxTokens) {
      chunks.push(currentChunk);
      currentChunk = [];
      currentTokens = 0;
    }

    currentChunk.push(entry);
    currentTokens += entryTokens;
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk);
  }

  return chunks;
}

function uniqueStrings(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function uniqueStreamEntryIds(entries: readonly StreamEntry[]): Episode["source_stream_ids"] {
  return [...new Set(entries.map((entry) => entry.id))];
}

function isEpisodicSourceEntry(entry: StreamEntry): boolean {
  return (EPISODIC_SOURCE_STREAM_KINDS as readonly string[]).includes(entry.kind);
}

function suppressedUserEntryId(entry: StreamEntry): string | null {
  if (entry.kind !== "agent_suppressed") {
    return null;
  }

  if (entry.content === null || typeof entry.content !== "object" || Array.isArray(entry.content)) {
    return null;
  }

  const userEntryId = (entry.content as { user_entry_id?: unknown }).user_entry_id;

  return typeof userEntryId === "string" && userEntryId.length > 0 ? userEntryId : null;
}

function streamOrderComparator(
  streamOrderById: ReadonlyMap<StreamEntry["id"], number>,
): (left: StreamEntry, right: StreamEntry) => number {
  return (left, right) => {
    const leftOrder = streamOrderById.get(left.id);
    const rightOrder = streamOrderById.get(right.id);

    if (leftOrder !== undefined && rightOrder !== undefined) {
      return leftOrder - rightOrder;
    }

    return left.timestamp - right.timestamp;
  };
}

function perceptionSignalFromEntry(entry: StreamEntry): AffectiveSignal | null {
  if (entry.kind !== "perception") {
    return null;
  }

  const parsed = perceptionAffectiveContentSchema.safeParse(entry.content);

  if (!parsed.success || parsed.data.affectiveSignalDegraded === true) {
    return null;
  }

  return parsed.data.affectiveSignal;
}

function perceptionSignalForUserEntry(
  userEntry: StreamEntry,
  contextEntries: readonly StreamEntry[],
  streamOrderById: ReadonlyMap<StreamEntry["id"], number>,
): AffectiveSignal | null {
  const userIndex = streamOrderById.get(userEntry.id);

  if (userIndex === undefined) {
    return null;
  }

  let nextSourceIndex = contextEntries.length;

  for (let index = userIndex + 1; index < contextEntries.length; index += 1) {
    const entry = contextEntries[index];

    if (entry !== undefined && isEpisodicSourceEntry(entry)) {
      nextSourceIndex = index;
      break;
    }
  }

  for (let index = userIndex + 1; index < nextSourceIndex; index += 1) {
    const entry = contextEntries[index];

    if (entry === undefined || (entry.audience ?? null) !== (userEntry.audience ?? null)) {
      continue;
    }

    const signal = perceptionSignalFromEntry(entry);

    if (signal !== null) {
      return signal;
    }
  }

  return null;
}

function perceptionContextLine(entry: StreamEntry): string | null {
  if (entry.kind !== "perception") {
    return null;
  }

  const parsed = perceptionContextContentSchema.safeParse(entry.content);

  if (!parsed.success) {
    return null;
  }

  return JSON.stringify({
    timestamp: entry.timestamp,
    mode: parsed.data.mode,
    entities: parsed.data.entities,
    temporalCue: parsed.data.temporalCue,
    affectiveSignal: parsed.data.affectiveSignal,
    affectiveSignalDegraded: parsed.data.affectiveSignalDegraded === true,
    audience: entry.audience,
  });
}

function perceptionContextEntriesForChunk(
  chunk: readonly StreamEntry[],
  contextEntries: readonly StreamEntry[],
): StreamEntry[] {
  if (chunk.length === 0) {
    return [];
  }

  const chunkIds = new Set(chunk.map((entry) => entry.id));
  const chunkIndexes = contextEntries.flatMap((entry, index) =>
    chunkIds.has(entry.id) ? [index] : [],
  );

  if (chunkIndexes.length === 0) {
    return [];
  }

  const sourceAudiences = new Set(chunk.map((entry) => entry.audience ?? null));
  const startIndex = Math.min(...chunkIndexes);
  const endIndex = Math.max(...chunkIndexes);

  return contextEntries
    .slice(startIndex, endIndex + 1)
    .filter(
      (entry) =>
        entry.kind === "perception" &&
        sourceAudiences.has(entry.audience ?? null) &&
        perceptionContextLine(entry) !== null,
    );
}

function buildEmotionalArc(
  sourceEntries: readonly StreamEntry[],
  contextEntries: readonly StreamEntry[],
): EmotionalArc | null {
  const streamOrderById = new Map(contextEntries.map((entry, index) => [entry.id, index]));
  const userEntries = sourceEntries
    .filter((entry) => entry.kind === "user_msg")
    .sort(streamOrderComparator(streamOrderById));

  if (userEntries.length === 0) {
    return null;
  }

  const signals = userEntries
    .map((entry) => perceptionSignalForUserEntry(entry, contextEntries, streamOrderById))
    .filter((signal): signal is AffectiveSignal => signal !== null);

  if (signals.length === 0) {
    return null;
  }
  const byPeak = [...signals].sort(
    (left, right) =>
      Math.abs(right.valence) + right.arousal - (Math.abs(left.valence) + left.arousal),
  );

  return {
    start: {
      valence: signals[0]?.valence ?? 0,
      arousal: signals[0]?.arousal ?? 0,
    },
    peak: {
      valence: byPeak[0]?.valence ?? 0,
      arousal: byPeak[0]?.arousal ?? 0,
    },
    end: {
      valence: signals.at(-1)?.valence ?? 0,
      arousal: signals.at(-1)?.arousal ?? 0,
    },
    dominant_emotion:
      signals.find((signal) => signal.dominant_emotion !== "neutral")?.dominant_emotion ?? null,
  };
}

function buildExtractorPrompt(
  chunk: readonly StreamEntry[],
  perceptionContextEntries: readonly StreamEntry[],
  relationalSlotSubjects: readonly RelationalSlotSubject[],
): string {
  const lines = chunk.map((entry) =>
    JSON.stringify({
      id: entry.id,
      timestamp: entry.timestamp,
      kind: entry.kind,
      content: entry.content,
      audience: entry.audience,
    }),
  );
  const perceptionLines = perceptionContextEntries.flatMap((entry) => {
    const line = perceptionContextLine(entry);

    return line === null ? [] : [line];
  });

  const promptLines = [
    "You extract episodic memories from a stream chunk.",
    `Emit your result by calling the ${EXTRACT_EPISODES_TOOL_NAME} tool exactly once.`,
    "source_stream_ids MUST only reference ids present in the chunk.",
    "Perception context is advisory only; NEVER include perception context entries in source_stream_ids.",
    "Narrative should be 2-5 concise sentences.",
    "For each episode, emit emotional_arc directly from the episode text and user signals. Use null only when there is no meaningful affective signal.",
    "Also emit relational_slot_updates for user-asserted relational attributes whose subject_entity_id appears in the supplied relational slot subject manifest.",
    "Use relational_slot_updates only for direct user assertions, not assistant statements, guesses, corrections, denials, or uncertainty.",
    "If an assistant introduced a person name and a later user merely reuses that name, do not emit a relational_slot_update for the name. Bare adoption is not explicit confirmation.",
    'Explicit confirmation can support a relational_slot_update, for example "her name is Marta", "yes, Marta", or "Marta is the tutor".',
    'For every relational_slot_update, emit confirmation_kind as "direct" for an ordinary user assertion or "explicit" when the user is explicitly confirming a previously uncertain, assistant-seeded, or contested value.',
    "relational_slot_updates.source_stream_entry_ids MUST reference user_msg ids present in the chunk.",
    'Use compact slot_key dot paths such as "partner.name", "partner.role", or "dog.name".',
    "Chunk:",
    ...lines,
  ];

  if (relationalSlotSubjects.length > 0) {
    promptLines.push(
      "<relational_slot_subjects>",
      ...relationalSlotSubjects.map((subject) => JSON.stringify(subject)),
      "</relational_slot_subjects>",
    );
  }

  if (perceptionLines.length > 0) {
    promptLines.push("<perception_context>", ...perceptionLines, "</perception_context>");
  }

  return promptLines.join("\n");
}

function parseLlmResponse(result: LLMCompleteResult): ExtractorResponse {
  const call = result.tool_calls.find((toolCall) => toolCall.name === EXTRACT_EPISODES_TOOL_NAME);

  if (call === undefined) {
    throw new LLMError(`Extractor did not emit tool ${EXTRACT_EPISODES_TOOL_NAME}`, {
      code: "EXTRACTOR_OUTPUT_INVALID",
    });
  }

  const parsed = extractorResponseSchema.safeParse(call.input);

  if (!parsed.success) {
    throw new LLMError("Extractor returned invalid episode payload", {
      cause: parsed.error,
      code: "EXTRACTOR_OUTPUT_INVALID",
    });
  }

  return parsed.data;
}

function sourceEntriesFromCandidate(
  candidate: ExtractorCandidate,
  chunkEntriesById: Map<string, StreamEntry>,
): StreamEntry[] {
  const entries = candidate.source_stream_ids
    .map((sourceId) => chunkEntriesById.get(sourceId))
    .filter((entry): entry is StreamEntry => entry !== undefined);

  if (entries.length !== candidate.source_stream_ids.length) {
    throw new LLMError("Extractor referenced stream ids outside the chunk", {
      code: "EXTRACTOR_SOURCE_ID_INVALID",
    });
  }

  return entries;
}

function sourceEntriesFromRelationalSlotUpdate(
  candidate: RelationalSlotUpdateCandidate,
  chunkEntriesById: Map<string, StreamEntry>,
): StreamEntry[] {
  const entries = candidate.source_stream_entry_ids
    .map((sourceId) => chunkEntriesById.get(sourceId))
    .filter((entry): entry is StreamEntry => entry !== undefined);

  if (entries.length !== candidate.source_stream_entry_ids.length) {
    throw new LLMError("Relational slot update referenced stream ids outside the chunk", {
      code: "EXTRACTOR_SOURCE_ID_INVALID",
    });
  }

  if (entries.some((entry) => entry.kind !== "user_msg")) {
    throw new LLMError("Relational slot update cited non-user stream evidence", {
      code: "EXTRACTOR_SOURCE_ID_INVALID",
    });
  }

  return entries;
}

function uniqueRelationalSlotSubjects(
  subjects: readonly RelationalSlotSubject[],
): RelationalSlotSubject[] {
  const unique: RelationalSlotSubject[] = [];

  for (const subject of subjects) {
    if (unique.some((existing) => existing.entity_id === subject.entity_id)) {
      continue;
    }

    unique.push(subject);
  }

  return unique;
}

function humanAudienceForRelationalSlot(
  sourceEntries: readonly StreamEntry[],
  relationalSlotSubjects: readonly RelationalSlotSubject[],
): RelationalSlotSubject | null {
  const audiences = uniqueStrings(
    sourceEntries.flatMap((entry) => {
      const audience = entry.audience?.trim();

      return audience === undefined || audience.length === 0 || audience === "self"
        ? []
        : [audience];
    }),
  );

  if (audiences.length !== 1) {
    return null;
  }

  const audience = audiences[0];

  return (
    relationalSlotSubjects.find(
      (subject) => subject.source === "audience" && subject.label === audience,
    ) ?? null
  );
}

function defaultUserRelationalSlotSubject(
  relationalSlotSubjects: readonly RelationalSlotSubject[],
): RelationalSlotSubject | null {
  return relationalSlotSubjects.find((subject) => subject.source === "default_user") ?? null;
}

function resolveRelationalSlotSubjectEntityId(
  subjectEntityId: string,
  sourceEntries: readonly StreamEntry[],
  relationalSlotSubjects: readonly RelationalSlotSubject[],
): EntityId {
  const candidate = subjectEntityId.trim();

  if (candidate === "user") {
    return (
      (
        humanAudienceForRelationalSlot(sourceEntries, relationalSlotSubjects) ??
        defaultUserRelationalSlotSubject(relationalSlotSubjects)
      )?.entity_id ?? (candidate as EntityId)
    );
  }

  return candidate as EntityId;
}

const ASSISTANT_SEEDED_USER_TURN_WINDOW = 2;

function streamEntryText(entry: StreamEntry): string {
  if (typeof entry.content === "string") {
    return entry.content;
  }

  try {
    return JSON.stringify(entry.content);
  } catch {
    return String(entry.content);
  }
}

function entryMentionsValue(entry: StreamEntry, value: string): boolean {
  return streamEntryText(entry).includes(value);
}

function priorContextWindow(
  sourceEntry: StreamEntry,
  contextEntries: readonly StreamEntry[],
): StreamEntry[] {
  const sourceIndex = contextEntries.findIndex((entry) => entry.id === sourceEntry.id);

  if (sourceIndex <= 0) {
    return [];
  }

  let startIndex = sourceIndex;
  let userTurns = 0;

  for (let index = sourceIndex - 1; index >= 0; index -= 1) {
    const entry = contextEntries[index];

    if (entry === undefined) {
      continue;
    }

    if (entry.kind === "user_msg") {
      userTurns += 1;

      if (userTurns > ASSISTANT_SEEDED_USER_TURN_WINDOW) {
        break;
      }
    }

    startIndex = index;
  }

  return contextEntries.slice(startIndex, sourceIndex);
}

function isAssistantSeededValue(input: {
  value: string;
  sourceEntries: readonly StreamEntry[];
  contextEntries: readonly StreamEntry[];
}): boolean {
  for (const sourceEntry of input.sourceEntries) {
    const window = priorContextWindow(sourceEntry, input.contextEntries);

    for (const [index, entry] of window.entries()) {
      if (entry.kind !== "agent_msg" || !entryMentionsValue(entry, input.value)) {
        continue;
      }

      const earlierUserMention = window
        .slice(0, index)
        .some(
          (candidate) =>
            candidate.kind === "user_msg" && entryMentionsValue(candidate, input.value),
        );

      if (!earlierUserMention) {
        return true;
      }
    }
  }

  return false;
}

function relationalSlotAssertionConfirmation(input: {
  candidate: RelationalSlotUpdateCandidate;
  sourceEntries: readonly StreamEntry[];
  contextEntries: readonly StreamEntry[];
}): RelationalSlotAssertionConfirmation {
  if (input.candidate.confirmation_kind === "explicit") {
    return "explicit";
  }

  if (
    isAssistantSeededValue({
      value: input.candidate.asserted_value,
      sourceEntries: input.sourceEntries,
      contextEntries: input.contextEntries,
    })
  ) {
    return "assistant_seeded";
  }

  return "direct";
}

function buildEpisodeFromCandidate(
  candidate: ExtractorCandidate,
  sourceEntries: readonly StreamEntry[],
  contextEntries: readonly StreamEntry[],
  access: Pick<Episode, "audience_entity_id" | "shared">,
  embedding: Float32Array,
  nowMs: number,
): Episode {
  const timestamps = sourceEntries.map((entry) => entry.timestamp);

  return normalizeEpisodeAccess({
    id: createEpisodeId(),
    title: candidate.title.trim(),
    narrative: candidate.narrative.trim(),
    participants: uniqueStrings(candidate.participants),
    location: candidate.location ?? null,
    start_time: Math.min(...timestamps),
    end_time: Math.max(...timestamps),
    source_stream_ids: uniqueStreamEntryIds(sourceEntries),
    significance: candidate.significance,
    tags: uniqueStrings(candidate.tags),
    confidence: candidate.confidence,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    emotional_arc: candidate.emotional_arc ?? buildEmotionalArc(sourceEntries, contextEntries),
    audience_entity_id: access.audience_entity_id,
    shared: access.shared,
    embedding,
    created_at: nowMs,
    updated_at: nowMs,
  });
}

type CandidateOutcome = "inserted" | "skipped";

export class EpisodicExtractor {
  private readonly clock: Clock;
  private readonly chunkTokenLimit: number;
  private readonly maxTokens: number;

  constructor(private readonly options: EpisodicExtractorOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.chunkTokenLimit = options.chunkTokenLimit ?? 16_000;
    this.maxTokens = options.maxTokens ?? 16_000;
  }

  private deriveEpisodeAccess(
    sourceEntries: readonly StreamEntry[],
  ): Pick<Episode, "audience_entity_id" | "shared"> | null {
    const audiences = uniqueStrings(
      sourceEntries.flatMap((entry) =>
        entry.audience === undefined || entry.audience.trim().length === 0 ? [] : [entry.audience],
      ),
    );

    if (audiences.length === 0) {
      return {
        audience_entity_id: null,
        shared: true,
      };
    }

    if (audiences.length > 1) {
      return null;
    }

    return {
      audience_entity_id: this.options.entityRepository.resolve(audiences[0] ?? ""),
      shared: false,
    };
  }

  private relationalSlotSubjectsForChunk(chunk: readonly StreamEntry[]): RelationalSlotSubject[] {
    if (this.options.relationalSlotRepository === undefined) {
      return [];
    }

    const subjects: RelationalSlotSubject[] = [];
    const defaultUser = this.options.defaultUser?.trim() ?? "user";

    if (defaultUser.length > 0) {
      subjects.push({
        entity_id: this.options.entityRepository.resolve(defaultUser),
        label: defaultUser,
        source: "default_user",
      });
    }

    for (const audience of uniqueStrings(
      chunk.flatMap((entry) =>
        entry.audience === undefined || entry.audience.trim().length === 0 ? [] : [entry.audience],
      ),
    )) {
      subjects.push({
        entity_id: this.options.entityRepository.resolve(audience),
        label: audience,
        source: "audience",
      });
    }

    return uniqueRelationalSlotSubjects(subjects);
  }

  private async processCandidate(
    candidate: ExtractorCandidate,
    chunkById: Map<string, StreamEntry>,
    contextEntries: readonly StreamEntry[],
  ): Promise<CandidateOutcome> {
    const sourceEntries = sourceEntriesFromCandidate(candidate, chunkById);
    const access = this.deriveEpisodeAccess(sourceEntries);

    if (access === null) {
      return "skipped";
    }

    const existing = await this.options.episodicRepository.findBySourceStreamIds(
      uniqueStreamEntryIds(sourceEntries),
    );

    if (existing !== null) {
      return "skipped";
    }

    const embedding = await this.options.embeddingClient.embed(
      `${candidate.title}\n${candidate.narrative}\n${candidate.tags.join(" ")}`,
    );
    const nowMs = this.clock.now();
    const nextEpisode = buildEpisodeFromCandidate(
      candidate,
      sourceEntries,
      contextEntries,
      access,
      embedding,
      nowMs,
    );
    await this.options.episodicRepository.insert(nextEpisode);
    return "inserted";
  }

  private processRelationalSlotUpdate(
    candidate: RelationalSlotUpdateCandidate,
    chunkById: Map<string, StreamEntry>,
    contextEntries: readonly StreamEntry[],
    relationalSlotSubjects: readonly RelationalSlotSubject[],
    sessionId: SessionId,
  ): void {
    const relationalSlotRepository = this.options.relationalSlotRepository;

    if (relationalSlotRepository === undefined) {
      return;
    }

    const sourceEntries = sourceEntriesFromRelationalSlotUpdate(candidate, chunkById);
    const subjectEntityId = resolveRelationalSlotSubjectEntityId(
      candidate.subject_entity_id,
      sourceEntries,
      relationalSlotSubjects,
    );
    const validSubjectIds = new Set(relationalSlotSubjects.map((subject) => subject.entity_id));

    if (!validSubjectIds.has(subjectEntityId)) {
      throw new LLMError("Relational slot update referenced an unknown subject entity", {
        code: "EXTRACTOR_SOURCE_ID_INVALID",
      });
    }

    const result = relationalSlotRepository.applyAssertion({
      subject_entity_id: subjectEntityId,
      slot_key: candidate.slot_key,
      asserted_value: candidate.asserted_value,
      source_stream_entry_ids: uniqueStreamEntryIds(sourceEntries) as StreamEntryId[],
      confirmation: relationalSlotAssertionConfirmation({
        candidate,
        sourceEntries,
        contextEntries,
      }),
    });

    if (result.constrained) {
      this.options.workingMemoryStore?.sanitizePendingActionsForRelationalSlot({
        sessionId,
        values: result.values_to_neutralize,
        neutralPhrase: result.neutral_phrase,
      });
    }
  }

  async extractFromStream(
    extractOptions: ExtractFromStreamOptions = {},
  ): Promise<ExtractFromStreamResult> {
    const session = extractOptions.session ?? DEFAULT_SESSION_ID;
    const reader = new StreamReader({
      dataDir: this.options.dataDir,
      sessionId: session,
    });
    const streamEntries: StreamEntry[] = [];
    const contextEntries: StreamEntry[] = [];

    for await (const entry of reader.iterate({
      kinds: EPISODIC_CONTEXT_STREAM_KINDS,
      sinceTs: extractOptions.sinceTs,
      sinceCursor: extractOptions.sinceCursor,
      untilTs: extractOptions.untilTs,
      untilCursor: extractOptions.untilCursor,
    })) {
      contextEntries.push(entry);

      if (isEpisodicSourceEntry(entry)) {
        streamEntries.push(entry);
      }
    }

    const activeContextEntries = filterActiveStreamEntries(contextEntries);
    const activeStreamEntryIds = new Set(activeContextEntries.map((entry) => entry.id));
    const suppressedUserEntryIds = new Set(
      activeContextEntries
        .map((entry) => suppressedUserEntryId(entry))
        .filter((entryId): entryId is string => entryId !== null),
    );
    const extractableStreamEntries = streamEntries
      .filter((entry) => activeStreamEntryIds.has(entry.id))
      .filter((entry) => entry.kind !== "user_msg" || !suppressedUserEntryIds.has(entry.id));

    if (extractableStreamEntries.length === 0) {
      return {
        inserted: 0,
        updated: 0,
        skipped: 0,
      };
    }

    let inserted = 0;
    let updated = 0;
    let skipped = 0;
    const chunks = chunkEntries(extractableStreamEntries, this.chunkTokenLimit);

    for (const chunk of chunks) {
      const chunkById = new Map(chunk.map((entry) => [entry.id, entry]));
      const perceptionContextEntries = perceptionContextEntriesForChunk(
        chunk,
        activeContextEntries,
      );
      const relationalSlotSubjects = this.relationalSlotSubjectsForChunk(chunk);
      const result = await this.options.llmClient.complete({
        model: this.options.model,
        system: "Extract episodic memories grounded only in the provided stream chunk.",
        messages: [
          {
            role: "user",
            content: buildExtractorPrompt(chunk, perceptionContextEntries, relationalSlotSubjects),
          },
        ],
        tools: [EXTRACT_EPISODES_TOOL],
        tool_choice: { type: "tool", name: EXTRACT_EPISODES_TOOL_NAME },
        max_tokens: this.maxTokens,
        budget: "episodic-extraction",
      });
      const extracted = parseLlmResponse(result);
      const candidates = extracted.episodes;

      for (const candidate of candidates) {
        const outcome = await this.processCandidate(candidate, chunkById, activeContextEntries);

        if (outcome === "inserted") {
          inserted += 1;
          continue;
        }

        skipped += 1;
      }

      for (const slotUpdate of extracted.relational_slot_updates) {
        this.processRelationalSlotUpdate(
          slotUpdate,
          chunkById,
          activeContextEntries,
          relationalSlotSubjects,
          session,
        );
      }
    }

    return {
      inserted,
      updated,
      skipped,
    };
  }
}
