import { existsSync, readdirSync } from "node:fs";
import { basename } from "node:path";

import type { FactualChallengeSignal } from "../cognition/types.js";
import type { EpisodicRepository } from "../memory/episodic/repository.js";
import type { EpisodeVisibilityOptions } from "../memory/episodic/types.js";
import {
  DEFAULT_SESSION_ID,
  getStreamDirectory,
  StreamReader,
  type StreamCursor,
  type StreamEntry,
} from "../stream/index.js";
import { parseSessionId, type EpisodeId, type SessionId, type StreamEntryId } from "../util/ids.js";

const DEFAULT_EPISODE_LIMIT = 5;
const DEFAULT_SNIPPET_LIMIT = 8;
const DEFAULT_SNIPPET_CHARS = 280;
const DEFAULT_TOTAL_SNIPPET_CHARS = 1_600;
const DEFAULT_STREAM_PAGE_SIZE = 200;

export type ChallengeEvidenceEpisode = {
  episode_id: EpisodeId;
  title: string;
  context: string;
  participants: string[];
  tags: string[];
  start_time: number;
  end_time: number;
  source_stream_ids: StreamEntryId[];
};

export type ChallengeEvidenceRawSnippet = {
  stream_entry_id: StreamEntryId;
  session_id: SessionId;
  timestamp: number;
  snippet: string;
  audience?: string;
};

export type ChallengeEvidence = {
  disputed_entity: string | null;
  disputed_property: string | null;
  user_position: string;
  episodes: ChallengeEvidenceEpisode[];
  raw_user_messages: ChallengeEvidenceRawSnippet[];
  reference_count: number;
};

export type RetrieveFactualChallengeEvidenceOptions = EpisodeVisibilityOptions & {
  challenge: FactualChallengeSignal | null | undefined;
  episodicRepository: Pick<EpisodicRepository, "searchByParticipantsOrTags">;
  dataDir: string;
  sessionIds?: readonly SessionId[];
  excludeStreamEntryIds?: readonly StreamEntryId[];
  audience?: string;
  episodeLimit?: number;
  snippetLimit?: number;
  maxSnippetChars?: number;
  maxTotalSnippetChars?: number;
  streamPageSize?: number;
};

function listSessionIds(dataDir: string, explicitSessionIds?: readonly SessionId[]): SessionId[] {
  if (explicitSessionIds !== undefined && explicitSessionIds.length > 0) {
    return [...explicitSessionIds];
  }

  const streamDir = getStreamDirectory(dataDir);

  if (!existsSync(streamDir)) {
    return [DEFAULT_SESSION_ID];
  }

  const sessionIds = readdirSync(streamDir)
    .map((filename) => {
      if (!filename.endsWith(".jsonl")) {
        return null;
      }

      try {
        return parseSessionId(basename(filename, ".jsonl"));
      } catch {
        return null;
      }
    })
    .filter((value): value is SessionId => value !== null)
    .sort();

  return sessionIds.length === 0 ? [DEFAULT_SESSION_ID] : sessionIds;
}

function contentToText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  return JSON.stringify(content) ?? "";
}

function compactText(text: string, maxChars: number): string {
  const normalized = text.replace(/\s+/g, " ").trim();

  if (normalized.length <= maxChars) {
    return normalized;
  }

  return `${normalized.slice(0, Math.max(0, maxChars - 3)).trimEnd()}...`;
}

function entryMatchesAudience(entry: StreamEntry, audience: string | undefined): boolean {
  return audience === undefined || entry.audience === undefined || entry.audience === audience;
}

function entryContainsEntity(entry: StreamEntry, normalizedEntity: string): boolean {
  return contentToText(entry.content).toLowerCase().includes(normalizedEntity);
}

async function collectRawUserMessageSnippets(input: {
  dataDir: string;
  sessionIds: readonly SessionId[];
  entity: string;
  excludeStreamEntryIds: ReadonlySet<string>;
  audience?: string;
  snippetLimit: number;
  maxSnippetChars: number;
  maxTotalSnippetChars: number;
  streamPageSize: number;
}): Promise<ChallengeEvidenceRawSnippet[]> {
  const snippets: ChallengeEvidenceRawSnippet[] = [];
  const normalizedEntity = input.entity.toLowerCase();
  let totalChars = 0;

  for (const sessionId of input.sessionIds) {
    const reader = new StreamReader({
      dataDir: input.dataDir,
      sessionId,
    });
    let cursor: StreamCursor | undefined;

    while (snippets.length < input.snippetLimit && totalChars < input.maxTotalSnippetChars) {
      const page: StreamEntry[] = [];

      for await (const entry of reader.iterate({
        kinds: ["user_msg"],
        limit: input.streamPageSize,
        ...(cursor === undefined ? {} : { sinceCursor: cursor }),
      })) {
        page.push(entry);
      }

      if (page.length === 0) {
        break;
      }

      for (const entry of page) {
        if (
          input.excludeStreamEntryIds.has(entry.id) ||
          !entryMatchesAudience(entry, input.audience) ||
          !entryContainsEntity(entry, normalizedEntity)
        ) {
          continue;
        }

        const remainingChars = input.maxTotalSnippetChars - totalChars;
        const snippet = compactText(
          contentToText(entry.content),
          Math.min(input.maxSnippetChars, remainingChars),
        );

        if (snippet.length === 0) {
          continue;
        }

        snippets.push({
          stream_entry_id: entry.id,
          session_id: entry.session_id,
          timestamp: entry.timestamp,
          snippet,
          ...(entry.audience === undefined ? {} : { audience: entry.audience }),
        });
        totalChars += snippet.length;

        if (snippets.length >= input.snippetLimit || totalChars >= input.maxTotalSnippetChars) {
          break;
        }
      }

      const lastEntry = page[page.length - 1];

      if (lastEntry === undefined || page.length < input.streamPageSize) {
        break;
      }

      cursor = {
        ts: lastEntry.timestamp,
        entryId: lastEntry.id,
      };
    }

    if (snippets.length >= input.snippetLimit || totalChars >= input.maxTotalSnippetChars) {
      break;
    }
  }

  return snippets;
}

export async function retrieveFactualChallengeEvidence(
  options: RetrieveFactualChallengeEvidenceOptions,
): Promise<ChallengeEvidence | null> {
  const challenge = options.challenge ?? null;

  if (challenge === null) {
    return null;
  }

  const disputedEntity = challenge.disputed_entity?.trim() || null;
  const episodeLimit = Math.max(1, options.episodeLimit ?? DEFAULT_EPISODE_LIMIT);
  const snippetLimit = Math.max(1, options.snippetLimit ?? DEFAULT_SNIPPET_LIMIT);
  const maxSnippetChars = Math.max(1, options.maxSnippetChars ?? DEFAULT_SNIPPET_CHARS);
  const maxTotalSnippetChars = Math.max(
    1,
    options.maxTotalSnippetChars ?? DEFAULT_TOTAL_SNIPPET_CHARS,
  );
  const streamPageSize = Math.max(1, options.streamPageSize ?? DEFAULT_STREAM_PAGE_SIZE);
  const episodeMatches =
    disputedEntity === null
      ? []
      : await options.episodicRepository.searchByParticipantsOrTags([disputedEntity], {
          limit: episodeLimit,
          audienceEntityId: options.audienceEntityId,
          crossAudience: options.crossAudience,
          globalIdentitySelfAudienceEntityId: options.globalIdentitySelfAudienceEntityId,
        });
  const rawUserMessages =
    disputedEntity === null
      ? []
      : await collectRawUserMessageSnippets({
          dataDir: options.dataDir,
          sessionIds: listSessionIds(options.dataDir, options.sessionIds),
          entity: disputedEntity,
          excludeStreamEntryIds: new Set(options.excludeStreamEntryIds ?? []),
          audience: options.audience,
          snippetLimit,
          maxSnippetChars,
          maxTotalSnippetChars,
          streamPageSize,
        });

  return {
    disputed_entity: disputedEntity,
    disputed_property: challenge.disputed_property,
    user_position: challenge.user_position,
    episodes: episodeMatches.map((match) => ({
      episode_id: match.episode.id,
      title: match.episode.title,
      context: compactText(match.episode.narrative, 320),
      participants: [...match.episode.participants],
      tags: [...match.episode.tags],
      start_time: match.episode.start_time,
      end_time: match.episode.end_time,
      source_stream_ids: [...match.episode.source_stream_ids],
    })),
    raw_user_messages: rawUserMessages,
    reference_count: episodeMatches.length + rawUserMessages.length,
  };
}
