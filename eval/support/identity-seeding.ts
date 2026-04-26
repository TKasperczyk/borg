import type {
  Borg,
  Clock,
  EmbeddingClient,
  Episode,
  TraitRecord,
  ValueRecord,
} from "../../src/index.js";
import type { EpisodicRepository } from "../../src/memory/episodic/index.js";

import { createEpisodeFixture } from "../../src/offline/test-support.js";

type BorgInternalsCarrier = {
  deps: {
    clock: Clock;
    embeddingClient: EmbeddingClient;
    episodicRepository: EpisodicRepository;
  };
};

type EpisodeStreamSeed = {
  kind: Parameters<Borg["stream"]["append"]>[0]["kind"];
  content: Parameters<Borg["stream"]["append"]>[0]["content"];
  audience?: string;
};

export function getBorgInternals(borg: Borg): BorgInternalsCarrier["deps"] {
  return (borg as unknown as BorgInternalsCarrier).deps;
}

export async function seedStreamBackedEpisode(
  borg: Borg,
  input: Omit<Episode, "source_stream_ids" | "embedding"> & {
    streamEntries: readonly EpisodeStreamSeed[];
    embeddingText?: string;
  },
): Promise<Episode> {
  const deps = getBorgInternals(borg);
  const appendedEntries = [];

  for (const entry of input.streamEntries) {
    appendedEntries.push(
      await borg.stream.append({
        kind: entry.kind,
        content: entry.content,
        ...(entry.audience === undefined ? {} : { audience: entry.audience }),
      }),
    );
  }

  const embedding = await deps.embeddingClient.embed(
    input.embeddingText ??
      [input.title, input.narrative, input.tags.join(" "), input.participants.join(" ")].join("\n"),
  );

  const episode = createEpisodeFixture({
    ...input,
    source_stream_ids: appendedEntries.map((entry) => entry.id),
    embedding,
  });
  await deps.episodicRepository.insert(episode);
  return episode;
}

export function seedEstablishedValue(
  borg: Borg,
  input: {
    id?: Parameters<Borg["self"]["values"]["add"]>[0]["id"];
    label: string;
    description: string;
    priority: number;
    episodeIds: readonly ValueRecord["evidence_episode_ids"][number][];
    createdAt: number;
  },
): ValueRecord {
  const [firstEpisodeId, ...restEpisodeIds] = input.episodeIds;

  if (firstEpisodeId === undefined) {
    throw new Error("seedEstablishedValue requires at least one episode id");
  }

  const value = borg.self.values.add({
    ...(input.id === undefined ? {} : { id: input.id }),
    label: input.label,
    description: input.description,
    priority: input.priority,
    provenance: {
      kind: "episodes",
      episode_ids: [firstEpisodeId],
    },
    createdAt: input.createdAt,
  });
  let current = value;

  for (const [index, episodeId] of restEpisodeIds.entries()) {
    current = borg.self.values.reinforce(
      value.id,
      {
        kind: "episodes",
        episode_ids: [episodeId],
      },
      input.createdAt + index + 1,
    );
  }

  return current;
}

export function seedEstablishedTrait(
  borg: Borg,
  input: {
    label: string;
    delta: number;
    episodeIds: readonly TraitRecord["evidence_episode_ids"][number][];
    timestampStart: number;
  },
): TraitRecord {
  if (input.episodeIds.length === 0) {
    throw new Error("seedEstablishedTrait requires at least one episode id");
  }

  let current: TraitRecord | null = null;

  for (const [index, episodeId] of input.episodeIds.entries()) {
    current = borg.self.traits.reinforce({
      label: input.label,
      delta: input.delta,
      provenance: {
        kind: "episodes",
        episode_ids: [episodeId],
      },
      timestamp: input.timestampStart + index,
    });
  }

  if (current === null) {
    throw new Error("Failed to seed established trait");
  }

  return current;
}
