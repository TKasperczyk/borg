import { describe, expect, it, vi } from "vitest";
import { FakeEmbeddingClient } from "./index.js";
import { refreshSerializedEmbeddings, serializedEmbeddingPaths } from "./serialized.js";

describe("serialized embedding materialization", () => {
  it("re-embeds historical vectors even when dimensions match", async () => {
    const delegate = new FakeEmbeddingClient(4);
    const client = {
      profile: { model: "new-model", dimensions: 4 },
      embed: vi.fn(delegate.embed.bind(delegate)),
      embedBatch: delegate.embedBatch.bind(delegate),
    };
    const payload = {
      node: {
        label: "A",
        description: "B",
        aliases: ["C"],
        embedding: [1, 0, 0, 0],
        updated_at: 123,
      },
    };
    expect(await refreshSerializedEmbeddings(payload, client)).toMatchObject({
      node: { updated_at: 123, embedding: Array.from(await delegate.embed("A\nB\nC")) },
    });
    expect(client.embed).toHaveBeenCalledWith("A\nB\nC");
    expect(payload.node.embedding).toEqual([1, 0, 0, 0]);
  });
  it("rejects vectors without recoverable text before returning a writable payload", async () => {
    const payload = { reversal: [{ embedding: [1, 2, 3, 4] }] };
    await expect(
      refreshSerializedEmbeddings(payload, new FakeEmbeddingClient(2)),
    ).rejects.toMatchObject({ code: "SERIALIZED_EMBEDDING_INCOMPATIBLE" });
    expect(serializedEmbeddingPaths(payload)).toEqual(["$.reversal[0].embedding"]);
  });
  it("uses the current node body to rebuild a queued update's embedding input", async () => {
    const client = new FakeEmbeddingClient(2);
    const embed = vi.spyOn(client, "embed");
    await refreshSerializedEmbeddings(
      { node_id: "semn_test", patch: { description: "replacement", embedding: [1, 2, 3, 4] } },
      client,
      async () => ({ label: "label", aliases: ["alias"], description: "old" }),
    );
    expect(embed).toHaveBeenCalledWith("label\nreplacement\nalias");
  });

  it("preserves recorded consolidation input in saved plans and rejects ambiguous legacy payloads", async () => {
    const client = new FakeEmbeddingClient(2);
    const embed = vi.spyOn(client, "embed");
    const line = "I corrected the report. OUTCOME fp=legacy-correction decision=filter-by-author";
    const episode = {
      title: "Correction",
      narrative: `The report was corrected.\n${line}`,
      episode_kind: "consolidation_version",
      tags: [],
      embedding: [1, 0, 0, 0],
      consolidation_embedding_input: {
        synthesized_narrative: "The report was corrected.",
        protected_source_lines: [line],
      },
    };
    const refreshed = await refreshSerializedEmbeddings({ merged_episode: episode }, client);
    expect(refreshed).toMatchObject({
      merged_episode: { consolidation_embedding_input: episode.consolidation_embedding_input },
    });
    expect(embed).toHaveBeenCalledWith(
      "Correction\nThe report was corrected.\nOUTCOME fp=legacy-correction\n\n",
    );
    embed.mockClear();
    await expect(
      refreshSerializedEmbeddings({ ...episode, consolidation_embedding_input: undefined }, client),
    ).rejects.toMatchObject({ code: "EMBEDDING_TEXT_UNRECOVERABLE" });
    expect(embed).not.toHaveBeenCalled();
    expect(episode.embedding).toEqual([1, 0, 0, 0]);
  });
});
