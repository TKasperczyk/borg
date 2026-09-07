import { describe, expect, it } from "vitest";

import { sidecarEmbeddingProfileFromEnv } from "./embedding-config.js";

describe("sidecar embedding configuration", () => {
  it("uses the explicitly configured model and dimensions", () => {
    expect(
      sidecarEmbeddingProfileFromEnv({ EMBEDDING_MODEL: "scw/bge-m3", EMBEDDING_DIMS: "1024" }),
    ).toEqual({ model: "scw/bge-m3", dimensions: 1024 });
  });

  it.each([undefined, "", " "])("requires the shared client's model (%s)", (model) => {
    expect(() =>
      sidecarEmbeddingProfileFromEnv({
        EMBEDDING_MODEL: model,
        EMBEDDING_DIMS: "1024",
        BORG_EMBEDDING_MODEL: "library-model",
      }),
    ).toThrow("EMBEDDING_MODEL is required");
  });

  it.each([undefined, "", " "])("requires the shared client's dimensions (%s)", (dims) => {
    expect(() =>
      sidecarEmbeddingProfileFromEnv({
        EMBEDDING_MODEL: "scw/bge-m3",
        EMBEDDING_DIMS: dims,
        BORG_EMBEDDING_DIMS: "1024",
      }),
    ).toThrow("EMBEDDING_DIMS is required");
  });

  it.each(["0", "-1", "1.5", "Infinity", "NaN", "invalid"])(
    "rejects invalid dimensions %s",
    (dims) => {
      expect(() =>
        sidecarEmbeddingProfileFromEnv({
          EMBEDDING_MODEL: "scw/bge-m3",
          EMBEDDING_DIMS: dims,
        }),
      ).toThrow("EMBEDDING_DIMS must be a positive integer");
    },
  );
});
