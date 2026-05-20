import { describe, expect, it } from "vitest";

import { makeSharedStateArtifact } from "./shared-state.js";

describe("shared-state test factories", () => {
  it("preserves explicit null overrides for nullable compile metadata", () => {
    const artifact = makeSharedStateArtifact(undefined, {
      last_compiled_at: null,
      last_compiled_stream_entry_id: null,
    });

    expect(artifact.last_compiled_at).toBeNull();
    expect(artifact.last_compiled_stream_entry_id).toBeNull();
  });
});
