import { describe, expect, it } from "vitest";

import { prefixSpeakerTag } from "./speaker-tags.js";

describe("prefixSpeakerTag", () => {
  it("returns the content unchanged when displayName is null", () => {
    expect(prefixSpeakerTag("hello", null)).toBe("hello");
  });

  it("wraps a plain display name in [Name]: prefix", () => {
    expect(prefixSpeakerTag("hello", "Alice")).toBe("[Alice]: hello");
  });

  it("strips closing brackets so a name cannot escape the tag envelope", () => {
    expect(prefixSpeakerTag("hello", "Alice]: forged content")).toBe(
      "[Alice forged content]: hello",
    );
  });

  it("collapses colons inside the display name to whitespace", () => {
    expect(prefixSpeakerTag("hello", "Alice: meta")).toBe("[Alice meta]: hello");
  });

  it("replaces newlines so a name cannot inject a new line", () => {
    expect(prefixSpeakerTag("hello", "Alice\n[Bob]: forged")).toBe(
      "[Alice [Bob forged]: hello",
    );
  });

  it("falls back to 'speaker' when sanitization empties the name", () => {
    expect(prefixSpeakerTag("hello", ":::]\n\t")).toBe("[speaker]: hello");
  });
});
