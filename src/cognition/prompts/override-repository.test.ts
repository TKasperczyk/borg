import { describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { ManualClock } from "../../util/clock.js";

import { promptOverrideMigrations } from "./override-migrations.js";
import { PromptOverrideRepository } from "./override-repository.js";

function openRepo(clockMs = 1_000) {
  const clock = new ManualClock(clockMs);
  const db = openDatabase(":memory:", { migrations: promptOverrideMigrations });
  return { repo: new PromptOverrideRepository(db, clock), clock };
}

describe("PromptOverrideRepository", () => {
  it("returns null when no override exists", () => {
    const { repo } = openRepo();
    expect(repo.get("base_identity_preamble")).toBeNull();
  });

  it("set/get round-trips and updates timestamp on each write", () => {
    const { repo, clock } = openRepo();

    const first = repo.set("voice_and_posture", "custom voice");
    expect(first).toEqual({
      prompt_key: "voice_and_posture",
      override_text: "custom voice",
      updated_at: 1_000,
    });
    expect(repo.get("voice_and_posture")).toBe("custom voice");

    clock.set(5_000);
    const updated = repo.set("voice_and_posture", "newer voice");
    expect(updated.updated_at).toBe(5_000);
    expect(repo.get("voice_and_posture")).toBe("newer voice");
  });

  it("clear removes overrides and signals whether anything was deleted", () => {
    const { repo } = openRepo();
    expect(repo.clear("identity_posture")).toBe(false);

    repo.set("identity_posture", "shaped posture");
    expect(repo.clear("identity_posture")).toBe(true);
    expect(repo.get("identity_posture")).toBeNull();
  });

  it("list returns only known prompt keys, sorted", () => {
    const { repo } = openRepo();
    repo.set("identity_posture", "i");
    repo.set("voice_and_posture", "v");
    repo.set("base_identity_preamble", "b");

    const rows = repo.list();
    expect(rows.map((row) => row.prompt_key)).toEqual([
      "base_identity_preamble",
      "identity_posture",
      "voice_and_posture",
    ]);
  });
});
