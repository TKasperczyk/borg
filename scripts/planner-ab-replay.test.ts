import {
  appendFileSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  assertReplayOAuthCredentialsOutsideDataDir,
  openPlannerCaptureSnapshot,
  parsePlannerAbReplayArgs,
  resolvePlannerAbReplayPaths,
} from "./planner-ab-replay.js";

describe("planner A/B replay filesystem boundaries", () => {
  const tempDirectories: string[] = [];

  function temp(name: string): string {
    const directory = mkdtempSync(join(tmpdir(), name));
    tempDirectories.push(directory);
    return directory;
  }

  afterEach(() => {
    for (const directory of tempDirectories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("keeps non-completed captures excluded unless explicitly requested", () => {
    expect(parsePlannerAbReplayArgs(["--live"]).includeNonCompleted).toBe(false);
    expect(
      parsePlannerAbReplayArgs(["--live", "--include-non-completed"]).includeNonCompleted,
    ).toBe(true);
  });

  it("defaults results below dataDir/captures even when input is elsewhere", () => {
    const dataDir = temp("borg-replay-data-");
    const external = temp("borg-replay-input-");
    const input = join(external, "captures.jsonl");
    writeFileSync(input, "");

    const paths = resolvePlannerAbReplayPaths({ dataDir, inputPath: input });

    expect(paths.outputPath).toBe(join(dataDir, "captures", "planner-ab-results.jsonl"));
  });

  it("resolves symlinks before rejecting output within dataDir but outside captures", () => {
    const dataDir = temp("borg-replay-contained-");
    const external = temp("borg-replay-external-");
    const input = join(external, "captures.jsonl");
    writeFileSync(input, "");
    mkdirSync(join(dataDir, "stream"));
    writeFileSync(join(dataDir, "stream", "result.jsonl"), "");
    symlinkSync(join(dataDir, "stream", "result.jsonl"), join(external, "result-link.jsonl"));

    expect(() =>
      resolvePlannerAbReplayPaths({
        dataDir,
        inputPath: input,
        outputPath: join(external, "result-link.jsonl"),
      }),
    ).toThrow("must stay within dataDir/captures");
  });

  it("resolves symlinks before checking input/output equality", () => {
    const dataDir = temp("borg-replay-equality-");
    const input = join(dataDir, "input.jsonl");
    const alias = join(dataDir, "input-alias.jsonl");
    writeFileSync(input, "");
    symlinkSync(input, alias);

    expect(() =>
      resolvePlannerAbReplayPaths({ dataDir, inputPath: input, outputPath: alias }),
    ).toThrow("must differ after resolving symlinks");
  });

  it("rejects a dangling output symlink instead of treating it as a creatable leaf", () => {
    const dataDir = temp("borg-replay-dangling-");
    const external = temp("borg-replay-dangling-outside-");
    const input = join(dataDir, "input.jsonl");
    const alias = join(dataDir, "captures", "result.jsonl");
    mkdirSync(join(dataDir, "captures"));
    writeFileSync(input, "");
    symlinkSync(join(external, "missing.jsonl"), alias);

    expect(() =>
      resolvePlannerAbReplayPaths({ dataDir, inputPath: input, outputPath: alias }),
    ).toThrow();
  });

  it("pins input byte length so later appends belong to the next cohort", async () => {
    const root = temp("borg-replay-snapshot-");
    const path = join(root, "captures.jsonl");
    writeFileSync(path, '{"capture":1}\n');
    const snapshot = openPlannerCaptureSnapshot(path);
    appendFileSync(path, '{"capture":2}\n');

    const lines: string[] = [];
    for await (const line of snapshot.lines) lines.push(line);

    expect(lines).toEqual(['{"capture":1}']);
  });

  it("rejects live OAuth credentials resolving within dataDir", () => {
    const dataDir = temp("borg-replay-oauth-");
    const credentials = join(dataDir, "private", "credentials.json");

    expect(() =>
      assertReplayOAuthCredentialsOutsideDataDir({
        dataDirectory: dataDir,
        env: { BORG_CLAUDE_CREDENTIALS_PATH: credentials },
      }),
    ).toThrow("must resolve outside");
  });

  it("resolves OAuth credential symlinks before containment checking", () => {
    const dataDir = temp("borg-replay-oauth-link-data-");
    const external = temp("borg-replay-oauth-link-external-");
    const credentials = join(dataDir, "credentials.json");
    const alias = join(external, "credentials-link.json");
    writeFileSync(credentials, "{}");
    symlinkSync(credentials, alias);

    expect(() =>
      assertReplayOAuthCredentialsOutsideDataDir({
        dataDirectory: dataDir,
        env: { BORG_CLAUDE_CREDENTIALS_PATH: alias },
      }),
    ).toThrow("must resolve outside");
  });
});
