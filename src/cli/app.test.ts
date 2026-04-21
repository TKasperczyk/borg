import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { runCli } from "./app.js";

function createOutputBuffer() {
  let value = "";

  return {
    stream: {
      write(chunk: string) {
        value += chunk;
        return true;
      },
    },
    read() {
      return value;
    },
  };
}

describe("cli", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("prints the version", async () => {
    const stdout = createOutputBuffer();
    const stderr = createOutputBuffer();

    const exitCode = await runCli(["node", "borg", "version"], {
      stdout: stdout.stream,
      stderr: stderr.stream,
    });

    expect(exitCode).toBe(0);
    expect(stdout.read()).toContain("borg 0.1.0");
    expect(stderr.read()).toBe("");
  });

  it("shows redacted config", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const stdout = createOutputBuffer();
    const stderr = createOutputBuffer();

    const exitCode = await runCli(["node", "borg", "config", "show"], {
      stdout: stdout.stream,
      stderr: stderr.stream,
      dataDir: tempDir,
      env: {
        ANTHROPIC_API_KEY: "secret",
      },
    });

    expect(exitCode).toBe(0);
    expect(JSON.parse(stdout.read())).toMatchObject({
      dataDir: tempDir,
      anthropic: {
        apiKey: "[REDACTED]",
      },
    });
    expect(stderr.read()).toBe("");
  });

  it("appends to and tails the stream", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const appendOut = createOutputBuffer();
    const appendErr = createOutputBuffer();

    expect(
      await runCli(
        ["node", "borg", "stream", "append", "--kind", "user_msg", "--content", "hello"],
        {
          stdout: appendOut.stream,
          stderr: appendErr.stream,
          dataDir: tempDir,
        },
      ),
    ).toBe(0);

    const tailOut = createOutputBuffer();
    const tailErr = createOutputBuffer();

    expect(
      await runCli(["node", "borg", "stream", "tail", "--n", "1"], {
        stdout: tailOut.stream,
        stderr: tailErr.stream,
        dataDir: tempDir,
      }),
    ).toBe(0);

    expect(JSON.parse(tailOut.read())).toMatchObject({
      kind: "user_msg",
      content: "hello",
    });
    expect(appendErr.read()).toBe("");
    expect(tailErr.read()).toBe("");
  });

  it("surfaces invalid session ids as clean cli errors", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const stdout = createOutputBuffer();
    const stderr = createOutputBuffer();

    const exitCode = await runCli(
      ["node", "borg", "stream", "tail", "--session", "not-a-session"],
      {
        stdout: stdout.stream,
        stderr: stderr.stream,
        dataDir: tempDir,
      },
    );

    expect(exitCode).toBe(1);
    expect(stderr.read()).toContain("Invalid session id");
  });
});
