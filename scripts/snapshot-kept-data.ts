import { cpSync, existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";

import { readOverseerAuditTranscript } from "../assessor/borg-transport.js";
import { Borg, DEFAULT_CONFIG, FakeEmbeddingClient } from "../src/index.js";
import { FakeLLMClient } from "../src/llm/test-support/fake-client.js";
import { buildMemorySnapshotMarkdown } from "../simulator/memory-snapshot.js";

function extractSection(markdown: string, title: string): string {
  const marker = `### ${title}`;
  const start = markdown.indexOf(marker);

  if (start === -1) {
    return `${marker}\nSection not found.`;
  }

  const next = markdown.indexOf("\n### ", start + marker.length);
  return markdown.slice(start, next === -1 ? undefined : next).trim();
}

function countCommitmentRows(section: string): number {
  return section
    .split("\n")
    .filter((line) => line.startsWith("- id=") && line.includes("directive=")).length;
}

const sourceArg = process.argv[2]?.trim();

if (sourceArg === undefined || sourceArg.length === 0) {
  process.stderr.write("usage: pnpm tsx scripts/snapshot-kept-data.ts <kept-data-dir>\n");
  process.exit(1);
}

const sourceDir = resolve(sourceArg);

if (!existsSync(sourceDir)) {
  process.stderr.write(`kept data dir not found: ${sourceDir}\n`);
  process.exit(1);
}

const tempRoot = mkdtempSync(join(tmpdir(), "borg-snapshot-kept-"));
const copiedDir = join(tempRoot, basename(sourceDir));
let borg: Borg | null = null;

try {
  cpSync(sourceDir, copiedDir, {
    recursive: true,
    dereference: true,
  });

  borg = await Borg.open({
    dataDir: copiedDir,
    embeddingDimensions: DEFAULT_CONFIG.embedding.dims,
    embeddingClient: new FakeEmbeddingClient(DEFAULT_CONFIG.embedding.dims),
    llmClient: new FakeLLMClient(),
    liveExtraction: false,
  });

  const transport = {
    getBorg: () => borg,
    readAuditTranscript: () => readOverseerAuditTranscript(copiedDir),
  } as unknown as Parameters<typeof buildMemorySnapshotMarkdown>[0]["transport"];
  const snapshot = await buildMemorySnapshotMarkdown({ transport });
  const commitmentsSection = extractSection(snapshot, "Commitments");

  process.stdout.write(`source=${sourceDir}\n`);
  process.stdout.write(`copy=${copiedDir}\n`);
  process.stdout.write(`visible_commitment_rows=${countCommitmentRows(commitmentsSection)}\n\n`);
  process.stdout.write(`${commitmentsSection}\n`);
} finally {
  if (borg !== null) {
    await borg.close();
  }

  rmSync(tempRoot, { recursive: true, force: true });
}
