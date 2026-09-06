/**
 * Usage: pnpm finalizer:cache-report -- <captures/finalizer-contexts.jsonl>
 *
 * Streams a frozen JSONL snapshot. By default compares consecutive autonomous
 * surfaces; --same-session compares all origins within each session instead.
 */
import { isPlainRecord } from "../src/util/guards.ts";
import { runAbCliEntrypoint } from "./ab-cli.ts";
import { openPlannerCaptureSnapshot } from "./planner-ab-replay.ts";

export const FINALIZER_CACHE_REPORT_USAGE =
  "Usage: pnpm finalizer:cache-report -- <captures/finalizer-contexts.jsonl> [--same-session]";

type AutonomousSurface = {
  captureId: string;
  blocks: readonly string[];
};

type TaggedSection = {
  blockIndex: number;
  tag: string;
  ordinal: number;
  text: string;
};

export type FinalizerCachePairReport = {
  previous_capture_id: string;
  current_capture_id: string;
  blocks: readonly {
    block_index: number;
    common_prefix_bytes: number;
    previous_chars: number;
    current_chars: number;
  }[];
  sections: readonly {
    block_index: number;
    tag: string;
    ordinal: number;
    byte_stable: boolean;
    previous_chars: number | null;
    current_chars: number | null;
  }[];
};

export type FinalizerCacheReportSummary = {
  autonomousCaptures: number;
  consecutivePairs: number;
};

const TOP_LEVEL_TAG_LINE = /^<([a-z][a-z0-9_]*)(?:[ \t][^<>\r\n]*)?>[ \t]*(?:\r?\n|$)/gm;

export function commonUtf8PrefixBytes(left: string, right: string): number {
  const leftBytes = Buffer.from(left, "utf8");
  const rightBytes = Buffer.from(right, "utf8");
  const limit = Math.min(leftBytes.length, rightBytes.length);
  let index = 0;
  while (index < limit && leftBytes[index] === rightBytes[index]) index += 1;
  return index;
}

export function extractTopLevelTaggedSections(block: string, blockIndex: number): TaggedSection[] {
  const sections: TaggedSection[] = [];
  const ordinals = new Map<string, number>();
  let consumedUntil = 0;
  TOP_LEVEL_TAG_LINE.lastIndex = 0;
  for (const opening of block.matchAll(TOP_LEVEL_TAG_LINE)) {
    const start = opening.index;
    if (start < consumedUntil) continue;
    const tag = opening[1]!;
    const ordinal = (ordinals.get(tag) ?? 0) + 1;
    const selfClosing = /\/>[ \t]*(?:\r?\n|$)$/.test(opening[0]);
    let end: number;
    if (selfClosing) {
      end = start + opening[0].replace(/\r?\n$/, "").length;
    } else {
      const close = `</${tag}>`;
      const closeStart = block.indexOf(close, start + opening[0].length);
      if (closeStart === -1) continue;
      end = closeStart + close.length;
    }
    ordinals.set(tag, ordinal);
    sections.push({ blockIndex, tag, ordinal, text: block.slice(start, end) });
    consumedUntil = end;
  }
  return sections;
}

function compactSystemBlocks(value: unknown, location: string): readonly string[] {
  if (typeof value === "string") return [value];
  if (!Array.isArray(value)) throw new Error(`Missing compact system blocks at ${location}`);
  return value.map((block, blockIndex) => {
    if (!isPlainRecord(block) || block.type !== "text" || typeof block.text !== "string") {
      throw new Error(`Invalid compact system block ${blockIndex} at ${location}`);
    }
    return block.text;
  });
}

function autonomousSurface(
  value: unknown,
  location: string,
  sameSession: boolean,
): AutonomousSurface | null {
  if (!isPlainRecord(value)) throw new Error(`Capture is not an object at ${location}`);
  if (!sameSession && value.turn_origin !== "autonomous") return null;
  if (typeof value.capture_id !== "string") {
    throw new Error(`Missing capture_id at ${location}`);
  }
  if (!isPlainRecord(value.surfaces) || !isPlainRecord(value.surfaces.compact)) {
    throw new Error(`Missing compact surface at ${location}`);
  }
  return {
    captureId: value.capture_id,
    blocks: compactSystemBlocks(value.surfaces.compact.system, location),
  };
}

function sectionKey(section: Pick<TaggedSection, "blockIndex" | "tag" | "ordinal">): string {
  return `${section.blockIndex}:${section.tag}:${section.ordinal}`;
}

export function compareAutonomousSurfaces(
  previous: AutonomousSurface,
  current: AutonomousSurface,
): FinalizerCachePairReport {
  const blockCount = Math.max(previous.blocks.length, current.blocks.length);
  const blocks = Array.from({ length: blockCount }, (_, blockIndex) => {
    const previousText = previous.blocks[blockIndex] ?? "";
    const currentText = current.blocks[blockIndex] ?? "";
    return {
      block_index: blockIndex,
      common_prefix_bytes: commonUtf8PrefixBytes(previousText, currentText),
      previous_chars: previousText.length,
      current_chars: currentText.length,
    };
  });
  const previousSections = previous.blocks.flatMap((block, blockIndex) =>
    extractTopLevelTaggedSections(block, blockIndex),
  );
  const currentSections = current.blocks.flatMap((block, blockIndex) =>
    extractTopLevelTaggedSections(block, blockIndex),
  );
  const previousByKey = new Map(previousSections.map((section) => [sectionKey(section), section]));
  const currentByKey = new Map(currentSections.map((section) => [sectionKey(section), section]));
  const orderedKeys = [
    ...previousByKey.keys(),
    ...[...currentByKey.keys()].filter((key) => !previousByKey.has(key)),
  ];
  const sections = orderedKeys.map((key) => {
    const previousSection = previousByKey.get(key);
    const currentSection = currentByKey.get(key);
    const identity = previousSection ?? currentSection!;
    return {
      block_index: identity.blockIndex,
      tag: identity.tag,
      ordinal: identity.ordinal,
      byte_stable:
        previousSection !== undefined &&
        currentSection !== undefined &&
        previousSection.text === currentSection.text,
      previous_chars: previousSection?.text.length ?? null,
      current_chars: currentSection?.text.length ?? null,
    };
  });
  return {
    previous_capture_id: previous.captureId,
    current_capture_id: current.captureId,
    blocks,
    sections,
  };
}

export async function analyzeFinalizerCaptureFile(
  path: string,
  onPair: (report: FinalizerCachePairReport) => void,
  options: { sameSession?: boolean } = {},
): Promise<FinalizerCacheReportSummary> {
  const { snapshotBytes, lines } = openPlannerCaptureSnapshot(path);
  let previousAutonomous: AutonomousSurface | null = null;
  const previousBySession = new Map<string, AutonomousSurface>();
  let autonomousCaptures = 0;
  let consecutivePairs = 0;
  let lineNumber = 0;

  for await (const line of lines) {
    lineNumber += 1;
    if (snapshotBytes === 0 || line.length === 0) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(line) as unknown;
    } catch (error) {
      throw new Error(
        `Invalid capture JSON at ${path}:${lineNumber}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    const current = autonomousSurface(
      parsed,
      `${path}:${lineNumber}`,
      options.sameSession === true,
    );
    if (current === null) continue;
    if (isPlainRecord(parsed) && parsed.turn_origin === "autonomous") autonomousCaptures += 1;
    let sessionId: string | null = null;
    if (options.sameSession) {
      if (!isPlainRecord(parsed) || typeof parsed.session_id !== "string") {
        throw new Error(`Missing session_id at ${path}:${lineNumber}`);
      }
      sessionId = parsed.session_id;
    }
    const previous =
      sessionId === null ? previousAutonomous : (previousBySession.get(sessionId) ?? null);
    if (previous !== null) {
      onPair(compareAutonomousSurfaces(previous, current));
      consecutivePairs += 1;
    }
    if (sessionId === null) previousAutonomous = current;
    else previousBySession.set(sessionId, current);
  }

  return { autonomousCaptures, consecutivePairs };
}

export async function runFinalizerCacheReportCli(argv: readonly string[]): Promise<void> {
  if (
    (argv.length !== 1 && !(argv.length === 2 && argv[1] === "--same-session")) ||
    argv[0] === "--help" ||
    argv[0] === "-h"
  ) {
    throw new Error(FINALIZER_CACHE_REPORT_USAGE);
  }
  const summary = await analyzeFinalizerCaptureFile(
    argv[0]!,
    (report) => {
      console.log(JSON.stringify(report));
    },
    { sameSession: argv[1] === "--same-session" },
  );
  console.error(
    `finalizer cache report complete: autonomous_captures=${summary.autonomousCaptures} consecutive_pairs=${summary.consecutivePairs}`,
  );
}

runAbCliEntrypoint(import.meta.url, runFinalizerCacheReportCli);
