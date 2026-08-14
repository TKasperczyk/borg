import { createHash } from "node:crypto";

import type { LLMConverseOptions, LLMSystemBlock } from "../../llm/index.js";

export type RequestSurfaceFingerprint = {
  systemChars: number;
  systemSha256: string;
  transportSha256: string;
  systemBlockCount: number;
  cacheBreakpointCount: number;
};

export type CanonicalRequestFingerprint = {
  canonicalChars: number;
  canonicalSha256: string;
};

function llmSystemBlocks(system: string | readonly LLMSystemBlock[]): readonly LLMSystemBlock[] {
  return typeof system === "string" ? [{ type: "text", text: system }] : system;
}

export function llmSystemText(system: string | readonly LLMSystemBlock[]): string {
  return llmSystemBlocks(system)
    .map((block) => block.text)
    .join("\n\n");
}

function sha256(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value === null || typeof value !== "object") return value;
  const record = value as Record<string, unknown>;
  return Object.fromEntries(
    Object.keys(record)
      .sort()
      .filter((key) => record[key] !== undefined)
      .map((key) => [key, canonicalize(record[key])]),
  );
}

export function fingerprintSystemSurface(
  system: NonNullable<LLMConverseOptions["system"]>,
): RequestSurfaceFingerprint {
  const blocks = llmSystemBlocks(system);
  const text = llmSystemText(system);
  return {
    systemChars: text.length,
    systemSha256: sha256(text),
    transportSha256: sha256(JSON.stringify(system)),
    systemBlockCount: blocks.length,
    cacheBreakpointCount: blocks.filter((block) => block.cache_control !== undefined).length,
  };
}

export function fingerprintCanonicalRequest(request: {
  system?: LLMConverseOptions["system"];
  messages: unknown;
  tools?: unknown;
  [key: string]: unknown;
}): CanonicalRequestFingerprint {
  const { system, messages, tools, ...callOptions } = request;
  const canonical = JSON.stringify(canonicalize({ system, messages, tools, callOptions }));
  return { canonicalChars: canonical.length, canonicalSha256: sha256(canonical) };
}

export function sha256Bytes(value: Uint8Array): string {
  return sha256(value);
}
