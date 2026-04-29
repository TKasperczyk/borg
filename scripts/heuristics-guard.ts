import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";

type Guard = {
  name: string;
  pattern: string;
  path: string;
  allowedUntilTokenizerRemoval?: boolean;
};

const guards: readonly Guard[] = [
  {
    name: "generic tokenizer usage",
    pattern: "tokenizeText|util/text/tokenize",
    path: "src",
  },
  {
    name: "affective/procedural/semantic heuristic constants",
    pattern:
      "POSITIVE_WORDS|NEGATIVE_WORDS|GRATITUDE_PATTERNS|DOMAIN_KEYWORDS|PROBLEM_KIND_RULES|DOMAIN_SYNONYMS",
    path: "src",
  },
  {
    name: "substring semantic label matching",
    pattern: "label\\.includes|alias\\.includes",
    path: "src",
  },
  {
    name: "semantic query label splitting",
    pattern: "split\\(/\\\\\\[,\\\\\\\\n",
    path: "src",
  },
];

function rg(pattern: string, path: string): string {
  const result = spawnSync("rg", ["--line-number", pattern, path], {
    encoding: "utf8",
  });

  if (result.status === 1) {
    return "";
  }

  if (result.status !== 0) {
    throw new Error(result.stderr.trim() || `rg failed for pattern ${pattern}`);
  }

  return result.stdout.trim();
}

const tokenizerRemoved =
  !existsSync("src/util/text/tokenize.ts") ||
  rg("^export function tokenizeText", "src/util/text/tokenize.ts").length === 0;
const failures: string[] = [];

for (const guard of guards) {
  if (guard.allowedUntilTokenizerRemoval === true && !tokenizerRemoved) {
    continue;
  }

  const matches = rg(guard.pattern, guard.path);

  if (matches.length > 0) {
    failures.push(`${guard.name}:\n${matches}`);
  }
}

if (failures.length > 0) {
  console.error(`Language-heuristics guard failed:\n\n${failures.join("\n\n")}`);
  process.exit(1);
}
