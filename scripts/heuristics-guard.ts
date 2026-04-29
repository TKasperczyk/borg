import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";

type Guard = {
  name: string;
  pattern: string;
  paths: readonly string[];
  allowedUntilTokenizerRemoval?: boolean;
};

const guards: readonly Guard[] = [
  {
    name: "generic tokenizer usage",
    pattern: "tokenizeText|util/text/tokenize",
    paths: ["src"],
  },
  {
    name: "affective/procedural/semantic heuristic constants",
    pattern:
      "POSITIVE_WORDS|NEGATIVE_WORDS|GRATITUDE_PATTERNS|DOMAIN_KEYWORDS|PROBLEM_KIND_RULES|DOMAIN_SYNONYMS",
    paths: ["src"],
  },
  {
    name: "substring semantic label matching",
    pattern: "\\b(label|alias)\\s*\\.\\s*(includes|indexOf)\\s*\\(",
    paths: ["src/memory/semantic", "src/retrieval"],
  },
  {
    name: "natural-language query punctuation splitting",
    pattern: "\\.split\\(\\s*/\\[[^\\]]*(,|\\\\n|\\\\r)[^\\]]*\\]",
    paths: ["src/retrieval", "src/cognition/perception"],
  },
  {
    name: "ASCII-only tokenization",
    pattern: "\\.split\\(\\s*/\\[\\^a-z",
    paths: ["src"],
  },
  {
    name: "affective English wordlist marker",
    pattern: "english wordlist|new Set\\s*\\(\\s*\\[",
    paths: ["src/memory/affective"],
  },
  {
    name: "generation English wordlist or role-label regex marker",
    pattern:
      "(?i)(const\\s+\\w+\\s*=\\s*\\[[^\\n\\]]*\\b(?:stop|stopping|responding|generate|output|human|assistant|user)\\b[^\\n\\]]*\\b(?:stop|stopping|responding|generate|output|human|assistant|user)\\b|/[^\\n/]*\\(\\?:[^\\n/]*\\b(?:human|assistant|user)\\b[^\\n/]*\\b(?:human|assistant|user)\\b[^\\n/]*\\))",
    paths: ["src/cognition/generation"],
  },
];

function rg(pattern: string, paths: readonly string[]): string {
  const result = spawnSync("rg", ["--line-number", pattern, ...paths], {
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
  rg("^export function tokenizeText", ["src/util/text/tokenize.ts"]).length === 0;
const failures: string[] = [];

for (const guard of guards) {
  if (guard.allowedUntilTokenizerRemoval === true && !tokenizerRemoved) {
    continue;
  }

  const matches = rg(guard.pattern, guard.paths);

  if (matches.length > 0) {
    failures.push(`${guard.name}:\n${matches}`);
  }
}

if (failures.length > 0) {
  console.error(`Language-heuristics guard failed:\n\n${failures.join("\n\n")}`);
  process.exit(1);
}
