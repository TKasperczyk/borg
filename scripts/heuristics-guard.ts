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
    name: "generation/extractor English wordlist or role-label regex marker",
    pattern:
      "(?i)(const\\s+\\w+\\s*=\\s*\\[[^\\n\\]]*\\b(?:stop|stopping|responding|generate|output|human|assistant|user)\\b[^\\n\\]]*\\b(?:stop|stopping|responding|generate|output|human|assistant|user)\\b|/[^\\n/]*\\(\\?:[^\\n/]*\\b(?:human|assistant|user)\\b[^\\n/]*\\b(?:human|assistant|user)\\b[^\\n/]*\\))",
    paths: [
      "src/cognition/generation",
      "src/cognition/commitments",
      "src/cognition/goals",
      "src/cognition/frame-anomaly",
    ],
  },
  {
    name: "generation/extractor English Set wordlist marker",
    pattern: "new Set\\s*\\(\\s*\\[",
    paths: [
      "src/cognition/generation",
      "src/cognition/commitments",
      "src/cognition/goals",
      "src/cognition/frame-anomaly",
    ],
  },
  {
    name: "frame-anomaly broad degraded fallback marker",
    pattern:
      "pattern:\\s*[\"'](?:as an ai|as a language model|i am an ai|i'm an ai|i am an artificial intelligence|i'm an artificial intelligence)[\"']",
    paths: ["src/cognition/frame-anomaly"],
  },
  {
    name: "commitment/goal extractor substring semantic matching",
    pattern: "\\.(includes|indexOf|startsWith|endsWith)\\s*\\(",
    paths: ["src/cognition/commitments", "src/cognition/goals"],
  },
  {
    name: "commitment/goal extractor regex literal marker",
    pattern: "(^|[=(:,\\[{!&|?;]|return\\s+)\\s*/[^/\\n]+/[dgimsuvy]*",
    paths: ["src/cognition/commitments", "src/cognition/goals"],
  },
  {
    name: "commitment/goal extractor capitalization heuristic marker",
    pattern: "\\[A-Z\\]|\\\\p\\{Lu\\}|toUpperCase\\s*\\(|isUpperCase\\s*\\(",
    paths: ["src/cognition/commitments", "src/cognition/goals"],
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
