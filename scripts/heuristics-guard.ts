import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { basename } from "node:path";

import ts from "typescript";

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

const disclosureGuardedPaths = [
  "src/cognition",
  "src/offline",
  "src/autonomy",
  "src/outbound",
  "src/tools/internal",
  "src/retrieval",
] as const;
const disclosureAllowedNamePattern =
  /Disclosure|Export|Admin|Public|CurrentAudienceStanding|ActionAuthorization/i;
const bannedDisclosureSymbols = new Set([
  "isEpisodeVisibleToAudience",
  "filterEpisodesByAudience",
  "searchWithContextForDisclosure",
  "isActionVisibleToSession",
  "isIdentityEventVisible",
  "visibleEpisodeIds",
  "visible_episode_ids",
]);
const deletedDisclosureFirewallSymbols = [
  "isEpisodeInGlobalIdentityScope",
  "episodeAccessScopeKey",
  "listVisibleActions",
] as const;
const disclosureCalleePattern = /ForDisclosure$/;

const labelCoverageGuardedPaths = [
  "src/cognition",
  "src/offline",
  "src/autonomy",
  "src/outbound",
  "src/tools/internal",
] as const;
const privateBearingModelFieldNames = new Set([
  "title",
  "narrative",
  "directive",
  "question",
  "description",
  "approach",
  "approach_summary",
  "problem_text",
  "value",
  "note",
  "content",
]);
const disclosureFieldNames = new Set(["disclosure", "disclosure_label"]);
const serializerHelperNamePattern = /(Payload|Row|Preview|LlmRecord|PromptRow|Messages?|serialize)/i;
const toSerializerHelperNamePattern = /to[A-Z]/;
const knownSerializerHelperNames = new Set([
  "episodeEvidencePromptRow",
  "semanticNodeLlmRecord",
  "semanticEdgeLlmRecord",
  "newInsightEpisodePayload",
  "newInsightProposedPayload",
  "semanticNodePromptPayload",
  "sourceEntryPromptPayload",
  "directivePromptRow",
  "directivePreview",
  "toSkillToolOutput",
]);

type LabelCoverageAllowlistEntry = {
  name: string;
  reason: string;
  matches: (input: {
    filePath: string;
    node: ts.ObjectLiteralExpression;
    privateFields: readonly string[];
  }) => boolean;
};

const labelCoverageAllowlist: readonly LabelCoverageAllowlistEntry[] = [
  {
    name: "llm-message-envelope-content",
    reason:
      "LLM message envelopes use `content` for the fully serialized prompt string; row labels live inside that prompt payload.",
    matches: ({ node, privateFields }) =>
      privateFields.length === 1 &&
      privateFields[0] === "content" &&
      objectLiteralPropertyNames(node).has("role"),
  },
  {
    name: "stream-append-content-record",
    reason:
      "These objects append already-produced agent text to the stream; they are persistence writes, not model-facing serializers.",
    matches: ({ filePath, node }) =>
      (filePath === "src/outbound/delivery.ts" &&
        enclosingFunctionName(node) === "appendAgentMessage") ||
      (filePath === "src/cognition/lifecycle/turn-phase/post-generation-phase.ts" &&
        enclosingFunctionName(node) === "persistMessageEmission"),
  },
  {
    name: "action-record-persistence-mapper",
    reason:
      "toActionRecord builds an internal ActionRecord for storage; prompt-facing action rows are labeled elsewhere.",
    matches: ({ filePath, node }) =>
      filePath === "src/cognition/actions/action-state-extractor.ts" &&
      enclosingFunctionName(node) === "toActionRecord",
  },
  {
    name: "corrective-preference-candidate-result",
    reason:
      "toCandidate converts a parsed tool result into an internal candidate, not a prompt row.",
    matches: ({ filePath, node }) =>
      filePath === "src/cognition/commitments/corrective-preference-extractor.ts" &&
      enclosingFunctionName(node) === "toCandidate",
  },
  {
    name: "llm-tool-result-protocol-content",
    reason:
      "Tool-result blocks use `content` as the Anthropic protocol field; labels are inside serialized tool outputs when those outputs carry memory rows.",
    matches: ({ filePath, node }) =>
      filePath === "src/cognition/turn-action/tool-loop.ts" &&
      [
        "buildToolResultBlock",
        "buildDroppedToolResultBlock",
        "buildUnavailableToolResultBlock",
      ].includes(enclosingFunctionName(node) ?? ""),
  },
  {
    name: "skill-split-review-internal-payload",
    reason:
      "Skill split payload helpers build and consume internal review-queue refs; they are not LLM prompt serializers.",
    matches: ({ filePath, node }) =>
      (filePath === "src/offline/procedural-synthesizer/index.ts" &&
        enclosingFunctionName(node) === "buildSkillSplitReviewPayload") ||
      (filePath === "src/offline/procedural-synthesizer/skill-split-review.ts" &&
        enclosingFunctionName(node) === "splitPartsFromPayload"),
  },
  {
    name: "creator-directive-private-operation-check",
    reason:
      "This structural authorization check passes a directive object into disclosure policy code; it does not serialize directive text to a model.",
    matches: ({ filePath, node }) =>
      filePath === "src/cognition/lifecycle/turn-phase/retrieval-phase.ts" &&
      enclosingFunctionName(node) === "canRenderCreatorDirectivePrivateOperation",
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

function rgFiles(paths: readonly string[]): string[] {
  const result = spawnSync("rg", ["--files", ...paths], {
    encoding: "utf8",
  });

  if (result.status === 1) {
    return [];
  }

  if (result.status !== 0) {
    throw new Error(result.stderr.trim() || `rg --files failed for ${paths.join(", ")}`);
  }

  return result.stdout
    .trim()
    .split("\n")
    .filter((file) => file.length > 0);
}

function nodeNameText(node: ts.Node): string | undefined {
  if (ts.isIdentifier(node) || ts.isStringLiteral(node)) {
    return node.text;
  }

  return undefined;
}

function isImportOrExportName(node: ts.Node): boolean {
  return (
    ts.isImportSpecifier(node.parent) ||
    ts.isImportClause(node.parent) ||
    ts.isNamespaceImport(node.parent) ||
    ts.isExportSpecifier(node.parent)
  );
}

function isDeclarationName(node: ts.Node): boolean {
  const parent = node.parent;

  if (
    ts.isFunctionDeclaration(parent) ||
    ts.isFunctionExpression(parent) ||
    ts.isMethodDeclaration(parent) ||
    ts.isPropertyDeclaration(parent) ||
    ts.isPropertySignature(parent) ||
    ts.isMethodSignature(parent) ||
    ts.isTypeAliasDeclaration(parent) ||
    ts.isInterfaceDeclaration(parent) ||
    ts.isClassDeclaration(parent)
  ) {
    return parent.name === node;
  }

  if (ts.isVariableDeclaration(parent)) {
    return parent.name === node;
  }

  if (ts.isPropertyAssignment(parent)) {
    return parent.name === node;
  }

  if (ts.isParameter(parent)) {
    return parent.name === node;
  }

  return false;
}

function isBannedDisclosureCallee(symbol: string): boolean {
  return bannedDisclosureSymbols.has(symbol) || disclosureCalleePattern.test(symbol);
}

function expressionSymbolName(node: ts.Expression): string | undefined {
  if (ts.isIdentifier(node)) {
    return node.text;
  }

  if (ts.isPropertyAccessExpression(node)) {
    return node.name.text;
  }

  if (ts.isElementAccessExpression(node)) {
    const argument = node.argumentExpression;
    return argument !== undefined && ts.isStringLiteral(argument) ? argument.text : undefined;
  }

  return undefined;
}

function propertyNameText(name: ts.PropertyName): string | undefined {
  if (ts.isIdentifier(name) || ts.isStringLiteral(name) || ts.isNumericLiteral(name)) {
    return name.text;
  }

  return undefined;
}

function objectLiteralPropertyNames(node: ts.ObjectLiteralExpression): Set<string> {
  const names = new Set<string>();

  for (const property of node.properties) {
    if (ts.isPropertyAssignment(property)) {
      const name = propertyNameText(property.name);

      if (name !== undefined) {
        names.add(name);
      }
    }

    if (ts.isShorthandPropertyAssignment(property)) {
      names.add(property.name.text);
    }
  }

  return names;
}

function collectDisclosureAliases(sourceFile: ts.SourceFile): Map<string, string> {
  const aliases = new Map<string, string>();

  function maybeAlias(localName: string, originalName: string): void {
    if (isBannedDisclosureCallee(originalName)) {
      aliases.set(localName, originalName);
    }
  }

  function visit(node: ts.Node): void {
    if (ts.isImportSpecifier(node)) {
      const originalName = (node.propertyName ?? node.name).text;
      maybeAlias(node.name.text, originalName);
    }

    if (ts.isBindingElement(node) && ts.isIdentifier(node.name)) {
      const propertyName = node.propertyName;

      if (
        propertyName !== undefined &&
        (ts.isIdentifier(propertyName) || ts.isStringLiteral(propertyName))
      ) {
        maybeAlias(node.name.text, propertyName.text);
      } else {
        maybeAlias(node.name.text, node.name.text);
      }
    }

    if (
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.initializer !== undefined
    ) {
      const originalName = expressionSymbolName(node.initializer);

      if (originalName !== undefined) {
        maybeAlias(node.name.text, originalName);
      }
    }

    ts.forEachChild(node, visit);
  }

  visit(sourceFile);
  return aliases;
}

function functionNameFromParent(parent: ts.Node | undefined): string | undefined {
  if (parent === undefined) {
    return undefined;
  }

  if (ts.isVariableDeclaration(parent) && ts.isIdentifier(parent.name)) {
    return parent.name.text;
  }

  if (ts.isPropertyAssignment(parent)) {
    return nodeNameText(parent.name);
  }

  if (ts.isMethodDeclaration(parent) || ts.isFunctionDeclaration(parent)) {
    return parent.name === undefined ? undefined : nodeNameText(parent.name);
  }

  return undefined;
}

function enclosingFunctionName(node: ts.Node): string | undefined {
  let current: ts.Node | undefined = node.parent;

  while (current !== undefined) {
    if (ts.isFunctionDeclaration(current) || ts.isMethodDeclaration(current)) {
      const directName = current.name === undefined ? undefined : nodeNameText(current.name);
      return directName ?? functionNameFromParent(current.parent);
    }

    if (ts.isFunctionExpression(current)) {
      const directName = current.name === undefined ? undefined : nodeNameText(current.name);
      const parentName = functionNameFromParent(current.parent);

      if (directName !== undefined || parentName !== undefined) {
        return directName ?? parentName;
      }
    }

    if (ts.isArrowFunction(current)) {
      const parentName = functionNameFromParent(current.parent);

      if (parentName !== undefined) {
        return parentName;
      }
    }

    current = current.parent;
  }

  return undefined;
}

function isAllowedDisclosureReference(filePath: string, node: ts.Node): boolean {
  if (disclosureAllowedNamePattern.test(basename(filePath))) {
    return true;
  }

  const functionName = enclosingFunctionName(node);
  return functionName !== undefined && disclosureAllowedNamePattern.test(functionName);
}

function reportDisclosureReference(
  sourceFile: ts.SourceFile,
  filePath: string,
  node: ts.Node,
  symbol: string,
): string {
  const position = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile));
  const functionName = enclosingFunctionName(node) ?? "<top-level>";
  return `${filePath}:${position.line + 1}:${position.character + 1} ${symbol} in ${functionName}`;
}

function calleeSymbolName(node: ts.CallExpression): string | undefined {
  return expressionSymbolName(node.expression);
}

function disclosureGuardFailures(): string[] {
  const sourceFiles = rgFiles(disclosureGuardedPaths).filter(
    (file) => file.endsWith(".ts") && !file.endsWith(".test.ts") && !file.endsWith(".d.ts"),
  );
  const matches: string[] = [];

  for (const filePath of sourceFiles) {
    const source = readFileSync(filePath, "utf8");
    const sourceFile = ts.createSourceFile(filePath, source, ts.ScriptTarget.Latest, true);
    const disclosureAliases = collectDisclosureAliases(sourceFile);

    function visit(node: ts.Node): void {
      if (ts.isCallExpression(node)) {
        const symbol = calleeSymbolName(node);
        const originalSymbol =
          symbol === undefined ? undefined : (disclosureAliases.get(symbol) ?? symbol);

        if (
          originalSymbol !== undefined &&
          isBannedDisclosureCallee(originalSymbol) &&
          !isAllowedDisclosureReference(filePath, node)
        ) {
          const reportedSymbol =
            symbol === originalSymbol ? originalSymbol : `${symbol} -> ${originalSymbol}`;
          matches.push(reportDisclosureReference(sourceFile, filePath, node, reportedSymbol));
        }
      }

      if (
        (ts.isIdentifier(node) || ts.isStringLiteral(node)) &&
        bannedDisclosureSymbols.has(node.text) &&
        !isImportOrExportName(node) &&
        !isDeclarationName(node) &&
        !isAllowedDisclosureReference(filePath, node)
      ) {
        matches.push(reportDisclosureReference(sourceFile, filePath, node, node.text));
      }

      ts.forEachChild(node, visit);
    }

    visit(sourceFile);
  }

  return matches;
}

function isStringLikeLiteral(node: ts.Expression): boolean {
  return ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node);
}

function isStaticPromptLiteral(node: ts.Expression): boolean {
  if (
    isStringLikeLiteral(node) ||
    ts.isNumericLiteral(node) ||
    node.kind === ts.SyntaxKind.TrueKeyword ||
    node.kind === ts.SyntaxKind.FalseKeyword ||
    node.kind === ts.SyntaxKind.NullKeyword
  ) {
    return true;
  }

  if (ts.isArrayLiteralExpression(node)) {
    return node.elements.every((element) => isStaticPromptLiteral(element));
  }

  if (ts.isObjectLiteralExpression(node)) {
    return node.properties.every((property) => {
      if (ts.isPropertyAssignment(property)) {
        return isStaticPromptLiteral(property.initializer);
      }

      return false;
    });
  }

  return false;
}

function hasMemoryDisclosurePayloadSpread(node: ts.ObjectLiteralExpression): boolean {
  return node.properties.some(
    (property) =>
      ts.isSpreadAssignment(property) &&
      ts.isCallExpression(property.expression) &&
      expressionSymbolName(property.expression.expression) === "memoryDisclosurePayloadFields",
  );
}

function objectLiteralHasDisclosureFields(node: ts.ObjectLiteralExpression): boolean {
  const names = objectLiteralPropertyNames(node);

  return (
    [...disclosureFieldNames].some((fieldName) => names.has(fieldName)) ||
    hasMemoryDisclosurePayloadSpread(node)
  );
}

function privateDynamicFields(node: ts.ObjectLiteralExpression): string[] {
  const fields: string[] = [];

  for (const property of node.properties) {
    if (ts.isPropertyAssignment(property)) {
      const name = propertyNameText(property.name);

      if (
        name !== undefined &&
        privateBearingModelFieldNames.has(name) &&
        !isStaticPromptLiteral(property.initializer)
      ) {
        fields.push(name);
      }
    }

    if (
      ts.isShorthandPropertyAssignment(property) &&
      privateBearingModelFieldNames.has(property.name.text)
    ) {
      fields.push(property.name.text);
    }
  }

  return fields;
}

function nodeWithin(child: ts.Node, parent: ts.Node): boolean {
  return child.getStart() >= parent.getStart() && child.getEnd() <= parent.getEnd();
}

function isJsonStringifyExpression(node: ts.Expression): boolean {
  return (
    ts.isPropertyAccessExpression(node) &&
    ts.isIdentifier(node.expression) &&
    node.expression.text === "JSON" &&
    node.name.text === "stringify"
  );
}

function isInsideJsonStringify(node: ts.Node): boolean {
  let current: ts.Node | undefined = node.parent;

  while (current !== undefined) {
    if (
      ts.isCallExpression(current) &&
      isJsonStringifyExpression(current.expression) &&
      current.arguments[0] !== undefined &&
      nodeWithin(node, current.arguments[0])
    ) {
      return true;
    }

    current = current.parent;
  }

  return false;
}

function isInsideNamedObjectProperty(node: ts.Node, propertyName: string): boolean {
  let current: ts.Node | undefined = node.parent;

  while (current !== undefined) {
    if (
      ts.isPropertyAssignment(current) &&
      propertyNameText(current.name) === propertyName &&
      nodeWithin(node, current.initializer)
    ) {
      return true;
    }

    current = current.parent;
  }

  return false;
}

function isFunctionLikeBoundary(node: ts.Node): boolean {
  return (
    ts.isFunctionDeclaration(node) ||
    ts.isFunctionExpression(node) ||
    ts.isArrowFunction(node) ||
    ts.isMethodDeclaration(node)
  );
}

function isInsideReturnStatement(node: ts.Node): boolean {
  let current: ts.Node | undefined = node.parent;

  while (current !== undefined) {
    if (ts.isReturnStatement(current)) {
      return true;
    }

    if (isFunctionLikeBoundary(current)) {
      return false;
    }

    current = current.parent;
  }

  return false;
}

function isAutonomyPayloadSerializer(filePath: string, node: ts.ObjectLiteralExpression): boolean {
  return filePath.startsWith("src/autonomy/") && isInsideNamedObjectProperty(node, "payload");
}

function isInternalToolInvokeSerializer(
  filePath: string,
  node: ts.ObjectLiteralExpression,
): boolean {
  return (
    filePath.startsWith("src/tools/internal/") &&
    enclosingFunctionName(node) === "invoke" &&
    isInsideReturnStatement(node)
  );
}

function isSerializerHelperName(name: string | undefined): boolean {
  return (
    name !== undefined &&
    (serializerHelperNamePattern.test(name) ||
      toSerializerHelperNamePattern.test(name) ||
      knownSerializerHelperNames.has(name))
  );
}

function isInsideArrowExpressionBody(node: ts.Node): boolean {
  let current: ts.Node | undefined = node.parent;

  while (current !== undefined) {
    if (ts.isArrowFunction(current)) {
      return !ts.isBlock(current.body) && nodeWithin(node, current.body);
    }

    if (isFunctionLikeBoundary(current)) {
      return false;
    }

    current = current.parent;
  }

  return false;
}

function isSerializerHelperReturn(node: ts.ObjectLiteralExpression): boolean {
  return (
    isSerializerHelperName(enclosingFunctionName(node)) &&
    (isInsideReturnStatement(node) || isInsideArrowExpressionBody(node))
  );
}

function isModelFacingSerializer(filePath: string, node: ts.ObjectLiteralExpression): boolean {
  return (
    isInsideJsonStringify(node) ||
    isAutonomyPayloadSerializer(filePath, node) ||
    isInternalToolInvokeSerializer(filePath, node) ||
    isSerializerHelperReturn(node)
  );
}

function isAllowedLabelCoverageOmission(
  filePath: string,
  node: ts.ObjectLiteralExpression,
  privateFields: readonly string[],
): boolean {
  return labelCoverageAllowlist.some((entry) =>
    entry.matches({
      filePath,
      node,
      privateFields,
    }),
  );
}

function reportLabelCoverageFailure(
  sourceFile: ts.SourceFile,
  filePath: string,
  node: ts.ObjectLiteralExpression,
  privateFields: readonly string[],
): string {
  const position = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile));
  const functionName = enclosingFunctionName(node) ?? "<top-level>";
  return `${filePath}:${position.line + 1}:${position.character + 1} ${privateFields.join(", ")} in ${functionName}`;
}

function labelCoverageGuardFailures(): string[] {
  const sourceFiles = rgFiles(labelCoverageGuardedPaths).filter(
    (file) => file.endsWith(".ts") && !file.endsWith(".test.ts") && !file.endsWith(".d.ts"),
  );
  const matches: string[] = [];

  for (const filePath of sourceFiles) {
    const source = readFileSync(filePath, "utf8");
    const sourceFile = ts.createSourceFile(filePath, source, ts.ScriptTarget.Latest, true);

    function visit(node: ts.Node): void {
      if (ts.isObjectLiteralExpression(node) && isModelFacingSerializer(filePath, node)) {
        const privateFields = privateDynamicFields(node);

        if (
          privateFields.length > 0 &&
          !objectLiteralHasDisclosureFields(node) &&
          !isAllowedLabelCoverageOmission(filePath, node, privateFields)
        ) {
          matches.push(reportLabelCoverageFailure(sourceFile, filePath, node, privateFields));
        }
      }

      ts.forEachChild(node, visit);
    }

    visit(sourceFile);
  }

  return matches;
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

for (const symbol of deletedDisclosureFirewallSymbols) {
  const matches = rg(`\\b${symbol}\\b`, ["src"]);

  if (matches.length > 0) {
    failures.push(`deleted disclosure firewall symbol reintroduced (${symbol}):\n${matches}`);
  }
}

const disclosureFailures = disclosureGuardFailures();

if (disclosureFailures.length > 0) {
  failures.push(
    `disclosure search symbols in cognition/recall paths:\n${disclosureFailures.join("\n")}`,
  );
}

const labelCoverageFailures = labelCoverageGuardFailures();

if (labelCoverageFailures.length > 0) {
  failures.push(
    `model-facing memory serializers missing disclosure labels:\n${labelCoverageFailures.join("\n")}`,
  );
}

if (failures.length > 0) {
  console.error(`Language-heuristics guard failed:\n\n${failures.join("\n\n")}`);
  process.exit(1);
}
