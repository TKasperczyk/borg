import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";
import { describe, expect, it } from "vitest";
import type { AnnotationTarget } from "../memory/identity/index.js";
import type { AnnotationPromptFields } from "../memory/common/evidence-prompt.js";
import fixtures from "./__fixtures__/stage-one-dev-renderers.json";
import { createEpisodeFixture, createSemanticNodeFixture } from "./test-support.js";
import { memoryDisclosureLabelFromEpisodeAccess } from "../memory/common/disclosure-label.js";
import {
  withEpisodeEvidenceContext,
  withSemanticNodeEvidenceContext,
  evidencePromptFields,
} from "../memory/common/episode-evidence.js";
import { evidencePromptSizes } from "./evidence-test-support.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
type Renderer = (...args: unknown[]) => string;

// Execute the actual renderer declarations, including their private helper dependencies.
// The dev declarations are frozen verbatim from b480dee9; no rendered fields are stripped.
function rendererSource(path: string, source: string, names: string[]): string {
  const file = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true);
  const declarations = new Map<string, ts.Statement>();
  for (const statement of file.statements) {
    if (ts.isFunctionDeclaration(statement) && statement.name)
      declarations.set(statement.name.text, statement);
    if (ts.isVariableStatement(statement))
      for (const declaration of statement.declarationList.declarations)
        if (ts.isIdentifier(declaration.name)) declarations.set(declaration.name.text, statement);
  }
  const selected = new Set<ts.Statement>();
  const add = (statement: ts.Statement | undefined) => {
    if (statement === undefined || selected.has(statement)) return;
    selected.add(statement);
    const visit = (node: ts.Node) => {
      if (ts.isIdentifier(node)) add(declarations.get(node.text));
      ts.forEachChild(node, visit);
    };
    ts.forEachChild(statement, visit);
  };
  names.forEach((name) => add(declarations.get(name)));
  return (
    file.statements
      .filter((statement) => ts.isImportDeclaration(statement) || selected.has(statement))
      .map((statement) => statement.getText(file))
      .join("\n\n") + `\nexport { ${names.join(", ")} };\n`
  );
}
const cache = new Map<string, Promise<Record<string, Renderer>>>();
async function loadRenderers(
  path: keyof typeof fixtures.modules,
  dev: boolean,
): Promise<Record<string, Renderer>> {
  const key = `${dev}:${path}`;
  const existing = cache.get(key);
  if (existing !== undefined) return existing;
  const load = async () => {
    const fixture = fixtures.modules[path];
    const source = dev
      ? fixture.source
      : rendererSource(path, readFileSync(resolve(root, path), "utf8"), fixture.exports);
    const compiled = ts.transpileModule(source, {
      compilerOptions: { target: ts.ScriptTarget.ES2023, module: ts.ModuleKind.CommonJS },
    }).outputText;
    const parsed = ts.createSourceFile(
      "compiled.js",
      compiled,
      ts.ScriptTarget.Latest,
      true,
      ts.ScriptKind.JS,
    );
    const imports = new Set<string>();
    const visit = (node: ts.Node) => {
      if (
        ts.isCallExpression(node) &&
        ts.isIdentifier(node.expression) &&
        node.expression.text === "require" &&
        node.arguments[0] !== undefined &&
        ts.isStringLiteral(node.arguments[0])
      )
        imports.add(node.arguments[0].text);
      ts.forEachChild(node, visit);
    };
    visit(parsed);
    const dependencies = new Map<string, unknown>();
    for (const specifier of imports) {
      const destination = specifier.startsWith(".")
        ? resolve(root, dirname(path), specifier).replace(/\.js$/, ".ts")
        : specifier;
      dependencies.set(
        specifier,
        dev && destination === resolve(root, "src/offline/evidence-labels.ts")
          ? await loadRenderers("src/offline/evidence-labels.ts", true)
          : await import(/* @vite-ignore */ destination),
      );
    }
    const exports: Record<string, Renderer> = {};
    new Function("require", "exports", compiled)(
      (specifier: string) => dependencies.get(specifier),
      exports,
    );
    return exports;
  };
  const result = load();
  cache.set(key, result);
  return result;
}

const first = createEpisodeFixture({
  title: "Synthetic first",
  narrative: "原文。 An ordinary observation.",
  shared: true,
});
const second = createEpisodeFixture({
  title: "Synthetic second",
  narrative: "Another ordinary observation.",
  shared: true,
});
const previous = {
  ...createEpisodeFixture({
    title: "Synthetic previous",
    narrative: "The previous summary.",
    shared: true,
    lineage: { derived_from: [first.id], supersedes: [] },
  }),
  episode_kind: "consolidation_version" as const,
};
const nodes = [
  createSemanticNodeFixture({ label: "Synthetic node one", source_episode_ids: [first.id] }),
  createSemanticNodeFixture({ label: "Synthetic node two", source_episode_ids: [second.id] }),
];

type Path = keyof typeof fixtures.modules;
function cases(
  episodes: (typeof first)[],
  version: typeof previous,
  candidates: typeof nodes,
): Array<{ name: string; path: Path; fn: string; args: unknown[] }> {
  const label = memoryDisclosureLabelFromEpisodeAccess(first);
  const labels = new Map([
    [first.id, label],
    [second.id, label],
  ]);
  const nodeRows = candidates.map((node) => ({
    ...node,
    ...evidencePromptFields(
      (node as { evidence_context?: AnnotationPromptFields }).evidence_context,
    ),
  }));
  // The belief reviser serializes the node record separately from its read projection.
  for (const row of nodeRows) delete (row as { evidence_context?: unknown }).evidence_context;
  const merge = { previousCurrentVersion: null, rawEpisodes: episodes, newRawEpisodes: episodes };
  const review = {
    item: {
      id: 1,
      kind: "contradiction",
      reason: "Synthetic comparison",
      refs: {},
      created_at: 1000,
      resolved_at: null,
      resolution: null,
    },
    loaded: {
      refs: { node_ids: candidates.map((node) => node.id) },
      nodes: candidates,
      labelsByEpisodeId: labels,
      evidence: candidates.map((node, i) => ({
        node_id: node.id,
        sampled_episode_ids: [episodes[i]!.id],
        total_source_episode_ids: 1,
        missing_sampled_episode_ids: [],
        episodes: [episodes[i]],
      })),
    },
  };
  return [
    {
      name: "initial merge",
      path: "src/offline/consolidator/index.ts",
      fn: "buildMergePrompt",
      args: [merge, null],
    },
    {
      name: "incremental merge",
      path: "src/offline/consolidator/index.ts",
      fn: "buildMergePrompt",
      args: [{ ...merge, previousCurrentVersion: version, newRawEpisodes: [episodes[1]] }, null],
    },
    {
      name: "extractor",
      path: "src/memory/semantic/extractor.ts",
      fn: "buildPrompt",
      args: [
        {
          episodes,
          identityAnchors: [],
          knownNodeKinds: [],
          participantRoster: null,
          selfEntityId: null,
        },
      ],
    },
    {
      name: "semantic-pair review",
      path: "src/offline/review-resolver/index.ts",
      fn: "semanticPairPromptPayload",
      args: [review],
    },
    {
      name: "belief review",
      path: "src/offline/belief-reviser/llm-reviewer.ts",
      fn: "promptPayload",
      args: [
        {
          review_id: 1,
          audience_entity_id: null,
          evidence_episode_ids: [first.id],
          target: { target_type: "semantic_node", record: nodeRows[0] },
          invalidated_edge: null,
          surviving_supports: [],
          evidence_episodes: episodes,
        },
      ],
    },
    {
      name: "procedural synthesis",
      path: "src/offline/procedural-synthesizer/index.ts",
      fn: "buildPrompt",
      args: [{ key: "synthetic", evidenceRows: [], sourceEpisodes: episodes }],
    },
    {
      name: "procedural split",
      path: "src/offline/procedural-synthesizer/index.ts",
      fn: "buildSplitPrompt",
      args: [
        {
          skill: {
            id: "skl_syntheticfixture",
            applies_when: "Synthetic comparison",
            approach: "Compare the observations",
            source_episode_ids: [first.id],
            alpha: 2,
            beta: 1,
            attempts: 1,
            successes: 1,
            failures: 0,
          },
          sourceDisclosureLabel: label,
          sourceEpisodes: episodes,
          buckets: [],
          divergence: 0.7,
          min_posterior_mean: 0.1,
          max_posterior_mean: 0.8,
        },
      ],
    },
    {
      name: "generic node review",
      path: "src/offline/review-resolver/index.ts",
      fn: "promptPayload",
      args: [
        {
          item: review.item,
          meaningChangingSemanticPatch: false,
          autonomous: false,
          loaded: {
            reviewPayload: {},
            targetPayload: { type: "semantic_node", content: nodeRows[0] },
            overseerFlagPayload: {},
            sourceEntries: [],
            missingSourceIds: [],
            taintedReviewedAssistantStreamIds: [],
          },
        },
      ],
    },
    {
      name: "new-insight review",
      path: "src/offline/review-resolver/index.ts",
      fn: "newInsightPromptPayload",
      args: [
        {
          item: review.item,
          loaded: {
            currentNode: null,
            labelsByEpisodeId: labels,
            sampledEvidenceEpisodeIds: [first.id],
            evidenceEpisodes: episodes,
            missingEvidenceEpisodeIds: [],
            totalEvidenceEpisodeIds: 2,
            refs: {
              episode_ids: [first.id],
              evidence_cluster_key: "synthetic",
              evidence_cluster_size: 2,
              reflector_pending_insight: {
                target: { mode: "insert", node: candidates[0] },
                evidence_cluster: { episode_ids: [first.id] },
                candidate_support_edges: [],
              },
            },
          },
        },
      ],
    },
    {
      name: "vector-duplicate review",
      path: "src/offline/review-resolver/index.ts",
      fn: "vectorDuplicatePromptPayload",
      args: [review],
    },
  ];
}

const baselineCases = cases([first, second], previous, nodes);
describe("stage-one prompt equivalence to dev b480dee9", () => {
  for (const [index, fixture] of baselineCases.entries())
    it(`${fixture.name}: ordinary bytes are identical; added context is measured against dev`, async () => {
      const reader = {
        episodicRepository: {
          getMany: async () => [],
          getLifecycleMany: (ids: readonly (typeof first.id)[]) =>
            new Map(ids.map((id) => [id, "active" as const])),
        },
        identityEventRepository: {
          listAnnotationsForTargets: (targets: readonly AnnotationTarget[]) =>
            targets.map((target) => ({ ...target, annotations: [] })),
        },
      };
      const ordinary = await withEpisodeEvidenceContext(reader, [first, second]);
      const currentCase = cases(ordinary, previous, withSemanticNodeEvidenceContext(reader, nodes))[
        index
      ]!;
      const dev = (await loadRenderers(fixture.path, true))[fixture.fn]!;
      const current = (await loadRenderers(fixture.path, false))[fixture.fn]!;
      const before = dev(...fixture.args);
      expect(Buffer.from(current(...currentCase.args))).toEqual(Buffer.from(before));
      const annotation = {
        text: "Synthetic correction. 更正。",
        written_at: "1970-01-01T00:00:01.000Z",
        authority: "Synthetic operator",
      };
      const annotated = cases(
        [{ ...first, evidence_context: { annotations: [annotation] } } as typeof first, second],
        previous,
        [
          {
            ...nodes[0]!,
            evidence_context: { annotations: [annotation] },
          } as (typeof nodes)[number],
          nodes[1]!,
        ],
      )[index]!;
      const after = current(...annotated.args);
      expect(evidencePromptSizes(before, after)).toMatchSnapshot();
    });
});
