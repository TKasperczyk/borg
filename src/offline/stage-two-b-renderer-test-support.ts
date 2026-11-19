import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";
import fixtures from "./__fixtures__/stage-two-b-dev-renderers.json";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
type Renderer = (...args: any[]) => any;

// Execute the actual renderer declarations, including their private helper dependencies.
// Dev declarations are frozen verbatim from cc7877c4; no rendered fields are stripped.
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
export async function loadStageTwoBRenderers(
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
      const relativePath = destination.slice(root.length + 1) as keyof typeof fixtures.modules;
      dependencies.set(
        specifier,
        dev && relativePath in fixtures.modules
          ? await loadStageTwoBRenderers(relativePath, true)
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
