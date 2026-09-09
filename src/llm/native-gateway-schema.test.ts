import { afterEach, expect, it, vi } from "vitest";

afterEach(() => vi.restoreAllMocks());

it("keeps native-path structured schemas, recall plans and emission schemas free of propertyNames", async () => {
  vi.resetModules();
  const llm = await import("./index.js");
  const convert = llm.toToolInputSchema;
  const schemas: ReturnType<typeof convert>[] = [];
  vi.spyOn(llm, "toToolInputSchema").mockImplementation((schema) => {
    const converted = convert(schema);
    schemas.push(converted);
    return converted;
  });
  // Includes private perception/extraction/planner/shared-state/guard/reflection
  // tools constructed when the production orchestrator module loads.
  await import("../cognition/turn-orchestrator.js");
  const { resolveAvailableEmissionTools } = await import("../cognition/deliberation/finalizer.js");
  for (const origin of ["user", "autonomous"] as const) {
    for (const tool of resolveAvailableEmissionTools(undefined, origin)) {
      llm.toToolInputSchema(tool.inputSchema);
    }
  }
  const { expandRecall } = await import("../retrieval/recall-expansion.js");
  const { FakeLLMClient } = await import("./test-support/fake-client.js");
  await expect(
    expandRecall({
      llmClient: new FakeLLMClient(),
      model: "probe",
      focus: "question",
      semanticVariantCount: 2,
    }),
  ).rejects.toThrow();
  expect(schemas.length).toBeGreaterThanOrEqual(20);
  for (const schema of schemas) {
    expect(schema.type).toBe("object");
    expect(JSON.stringify(schema)).not.toContain('"propertyNames":');
  }
});
