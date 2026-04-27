type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export type EvalCaseResult = {
  name: string;
  passed: boolean;
  actual?: JsonValue;
  expected?: JsonValue;
  note?: string;
};

export type EvalMetricResult = {
  name: string;
  description: string;
  passed: boolean;
  actual: Record<string, JsonValue>;
  expected: Record<string, JsonValue>;
  duration_ms: number;
  cases: EvalCaseResult[];
};

export type EvalMetricScorecardEntry = Omit<EvalMetricResult, "duration_ms"> & {
  duration_ms?: number;
};

export type EvalMetricModule = {
  name: string;
  description: string;
  run: () => Promise<EvalMetricResult>;
};

export type EvalScorecard = {
  suite: "borg";
  generated_at?: string;
  summary: {
    total: number;
    passed: number;
    failed: number;
  };
  metrics: EvalMetricScorecardEntry[];
};

function formatJsonValue(value: JsonValue): string {
  if (typeof value === "string") {
    return value;
  }

  return JSON.stringify(value);
}

function formatRecord(value: Record<string, JsonValue>): string {
  return Object.entries(value)
    .map(([key, entry]) => `${key}=${formatJsonValue(entry)}`)
    .join(", ");
}

export function buildScorecard(
  metrics: EvalMetricResult[],
  options: {
    includeTiming?: boolean;
  } = {},
): EvalScorecard {
  const includeTiming = options.includeTiming ?? true;
  const passed = metrics.filter((metric) => metric.passed).length;

  return {
    suite: "borg",
    ...(includeTiming ? { generated_at: new Date().toISOString() } : {}),
    summary: {
      total: metrics.length,
      passed,
      failed: metrics.length - passed,
    },
    metrics: metrics.map((metric) =>
      includeTiming
        ? metric
        : {
            name: metric.name,
            description: metric.description,
            passed: metric.passed,
            actual: metric.actual,
            expected: metric.expected,
            cases: metric.cases,
          },
    ),
  };
}

export function formatHumanScorecard(scorecard: EvalScorecard): string {
  const lines = [`${scorecard.suite} eval scorecard`, ""];

  for (const metric of scorecard.metrics) {
    lines.push(
      `${metric.name}: ${metric.passed ? "PASS" : "FAIL"} | actual ${formatRecord(metric.actual)} | expected ${formatRecord(metric.expected)}${metric.duration_ms === undefined ? "" : ` | ${metric.duration_ms}ms`}`,
    );

    for (const testCase of metric.cases) {
      if (metric.passed && testCase.passed) {
        continue;
      }

      const details = [
        testCase.actual === undefined ? null : `actual=${formatJsonValue(testCase.actual)}`,
        testCase.expected === undefined ? null : `expected=${formatJsonValue(testCase.expected)}`,
        testCase.note ?? null,
      ]
        .filter((part): part is string => part !== null)
        .join(" | ");

      lines.push(
        `  - ${testCase.name}: ${testCase.passed ? "PASS" : "FAIL"}${details ? ` | ${details}` : ""}`,
      );
    }
  }

  lines.push("");
  lines.push(
    `summary: total=${scorecard.summary.total}, passed=${scorecard.summary.passed}, failed=${scorecard.summary.failed}`,
  );
  return `${lines.join("\n")}\n`;
}
