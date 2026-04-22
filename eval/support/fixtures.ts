import { readdirSync } from "node:fs";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";

import { z } from "zod";

import { readJsonFile } from "../../src/util/atomic-write.js";

export type LoadedFixture<T> = {
  name: string;
  path: string;
  data: T;
};

const FIXTURES_ROOT = fileURLToPath(new URL("../fixtures", import.meta.url));

export function loadMetricFixtures<T>(
  metricName: string,
  schema: z.ZodType<T>,
): LoadedFixture<T>[] {
  const directory = join(FIXTURES_ROOT, metricName);
  const entries = readdirSync(directory, {
    withFileTypes: true,
  })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
    .sort((left, right) => left.name.localeCompare(right.name));

  return entries.map((entry) => {
    const path = join(directory, entry.name);
    const raw = readJsonFile<unknown>(path);

    if (raw === undefined) {
      throw new Error(`Fixture file could not be read: ${path}`);
    }

    return {
      name: basename(entry.name, ".json"),
      path,
      data: schema.parse(raw),
    };
  });
}
