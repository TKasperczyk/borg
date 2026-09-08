// One-off import over the normal API. The input file is only ever read; Borg's
// live storage is never opened directly. Dry-run is the default.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { pathToFileURL } from "node:url";

import { z } from "zod";

import {
  operatorAttentionRecordSchema,
  type OperatorAttentionRecord,
} from "../src/memory/operator-attention/types.js";

const storedEnvelopeSchema = z.object({
  ts: operatorAttentionRecordSchema.shape.filed_at,
  record_key: operatorAttentionRecordSchema.shape.record_key.optional(),
  filer_entity_id: operatorAttentionRecordSchema.shape.filer_entity_id.optional(),
  subject: operatorAttentionRecordSchema.shape.subject.optional(),
});

export function readOperatorAttentionBackfill(
  path: string,
  filerEntityId: string,
): OperatorAttentionRecord[] {
  const fallbackFiler = operatorAttentionRecordSchema.shape.filer_entity_id.parse(filerEntityId);
  const occurrences = new Map<string, number>();
  const records: OperatorAttentionRecord[] = [];
  // JSONL delimiters and envelope fields are protocol structure. Forward stored
  // subjects, but never inspect reason/body to infer one or include them in a payload.
  for (const [lineIndex, line] of readFileSync(path, "utf8").split("\n").entries()) {
    if (line.trim() === "") continue;
    let envelope: z.infer<typeof storedEnvelopeSchema>;
    try {
      envelope = storedEnvelopeSchema.parse(JSON.parse(line));
    } catch {
      // Do not echo a malformed line (or a JSON parser error containing its body).
      throw new Error(`Invalid attention envelope on line ${lineIndex + 1}`);
    }
    const filer = envelope.filer_entity_id ?? fallbackFiler;
    const identity = JSON.stringify([filer, envelope.ts]);
    const ordinal = occurrences.get(identity) ?? 0;
    occurrences.set(identity, ordinal + 1);
    const recordKey =
      envelope.record_key ??
      `cclink:legacy:${createHash("sha256")
        .update(JSON.stringify([filer, envelope.ts, ordinal]))
        .digest("hex")}`;
    records.push(
      operatorAttentionRecordSchema.parse({
        record_key: recordKey,
        filed_at: envelope.ts,
        filer_entity_id: filer,
        subject: envelope.subject ?? null,
      }),
    );
  }
  return records;
}

export async function backfillOperatorAttention(input: {
  path: string;
  filerEntityId: string;
  apply?: boolean;
  borgUrl?: string;
  fetchImpl?: typeof fetch;
}): Promise<{ records: OperatorAttentionRecord[]; inserted: number; duplicates: number }> {
  // Validate the whole import before the first write.
  const records = readOperatorAttentionBackfill(input.path, input.filerEntityId);
  let inserted = 0;
  let duplicates = 0;
  if (input.apply === true) {
    if (input.borgUrl === undefined) throw new Error("--borg-url is required with --apply");
    const base = input.borgUrl.endsWith("/") ? input.borgUrl : `${input.borgUrl}/`;
    for (const record of records) {
      const response = await (input.fetchImpl ?? fetch)(new URL("api/operator-attention", base), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(record),
        signal: AbortSignal.timeout(5_000),
      });
      if (!response.ok) {
        await response.body?.cancel();
        throw new Error(
          `Attention import failed for ${record.record_key}: HTTP ${response.status}`,
        );
      }
      const result = z.object({ inserted: z.boolean() }).parse(await response.json());
      if (result.inserted) inserted += 1;
      else duplicates += 1;
    }
  }
  return { records, inserted, duplicates };
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      file: { type: "string" },
      "filer-entity-id": { type: "string" },
      "borg-url": { type: "string" },
      apply: { type: "boolean", default: false },
    },
  });
  if (!values.file || !values["filer-entity-id"]) {
    throw new Error(
      "Usage: pnpm exec tsx scripts/backfill-operator-attention.ts --file <jsonl> --filer-entity-id <entity id> [--apply --borg-url <base URL>]",
    );
  }
  const result = await backfillOperatorAttention({
    path: values.file,
    filerEntityId: values["filer-entity-id"],
    apply: values.apply,
    borgUrl: values["borg-url"],
  });
  console.log(JSON.stringify({ mode: values.apply ? "apply" : "dry-run", ...result }, null, 2));
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    await main();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
