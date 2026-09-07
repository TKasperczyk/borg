import { existsSync, lstatSync, realpathSync } from "node:fs";
import { dirname, join, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import OpenAI from "openai";
import { z } from "zod";
import { OpenAICompatibleEmbeddingClient } from "../src/embeddings/index.js";
import { embeddingProfileSchema } from "../src/embeddings/bank-profile.js";
import { DEFAULT_TENANT_ID_PATTERN, listBankTenantIds } from "../src/borg/tenant-directories.js";
import { DEFAULT_GATEWAY_BASE_URL } from "../src/sidecar/gateway-config.js";
import { parsePositiveIntegerValue } from "../src/util/parse.js";
import { BorgError, ConfigError } from "../src/util/errors.js";
import { migrateTenant, MigrationInputBlockedError } from "./embedding-migration/migrate.js";

export function parseEmbeddingMigrationArgs(args: string[]) {
  const { values } = parseArgs({
    args,
    strict: true,
    options: {
      "data-root": { type: "string" },
      tenant: { type: "string", multiple: true },
      "all-tenants": { type: "boolean" },
      "target-model": { type: "string" },
      "target-dims": { type: "string" },
      "source-model": { type: "string" },
      "dry-run": { type: "boolean" },
      resume: { type: "boolean" },
      "verify-only": { type: "boolean" },
      "backup-dir": { type: "string" },
      "batch-size": { type: "string", default: "32" },
      concurrency: { type: "string", default: "2" },
      help: { type: "boolean" },
    },
  });
  if (values.help) return null;
  if (Boolean(values["all-tenants"]) === Boolean(values.tenant?.length))
    throw new ConfigError("Choose repeatable --tenant OR --all-tenants");
  if (values["dry-run"] && (values.resume || values["verify-only"]))
    throw new ConfigError("--dry-run cannot be combined with --resume or --verify-only");
  const dataRoot = realpathSync(resolve(z.string().min(1).parse(values["data-root"])));
  const target = embeddingProfileSchema.parse({
    model: values["target-model"],
    dimensions: parsePositiveIntegerValue(values["target-dims"] ?? ""),
  });
  const batchSize = z
    .number()
    .int()
    .positive()
    .parse(parsePositiveIntegerValue(values["batch-size"]!));
  const concurrency = z
    .number()
    .int()
    .positive()
    .parse(parsePositiveIntegerValue(values.concurrency!));
  // Bound operator mistakes before allocating workers or gateway requests.
  if (batchSize > 1024 || concurrency > 32)
    throw new ConfigError("--batch-size must be <= 1024 and --concurrency <= 32");
  const backupDir = resolve(values["backup-dir"] ?? join(dataRoot, "backups"));
  return {
    dataRoot,
    backupDir,
    target,
    batchSize,
    concurrency,
    tenants: [...new Set(values.tenant ?? [])],
    allTenants: values["all-tenants"] ?? false,
    sourceModel: values["source-model"],
    dryRun: values["dry-run"] ?? false,
    resume: values.resume ?? false,
    verifyOnly: values["verify-only"] ?? false,
  };
}

export async function embeddingMigrationMain(args = process.argv.slice(2)): Promise<number> {
  const options = parseEmbeddingMigrationArgs(args);
  if (!options) {
    process.stdout.write(
      "Usage: node --import tsx scripts/migrate-embeddings.ts --data-root /data (--tenant ID ... | --all-tenants) --target-model MODEL --target-dims N [--source-model MODEL] [--dry-run | --resume | --verify-only] [--backup-dir PATH] [--batch-size 32] [--concurrency 2]\n",
    );
    return 0;
  }
  const banked = await listBankTenantIds(options.dataRoot, DEFAULT_TENANT_ID_PATTERN, {
    strict: true,
  });
  const tenants = options.allTenants ? banked : options.tenants;
  if (tenants.length === 0) throw new ConfigError("No tenant banks selected");
  for (const tenant of tenants) {
    if (!DEFAULT_TENANT_ID_PATTERN.test(tenant) || !banked.includes(tenant))
      throw new ConfigError(`Unknown tenant bank: ${tenant}`);
    const path = join(options.dataRoot, tenant);
    if (lstatSync(path).isSymbolicLink() || realpathSync(path) !== path)
      throw new ConfigError(`Tenant directory must not be a symlink: ${tenant}`);
  }
  let backupParent = options.backupDir;
  while (!existsSync(backupParent)) backupParent = dirname(backupParent);
  if (realpathSync(backupParent) !== backupParent)
    throw new ConfigError("Backup path must not contain symlinks");
  if (
    options.backupDir === options.dataRoot ||
    banked.some(
      (tenant) =>
        options.backupDir === join(options.dataRoot, tenant) ||
        options.backupDir.startsWith(`${join(options.dataRoot, tenant)}${sep}`),
    )
  )
    throw new ConfigError(
      "--backup-dir must be outside every tenant directory and tenant discovery",
    );
  const client =
    options.dryRun || options.verifyOnly
      ? undefined
      : new OpenAICompatibleEmbeddingClient({
          model: options.target.model,
          dims: options.target.dimensions,
          maxBatchSize: options.batchSize,
          modelReloadRetryDelaysMs: [],
          client: new OpenAI({
            apiKey: z
              .string()
              .min(1)
              .parse(process.env.LLM_API_KEY ?? process.env.BORG_EMBEDDING_API_KEY),
            baseURL:
              process.env.KRATOS_BASE_URL ??
              process.env.BORG_EMBEDDING_BASE_URL ??
              DEFAULT_GATEWAY_BASE_URL,
            timeout: 60_000,
            maxRetries: 0,
          }),
        });
  // Always migrate one tenant at a time, even with --all-tenants.
  for (const tenant of tenants) {
    const report = await migrateTenant(
      { ...options, tenantDir: join(options.dataRoot, tenant) },
      {
        client,
        progress: (event) => process.stdout.write(`${JSON.stringify({ tenant, ...event })}\n`),
      },
    );
    process.stdout.write(`${JSON.stringify({ report })}\n`);
    if (!options.dryRun && report.complete !== true) return 1;
  }
  return 0;
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    process.exitCode = await embeddingMigrationMain();
  } catch (error) {
    process.stderr.write(
      `${JSON.stringify({ error: error instanceof Error ? error.message : String(error), code: error instanceof BorgError ? error.code : "EMBEDDING_MIGRATION_FAILED", complete: false, ...(error instanceof MigrationInputBlockedError ? { report: error.report } : {}) })}\n`,
    );
    process.exitCode = 1;
  }
}
