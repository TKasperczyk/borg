import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { z } from "zod";

import { readJsonFile } from "../util/atomic-write.js";
import { ConfigError } from "../util/errors.js";

const DEFAULT_DATA_DIR = "~/.borg";

const configFileSchema = z
  .object({
    dataDir: z.string().min(1).optional(),
    embedding: z
      .object({
        baseUrl: z.string().url().optional(),
        apiKey: z.string().min(1).optional(),
        model: z.string().min(1).optional(),
        dims: z.number().int().positive().optional(),
      })
      .partial()
      .optional(),
    anthropic: z
      .object({
        apiKey: z.string().min(1).optional(),
        models: z
          .object({
            cognition: z.string().min(1).optional(),
            background: z.string().min(1).optional(),
            extraction: z.string().min(1).optional(),
          })
          .partial()
          .optional(),
      })
      .partial()
      .optional(),
  })
  .partial();

export const configSchema = z.object({
  dataDir: z.string().min(1),
  embedding: z.object({
    baseUrl: z.string().url(),
    apiKey: z.string().min(1).optional(),
    model: z.string().min(1),
    dims: z.number().int().positive(),
  }),
  anthropic: z.object({
    apiKey: z.string().min(1).optional(),
    models: z.object({
      cognition: z.string().min(1),
      background: z.string().min(1),
      extraction: z.string().min(1),
    }),
  }),
});

export type Config = z.infer<typeof configSchema>;

export const DEFAULT_CONFIG: Config = {
  dataDir: expandPath(DEFAULT_DATA_DIR),
  embedding: {
    baseUrl: "http://localhost:1234/v1",
    apiKey: "lm-studio",
    model: "text-embedding-qwen3-embedding-8b",
    dims: 4096,
  },
  anthropic: {
    apiKey: undefined,
    models: {
      cognition: "claude-sonnet-4-5",
      background: "claude-haiku-4-5",
      extraction: "claude-haiku-4-5",
    },
  },
};

export type LoadConfigOptions = {
  dataDir?: string;
  env?: NodeJS.ProcessEnv;
};

export function expandPath(pathLike: string): string {
  if (pathLike === "~") {
    return homedir();
  }

  if (pathLike.startsWith("~/")) {
    return join(homedir(), pathLike.slice(2));
  }

  return isAbsolute(pathLike) ? pathLike : resolve(pathLike);
}

function readOptionalEnvString(env: NodeJS.ProcessEnv, name: string): string | undefined {
  const value = env[name]?.trim();
  return value ? value : undefined;
}

function readOptionalEnvNumber(env: NodeJS.ProcessEnv, name: string): number | undefined {
  const raw = readOptionalEnvString(env, name);

  if (raw === undefined) {
    return undefined;
  }

  const value = Number(raw);

  if (!Number.isInteger(value) || value <= 0) {
    throw new ConfigError(`Environment variable ${name} must be a positive integer`);
  }

  return value;
}

function isNodeError(error: unknown): error is NodeJS.ErrnoException & { code: string } {
  return error instanceof Error && typeof (error as NodeJS.ErrnoException).code === "string";
}

function parseConfigFile(dataDir: string): z.infer<typeof configFileSchema> {
  const configPath = join(dataDir, "config.json");

  try {
    const rawConfig = readJsonFile<unknown>(configPath);

    if (rawConfig === undefined) {
      return {};
    }

    const parsed = configFileSchema.safeParse(rawConfig);

    if (!parsed.success) {
      throw new ConfigError(`Invalid config file at ${configPath}`, {
        cause: parsed.error,
        code: "CONFIG_FILE_INVALID",
      });
    }

    return parsed.data;
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return {};
    }

    if (error instanceof ConfigError) {
      throw error;
    }

    throw new ConfigError(`Invalid config file at ${configPath}`, {
      cause: error,
      code: "CONFIG_FILE_INVALID",
    });
  }
}

export function loadConfig(options: LoadConfigOptions = {}): Config {
  const env = options.env ?? process.env;
  const lookupDataDir = expandPath(
    options.dataDir ?? readOptionalEnvString(env, "BORG_DATA_DIR") ?? DEFAULT_DATA_DIR,
  );
  const fileConfig = parseConfigFile(lookupDataDir);

  const candidate = {
    dataDir: expandPath(
      options.dataDir ??
        readOptionalEnvString(env, "BORG_DATA_DIR") ??
        fileConfig.dataDir ??
        DEFAULT_CONFIG.dataDir,
    ),
    embedding: {
      baseUrl:
        readOptionalEnvString(env, "BORG_EMBEDDING_BASE_URL") ??
        fileConfig.embedding?.baseUrl ??
        DEFAULT_CONFIG.embedding.baseUrl,
      apiKey:
        readOptionalEnvString(env, "BORG_EMBEDDING_API_KEY") ??
        fileConfig.embedding?.apiKey ??
        DEFAULT_CONFIG.embedding.apiKey,
      model:
        readOptionalEnvString(env, "BORG_EMBEDDING_MODEL") ??
        fileConfig.embedding?.model ??
        DEFAULT_CONFIG.embedding.model,
      dims:
        readOptionalEnvNumber(env, "BORG_EMBEDDING_DIMS") ??
        fileConfig.embedding?.dims ??
        DEFAULT_CONFIG.embedding.dims,
    },
    anthropic: {
      apiKey:
        readOptionalEnvString(env, "ANTHROPIC_API_KEY") ??
        fileConfig.anthropic?.apiKey ??
        DEFAULT_CONFIG.anthropic.apiKey,
      models: {
        cognition:
          readOptionalEnvString(env, "BORG_MODEL_COGNITION") ??
          fileConfig.anthropic?.models?.cognition ??
          DEFAULT_CONFIG.anthropic.models.cognition,
        background:
          readOptionalEnvString(env, "BORG_MODEL_BACKGROUND") ??
          fileConfig.anthropic?.models?.background ??
          DEFAULT_CONFIG.anthropic.models.background,
        extraction:
          readOptionalEnvString(env, "BORG_MODEL_EXTRACTION") ??
          fileConfig.anthropic?.models?.extraction ??
          DEFAULT_CONFIG.anthropic.models.extraction,
      },
    },
  };

  const parsed = configSchema.safeParse(candidate);

  if (!parsed.success) {
    throw new ConfigError("Invalid borg configuration", {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

function redactSecret(value: string | undefined): string | undefined {
  return value === undefined ? undefined : "[REDACTED]";
}

export function redactConfig(config: Config): Config {
  return {
    ...config,
    embedding: {
      ...config.embedding,
      apiKey: redactSecret(config.embedding.apiKey),
    },
    anthropic: {
      ...config.anthropic,
      apiKey: redactSecret(config.anthropic.apiKey),
    },
  };
}
