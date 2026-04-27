// Opens Borg's configured storage engines and LanceDB tables.

import { join } from "node:path";

import { autonomyMigrations } from "../autonomy/index.js";
import { DEFAULT_CONFIG, configSchema, loadConfig, type Config } from "../config/index.js";
import { executiveMigrations } from "../executive/index.js";
import { affectiveMigrations } from "../memory/affective/index.js";
import { commitmentMigrations } from "../memory/commitments/index.js";
import { createEpisodesTableSchema, episodicMigrations } from "../memory/episodic/index.js";
import { identityMigrations } from "../memory/identity/index.js";
import { createSkillsTableSchema, proceduralMigrations } from "../memory/procedural/index.js";
import { selfMigrations } from "../memory/self/index.js";
import { createSemanticNodesTableSchema, semanticMigrations } from "../memory/semantic/index.js";
import { socialMigrations } from "../memory/social/index.js";
import { offlineMigrations } from "../offline/index.js";
import { retrievalMigrations } from "../retrieval/index.js";
import { LanceDbStore, type LanceDbTable } from "../storage/lancedb/index.js";
import { openDatabase, type Migration, type SqliteDatabase } from "../storage/sqlite/index.js";
import { streamEntryIndexMigrations, streamWatermarkMigrations } from "../stream/index.js";

export type BorgStorage = {
  sqlite: SqliteDatabase;
  lance: LanceDbStore;
};

export type BorgLanceTables = {
  episodesTable: LanceDbTable;
  semanticNodesTable: LanceDbTable;
  skillsTable: LanceDbTable;
};

export function resolveBorgConfig(options: {
  config?: Config;
  env?: NodeJS.ProcessEnv;
  dataDir?: string;
}): Config {
  const rawConfig = options.config ?? loadConfig({ env: options.env, dataDir: options.dataDir });

  return configSchema.parse({
    ...DEFAULT_CONFIG,
    ...rawConfig,
    dataDir: options.dataDir ?? rawConfig.dataDir ?? DEFAULT_CONFIG.dataDir,
    defaultUser: rawConfig.defaultUser ?? DEFAULT_CONFIG.defaultUser,
    perception: {
      ...DEFAULT_CONFIG.perception,
      ...rawConfig.perception,
    },
    affective: {
      ...DEFAULT_CONFIG.affective,
      ...(rawConfig as Partial<Config>).affective,
    },
    embedding: {
      ...DEFAULT_CONFIG.embedding,
      ...rawConfig.embedding,
    },
    anthropic: {
      ...DEFAULT_CONFIG.anthropic,
      ...rawConfig.anthropic,
      models: {
        ...DEFAULT_CONFIG.anthropic.models,
        ...rawConfig.anthropic?.models,
      },
    },
    procedural: {
      ...DEFAULT_CONFIG.procedural,
      ...(rawConfig as Partial<Config>).procedural,
    },
    executive: {
      ...DEFAULT_CONFIG.executive,
      ...(rawConfig as Partial<Config>).executive,
    },
    offline: {
      ...DEFAULT_CONFIG.offline,
      ...rawConfig.offline,
      consolidator: {
        ...DEFAULT_CONFIG.offline.consolidator,
        ...rawConfig.offline?.consolidator,
      },
      reflector: {
        ...DEFAULT_CONFIG.offline.reflector,
        ...rawConfig.offline?.reflector,
      },
      proceduralSynthesizer: {
        ...DEFAULT_CONFIG.offline.proceduralSynthesizer,
        ...rawConfig.offline?.proceduralSynthesizer,
      },
      curator: {
        ...DEFAULT_CONFIG.offline.curator,
        ...rawConfig.offline?.curator,
      },
      overseer: {
        ...DEFAULT_CONFIG.offline.overseer,
        ...rawConfig.offline?.overseer,
      },
      ruminator: {
        ...DEFAULT_CONFIG.offline.ruminator,
        ...rawConfig.offline?.ruminator,
      },
      selfNarrator: {
        ...DEFAULT_CONFIG.offline.selfNarrator,
        ...rawConfig.offline?.selfNarrator,
      },
    },
    autonomy: {
      ...DEFAULT_CONFIG.autonomy,
      ...rawConfig.autonomy,
      triggers: {
        ...DEFAULT_CONFIG.autonomy.triggers,
        ...rawConfig.autonomy?.triggers,
        commitmentExpiring: {
          ...DEFAULT_CONFIG.autonomy.triggers.commitmentExpiring,
          ...rawConfig.autonomy?.triggers?.commitmentExpiring,
        },
        openQuestionDormant: {
          ...DEFAULT_CONFIG.autonomy.triggers.openQuestionDormant,
          ...rawConfig.autonomy?.triggers?.openQuestionDormant,
        },
        scheduledReflection: {
          ...DEFAULT_CONFIG.autonomy.triggers.scheduledReflection,
          ...rawConfig.autonomy?.triggers?.scheduledReflection,
        },
      },
    },
  });
}

function createMigrations(): Migration[] {
  return [
    ...episodicMigrations,
    ...selfMigrations,
    ...executiveMigrations,
    ...identityMigrations,
    ...affectiveMigrations,
    ...retrievalMigrations,
    ...semanticMigrations,
    ...commitmentMigrations,
    ...socialMigrations,
    ...proceduralMigrations,
    ...offlineMigrations,
    ...autonomyMigrations,
    ...streamWatermarkMigrations,
    ...streamEntryIndexMigrations,
  ];
}

export function openBorgStorage(config: Config): BorgStorage {
  return {
    sqlite: openDatabase(join(config.dataDir, "borg.db"), {
      migrations: createMigrations(),
    }),
    lance: new LanceDbStore({
      uri: join(config.dataDir, "lancedb"),
    }),
  };
}

export async function openBorgLanceTables(options: {
  lance: LanceDbStore;
  embeddingDimensions: number;
}): Promise<BorgLanceTables> {
  const episodesTable = await options.lance.openTable({
    name: "episodes",
    schema: createEpisodesTableSchema(options.embeddingDimensions),
  });
  const semanticNodesTable = await options.lance.openTable({
    name: "semantic_nodes",
    schema: createSemanticNodesTableSchema(options.embeddingDimensions),
  });
  const skillsTable = await options.lance.openTable({
    name: "skills",
    schema: createSkillsTableSchema(options.embeddingDimensions),
  });

  return {
    episodesTable,
    semanticNodesTable,
    skillsTable,
  };
}
