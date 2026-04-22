import { createHash } from "node:crypto";
import { createInterface } from "node:readline";

import {
  Borg,
  OFFLINE_PROCESS_NAMES,
  WorkingMemoryStore,
  computeBetaStats,
  parseSessionId,
  type OfflineProcessName,
  type SessionId,
  type TurnStakes,
} from "../src/index.ts";
import { BorgError } from "../src/util/errors.js";
import { createAnsi } from "./_ansi.ts";
import { selectScriptClients, type ScriptClientSelectionMode } from "./_clients.ts";

type SessionTarget = {
  label: string;
  id: SessionId;
};

type ChatOptions = {
  session: SessionTarget;
  audience: string;
  stakes: TurnStakes;
  clientMode: ScriptClientSelectionMode;
};

const ansi = createAnsi();
const OFFLINE_PROCESS_SET = new Set<OfflineProcessName>(OFFLINE_PROCESS_NAMES);
const FORCE_EXIT_WINDOW_MS = 2_000;
const DEFAULT_EXTRACT_EVERY = 6;

function writeLine(line: string): void {
  process.stdout.write(`${line}\n`);
}

function writeWarn(line: string): void {
  writeLine(ansi.yellow(`WARN ${line}`));
}

function truncate(text: string, limit = 160): string {
  const collapsed = text.replace(/\s+/g, " ").trim();
  return collapsed.length <= limit ? collapsed : `${collapsed.slice(0, limit - 1)}…`;
}

function hashSessionLabel(label: string): SessionId {
  const hash = createHash("sha256").update(label).digest("hex").slice(0, 16);
  return parseSessionId(`sess_${hash}`);
}

function resolveSessionTarget(raw: string): SessionTarget {
  const label = raw.trim();

  if (label.length === 0) {
    throw new Error("Session is required");
  }

  try {
    return {
      label,
      id: parseSessionId(label),
    };
  } catch {
    return {
      label,
      id: hashSessionLabel(label.toLowerCase()),
    };
  }
}

function parseStakes(value: string): TurnStakes {
  if (value === "low" || value === "medium" || value === "high") {
    return value;
  }

  throw new Error(`Invalid stakes: ${value}`);
}

function parseStartupArgs(argv: readonly string[]): ChatOptions {
  let session = resolveSessionTarget("chat");
  let audience = "user";
  let stakes: TurnStakes = "medium";
  let clientMode: ScriptClientSelectionMode = "auto";

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === undefined || arg === "--") {
      continue;
    }

    if (arg === "--session") {
      const value = argv[index + 1];

      if (value === undefined) {
        throw new Error("--session requires a value");
      }

      session = resolveSessionTarget(value);
      index += 1;
      continue;
    }

    if (arg === "--audience") {
      const value = argv[index + 1];

      if (value === undefined || value.trim() === "") {
        throw new Error("--audience requires a value");
      }

      audience = value.trim();
      index += 1;
      continue;
    }

    if (arg === "--stakes") {
      const value = argv[index + 1];

      if (value === undefined) {
        throw new Error("--stakes requires a value");
      }

      stakes = parseStakes(value.trim().toLowerCase());
      index += 1;
      continue;
    }

    if (arg === "--real") {
      clientMode = "real";
      continue;
    }

    if (arg === "--fakes") {
      clientMode = "fakes";
      continue;
    }

    throw new Error(`Unknown argument: ${arg}`);
  }

  return {
    session,
    audience,
    stakes,
    clientMode,
  };
}

function splitQuotedArgs(input: string): string[] {
  const parts: string[] = [];
  let current = "";
  let quote: '"' | "'" | null = null;

  for (let index = 0; index < input.length; index += 1) {
    const character = input[index];

    if (character === undefined) {
      continue;
    }

    if (quote !== null) {
      if (character === quote) {
        quote = null;
      } else if (character === "\\" && index + 1 < input.length) {
        current += input[index + 1] ?? "";
        index += 1;
      } else {
        current += character;
      }

      continue;
    }

    if (character === '"' || character === "'") {
      quote = character;
      continue;
    }

    if (/\s/.test(character)) {
      if (current.length > 0) {
        parts.push(current);
        current = "";
      }

      continue;
    }

    current += character;
  }

  if (quote !== null) {
    throw new Error("Unclosed quote");
  }

  if (current.length > 0) {
    parts.push(current);
  }

  return parts;
}

function takeOption(args: string[], flag: string): string | undefined {
  const lowerFlag = flag.toLowerCase();
  const index = args.findIndex((value) => value.toLowerCase() === lowerFlag);

  if (index === -1) {
    return undefined;
  }

  const value = args[index + 1];

  if (value === undefined) {
    throw new Error(`${flag} requires a value`);
  }

  args.splice(index, 2);
  return value;
}

function takeBooleanFlag(args: string[], flag: string): boolean {
  const lowerFlag = flag.toLowerCase();
  const index = args.findIndex((value) => value.toLowerCase() === lowerFlag);

  if (index === -1) {
    return false;
  }

  args.splice(index, 1);
  return true;
}

function parsePositiveInteger(value: string, label: string): number {
  const parsed = Number(value);

  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }

  return parsed;
}

function parseSinceToTimestamp(raw: string): number {
  const trimmed = raw.trim();

  if (/^\d+$/.test(trimmed)) {
    const value = Number(trimmed);

    if (!Number.isFinite(value)) {
      throw new Error(`Invalid timestamp: ${raw}`);
    }

    return value;
  }

  const match = trimmed.match(/^(\d+)(ms|s|m|h|d|w)$/i);

  if (match === null) {
    throw new Error(`Invalid duration: ${raw}`);
  }

  const value = Number(match[1]);
  const unit = (match[2] ?? "").toLowerCase();
  const multiplier =
    unit === "ms"
      ? 1
      : unit === "s"
        ? 1_000
        : unit === "m"
          ? 60_000
          : unit === "h"
            ? 3_600_000
            : unit === "d"
              ? 86_400_000
              : 604_800_000;

  return Date.now() - value * multiplier;
}

function parseExtractEvery(raw: string | undefined): number {
  if (raw === undefined || raw.trim() === "") {
    return DEFAULT_EXTRACT_EVERY;
  }

  const value = Number(raw);

  if (!Number.isInteger(value) || value < 0) {
    throw new Error("BORG_CHAT_EXTRACT_EVERY must be a non-negative integer");
  }

  return value;
}

function flattenGoals(
  goals: ReadonlyArray<{
    description: string;
    priority: number;
    status: string;
    children?: unknown;
  }>,
): Array<{
  description: string;
  priority: number;
  status: string;
}> {
  const flattened: Array<{
    description: string;
    priority: number;
    status: string;
  }> = [];
  const queue = [...goals];

  while (queue.length > 0) {
    const next = queue.shift();

    if (next === undefined) {
      continue;
    }

    flattened.push(next);

    if (Array.isArray(next.children)) {
      queue.push(...(next.children as typeof queue));
    }
  }

  return flattened;
}

function formatError(error: unknown): string {
  if (error instanceof BorgError) {
    const cause =
      error.cause instanceof Error
        ? error.cause.message
        : error.cause === undefined
          ? ""
          : String(error.cause);
    return cause.length === 0 ? error.message : `${error.message} (cause: ${cause})`;
  }

  if (error instanceof Error) {
    const cause =
      "cause" in error && error.cause instanceof Error ? ` (cause: ${error.cause.message})` : "";
    return `${error.message}${cause}`;
  }

  return String(error);
}

function printDim(line: string): void {
  writeLine(ansi.dim(line));
}

function formatThoughtLines(thought: string): string[] {
  const prefix = "  thought: ";
  const indent = " ".repeat(prefix.length);

  return thought
    .split(/\r?\n/)
    .map((line) => line.trimEnd())
    .filter((line) => line.length > 0)
    .map((line, index) => `${index === 0 ? prefix : indent}${line}`);
}

function printHelp(): void {
  writeLine("Commands:");
  writeLine("/                              Same as /help.");
  writeLine("/help                          Show this help.");
  writeLine("/exit, /quit                   Clean shutdown.");
  writeLine("/who                           Show session, audience, stakes, and client sources.");
  writeLine("/session <id>                  Switch session.");
  writeLine("/audience <name>               Set audience for subsequent turns.");
  writeLine("/stakes <low|medium|high>      Set default stakes.");
  writeLine("/tail [n=10] [--session <id>]  Show recent stream entries.");
  writeLine("/episodes <query> [--limit N]  Search episodic memory.");
  writeLine("/questions [--status open|all] Show open questions.");
  writeLine("/commitments [--audience X]    Show active commitments.");
  writeLine("/goals | /values | /traits     Show current self state.");
  writeLine("/mood | /period | /growth      Show affective and narrative state.");
  writeLine("/skills                        Show procedural skills.");
  writeLine("/dream [--apply] [--process X,Y,...] [--budget N]  Run maintenance.");
  writeLine("/extract [--since <rel>]       Force stream extraction for the current session.");
  writeLine("/save                          Force working-memory save.");
}

async function main(): Promise<void> {
  const options = parseStartupArgs(process.argv.slice(2));
  const extractEvery = parseExtractEvery(process.env.BORG_CHAT_EXTRACT_EVERY);
  const selection = await selectScriptClients({
    dataDir: process.env.BORG_DATA_DIR,
    mode: options.clientMode,
    warn: (message) => writeWarn(message),
  });
  const borg = await Borg.open({
    config: selection.config,
    embeddingDimensions: selection.embeddingDimensions,
    embeddingClient: selection.embeddings,
    llmClient: selection.llm,
    // Live episodic extraction: every completed turn walks the stream past
    // the per-session watermark and extracts new episodes. Next turn's
    // retrieval sees material from the turn that just ran, not just
    // episodes from prior sessions or manual extract runs.
    liveExtraction: selection.llmMode === "real",
  });
  const persistenceStore = new WorkingMemoryStore({
    dataDir: selection.config.dataDir,
  });
  const state = {
    session: options.session,
    audience: options.audience,
    stakes: options.stakes,
  };
  let closed = false;
  let shuttingDown: Promise<void> | null = null;
  let lastSigintAt = 0;
  let extractionInFlight: {
    session: SessionTarget;
    promise: Promise<void>;
  } | null = null;

  borg.workmem.load(state.session.id);
  writeLine(
    `borg chat ready. session=${state.session.label} audience=${state.audience} stakes=${state.stakes} (${selection.llmMode} llm, ${selection.embeddingMode} embeddings). /help for commands.`,
  );

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: process.stdin.isTTY === true && process.stdout.isTTY === true,
    prompt: "you > ",
  });

  const saveCurrentSession = (): void => {
    const snapshot = borg.workmem.load(state.session.id);
    persistenceStore.save(snapshot);
  };

  const startExtraction = (
    session: SessionTarget,
    options: {
      awaitResult: boolean;
      skipIfRunning: boolean;
      sinceTs?: number;
    },
  ): Promise<void> | null => {
    if (extractionInFlight !== null) {
      if (options.skipIfRunning) {
        return null;
      }

      return extractionInFlight.promise;
    }

    const run = (async () => {
      try {
        const result = await borg.episodic.extract({
          session: session.id,
          sinceTs: options.sinceTs,
        });
        printDim(
          `extracted: inserted=${result.inserted} updated=${result.updated} skipped=${result.skipped}`,
        );
      } catch (error) {
        writeWarn(`extract failed: ${formatError(error)}`);
      } finally {
        if (extractionInFlight?.promise === run) {
          extractionInFlight = null;
        }
      }
    })();

    extractionInFlight = {
      session,
      promise: run,
    };

    if (options.awaitResult) {
      return run;
    }

    void run;
    return run;
  };

  const awaitExtractionForSession = async (session: SessionTarget): Promise<void> => {
    if (extractionInFlight?.session.id === session.id) {
      await extractionInFlight.promise;
      return;
    }

    if (extractionInFlight !== null) {
      await extractionInFlight.promise;
    }

    await startExtraction(session, {
      awaitResult: true,
      skipIfRunning: false,
    });
  };

  const shutdown = async (): Promise<void> => {
    if (shuttingDown !== null) {
      return shuttingDown;
    }

    shuttingDown = (async () => {
      try {
        saveCurrentSession();
        await awaitExtractionForSession(state.session);
      } finally {
        await borg.close();
        writeLine(`saved working memory, session=${state.session.label}`);
      }
    })();

    return shuttingDown;
  };

  const exitCleanly = async (code = 0): Promise<void> => {
    try {
      await shutdown();
      process.exitCode = code;
    } catch (error) {
      writeLine(ansi.red(formatError(error)));
      process.exitCode = 1;
    }
  };

  const processDream = async (args: string[]): Promise<void> => {
    const apply = takeBooleanFlag(args, "--apply");
    const processList = takeOption(args, "--process");
    const budget = takeOption(args, "--budget");

    if (args.length > 0) {
      throw new Error(`Unexpected arguments: ${args.join(" ")}`);
    }

    const processes =
      processList === undefined
        ? [...OFFLINE_PROCESS_NAMES]
        : processList.split(",").map((value) => {
            const trimmed = value.trim() as OfflineProcessName;

            if (!OFFLINE_PROCESS_SET.has(trimmed)) {
              throw new Error(`Unknown process: ${value}`);
            }

            return trimmed;
          });
    const result = await borg.dream({
      dryRun: !apply,
      processes,
      budget: budget === undefined ? undefined : parsePositiveInteger(budget, "--budget"),
    });

    for (const process of result.results) {
      writeLine(
        `${process.process} ${process.dryRun ? "dry-run" : "apply"} changes=${process.changes.length} tokens=${process.tokens_used} budget_exhausted=${process.budget_exhausted} errors=${process.errors.length}`,
      );
    }
  };

  const requestClose = (): void => {
    if (closed) {
      return;
    }

    closed = true;
    rl.close();
  };

  const handleCommand = async (line: string): Promise<void> => {
    const tokens = splitQuotedArgs(line.slice(1).trim());
    const command = (tokens.shift() ?? "help").toLowerCase();

    if (command === "" || command === "help") {
      printHelp();
      return;
    }

    if (command === "exit" || command === "quit") {
      requestClose();
      return;
    }

    if (command === "who") {
      writeLine(
        `session=${state.session.label} audience=${state.audience} stakes=${state.stakes} llm=${selection.llmMode} embeddings=${selection.embeddingMode}`,
      );
      return;
    }

    if (command === "session") {
      const value = tokens[0];

      if (value === undefined) {
        throw new Error("/session requires an id");
      }

      saveCurrentSession();
      state.session = resolveSessionTarget(value);
      borg.workmem.load(state.session.id);
      writeLine(`session=${state.session.label}`);
      return;
    }

    if (command === "audience") {
      const value = tokens.join(" ").trim();

      if (value.length === 0) {
        throw new Error("/audience requires a value");
      }

      state.audience = value;
      writeLine(`audience=${state.audience}`);
      return;
    }

    if (command === "stakes") {
      const value = tokens[0]?.toLowerCase();

      if (value === undefined) {
        throw new Error("/stakes requires low, medium, or high");
      }

      state.stakes = parseStakes(value);
      writeLine(`stakes=${state.stakes}`);
      return;
    }

    if (command === "tail") {
      const args = [...tokens];
      const sessionArg = takeOption(args, "--session");
      const limit = args[0] === undefined ? 10 : parsePositiveInteger(args[0], "/tail limit");
      const session = sessionArg === undefined ? state.session : resolveSessionTarget(sessionArg);
      const entries = borg.stream.tail(limit, { session: session.id });

      if (entries.length === 0) {
        writeLine("no stream entries");
        return;
      }

      for (const entry of entries) {
        const content =
          typeof entry.content === "string" ? entry.content : JSON.stringify(entry.content);
        writeLine(
          `${new Date(entry.timestamp).toISOString()} ${entry.kind} ${truncate(content, 100)}`,
        );
      }

      return;
    }

    if (command === "episodes") {
      const args = [...tokens];
      const limitArg = takeOption(args, "--limit");
      const query = args.join(" ").trim();

      if (query.length === 0) {
        throw new Error("/episodes requires a query");
      }

      const results = await borg.episodic.search(query, {
        limit: limitArg === undefined ? 5 : parsePositiveInteger(limitArg, "--limit"),
      });

      if (results.length === 0) {
        writeLine("no episodes matched");
        return;
      }

      for (const result of results.slice(0, 8)) {
        writeLine(
          `${result.episode.title} score=${result.score.toFixed(3)} cites=${result.citationChain.length}`,
        );
      }

      if (results.length > 8) {
        writeLine(`... (${results.length - 8} more)`);
      }

      return;
    }

    if (command === "questions") {
      const args = [...tokens];
      const statusArg = takeOption(args, "--status")?.toLowerCase();

      if (args.length > 0) {
        throw new Error(`Unexpected arguments: ${args.join(" ")}`);
      }

      if (statusArg !== undefined && statusArg !== "open" && statusArg !== "all") {
        throw new Error("--status must be open or all");
      }

      const questions = borg.self.openQuestions.list({
        status: statusArg === undefined || statusArg === "all" ? undefined : "open",
        limit: 10,
      });

      if (questions.length === 0) {
        writeLine("no questions");
        return;
      }

      for (const question of questions) {
        writeLine(
          `${question.id} ${question.status} urgency=${question.urgency.toFixed(2)} ${truncate(question.question, 80)}`,
        );
      }

      return;
    }

    if (command === "commitments") {
      const args = [...tokens];
      const audience = takeOption(args, "--audience") ?? state.audience;

      if (args.length > 0) {
        throw new Error(`Unexpected arguments: ${args.join(" ")}`);
      }

      const commitments = borg.commitments.list({
        activeOnly: true,
        audience,
      });

      if (commitments.length === 0) {
        writeLine("no active commitments");
        return;
      }

      for (const commitment of commitments.slice(0, 8)) {
        writeLine(
          `${commitment.id} ${commitment.type} p=${commitment.priority} ${truncate(commitment.directive, 90)}`,
        );
      }

      if (commitments.length > 8) {
        writeLine(`... (${commitments.length - 8} more)`);
      }

      return;
    }

    if (command === "goals") {
      const goals = flattenGoals(borg.self.goals.list({ status: "active" }));

      if (goals.length === 0) {
        writeLine("no active goals");
        return;
      }

      for (const goal of goals.slice(0, 8)) {
        writeLine(`p=${goal.priority} ${truncate(goal.description, 100)}`);
      }

      if (goals.length > 8) {
        writeLine(`... (${goals.length - 8} more)`);
      }

      return;
    }

    if (command === "values") {
      const values = borg.self.values.list();

      if (values.length === 0) {
        writeLine("no values");
        return;
      }

      for (const value of values.slice(0, 8)) {
        writeLine(`p=${value.priority} ${value.label} — ${truncate(value.description, 80)}`);
      }

      if (values.length > 8) {
        writeLine(`... (${values.length - 8} more)`);
      }

      return;
    }

    if (command === "traits") {
      const traits = borg.self.traits
        .list()
        .slice()
        .sort(
          (left, right) => right.strength - left.strength || left.label.localeCompare(right.label),
        );

      if (traits.length === 0) {
        writeLine("no traits");
        return;
      }

      for (const trait of traits.slice(0, 8)) {
        writeLine(`${trait.label} strength=${trait.strength.toFixed(2)}`);
      }

      if (traits.length > 8) {
        writeLine(`... (${traits.length - 8} more)`);
      }

      return;
    }

    if (command === "mood") {
      const current = borg.mood.current(state.session.id);
      const history = borg.mood.history(state.session.id, { limit: 50 });
      writeLine(
        `mood valence=${current.valence.toFixed(2)} arousal=${current.arousal.toFixed(2)} history=${history.length}`,
      );
      return;
    }

    if (command === "period") {
      const period = borg.self.autobiographical.currentPeriod();

      if (period === null) {
        writeLine("no current period");
        return;
      }

      writeLine(`${period.label} ${truncate(period.narrative, 120)}`);
      return;
    }

    if (command === "growth") {
      const limit = tokens[0] === undefined ? 5 : parsePositiveInteger(tokens[0], "/growth limit");
      const markers = borg.self.growthMarkers.list({ limit });

      if (markers.length === 0) {
        writeLine("no growth markers");
        return;
      }

      for (const marker of markers.slice(0, limit)) {
        writeLine(
          `${marker.category} conf=${marker.confidence.toFixed(2)} ${truncate(marker.what_changed, 90)}`,
        );
      }

      return;
    }

    if (command === "skills") {
      const skills = borg.skills.list(8);

      if (skills.length === 0) {
        writeLine("no skills");
        return;
      }

      for (const skill of skills) {
        const stats = computeBetaStats(skill.alpha, skill.beta);
        writeLine(
          `${skill.id} ${truncate(skill.applies_when, 42)} mean=${stats.mean.toFixed(2)} ci=${stats.ci_95[0].toFixed(2)}-${stats.ci_95[1].toFixed(2)}`,
        );
      }

      return;
    }

    if (command === "dream") {
      await processDream([...tokens]);
      return;
    }

    if (command === "extract") {
      const args = [...tokens];
      const sinceArg = takeOption(args, "--since");

      if (args.length > 0) {
        throw new Error(`Unexpected arguments: ${args.join(" ")}`);
      }

      const result = await borg.episodic.extract({
        session: state.session.id,
        sinceTs: sinceArg === undefined ? undefined : parseSinceToTimestamp(sinceArg),
      });
      writeLine(
        `extract inserted=${result.inserted} updated=${result.updated} skipped=${result.skipped}`,
      );
      return;
    }

    if (command === "save") {
      saveCurrentSession();
      writeLine(`saved working memory, session=${state.session.label}`);
      return;
    }

    throw new Error(`Unknown command: /${command}`);
  };

  const handleTurn = async (message: string): Promise<void> => {
    printDim("[thinking]");
    const result = await borg.turn({
      userMessage: message,
      sessionId: state.session.id,
      audience: state.audience,
      stakes: state.stakes,
    });

    writeLine(`${ansi.strong("borg >")} ${result.response}`);
    printDim(
      `[mode=${result.mode} path=${result.path === "system_1" ? "s1" : "s2"} tokens=${result.usage.input_tokens}/${result.usage.output_tokens} retrieved=${result.retrievedEpisodeIds.length} intents=${result.intents.length} thoughts=${result.thoughts.length}]`,
    );

    if (result.path === "system_2" && result.thoughts.length > 0) {
      for (const thought of result.thoughts) {
        for (const line of formatThoughtLines(thought)) {
          printDim(line);
        }
      }
    }

    if (extractEvery > 0) {
      const turnCounter = borg.workmem.load(state.session.id).turn_counter;

      if (turnCounter > 0 && turnCounter % extractEvery === 0) {
        startExtraction({ ...state.session }, { awaitResult: false, skipIfRunning: true });
      }
    }
  };

  let queue = Promise.resolve();

  const prompt = (): void => {
    if (!closed && shuttingDown === null) {
      rl.prompt();
    }
  };

  rl.on("line", (rawLine) => {
    queue = queue
      .then(async () => {
        const line = rawLine.trim();

        if (line.length === 0) {
          return;
        }

        if (line.startsWith("/")) {
          await handleCommand(line);
          return;
        }

        await handleTurn(line);
      })
      .catch((error: unknown) => {
        printDim(formatError(error));
      })
      .finally(() => {
        prompt();
      });
  });

  rl.on("close", () => {
    closed = true;
    void queue.finally(() => exitCleanly(0));
  });

  process.on("SIGINT", () => {
    const now = Date.now();

    if (closed && now - lastSigintAt <= FORCE_EXIT_WINDOW_MS) {
      process.exit(130);
    }

    lastSigintAt = now;
    requestClose();
  });

  prompt();
}

void main().catch((error: unknown) => {
  writeLine(ansi.red(formatError(error)));
  process.exitCode = 1;
});
