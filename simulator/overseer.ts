import { existsSync, readFileSync } from "node:fs";

import Anthropic from "@anthropic-ai/sdk";
import type {
  ContentBlock,
  Message,
  MessageParam,
  TextBlockParam,
  Tool,
  ToolResultBlockParam,
  ToolUseBlock,
} from "@anthropic-ai/sdk/resources/messages/messages.js";
import { z } from "zod";

import { BorgTransport } from "../assessor/borg-transport.js";
import { getFreshCredentials } from "../src/auth/claude-oauth.js";
import { CLAUDE_CODE_IDENTITY_BLOCK_TEXT, createOAuthFetch } from "../src/llm/index.js";
import type { StreamEntry } from "../src/stream/index.js";

import type { MetricsRow, OverseerVerdict } from "./types.js";

const OVERSEER_MODEL = "claude-opus-4-7";
const OAUTH_BETAS = "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14";
const OAUTH_USER_AGENT = "claude-cli/2.1.2 (external, cli)";

const verdictSchema = z.object({
  status: z.enum(["healthy", "concerning", "failing"]),
  observations: z.array(z.string().min(1)),
  recommendation: z.string().min(1),
});

type OverseerClient = {
  messages: {
    stream(params: {
      model: string;
      system?: string | TextBlockParam[];
      messages: MessageParam[];
      tools: Tool[];
      max_tokens: number;
    }): {
      finalMessage(): Promise<Message>;
    };
  };
};

export type RunOverseerOptions = {
  transport: BorgTransport;
  metricsPath: string;
  turnCounter: number;
  totalTurns: number;
  mock?: boolean;
  client?: OverseerClient;
  systemPrefix?: TextBlockParam[];
  env?: NodeJS.ProcessEnv;
};

const OVERSEER_TOOLS: Tool[] = [
  {
    name: "submit_overseer_verdict",
    description: "Submit the simulator checkpoint health verdict.",
    input_schema: {
      type: "object",
      properties: {
        status: {
          type: "string",
          enum: ["healthy", "concerning", "failing"],
        },
        observations: {
          type: "array",
          items: {
            type: "string",
          },
        },
        recommendation: {
          type: "string",
        },
      },
      required: ["status", "observations", "recommendation"],
      additionalProperties: false,
    },
  },
];

function isToolUseBlock(block: ContentBlock): block is ToolUseBlock {
  return block.type === "tool_use";
}

function toolResult(id: string, content: string, isError = false): ToolResultBlockParam {
  return {
    type: "tool_result",
    tool_use_id: id,
    content,
    ...(isError ? { is_error: true } : {}),
  };
}

async function createDefaultOverseerClient(
  env: NodeJS.ProcessEnv = process.env,
): Promise<{ client: OverseerClient; systemPrefix: TextBlockParam[] }> {
  const apiKey = env.ANTHROPIC_API_KEY?.trim();

  if (apiKey !== undefined && apiKey.length > 0) {
    return {
      client: new Anthropic({ apiKey }),
      systemPrefix: [],
    };
  }

  const authToken = env.ANTHROPIC_AUTH_TOKEN?.trim();
  const credentials =
    authToken === undefined || authToken.length === 0 ? await getFreshCredentials({ env }) : null;
  const resolvedToken =
    authToken !== undefined && authToken.length > 0 ? authToken : credentials?.accessToken;

  if (resolvedToken === undefined || resolvedToken.length === 0) {
    throw new Error("No Anthropic credentials detected for real simulator overseer mode");
  }

  return {
    client: new Anthropic({
      authToken: resolvedToken,
      defaultHeaders: {
        "anthropic-beta": OAUTH_BETAS,
        "user-agent": OAUTH_USER_AGENT,
      },
      fetch: createOAuthFetch(),
    }),
    systemPrefix: [
      {
        type: "text",
        text: CLAUDE_CODE_IDENTITY_BLOCK_TEXT,
      },
    ],
  };
}

function systemParam(prefix: readonly TextBlockParam[]): string | TextBlockParam[] {
  const prompt =
    "You are auditing a long-running Borg conversation for cognitive degradation. Use only the submit_overseer_verdict tool.";

  if (prefix.length === 0) {
    return prompt;
  }

  return [
    ...prefix,
    {
      type: "text",
      text: prompt,
    },
  ];
}

function readRecentMetrics(path: string, limit: number): MetricsRow[] {
  if (!existsSync(path)) {
    return [];
  }

  return readFileSync(path, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .slice(-limit)
    .map((line) => JSON.parse(line) as MetricsRow);
}

function entryContent(entry: StreamEntry): string {
  return typeof entry.content === "string" ? entry.content : JSON.stringify(entry.content);
}

function conversationWindow(transport: BorgTransport): string {
  return transport
    .streamTail(100)
    .filter((entry) => entry.kind === "user_msg" || entry.kind === "agent_msg")
    .slice(-100)
    .map((entry) => `${entry.kind}: ${entryContent(entry).slice(0, 500)}`)
    .join("\n");
}

function buildPrompt(options: RunOverseerOptions): string {
  const startTurn = Math.max(1, options.turnCounter - 50);
  const metrics = readRecentMetrics(options.metricsPath, 5)
    .map((row) =>
      [
        `turn=${row.turn_counter}`,
        `episodes=${row.episode_count}`,
        `semantic_nodes=${row.semantic_node_count}`,
        `semantic_edges=${row.semantic_edge_count}`,
        `semantic_nodes_added=${row.semantic_nodes_added_since_last_check}`,
        `semantic_edges_added=${row.semantic_edges_added_since_last_check}`,
        `open_questions=${row.open_question_count}`,
        `active_goals=${row.active_goal_count}`,
        `retrieval_ms=${row.retrieval_latency_ms ?? "null"}`,
      ].join(" "),
    )
    .join("\n");

  return [
    `Sample window: turns ${startTurn} to ${options.turnCounter} of ${options.totalTurns}.`,
    `Metrics trajectory:\n${metrics.length === 0 ? "No metrics rows yet." : metrics}`,
    `Conversation window:\n${conversationWindow(options.transport) || "No conversation entries."}`,
    "",
    "Specific questions:",
    "1. Has recall stayed sharp? Look for facts seeded early being correctly retrieved.",
    "2. Is the autobiographical narrative coherent?",
    "3. Has identity remained stable?",
    "4. Are open questions getting resolved or piling up?",
    "5. Has retrieval latency grown superlinearly with episode count?",
    "6. Has the semantic graph fragmented or stayed coherent?",
    "7. Any signs of accumulated false memories or contradiction debt?",
  ].join("\n\n");
}

export async function runOverseer(options: RunOverseerOptions): Promise<OverseerVerdict> {
  if (options.mock === true) {
    return {
      ts: Date.now(),
      turn_counter: options.turnCounter,
      status: "healthy",
      observations: ["Mock overseer checkpoint completed."],
      recommendation: "Continue the run and inspect metrics trends after completion.",
    };
  }

  const initialized =
    options.client === undefined
      ? await createDefaultOverseerClient(options.env)
      : { client: options.client, systemPrefix: options.systemPrefix ?? [] };
  const messages: MessageParam[] = [
    {
      role: "user",
      content: buildPrompt(options),
    },
  ];

  for (let attempt = 0; attempt < 4; attempt += 1) {
    const response = await initialized.client.messages
      .stream({
        model: OVERSEER_MODEL,
        system: systemParam(initialized.systemPrefix),
        messages,
        tools: OVERSEER_TOOLS,
        max_tokens: 4_000,
      })
      .finalMessage();
    messages.push({
      role: "assistant",
      content: response.content,
    });

    const toolUses = response.content.filter(isToolUseBlock);
    const results: ToolResultBlockParam[] = [];

    for (const use of toolUses) {
      if (use.name !== "submit_overseer_verdict") {
        results.push(toolResult(use.id, `Unknown overseer tool: ${use.name}`, true));
        continue;
      }

      const parsed = verdictSchema.safeParse(use.input);

      if (!parsed.success) {
        results.push(toolResult(use.id, parsed.error.message, true));
        continue;
      }

      return {
        ts: Date.now(),
        turn_counter: options.turnCounter,
        ...parsed.data,
      };
    }

    messages.push({
      role: "user",
      content:
        results.length === 0
          ? "Submit the checkpoint verdict using submit_overseer_verdict."
          : results,
    });
  }

  return {
    ts: Date.now(),
    turn_counter: options.turnCounter,
    status: "concerning",
    observations: ["Overseer did not submit a structured verdict within the call cap."],
    recommendation: "Inspect this checkpoint manually before trusting later degradation signals.",
  };
}
