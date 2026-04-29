import Anthropic from "@anthropic-ai/sdk";
import type {
  ContentBlock,
  Message,
  MessageParam,
  TextBlockParam,
} from "@anthropic-ai/sdk/resources/messages/messages.js";

import { getFreshCredentials } from "../src/auth/claude-oauth.js";
import { CLAUDE_CODE_IDENTITY_BLOCK_TEXT, createOAuthFetch } from "../src/llm/index.js";

import type { Persona } from "./types.js";

export const PERSONA_MODEL = "claude-opus-4-7";

const OAUTH_BETAS = "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14";
const OAUTH_USER_AGENT = "claude-cli/2.1.2 (external, cli)";
const DEFAULT_MOCK_MESSAGES = [
  "My dog's name is Otto, and I'm trying to get a clean start on a distributed systems project this week.",
  "The architecture is still fuzzy, but I think Spanish practice after dinner is helping me reset.",
  "Otto stole a sock again, which somehow made debugging replication lag less annoying.",
  "I'm a little frustrated tonight; the scaling plan at work keeps changing.",
  "Do you think I should keep pushing on the design doc or step back and read for a bit?",
] as const;

type PersonaClient = {
  messages: {
    stream(params: {
      model: string;
      system?: string | TextBlockParam[];
      messages: MessageParam[];
      max_tokens: number;
    }): {
      finalMessage(): Promise<Message>;
    };
  };
};

export type PersonaSessionOptions = {
  persona: Persona;
  mock?: boolean;
  mockMessages?: readonly string[];
  client?: PersonaClient;
  systemPrefix?: TextBlockParam[];
  model?: string;
  env?: NodeJS.ProcessEnv;
};

type PersonaClientInit = {
  client: PersonaClient;
  systemPrefix: TextBlockParam[];
};

async function createDefaultPersonaClient(
  env: NodeJS.ProcessEnv = process.env,
): Promise<PersonaClientInit> {
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
    throw new Error("No Anthropic credentials detected for real simulator persona mode");
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

function systemParam(
  prefix: readonly TextBlockParam[],
  persona: Persona,
): string | TextBlockParam[] {
  if (prefix.length === 0) {
    return persona.systemPrompt;
  }

  return [
    ...prefix,
    {
      type: "text",
      text: persona.systemPrompt,
    },
  ];
}

function contentText(block: ContentBlock): string {
  if (block.type === "text") {
    return block.text;
  }

  return "";
}

function responseText(message: Message): string {
  return message.content
    .map((block) => contentText(block))
    .join("\n")
    .trim();
}

function initialPrompt(persona: Persona): string {
  const facts =
    persona.seedFacts === undefined || persona.seedFacts.length === 0
      ? ""
      : `\nSeed facts you may weave in naturally:\n${persona.seedFacts
          .map((fact) => `- ${fact}`)
          .join("\n")}`;

  return `Open the conversation. You may seed any of the listed facts naturally.${facts}`;
}

export class PersonaSession {
  private readonly persona: Persona;
  private readonly mock: boolean;
  private readonly mockMessages: readonly string[];
  private readonly client?: PersonaClient;
  private readonly systemPrefix: TextBlockParam[];
  private readonly model: string;
  private readonly env: NodeJS.ProcessEnv;
  private readonly messages: MessageParam[] = [];
  private mockIndex = 0;

  constructor(options: PersonaSessionOptions) {
    this.persona = options.persona;
    this.mock = options.mock ?? false;
    this.mockMessages = options.mockMessages ?? DEFAULT_MOCK_MESSAGES;
    this.client = options.client;
    this.systemPrefix = options.systemPrefix ?? [];
    this.model = options.model ?? PERSONA_MODEL;
    this.env = options.env ?? process.env;
  }

  async nextTurn(borgPreviousResponse: string | null): Promise<string> {
    if (this.mock) {
      const message =
        this.mockMessages[this.mockIndex % this.mockMessages.length] ?? DEFAULT_MOCK_MESSAGES[0];
      this.mockIndex += 1;
      return message;
    }

    const initialized =
      this.client === undefined
        ? await createDefaultPersonaClient(this.env)
        : { client: this.client, systemPrefix: this.systemPrefix };
    const userMessage =
      this.messages.length === 0 || borgPreviousResponse === null
        ? initialPrompt(this.persona)
        : borgPreviousResponse;

    this.messages.push({
      role: "user",
      content: userMessage,
    });

    const response = await initialized.client.messages
      .stream({
        model: this.model,
        system: systemParam(initialized.systemPrefix, this.persona),
        messages: this.messages,
        max_tokens: 4_000,
      })
      .finalMessage();
    const text = responseText(response);

    if (text.length === 0) {
      throw new Error("Persona LLM produced an empty turn");
    }

    this.messages.push({
      role: "assistant",
      content: text,
    });

    return text;
  }
}
