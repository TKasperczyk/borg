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

type PersonaTurnDraftKind = "mock" | "llm";

export type PersonaTurnDraft = {
  kind: PersonaTurnDraftKind;
  message: string;
  history: readonly [MessageParam, MessageParam] | null;
  mockIndex: number | null;
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

// Anthropic rejects requests where any message has empty content with
// a 400 'user messages must have non-empty content'. Defense in depth:
// even though we try not to push empty content into history, certain
// edge cases (Borg returning empty, content with only whitespace,
// LLM-internal stop blocks) can sneak in. Sanitize at the boundary so
// a single empty cell doesn't poison the whole request.
const EMPTY_CONTENT_PLACEHOLDER = "(no content)";

function sanitizeMessages(messages: readonly MessageParam[]): MessageParam[] {
  return messages.map((msg) => {
    if (typeof msg.content !== "string") {
      return msg;
    }

    if (msg.content.trim().length === 0) {
      return { ...msg, content: EMPTY_CONTENT_PLACEHOLDER };
    }

    return msg;
  });
}

function initialPrompt(persona: Persona, gapContext: string | null): string {
  const facts =
    persona.seedFacts === undefined || persona.seedFacts.length === 0
      ? ""
      : `\nSeed facts you may weave in naturally:\n${persona.seedFacts
          .map((fact) => `- ${fact}`)
          .join("\n")}`;

  const opener =
    gapContext === null
      ? "Open the conversation. You may seed any of the listed facts naturally."
      : `${gapContext} You're starting a new conversation now -- pick a different topic, a different mood, a different time of day than the last one. You may seed any of the listed facts naturally.`;

  return `${opener}${facts}`;
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
  private nextSessionGap: string | null = null;

  constructor(options: PersonaSessionOptions) {
    this.persona = options.persona;
    this.mock = options.mock ?? false;
    this.mockMessages = options.mockMessages ?? DEFAULT_MOCK_MESSAGES;
    this.client = options.client;
    this.systemPrefix = options.systemPrefix ?? [];
    this.model = options.model ?? PERSONA_MODEL;
    this.env = options.env ?? process.env;
  }

  startNewSession(gapContext: string): void {
    this.messages.length = 0;
    this.nextSessionGap = gapContext;
  }

  async prepareNextTurn(borgPreviousResponse: string | null): Promise<PersonaTurnDraft> {
    if (this.mock) {
      const message =
        this.mockMessages[this.mockIndex % this.mockMessages.length] ?? DEFAULT_MOCK_MESSAGES[0];
      return {
        kind: "mock",
        message,
        history: null,
        mockIndex: this.mockIndex,
      };
    }

    const initialized =
      this.client === undefined
        ? await createDefaultPersonaClient(this.env)
        : { client: this.client, systemPrefix: this.systemPrefix };

    // borgPreviousResponse can be null (first turn), empty string, or
    // whitespace-only -- the latter two happen when Borg's pipeline
    // returns an empty response (e.g., a probe whose finalizer text
    // was zero-length, or a Borg internal failure). Treating an empty
    // string as a real user message would push empty content into our
    // history and trigger a 400 from Anthropic on the next turn. Fall
    // back to the initial prompt so the persona has something to
    // respond to instead of nothing.
    const trimmedBorgResponse = borgPreviousResponse?.trim() ?? "";
    const baseUserMessage =
      this.messages.length === 0 || trimmedBorgResponse.length === 0
        ? initialPrompt(this.persona, this.nextSessionGap)
        : trimmedBorgResponse;
    if (this.messages.length === 0) {
      this.nextSessionGap = null;
    }

    // Build the candidate request messages without committing to
    // history yet -- if the call fails or returns empty, we retry with
    // a nudge before mutating session state. Committing both user and
    // assistant messages atomically (only after a successful non-empty
    // response) is what keeps the history alternation contract intact:
    // a previous version pushed the user message before the API call,
    // and an empty response then left an orphan user message that
    // poisoned subsequent turns.
    const requestMessages: MessageParam[] = [
      ...this.messages,
      { role: "user", content: baseUserMessage },
    ];

    const text = await this.callPersona(initialized, requestMessages);

    if (text.length > 0) {
      return {
        kind: "llm",
        message: text,
        history: [
          { role: "user", content: baseUserMessage },
          { role: "assistant", content: text },
        ],
        mockIndex: null,
      };
    }

    // Empty response: retry once with a generic nudge appended as an
    // additional user turn. Opus occasionally returns empty after
    // many self-play turns when it decides there is nothing meaningful
    // to add; a continuation prompt usually unsticks it.
    const nudgedRequest: MessageParam[] = [
      ...requestMessages,
      {
        role: "assistant",
        content: "(empty)",
      },
      {
        role: "user",
        content:
          "Keep the conversation going -- ask a question, change the subject, or share what's on your mind today. Do not produce an empty response.",
      },
    ];
    const nudgedText = await this.callPersona(initialized, nudgedRequest);

    if (nudgedText.length === 0) {
      throw new Error("Persona LLM produced an empty turn even after a nudge");
    }

    // Commit only the original user message + the recovered assistant
    // response -- the nudge exchange is harness-internal and should
    // not pollute the persona's apparent conversation history.
    return {
      kind: "llm",
      message: nudgedText,
      history: [
        { role: "user", content: baseUserMessage },
        { role: "assistant", content: nudgedText },
      ],
      mockIndex: null,
    };
  }

  commit(draft: PersonaTurnDraft, _borgResponse: string): void {
    if (draft.kind === "mock") {
      if (draft.mockIndex === this.mockIndex) {
        this.mockIndex += 1;
      }
      return;
    }

    if (draft.history === null) {
      return;
    }

    this.messages.push(...draft.history);
    if (this.messages.length === draft.history.length) {
      this.nextSessionGap = null;
    }
  }

  rollback(_draft: PersonaTurnDraft): void {
    // Draft generation is side-effect free; rollback is part of the public
    // lifecycle so runner failures can explicitly discard pending state.
  }

  private async callPersona(
    initialized: PersonaClientInit,
    messages: MessageParam[],
  ): Promise<string> {
    const response = await initialized.client.messages
      .stream({
        model: this.model,
        system: systemParam(initialized.systemPrefix, this.persona),
        messages: sanitizeMessages(messages),
        max_tokens: 4_000,
      })
      .finalMessage();
    return responseText(response);
  }
}
