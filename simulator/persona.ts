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

type PersonaRoleBleedRetry = {
  retry?: "persona_role_bleed";
};

export type PriorBorgTurn =
  | ({
      kind: "new_session";
      gapContext?: string;
    } & PersonaRoleBleedRetry)
  | ({
      kind: "normal";
      text: string;
    } & PersonaRoleBleedRetry)
  | ({
      kind: "continued_suppression";
      reason: string;
    } & PersonaRoleBleedRetry);

type PersonaClientInit = {
  client: PersonaClient;
  systemPrefix: TextBlockParam[];
};

export const PERSONA_ROLE_BLEED_PATTERNS = [
  "i don't carry memory",
  "as an ai",
  "i should have said",
  "you deserve a straight answer",
  "i've been tom",
  "you've been borg",
  "i had the role assignment inverted",
] as const;

function personaRoleBleedRetryPrompt(persona: Persona): string {
  return `Your previous draft shifted into Borg's role. Discard it. Write only ${persona.displayName}'s next user-side message. Do not answer as Borg, explain Borg, or mention role assignment.`;
}

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

function continuedSuppressionPrompt(persona: Persona): string {
  return [
    "Borg produced no visible response to your last message.",
    `Continue as ${persona.displayName}.`,
    "Do not answer your own previous question.",
    "Do not speak as Borg or describe Borg's hidden behavior.",
    "You may rephrase, move on, react to the silence, or introduce a new user-side thought.",
  ].join(" ");
}

function baseUserMessageForPriorTurn(persona: Persona, priorTurn: PriorBorgTurn): string {
  if (priorTurn.kind === "new_session") {
    return initialPrompt(persona, priorTurn.gapContext ?? null);
  }

  if (priorTurn.kind === "continued_suppression") {
    return continuedSuppressionPrompt(persona);
  }

  const trimmedBorgResponse = priorTurn.text.trim();
  return trimmedBorgResponse.length === 0
    ? continuedSuppressionPrompt(persona)
    : trimmedBorgResponse;
}

function requestUserMessageForPriorTurn(persona: Persona, priorTurn: PriorBorgTurn): string {
  const baseUserMessage = baseUserMessageForPriorTurn(persona, priorTurn);

  if (priorTurn.retry !== "persona_role_bleed") {
    return baseUserMessage;
  }

  return `${baseUserMessage}\n\n${personaRoleBleedRetryPrompt(persona)}`;
}

export function personaRoleBleedPattern(message: string): string | null {
  const normalized = message.toLowerCase();

  for (const pattern of PERSONA_ROLE_BLEED_PATTERNS) {
    if (normalized.includes(pattern)) {
      return pattern;
    }
  }

  return null;
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

  startNewSession(): void {
    this.messages.length = 0;
  }

  async prepareNextTurn(priorBorgTurn: PriorBorgTurn): Promise<PersonaTurnDraft> {
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

    const historyUserMessage = baseUserMessageForPriorTurn(this.persona, priorBorgTurn);
    const requestUserMessage = requestUserMessageForPriorTurn(this.persona, priorBorgTurn);

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
      { role: "user", content: requestUserMessage },
    ];

    const text = await this.callPersona(initialized, requestMessages);

    if (text.length > 0) {
      return {
        kind: "llm",
        message: text,
        history: [
          { role: "user", content: historyUserMessage },
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
        { role: "user", content: historyUserMessage },
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
