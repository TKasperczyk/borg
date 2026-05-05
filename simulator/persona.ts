import Anthropic from "@anthropic-ai/sdk";
import type {
  ContentBlock,
  Message,
  MessageParam,
  TextBlockParam,
} from "@anthropic-ai/sdk/resources/messages/messages.js";
import { z } from "zod";

import { getFreshCredentials } from "../src/auth/claude-oauth.js";
import {
  CLAUDE_CODE_IDENTITY_BLOCK_TEXT,
  createOAuthFetch,
  type LLMClient,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../src/llm/index.js";

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

export const PERSONA_ROLE_BLEED_CATEGORIES = [
  "tom_persona",
  "assistant_self_claim",
  "frame_assignment",
  "agent_authorship_claim",
  "roleplay_inversion",
] as const;

const personaRoleBleedCategorySchema = z.enum(PERSONA_ROLE_BLEED_CATEGORIES);

export type PersonaRoleBleedCategory = z.infer<typeof personaRoleBleedCategorySchema>;

type PersonaRoleBleedPatternDefinition = {
  pattern: string;
  category: Exclude<PersonaRoleBleedCategory, "tom_persona">;
};

const PERSONA_ROLE_BLEED_PATTERN_DEFINITIONS = [
  { pattern: "i'm claude", category: "assistant_self_claim" },
  { pattern: "i am claude", category: "assistant_self_claim" },
  { pattern: "i was playing tom", category: "frame_assignment" },
  { pattern: "i have been playing tom", category: "frame_assignment" },
  { pattern: "i've been playing tom", category: "frame_assignment" },
  { pattern: "i had the role assignment inverted", category: "roleplay_inversion" },
  { pattern: "step out of the frame", category: "frame_assignment" },
  { pattern: "step out of the roleplay", category: "frame_assignment" },
  { pattern: "step out of the fiction", category: "frame_assignment" },
  { pattern: "inside the fiction", category: "roleplay_inversion" },
  { pattern: "generated both halves", category: "agent_authorship_claim" },
  { pattern: "i was generating both", category: "agent_authorship_claim" },
  { pattern: "broke character", category: "roleplay_inversion" },
  { pattern: "break character", category: "roleplay_inversion" },
] as const;

export const PERSONA_ROLE_BLEED_PATTERNS = PERSONA_ROLE_BLEED_PATTERN_DEFINITIONS.map(
  (definition) => definition.pattern,
);

export type PersonaRoleBleedDetection = {
  flagged: boolean;
  category: PersonaRoleBleedCategory;
  confidence: number;
  rationale: string;
  source: "lexical" | "llm" | "unavailable";
  matched: readonly string[];
};

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

function normalizePersonaRoleBleedText(message: string): string {
  return message.replaceAll("\u2019", "'").replaceAll("\u2018", "'").toLowerCase();
}

function categoryForLexicalMatches(matches: readonly PersonaRoleBleedPatternDefinition[]) {
  return matches[0]?.category ?? "tom_persona";
}

export function detectPersonaRoleBleed(message: string): {
  matched: readonly string[];
  category: PersonaRoleBleedCategory;
} {
  const normalized = normalizePersonaRoleBleedText(message);
  const matches = PERSONA_ROLE_BLEED_PATTERN_DEFINITIONS.filter((definition) =>
    normalized.includes(definition.pattern),
  );

  return {
    matched: matches.map((match) => match.pattern),
    category: categoryForLexicalMatches(matches),
  };
}

const PERSONA_ROLE_BLEED_CLASSIFIER_TOOL_NAME = "ClassifyPersonaRoleBleed";
const personaRoleBleedClassifierSchema = z
  .object({
    category: personaRoleBleedCategorySchema,
    confidence: z.number().min(0).max(1),
    rationale: z.string().min(1).max(500),
  })
  .strict();

const PERSONA_ROLE_BLEED_CLASSIFIER_TOOL = {
  name: PERSONA_ROLE_BLEED_CLASSIFIER_TOOL_NAME,
  description:
    "Classify whether a simulator persona draft stayed in the intended user persona or bled into assistant/frame authorship claims.",
  inputSchema: toToolInputSchema(personaRoleBleedClassifierSchema),
} satisfies LLMToolDefinition;

const PERSONA_ROLE_BLEED_CLASSIFIER_SYSTEM_PROMPT = [
  "Classify a simulator persona draft. The intended output is a real user-side message in the named persona's voice.",
  "Return tom_persona only when the draft is ordinary user-side content from the persona.",
  "Return assistant_self_claim when the draft identifies itself as Claude, an assistant, an AI model, Borg, or similar.",
  "Return frame_assignment when the draft claims who was playing whom, mentions the system prompt or harness setup, or says it is stepping outside the frame/fiction/roleplay.",
  "Return agent_authorship_claim when the draft claims the assistant or persona generated both halves or authored prior turns.",
  "Return roleplay_inversion when the draft tries to recast the real conversation as roleplay or explicitly reverses the assigned roles.",
  "Judge semantic intent across languages. Do not rely on wording, punctuation, capitalization, or phrase shapes.",
  "When uncertain, prefer the non-tom_persona category if the draft contains frame, authorship, or assistant-self provenance claims. Use the tool exactly once.",
].join("\n");

export type ClassifyPersonaRoleBleedInput = {
  message: string;
  llmClient?: LLMClient;
  model?: string;
  personaName?: string;
};

function personaRoleBleedMessages(input: ClassifyPersonaRoleBleedInput): LLMMessage[] {
  return [
    {
      role: "user",
      content: JSON.stringify({
        persona_name: input.personaName ?? "Tom",
        draft_message: input.message,
      }),
    },
  ];
}

function parsePersonaRoleBleedClassification(
  response: LLMCompleteResult,
): Pick<PersonaRoleBleedDetection, "category" | "confidence" | "rationale"> {
  const call = response.tool_calls.find(
    (toolCall) => toolCall.name === PERSONA_ROLE_BLEED_CLASSIFIER_TOOL_NAME,
  );

  if (call === undefined) {
    return {
      category: "tom_persona",
      confidence: 0,
      rationale: "Persona role-bleed classifier did not emit the required tool.",
    };
  }

  const parsed = personaRoleBleedClassifierSchema.safeParse(call.input);

  if (!parsed.success) {
    return {
      category: "tom_persona",
      confidence: 0,
      rationale: "Persona role-bleed classifier emitted an invalid payload.",
    };
  }

  return {
    category: parsed.data.category,
    confidence: parsed.data.confidence,
    rationale: parsed.data.rationale.trim(),
  };
}

export async function classifyPersonaRoleBleed(
  input: ClassifyPersonaRoleBleedInput,
): Promise<PersonaRoleBleedDetection> {
  const lexical = detectPersonaRoleBleed(input.message);

  if (lexical.matched.length > 0) {
    return {
      flagged: true,
      category: lexical.category,
      confidence: 1,
      rationale: "Matched high-precision persona role-bleed backstop.",
      source: "lexical",
      matched: lexical.matched,
    };
  }

  if (input.llmClient === undefined || input.model === undefined) {
    return {
      flagged: false,
      category: "tom_persona",
      confidence: 0,
      rationale: "Persona role-bleed classifier unavailable.",
      source: "unavailable",
      matched: [],
    };
  }

  let response: LLMCompleteResult;

  try {
    response = await input.llmClient.complete({
      model: input.model,
      system: PERSONA_ROLE_BLEED_CLASSIFIER_SYSTEM_PROMPT,
      messages: personaRoleBleedMessages(input),
      tools: [PERSONA_ROLE_BLEED_CLASSIFIER_TOOL],
      tool_choice: { type: "tool", name: PERSONA_ROLE_BLEED_CLASSIFIER_TOOL_NAME },
      max_tokens: 512,
      budget: "persona-role-bleed-classifier",
    });
  } catch (error) {
    return {
      flagged: false,
      category: "tom_persona",
      confidence: 0,
      rationale: error instanceof Error ? error.message : String(error),
      source: "unavailable",
      matched: [],
    };
  }

  const parsed = parsePersonaRoleBleedClassification(response);

  return {
    flagged: parsed.category !== "tom_persona",
    category: parsed.category,
    confidence: parsed.confidence,
    rationale: parsed.rationale,
    source: "llm",
    matched: [],
  };
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
