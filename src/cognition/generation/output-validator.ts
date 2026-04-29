import type { GenerationSuppressionReason } from "./types.js";

const ROLE_LABEL_LINE_PATTERN = /^\s*(human|assistant|user|ai|borg)\s*:/iu;
const FENCE_PATTERN = /^\s*(```+|~~~+)/u;
const NON_GENERATION_TEXT = new Set([
  "(no response)",
  "[no response]",
  "no response",
  "(silence)",
  "[silence]",
  "(stopping.)",
  "(stopping)",
  "[stopping]",
  "stopping.",
  "...",
  ".",
]);

export type OutputValidationFailure = {
  reason: GenerationSuppressionReason;
  kind: "empty" | "non_generation_text" | "role_label";
  message: string;
  line?: number;
  label?: string;
};

export type OutputValidationResult =
  | {
      ok: true;
    }
  | {
      ok: false;
      failure: OutputValidationFailure;
    };

function isFenceLine(line: string): boolean {
  return FENCE_PATTERN.test(line);
}

function isBlockQuoteLine(line: string): boolean {
  return /^\s*>/u.test(line);
}

function isIndentedCodeLine(line: string): boolean {
  return /^(?: {4,}|\t)/u.test(line);
}

export function validateAssistantOutput(text: string): OutputValidationResult {
  const trimmed = text.trim();

  if (trimmed.length === 0) {
    return {
      ok: false,
      failure: {
        reason: "empty_finalizer",
        kind: "empty",
        message: "The assistant draft was empty.",
      },
    };
  }

  if (NON_GENERATION_TEXT.has(trimmed.toLowerCase())) {
    return {
      ok: false,
      failure: {
        reason: "invalid_non_generation_text",
        kind: "non_generation_text",
        message: "The assistant draft narrated non-generation instead of emitting no message.",
      },
    };
  }

  let inFence = false;
  const lines = text.split(/\r?\n/u);

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index] ?? "";

    if (isFenceLine(line)) {
      inFence = !inFence;
      continue;
    }

    if (inFence || isBlockQuoteLine(line) || isIndentedCodeLine(line)) {
      continue;
    }

    const match = ROLE_LABEL_LINE_PATTERN.exec(line);

    if (match !== null) {
      return {
        ok: false,
        failure: {
          reason: "output_validator",
          kind: "role_label",
          message: "The assistant draft contained an unquoted role label at line start.",
          line: index + 1,
          label: match[1] ?? "role",
        },
      };
    }
  }

  return { ok: true };
}

export function renderOutputValidatorRetrySection(failure: OutputValidationFailure): string {
  const location = failure.line === undefined ? "" : ` at line ${failure.line}`;

  return [
    "<borg_output_validator_feedback>",
    "The previous assistant draft was invalid and was not emitted.",
    `Violation${location}: ${failure.message}`,
    "Write a normal assistant response. Role labels such as Human: or Assistant: are never response content at line start.",
    "If you need to discuss a transcript or code, quote it or put it in a fenced code block.",
    "If no message should be emitted, the system will suppress output; do not narrate silence.",
    "</borg_output_validator_feedback>",
  ].join("\n");
}
