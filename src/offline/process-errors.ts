import { findInErrorCauseChain } from "../util/errors.js";
import { formatZodErrorIssues, isZodError } from "../util/zod-errors.js";

import type { OfflineProcessError, OfflineProcessName } from "./types.js";

export const MAX_OFFLINE_ERROR_DETAILS = 3;
export const MAX_OFFLINE_ERROR_MESSAGE_LENGTH = 300;

const MAX_CAUSE_ERROR_DETAIL_LENGTH = 220;

type OfflineProcessErrorOptions = {
  code?: string;
  includeErrorCode?: boolean;
  target_type?: OfflineProcessError["target_type"];
  target_id?: string;
};

function errorCode(error: unknown): string | undefined {
  return error instanceof Error && "code" in error ? String(error.code) : undefined;
}

function truncateMessage(message: string, maxCharacters: number): string {
  if (message.length <= maxCharacters) {
    return message;
  }

  if (maxCharacters <= 1) {
    return maxCharacters === 1 ? "…" : "";
  }

  return `${message.slice(0, maxCharacters - 1)}…`;
}

function safeScalarDetail(value: unknown): string | undefined {
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean"
    ? String(value)
    : undefined;
}

function redactedBodyShape(body: unknown): string {
  return `[response body omitted: ${Array.isArray(body) ? "array" : body === null ? "null" : typeof body}]`;
}

function httpBodyDetail(body: unknown): string {
  if (body !== null && typeof body === "object" && !Array.isArray(body)) {
    const record = body as { message?: unknown; detail?: unknown; error?: unknown };
    const message = safeScalarDetail(record.message);

    if (message !== undefined) {
      return truncateMessage(message, MAX_CAUSE_ERROR_DETAIL_LENGTH);
    }

    if (record.error !== null && typeof record.error === "object" && !Array.isArray(record.error)) {
      const nested = record.error as { message?: unknown; detail?: unknown };
      const nestedMessage = safeScalarDetail(nested.message);

      if (nestedMessage !== undefined) {
        return truncateMessage(nestedMessage, MAX_CAUSE_ERROR_DETAIL_LENGTH);
      }

      const nestedDetail = safeScalarDetail(nested.detail);

      if (nestedDetail !== undefined) {
        return truncateMessage(nestedDetail, MAX_CAUSE_ERROR_DETAIL_LENGTH);
      }
    }

    const detail = safeScalarDetail(record.detail);

    if (detail !== undefined) {
      return truncateMessage(detail, MAX_CAUSE_ERROR_DETAIL_LENGTH);
    }
  }

  return redactedBodyShape(body);
}

type HttpErrorCause = {
  status: number | string;
  error?: unknown;
  body?: unknown;
};

function isHttpErrorCause(error: unknown): error is HttpErrorCause {
  if (error === null || (typeof error !== "object" && typeof error !== "function")) {
    return false;
  }

  const status = (error as { status?: unknown }).status;
  return typeof status === "number" || typeof status === "string";
}

function httpErrorDetail(error: unknown): string | undefined {
  const httpError = findInErrorCauseChain(error, isHttpErrorCause);

  if (httpError === undefined) {
    return undefined;
  }

  const body = httpError.error ?? httpError.body;
  const detail =
    body === undefined
      ? httpError instanceof Error
        ? httpError.message
        : "request failed"
      : httpBodyDetail(body);
  const status = truncateMessage(String(httpError.status), 24);
  const prefix = `HTTP ${status}: `;

  return `${prefix}${truncateMessage(detail, MAX_CAUSE_ERROR_DETAIL_LENGTH - prefix.length)}`;
}

export function formatOfflineProcessErrorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  const validationDetail = formatZodErrorIssues(error, {
    maxIssues: MAX_OFFLINE_ERROR_DETAILS,
    maxCharacters: MAX_CAUSE_ERROR_DETAIL_LENGTH,
  });
  const causeDetail = validationDetail ?? httpErrorDetail(error);

  if (causeDetail === undefined) {
    return truncateMessage(message, MAX_OFFLINE_ERROR_MESSAGE_LENGTH);
  }

  if (isZodError(error) || message === causeDetail) {
    return truncateMessage(causeDetail, MAX_OFFLINE_ERROR_MESSAGE_LENGTH);
  }

  const messageCharacters = MAX_OFFLINE_ERROR_MESSAGE_LENGTH - causeDetail.length - 2;
  return `${truncateMessage(message, messageCharacters)}: ${causeDetail}`;
}

export function offlineProcessError<ProcessName extends OfflineProcessName>(
  process: ProcessName,
  error: unknown,
  options: OfflineProcessErrorOptions = {},
): OfflineProcessError & { process: ProcessName } {
  const resolvedCode =
    options.code ?? (options.includeErrorCode === false ? undefined : errorCode(error));

  return {
    process,
    message: formatOfflineProcessErrorMessage(error),
    ...(resolvedCode === undefined ? {} : { code: resolvedCode }),
    ...(options.target_type === undefined ? {} : { target_type: options.target_type }),
    ...(options.target_id === undefined ? {} : { target_id: options.target_id }),
  };
}
