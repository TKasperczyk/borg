import { z } from "zod";

import { findInErrorCauseChain } from "./errors.js";

export type ZodErrorMessageOptions = {
  maxIssues?: number;
  maxCharacters?: number;
};

export function isZodError(error: unknown): error is z.ZodError {
  return error instanceof z.ZodError;
}

function truncateIssue(message: string, maxCharacters: number): string {
  if (message.length <= maxCharacters) {
    return message;
  }

  if (maxCharacters <= 1) {
    return maxCharacters === 1 ? "…" : "";
  }

  return `${message.slice(0, maxCharacters - 1)}…`;
}

function omittedIssuesSuffix(count: number): string {
  return count > 0 ? `; (+${count} more issues)` : "";
}

export function formatZodErrorIssues(
  error: unknown,
  options: ZodErrorMessageOptions = {},
): string | undefined {
  const zodError = findInErrorCauseChain(error, isZodError);

  if (zodError === undefined) {
    return undefined;
  }

  const issueMessages = zodError.issues.map(
    (issue) => `${issue.path.map(String).join(".") || "(root)"}: ${issue.message}`,
  );
  const maxIssues = Math.max(0, options.maxIssues ?? issueMessages.length);
  const maxCharacters =
    options.maxCharacters === undefined ? undefined : Math.max(0, options.maxCharacters);

  if (maxCharacters === undefined) {
    const shownIssues = issueMessages.slice(0, maxIssues);
    const remainingIssues = issueMessages.length - shownIssues.length;
    const message = shownIssues.join("; ");

    return remainingIssues > 0 ? `${message}${omittedIssuesSuffix(remainingIssues)}` : message;
  }

  const shownIssues: string[] = [];

  for (const issue of issueMessages.slice(0, maxIssues)) {
    const nextIssues = [...shownIssues, issue];
    const remainingIssues = issueMessages.length - nextIssues.length;
    const rendered = `${nextIssues.join("; ")}${omittedIssuesSuffix(remainingIssues)}`;

    if (rendered.length > maxCharacters) {
      break;
    }

    shownIssues.push(issue);
  }

  if (shownIssues.length > 0) {
    return `${shownIssues.join("; ")}${omittedIssuesSuffix(
      issueMessages.length - shownIssues.length,
    )}`;
  }

  const firstIssue = issueMessages[0];

  if (firstIssue === undefined || maxIssues === 0) {
    return truncateIssue(`(+${issueMessages.length} issues)`, maxCharacters);
  }

  const suffix = omittedIssuesSuffix(issueMessages.length - 1);

  if (suffix.length >= maxCharacters) {
    return truncateIssue(firstIssue, maxCharacters);
  }

  return `${truncateIssue(firstIssue, Math.max(0, maxCharacters - suffix.length))}${suffix}`;
}

export function parseErrorMessage(error: unknown, options: ZodErrorMessageOptions = {}): string {
  const zodIssues = formatZodErrorIssues(error, options);

  if (zodIssues !== undefined) {
    return zodIssues;
  }

  return error instanceof Error ? error.message : String(error);
}
