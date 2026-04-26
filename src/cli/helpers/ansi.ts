import type { Output } from "../types.js";

type AnsiOutput = Output & {
  isTTY?: boolean;
};

export function createAnsi(output: AnsiOutput) {
  const colorEnabled = output.isTTY === true;
  const style = (text: string, code: string): string =>
    colorEnabled ? `\u001b[${code}m${text}\u001b[0m` : text;

  return {
    colorEnabled,
    dim: (text: string) => style(text, "2"),
    green: (text: string) => style(text, "32"),
    yellow: (text: string) => style(text, "33"),
    red: (text: string) => style(text, "31"),
    strong: (text: string) => style(text, "1"),
    accent: (text: string) => style(text, "1;36"),
  };
}
