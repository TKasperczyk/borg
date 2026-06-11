import type { StreamEntry, TurnHistoryRow, TurnTerminalFrame } from "../../api/types";
import { isDeliberateSilence, type ThreadItem } from "./artifacts";

export type OutcomeTone = "idle" | "running" | "ok" | "dim" | "red";

export type OutcomeDisplay = {
  text: string;
  tone: OutcomeTone;
  pulse: boolean;
};

export type TerminalSummary = {
  turnId: string;
  outcome: TurnTerminalFrame["data"]["outcome"];
};

export function terminalSummaryFromFrame(frame: TurnTerminalFrame): TerminalSummary {
  return {
    turnId: frame.data.turn_id,
    outcome: frame.data.outcome,
  };
}

function isStreamEntry(entry: StreamEntry | ThreadItem): entry is StreamEntry {
  return "kind" in entry;
}

function isThreadItem(entry: StreamEntry | ThreadItem): entry is ThreadItem {
  return "type" in entry;
}

export function outcomeFromTerminalAndEntry(
  terminal: TerminalSummary | null,
  entry: StreamEntry | ThreadItem | null,
  running: boolean,
  turn: TurnHistoryRow | undefined = undefined,
): OutcomeDisplay {
  if (running) {
    return { text: "turn in flight…", tone: "running", pulse: true };
  }

  if (terminal === null) {
    return { text: "awaiting turn — core at rest", tone: "idle", pulse: false };
  }

  if (terminal.outcome === "reflected") {
    if (entry === null) {
      return { text: "turn complete", tone: "dim", pulse: false };
    }
    if (
      (isStreamEntry(entry) && entry.kind === "agent_msg") ||
      (isThreadItem(entry) && entry.type === "agent")
    ) {
      return { text: "TURN COMPLETE — answered", tone: "ok", pulse: false };
    }
    if (
      (isStreamEntry(entry) && entry.kind === "agent_observed") ||
      (isThreadItem(entry) && entry.type === "observed")
    ) {
      return { text: "TURN COMPLETE — observing", tone: "dim", pulse: false };
    }
    if (isStreamEntry(entry) && entry.kind === "agent_suppressed") {
      return isDeliberateSilence(entry, turn)
        ? { text: "TURN COMPLETE — deliberate silence", tone: "dim", pulse: false }
        : { text: "TURN COMPLETE — suppressed", tone: "red", pulse: false };
    }
    if (isThreadItem(entry) && entry.type === "silence") {
      return { text: "TURN COMPLETE — deliberate silence", tone: "dim", pulse: false };
    }
    if (isThreadItem(entry) && entry.type === "suppressed") {
      return { text: "TURN COMPLETE — suppressed", tone: "red", pulse: false };
    }

    return { text: "turn complete", tone: "dim", pulse: false };
  }

  if (
    terminal.outcome === "suppressed_closure" ||
    terminal.outcome === "suppressed_generation_gate" ||
    terminal.outcome === "suppressed_action"
  ) {
    return {
      text: `OUTPUT WITHHELD — ${terminal.outcome}`,
      tone: "red",
      pulse: false,
    };
  }

  return {
    text: `TURN FAILED — ${terminal.outcome}`,
    tone: "red",
    pulse: false,
  };
}
