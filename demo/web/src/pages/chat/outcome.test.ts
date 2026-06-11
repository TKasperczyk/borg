import type { StreamEntry, TurnHistoryRow } from "../../api/types";
import { outcomeFromTerminalAndEntry, type TerminalSummary } from "./outcome";

function terminal(outcome: TerminalSummary["outcome"]): TerminalSummary {
  return { turnId: "t1", outcome };
}

function entry(kind: StreamEntry["kind"], content: unknown = ""): StreamEntry {
  return {
    id: `e_${kind}`,
    timestamp: 1,
    kind,
    content,
    turn_id: "t1",
    sender_entity_id: null,
    reply_target_entity_id: null,
    session_id: "s1",
    sender_label: null,
    session_label: null,
    audience_label: null,
  };
}

function turn(outcome: TurnHistoryRow["outcome"]): TurnHistoryRow {
  return {
    turn_id: "t1",
    started_at: 1,
    audience: null,
    outcome,
    suppression_reason: outcome === "deliberate-silence" ? "finalizer_no_output" : null,
  };
}

describe("outcome resolution", () => {
  it("resolves reflected terminal outcomes from subsequent terminal entry kind", () => {
    expect(outcomeFromTerminalAndEntry(terminal("reflected"), null, false).text).toBe(
      "turn complete",
    );
    expect(outcomeFromTerminalAndEntry(terminal("reflected"), entry("agent_msg", "hi"), false).text).toBe(
      "TURN COMPLETE — answered",
    );
    expect(
      outcomeFromTerminalAndEntry(
        terminal("reflected"),
        entry("agent_suppressed", { reason: "finalizer_no_output" }),
        false,
        turn("deliberate-silence"),
      ).text,
    ).toBe("TURN COMPLETE — deliberate silence");
    expect(
      outcomeFromTerminalAndEntry(
        terminal("reflected"),
        entry("agent_suppressed", { reason: "finalizer_no_output" }),
        false,
      ).text,
    ).toBe("TURN COMPLETE — suppressed");
    expect(
      outcomeFromTerminalAndEntry(
        terminal("reflected"),
        entry("agent_observed", { reason: "not addressed" }),
        false,
      ).text,
    ).toBe("TURN COMPLETE — observing");
  });

  it("renders structural suppression and failure terminal outcomes directly", () => {
    expect(outcomeFromTerminalAndEntry(terminal("suppressed_action"), null, false)).toMatchObject({
      text: "OUTPUT WITHHELD — suppressed_action",
      tone: "red",
    });
    expect(outcomeFromTerminalAndEntry(terminal("error"), null, false)).toMatchObject({
      text: "TURN FAILED — error",
      tone: "red",
    });
  });
});
