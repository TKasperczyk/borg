import { fireEvent, render, screen } from "@testing-library/react";

import type { StreamEntry, TurnHistoryRow } from "../../api/types";
import { ThreadArtifactList, threadItemsFromEntries } from "./artifacts";

function entry(input: Partial<StreamEntry> & Pick<StreamEntry, "id" | "kind" | "timestamp" | "content">): StreamEntry {
  return {
    sender_entity_id: null,
    reply_target_entity_id: null,
    session_id: "s1",
    sender_label: null,
    session_label: null,
    audience_label: null,
    ...input,
  };
}

const turns: TurnHistoryRow[] = [
  {
    turn_id: "t_silence",
    started_at: Date.UTC(2026, 5, 11, 10),
    audience: null,
    outcome: "deliberate-silence",
    suppression_reason: "finalizer_no_output",
  },
  {
    turn_id: "t_guard",
    started_at: Date.UTC(2026, 5, 12, 10),
    audience: null,
    outcome: "guard-blocked",
    suppression_reason: "commitment_violation_after_regenerate",
  },
];

describe("chat artifact rendering", () => {
  it("renders known artifacts, day separators, and display_content preference", () => {
    const items = threadItemsFromEntries(
      [
        entry({
          id: "u1",
          kind: "user_msg",
          timestamp: Date.UTC(2026, 5, 11, 10, 0),
          content: "raw wrapper",
          display_content: "shown body",
          sender_label: "operator",
        }),
        entry({
          id: "a1",
          kind: "agent_msg",
          timestamp: Date.UTC(2026, 5, 11, 10, 1),
          content: "answer",
          turn_id: "t_answer",
        }),
        entry({
          id: "s1",
          kind: "agent_suppressed",
          timestamp: Date.UTC(2026, 5, 11, 10, 2),
          content: { reason: "finalizer_no_output" },
          turn_id: "t_silence",
        }),
        entry({
          id: "g1",
          kind: "agent_suppressed",
          timestamp: Date.UTC(2026, 5, 12, 10, 3),
          content: { reason: "commitment_violation_after_regenerate" },
          turn_id: "t_guard",
        }),
        entry({
          id: "n1",
          kind: "agent_suppressed",
          timestamp: Date.UTC(2026, 5, 12, 10, 3, 30),
          content: { reason: "finalizer_no_output" },
          turn_id: "t_unjoined",
        }),
        entry({
          id: "o1",
          kind: "agent_observed",
          timestamp: Date.UTC(2026, 5, 12, 10, 4),
          content: { reason: "not addressed" },
          turn_id: "t_observed",
        }),
        entry({
          id: "d1",
          kind: "dream_report",
          timestamp: Date.UTC(2026, 5, 12, 10, 5),
          content: { process: "consolidator" },
        }),
        entry({
          id: "x1",
          kind: "thought",
          timestamp: Date.UTC(2026, 5, 12, 10, 6),
          content: "private",
        }),
      ],
      turns,
      { t_guard: "withheld text" },
    );

    render(<ThreadArtifactList items={items} />);

    expect(screen.getByText("— JUN 11 —")).toBeTruthy();
    expect(screen.getByText("— JUN 12 —")).toBeTruthy();
    expect(screen.getByText("shown body")).toBeTruthy();
    expect(screen.queryByText("raw wrapper")).toBeNull();
    expect(screen.getByText("NO OUTPUT — DELIBERATE SILENCE")).toBeTruthy();
    expect(screen.getByText("SUPPRESSED — guard-blocked")).toBeTruthy();
    expect(screen.getByText("SUPPRESSED")).toBeTruthy();
    expect(screen.queryByText("SUPPRESSED — finalizer_no_output")).toBeNull();
    expect(screen.queryByText("finalizer_no_output")).toBeNull();
    expect(screen.getByText("withheld text")).toBeTruthy();
    expect(screen.getByText(/◎ OBSERVED/)).toBeTruthy();
    expect(screen.getByText(/dream report/)).toBeTruthy();
    expect(screen.queryByText("private")).toBeNull();
  });

  it("expands deliberate silence rows with sibling plan rationale and structured reasons", () => {
    const items = threadItemsFromEntries(
      [
        entry({
          id: "p1",
          kind: "thought",
          timestamp: Date.UTC(2026, 5, 11, 10, 1),
          content:
            "plan: uncertainty: The current turn is empty -- an autonomous executive_focus_due wake with nothing addressed to me ; verify: wait ; emission: no_output",
          turn_id: "t_silence",
        }),
        entry({
          id: "s1",
          kind: "agent_suppressed",
          timestamp: Date.UTC(2026, 5, 11, 10, 2),
          content: {
            reason: "finalizer_no_output",
            primary_no_output_reason: "low_value_echo",
            no_output_categories: ["with_open_question"],
          },
          turn_id: "t_silence",
        }),
      ],
      turns,
    );

    render(<ThreadArtifactList items={items} />);

    const toggle = screen.getByRole("button", { name: /NO OUTPUT — DELIBERATE SILENCE/ });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByText(/The current turn is empty/)).toBeNull();

    fireEvent.click(toggle);

    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getAllByText(/The current turn is empty/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/thought: plan: uncertainty:/)).toBeTruthy();
    expect(screen.getByText("finalizer_no_output")).toBeTruthy();
    expect(screen.getByText("low-value echo")).toBeTruthy();
    expect(screen.getByText("open question pending")).toBeTruthy();
  });

  it("keeps semicolons inside the extracted uncertainty rationale", () => {
    const items = threadItemsFromEntries(
      [
        entry({
          id: "p1",
          kind: "thought",
          timestamp: Date.UTC(2026, 5, 11, 10, 1),
          content: "plan: uncertainty: first clause; still the why ; emission: no_output",
          turn_id: "t_silence",
        }),
        entry({
          id: "s1",
          kind: "agent_suppressed",
          timestamp: Date.UTC(2026, 5, 11, 10, 2),
          content: { reason: "finalizer_no_output" },
          turn_id: "t_silence",
        }),
      ],
      turns,
    );

    render(<ThreadArtifactList items={items} />);

    fireEvent.click(screen.getByRole("button", { name: /NO OUTPUT — DELIBERATE SILENCE/ }));

    expect(screen.getByText("first clause; still the why")).toBeTruthy();
  });

  it("does not attach an unrelated plan thought from a different turn", () => {
    const items = threadItemsFromEntries(
      [
        entry({
          id: "p1",
          kind: "thought",
          timestamp: Date.UTC(2026, 5, 11, 10, 1),
          content: "plan: uncertainty: wrong turn why ; emission: no_output",
          turn_id: "t_other",
        }),
        entry({
          id: "s1",
          kind: "agent_suppressed",
          timestamp: Date.UTC(2026, 5, 11, 10, 2),
          content: { reason: "finalizer_no_output" },
          turn_id: "t_silence",
        }),
      ],
      turns,
    );

    render(<ThreadArtifactList items={items} />);

    fireEvent.click(screen.getByRole("button", { name: /NO OUTPUT — DELIBERATE SILENCE/ }));

    expect(screen.queryByText(/wrong turn why/)).toBeNull();
    expect(screen.queryByText("WHY")).toBeNull();
  });

  it("renders deliberate silence details without a sibling plan thought", () => {
    const items = threadItemsFromEntries(
      [
        entry({
          id: "s1",
          kind: "agent_suppressed",
          timestamp: Date.UTC(2026, 5, 11, 10, 2),
          content: {
            reason: "finalizer_no_output",
            primary_no_output_reason: "closure",
            no_output_categories: ["custom_category"],
          },
          turn_id: "t_silence",
        }),
      ],
      turns,
    );

    render(<ThreadArtifactList items={items} />);

    const toggle = screen.getByRole("button", { name: /NO OUTPUT — DELIBERATE SILENCE/ });
    fireEvent.click(toggle);

    expect(screen.getByText("finalizer_no_output")).toBeTruthy();
    expect(screen.getByText("closure")).toBeTruthy();
    expect(screen.getByText("custom category")).toBeTruthy();
    expect(screen.queryByText("WHY")).toBeNull();
  });
});
