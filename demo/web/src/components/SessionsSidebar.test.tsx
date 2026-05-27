import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { SessionRecord } from "../api/types";
import { useSession } from "../hooks/use-session";
import { SessionsSidebar } from "./SessionsSidebar";

function session(
  input: Partial<SessionRecord> & Pick<SessionRecord, "session_id" | "label">,
): SessionRecord {
  return {
    source_type: "demo",
    source_external_id: null,
    source_url: null,
    audience_label: "alice",
    audience_entity_id: null,
    conversation_kind: "demo",
    created_at: 1_000,
    last_activity_at: 1_000,
    last_turn_id: null,
    message_count: 0,
    status: "active",
    privacy_level: "payload_off",
    participation_policy: "active",
    audience_role: "participant",
    ...input,
  };
}

function Harness({ sessions }: { sessions: readonly SessionRecord[] }) {
  const { sessionId, setSessionId } = useSession();
  return (
    <SessionsSidebar sessions={sessions} activeSessionId={sessionId} onSelect={setSessionId} />
  );
}

afterEach(() => {
  vi.restoreAllMocks();
  window.history.replaceState(null, "", "/");
});

describe("SessionsSidebar", () => {
  it("renders sessions and switches the URL session when clicked", () => {
    vi.spyOn(Date, "now").mockReturnValue(61_000);
    window.history.replaceState(null, "", "/");
    const rows = [
      session({ session_id: "default", label: "demo (default)", message_count: 2 }),
      session({
        session_id: "sess_aaaaaaaaaaaaaaaa",
        source_type: "slack",
        conversation_kind: "thread",
        label: "Slack #planning",
        audience_label: "#planning",
      }),
    ];

    render(<Harness sessions={rows} />);

    expect(screen.getByText("demo (default)")).toBeInTheDocument();
    expect(screen.getByText("Slack #planning")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Slack #planning/ }));

    expect(new URL(window.location.href).searchParams.get("session")).toBe("sess_aaaaaaaaaaaaaaaa");
  });

  it("renders the operator chat preset and opens it when clicked", async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve(
        new Response(JSON.stringify(session({ session_id: "sess_operator", label: "operator" })), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    render(
      <SessionsSidebar
        sessions={[session({ session_id: "default", label: "demo (default)" })]}
        activeSessionId="default"
        onSelect={() => undefined}
        onOpenOperatorChat={async () => {
          await fetch("/api/sessions/operator", {
            method: "POST",
          });
        }}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "operator chat" }));

    expect(fetchMock).toHaveBeenCalledWith("/api/sessions/operator", {
      method: "POST",
    });
  });
});
