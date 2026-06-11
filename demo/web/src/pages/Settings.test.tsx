import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import type {
  ApiState,
  DreamStateResponse,
  EntityRecord,
  PromptBlock,
  SessionRecord,
} from "../api/types";
import { StateProvider } from "../state/app-state";
import { SettingsPage } from "./Settings";

const now = Date.UTC(2026, 5, 11, 12);

const blocks: PromptBlock[] = [
  {
    key: "base_identity_preamble",
    label: "Base identity preamble",
    description: "Opening frame.",
    default_text: "default one",
    current_text: "current one",
    current_text_kind: "stored_override",
    overridden: true,
    updated_at: now,
  },
  {
    key: "self_architecture",
    label: "Self architecture",
    description: "Loop frame.",
    default_text: "second text",
    current_text: "second text",
    current_text_kind: "static_default",
    overridden: false,
    updated_at: null,
  },
];

const state: ApiState = {
  active_session: "sess_1",
  audiences: ["operator"],
  counts: {
    turns: 0,
    commitments: 0,
    open_qs: 0,
    open_reviews: 0,
    dream_audit_rows: 0,
  },
  current_mood: null,
  runtime: {
    model: "claude-opus",
    embedding: {
      model: "qwen",
      dims: 4096,
    },
  },
  version: "test",
};

const dream: DreamStateResponse = {
  processes: [],
  pending_extraction_episodes: 0,
  schedule: [],
  dream_reports: [],
  audit_rows: [],
  belief_revision_rows: [],
  scheduler: {
    enabled: true,
    light_interval_ms: 60_000,
    heavy_interval_ms: 3_600_000,
    optimize_storage: false,
    light_processes: ["consolidator"],
    heavy_processes: ["reflector", "semantic-extractor"],
    process_budgets: {},
  },
};

const creator: EntityRecord = {
  id: "ent_1",
  canonical_name: "Tom",
  aliases: [],
  kind: "person",
  borg_role: "creator",
  created_at: now,
};

const session: SessionRecord = {
  session_id: "sess_1",
  source_type: "demo",
  source_external_id: null,
  source_url: null,
  label: "Session One",
  audience_label: "operator",
  audience_entity_id: null,
  conversation_kind: "demo",
  created_at: now,
  last_activity_at: now,
  last_turn_id: null,
  message_count: 0,
  status: "active",
  privacy_level: "payload_on",
  participation_policy: "active",
  audience_role: "operator",
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function renderSettings() {
  const requests: Array<{ url: string; method: string; body: unknown }> = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    requests.push({
      url,
      method,
      body: init?.body === undefined ? null : JSON.parse(String(init.body)),
    });

    if (url === "/api/state") {
      return json(state);
    }
    if (url === "/api/prompts") {
      return json({ blocks });
    }
    if (url === "/api/prompts/assembled") {
      return json({
        text: "Alpha framing\n\nBeta framing",
        sections: ["alpha", "beta"],
        segments: [
          { id: "alpha", label: "alpha", editable_key: null, start: 0, end: 13 },
          { id: "beta", label: "beta", editable_key: null, start: 15, end: 27 },
        ],
      });
    }
    if (url === "/api/dream/state") {
      return json(dream);
    }
    if (url === "/api/entities/creator") {
      return json(creator);
    }
    if (url === "/api/sessions") {
      return json({ sessions: [session] });
    }
    if (url === "/api/prompts/base_identity_preamble" && method === "PUT") {
      return json({ ...blocks[0], current_text: "new text", overridden: true });
    }
    if (url === "/api/prompts/base_identity_preamble" && method === "DELETE") {
      return json({ ...blocks[0], current_text: "default one", overridden: false });
    }
    if (url === "/api/sessions/sess_1/participation" && method === "POST") {
      return json({ ...session, participation_policy: "paused" });
    }
    if (url === "/api/admin/reset" && method === "POST") {
      return json({ ok: true });
    }
    if (url === "/api/entities/creator" && method === "POST") {
      return json({ ...creator, canonical_name: "Dana" });
    }

    return json({ message: "not found" }, 404);
  });

  render(
    <StateProvider>
      <SettingsPage />
    </StateProvider>,
  );
  return { requests };
}

describe("Settings page", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("saves and resets prompt overrides, with over-limit save disabled", async () => {
    const { requests } = renderSettings();
    const textarea = (await screen.findByDisplayValue("current one")) as HTMLTextAreaElement;

    fireEvent.change(textarea, { target: { value: "   \n  " } });
    expect(screen.getByText("0 trimmed / 50,000")).toBeTruthy();
    expect((screen.getByRole("button", { name: "SAVE OVERRIDE" }) as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(textarea, { target: { value: ` ${"x".repeat(50_000)} ` } });
    expect(screen.getByText("50,000 trimmed / 50,000")).toBeTruthy();
    expect((screen.getByRole("button", { name: "SAVE OVERRIDE" }) as HTMLButtonElement).disabled).toBe(false);

    fireEvent.change(textarea, { target: { value: "x".repeat(50_001) } });
    expect((screen.getByRole("button", { name: "SAVE OVERRIDE" }) as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(textarea, { target: { value: "new text" } });
    fireEvent.click(screen.getByRole("button", { name: "SAVE OVERRIDE" }));

    await waitFor(() =>
      expect(requests.some((request) => request.url === "/api/prompts/base_identity_preamble" && request.method === "PUT")).toBe(true),
    );
    expect(requests.find((request) => request.method === "PUT")?.body).toEqual({ text: "new text" });

    fireEvent.click(screen.getByRole("button", { name: "RESET TO DEFAULT" }));
    fireEvent.click(screen.getByRole("button", { name: "CONFIRM RESET" }));

    await waitFor(() =>
      expect(requests.some((request) => request.url === "/api/prompts/base_identity_preamble" && request.method === "DELETE")).toBe(true),
    );
  });

  it("guards dirty drafts before switching prompt selection", async () => {
    renderSettings();
    const textarea = (await screen.findByDisplayValue("current one")) as HTMLTextAreaElement;

    fireEvent.change(textarea, { target: { value: "dirty draft" } });
    fireEvent.click(screen.getByText("self_architecture"));

    expect(screen.getByText("discard unsaved prompt changes?")).toBeTruthy();
    expect(textarea.value).toBe("dirty draft");

    fireEvent.click(screen.getByRole("button", { name: "DISCARD DRAFT" }));

    await waitFor(() => expect((screen.getByDisplayValue("second text") as HTMLTextAreaElement).value).toBe("second text"));
  });

  it("routes dirty reset-to-default through discard-draft confirmation", async () => {
    const { requests } = renderSettings();
    const textarea = (await screen.findByDisplayValue("current one")) as HTMLTextAreaElement;

    fireEvent.change(textarea, { target: { value: "dirty override draft" } });
    fireEvent.click(screen.getByRole("button", { name: "RESET TO DEFAULT" }));

    expect(screen.getByText("reset to default discards the unsaved draft first")).toBeTruthy();
    expect(requests.some((request) => request.method === "DELETE")).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "DISCARD DRAFT" }));
    expect(screen.getByRole("button", { name: "CONFIRM RESET" })).toBeTruthy();
    expect(requests.some((request) => request.method === "DELETE")).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "CONFIRM RESET" }));
    await waitFor(() =>
      expect(requests.some((request) => request.url === "/api/prompts/base_identity_preamble" && request.method === "DELETE")).toBe(true),
    );
  });

  it("renders assembled framing preview from real segment offsets", async () => {
    renderSettings();

    await screen.findByDisplayValue("current one");
    fireEvent.click(screen.getByText("▸ PREVIEW ASSEMBLED FRAMING"));

    expect(await screen.findByText("exactly what deliberation sees")).toBeTruthy();
    expect(screen.getByText("Alpha framing")).toBeTruthy();
    expect(screen.getByText("Beta framing")).toBeTruthy();
  });

  it("posts session participation policy and reason", async () => {
    const { requests } = renderSettings();

    expect(await screen.findByText("Session One")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "active" }));
    fireEvent.change(screen.getByPlaceholderText("reason"), { target: { value: "quiet" } });
    fireEvent.click(screen.getByRole("button", { name: "APPLY" }));

    await waitFor(() =>
      expect(requests.some((request) => request.url === "/api/sessions/sess_1/participation")).toBe(true),
    );
    expect(requests.find((request) => request.url === "/api/sessions/sess_1/participation")?.body).toEqual({
      policy: "paused",
      reason: "quiet",
    });
  });

  it("renders present runtime fields and read-only scheduler state", async () => {
    renderSettings();

    expect(await screen.findByText("claude-opus")).toBeTruthy();
    expect(screen.getByText("qwen · 4096 dims")).toBeTruthy();
    expect(screen.getByText("read-only")).toBeTruthy();
    expect(screen.getByText("light · 1m")).toBeTruthy();
    expect(screen.queryByText(/data dir/i)).toBeNull();
  });

  it("arms reset only with the exact token and posts confirm body", async () => {
    const { requests } = renderSettings();
    const button = await screen.findByRole("button", { name: "RESET" });

    expect((button as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByPlaceholderText("type RESET"), { target: { value: "reset" } });
    expect((button as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByPlaceholderText("type RESET"), { target: { value: "RESET" } });
    expect((button as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(button);

    await waitFor(() =>
      expect(requests.some((request) => request.url === "/api/admin/reset" && request.method === "POST")).toBe(true),
    );
    expect(requests.find((request) => request.url === "/api/admin/reset")?.body).toEqual({
      confirm: "RESET",
    });
  });
});
