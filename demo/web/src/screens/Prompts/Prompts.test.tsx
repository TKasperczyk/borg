import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Inspector } from "../../components/Inspector/Inspector";
import { InspectorProvider } from "../../components/Inspector/inspector-context";
import { PromptsScreen } from "./index";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function defaultPrompts() {
  return {
    blocks: [
      {
        key: "base_identity_preamble",
        label: "Base identity preamble",
        description: "Opening framing block.",
        default_text: "DEFAULT PREAMBLE",
        current_text: "DEFAULT PREAMBLE",
        current_text_kind: "static_default",
        overridden: false,
        updated_at: null,
      },
      {
        key: "voice_and_posture",
        label: "Voice and posture",
        description: "Speaking style.",
        default_text: "DEFAULT VOICE",
        current_text: "DEFAULT VOICE",
        current_text_kind: "static_default",
        overridden: false,
        updated_at: null,
      },
      {
        key: "epistemic_posture",
        label: "Epistemic posture",
        description: "Evidence posture.",
        default_text: "DEFAULT EPISTEMIC",
        current_text: "DEFAULT EPISTEMIC",
        current_text_kind: "static_default",
        overridden: false,
        updated_at: null,
      },
      {
        key: "identity_posture",
        label: "Identity posture",
        description: "Identity stance.",
        default_text: "DEFAULT IDENTITY",
        current_text: "DEFAULT IDENTITY",
        current_text_kind: "static_default",
        overridden: false,
        updated_at: null,
      },
      {
        key: "host_capabilities",
        label: "Host capabilities",
        description: "Runtime capabilities.",
        default_text: "DEFAULT HOST",
        current_text: "DEFAULT HOST",
        current_text_kind: "static_default",
        overridden: false,
        updated_at: null,
      },
    ],
  };
}

function assembledPrompt() {
  return {
    text: [
      "DEFAULT PREAMBLE",
      "DEFAULT VOICE",
      "The following tagged blocks mix substrate-owned guidance with memory-derived self-model records.",
      "<borg_host_capabilities>",
      "HOST CAPABILITIES",
      "</borg_host_capabilities>",
    ].join("\n\n"),
    sections: [
      "base_identity_preamble",
      "voice_and_posture",
      "trusted_guidance_preamble",
      "borg_host_capabilities",
    ],
  };
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

function requestMethod(init: RequestInit | undefined): string {
  return init?.method ?? "GET";
}

function requestCount(
  fetchMock: ReturnType<typeof vi.fn>,
  path: string,
  method: string = "GET",
): number {
  return fetchMock.mock.calls.filter(
    (call) =>
      requestPath(call[0] as RequestInfo | URL) === path &&
      requestMethod(call[1] as RequestInit | undefined) === method,
  ).length;
}

function renderWithInspector(children: ReactNode) {
  return render(
    <InspectorProvider
      setView={vi.fn()}
      setSessionId={vi.fn()}
      sessionId="default"
      audience="operator"
    >
      {children}
      <Inspector />
    </InspectorProvider>,
  );
}

function renderPromptsScreen() {
  return renderWithInspector(<PromptsScreen />);
}

function mockClipboard() {
  const writeText = vi.fn<Clipboard["writeText"]>().mockResolvedValue(undefined);
  Object.defineProperty(window.navigator, "clipboard", {
    configurable: true,
    value: { writeText },
  });
  return writeText;
}

afterEach(() => {
  Reflect.deleteProperty(window.navigator, "clipboard");
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("PromptsScreen", () => {
  it("loads blocks, disables save, and hides reset when nothing is overridden", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(defaultPrompts()));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();

    await screen.findByRole("heading", { name: "Base identity preamble" });

    expect(screen.getAllByTestId("prompt-block-row")).toHaveLength(5);
    expect(screen.getByRole("button", { name: "save" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "reset to default" })).not.toBeInTheDocument();
  });

  it("saves an override via PUT and refetches", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(defaultPrompts()));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      if (path === "/api/prompts/voice_and_posture" && init?.method === "PUT") {
        return Promise.resolve(
          jsonResponse({
            key: "voice_and_posture",
            label: "Voice and posture",
            description: "Speaking style.",
            default_text: "DEFAULT VOICE",
            current_text: "CUSTOM VOICE",
            current_text_kind: "stored_override",
            overridden: true,
            updated_at: 123,
          }),
        );
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    await screen.findByRole("button", { name: /Voice and posture/ });

    fireEvent.click(screen.getByRole("button", { name: /Voice and posture/ }));
    const voiceTextarea = screen.getByLabelText("edited override") as HTMLTextAreaElement;
    fireEvent.change(voiceTextarea, { target: { value: "CUSTOM VOICE" } });

    const saveButton = screen.getByRole("button", { name: "save" });
    expect(saveButton).not.toBeDisabled();
    fireEvent.click(saveButton);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    await waitFor(() => {
      const putCall = fetchMock.mock.calls.find(
        (call) =>
          requestPath(call[0] as RequestInfo | URL) === "/api/prompts/voice_and_posture" &&
          (call[1] as RequestInit | undefined)?.method === "PUT",
      );
      expect(putCall).toBeDefined();
      const init = putCall![1] as RequestInit;
      expect(JSON.parse(String(init.body))).toEqual({ text: "CUSTOM VOICE" });
    });
    await waitFor(() => {
      expect(requestCount(fetchMock, "/api/prompts")).toBeGreaterThanOrEqual(2);
      expect(requestCount(fetchMock, "/api/prompts/assembled")).toBeGreaterThanOrEqual(2);
    });
  });

  it("resets an override via DELETE", async () => {
    const overriddenPrompts = {
      blocks: [
        {
          ...defaultPrompts().blocks[1]!,
          current_text: "CUSTOM VOICE",
          current_text_kind: "stored_override",
          overridden: true,
          updated_at: 123,
        },
      ],
    };
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(overriddenPrompts));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      if (path === "/api/prompts/voice_and_posture" && init?.method === "DELETE") {
        return Promise.resolve(
          jsonResponse({
            ...overriddenPrompts.blocks[0]!,
            current_text: "DEFAULT VOICE",
            current_text_kind: "static_default",
            overridden: false,
            updated_at: null,
          }),
        );
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    await screen.findByRole("heading", { name: "Voice and posture" });

    const resetButton = screen.getByRole("button", { name: "reset to default" });
    expect(resetButton).not.toBeDisabled();
    fireEvent.click(resetButton);

    await waitFor(() => {
      const deleteCall = fetchMock.mock.calls.find(
        (call) => (call[1] as RequestInit | undefined)?.method === "DELETE",
      );
      expect(deleteCall).toBeDefined();
      expect(requestPath(deleteCall![0] as RequestInfo | URL)).toBe(
        "/api/prompts/voice_and_posture",
      );
    });
    await waitFor(() => {
      expect(requestCount(fetchMock, "/api/prompts")).toBeGreaterThanOrEqual(2);
      expect(requestCount(fetchMock, "/api/prompts/assembled")).toBeGreaterThanOrEqual(2);
    });
  });

  it("shows runtime-composed host capabilities as connector-injected", async () => {
    const prompts = {
      blocks: [
        {
          key: "host_capabilities",
          label: "Host capabilities",
          description: "Runtime capabilities.",
          default_text: "STATIC HOST DEFAULT",
          current_text: "STATIC HOST DEFAULT\n\nHost-wired outbound capabilities available now.",
          current_text_kind: "runtime_composed",
          overridden: false,
          updated_at: null,
        },
      ],
    };
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(prompts));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();

    await screen.findByRole("heading", { name: "Host capabilities" });
    expect(screen.getByText("runtime composed (connector-injected)")).toBeInTheDocument();
  });

  it("blocks a runtime-composed host capabilities save until confirmed", async () => {
    const prompts = {
      blocks: [
        {
          key: "host_capabilities",
          label: "Host capabilities",
          description: "Runtime capabilities.",
          default_text: "STATIC HOST DEFAULT",
          current_text: "STATIC HOST DEFAULT\n\nHost-wired outbound capabilities available now.",
          current_text_kind: "runtime_composed",
          overridden: false,
          updated_at: null,
        },
      ],
    };
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(prompts));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      if (path === "/api/prompts/host_capabilities" && init?.method === "PUT") {
        return Promise.resolve(
          jsonResponse({
            ...prompts.blocks[0]!,
            current_text: "EDITED HOST CAPABILITIES",
            current_text_kind: "stored_override",
            overridden: true,
            updated_at: 123,
          }),
        );
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    await screen.findByRole("heading", { name: "Host capabilities" });

    const textarea = screen.getByLabelText("edited override") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "EDITED HOST CAPABILITIES" } });
    fireEvent.click(screen.getByRole("button", { name: "save" }));

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("freeze host capabilities override?")).toBeInTheDocument();
    expect(
      within(dialog).getByText(/Saving freezes the current live connector-composed/),
    ).toBeInTheDocument();
    expect(
      within(dialog).getByLabelText("static default versus saved override diff"),
    ).toBeInTheDocument();
    expect(within(dialog).getByText("STATIC HOST DEFAULT")).toBeInTheDocument();
    expect(within(dialog).getByText("EDITED HOST CAPABILITIES")).toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some(
        (call) =>
          requestPath(call[0] as RequestInfo | URL) === "/api/prompts/host_capabilities" &&
          (call[1] as RequestInit | undefined)?.method === "PUT",
      ),
    ).toBe(false);

    fireEvent.click(within(dialog).getByRole("button", { name: "save static override" }));

    await waitFor(() => {
      const putCall = fetchMock.mock.calls.find(
        (call) =>
          requestPath(call[0] as RequestInfo | URL) === "/api/prompts/host_capabilities" &&
          (call[1] as RequestInit | undefined)?.method === "PUT",
      );
      expect(putCall).toBeDefined();
      const init = putCall![1] as RequestInit;
      expect(JSON.parse(String(init.body))).toEqual({ text: "EDITED HOST CAPABILITIES" });
    });
  });

  it("does not require confirmation when host capabilities is already overridden", async () => {
    const prompts = {
      blocks: [
        {
          key: "host_capabilities",
          label: "Host capabilities",
          description: "Runtime capabilities.",
          default_text: "STATIC HOST DEFAULT",
          current_text: "CUSTOM HOST",
          current_text_kind: "stored_override",
          overridden: true,
          updated_at: 123,
        },
      ],
    };
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(prompts));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      if (path === "/api/prompts/host_capabilities" && init?.method === "PUT") {
        return Promise.resolve(
          jsonResponse({
            ...prompts.blocks[0]!,
            current_text: "CUSTOM HOST 2",
            current_text_kind: "stored_override",
          }),
        );
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    await screen.findByRole("heading", { name: "Host capabilities" });

    const textarea = screen.getByLabelText("edited override") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "CUSTOM HOST 2" } });
    fireEvent.click(screen.getByRole("button", { name: "save" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          (call) =>
            requestPath(call[0] as RequestInfo | URL) === "/api/prompts/host_capabilities" &&
            (call[1] as RequestInit | undefined)?.method === "PUT",
        ),
      ).toBe(true);
    });
  });

  it("renders the assembled prompt preview with the per-turn context omission label", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(defaultPrompts()));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    await screen.findByRole("heading", { name: "Base identity preamble" });

    expect(
      screen.getByText(
        /per-turn dynamic context \(retrieval, evidence ledger, commitments, current message\)/,
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("borg_host_capabilities")).toBeInTheDocument();
    expect(screen.getByText(/<borg_host_capabilities>/)).toBeInTheDocument();
  });

  it("selects a block row and drives the editor", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(defaultPrompts()));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    await screen.findByRole("heading", { name: "Base identity preamble" });

    fireEvent.click(screen.getByRole("button", { name: /Voice and posture/ }));

    expect(screen.getByRole("heading", { name: "Voice and posture" })).toBeInTheDocument();
    expect(screen.getByLabelText("edited override")).toHaveValue("DEFAULT VOICE");
  });

  it("preserves an unsaved draft when switching away and back", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(defaultPrompts()));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    await screen.findByRole("heading", { name: "Base identity preamble" });

    fireEvent.change(screen.getByLabelText("edited override"), {
      target: { value: "UNSAVED PREAMBLE DRAFT" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Voice and posture/ }));
    expect(screen.getByLabelText("edited override")).toHaveValue("DEFAULT VOICE");

    fireEvent.click(screen.getByRole("button", { name: /Base identity preamble/ }));
    expect(screen.getByLabelText("edited override")).toHaveValue("UNSAVED PREAMBLE DRAFT");
  });

  it("does not bleed a draft between blocks with identical current metadata", async () => {
    const prompts = {
      blocks: [
        {
          ...defaultPrompts().blocks[0]!,
          default_text: "DEFAULT FIRST",
          current_text: "SAME CURRENT",
          current_text_kind: "static_default",
          overridden: false,
          updated_at: null,
        },
        {
          ...defaultPrompts().blocks[1]!,
          default_text: "DEFAULT SECOND",
          current_text: "SAME CURRENT",
          current_text_kind: "static_default",
          overridden: false,
          updated_at: null,
        },
      ],
    };
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(prompts));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    await screen.findByRole("heading", { name: "Base identity preamble" });

    fireEvent.change(screen.getByLabelText("edited override"), {
      target: { value: "FIRST BLOCK DRAFT" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Voice and posture/ }));

    expect(screen.getByRole("heading", { name: "Voice and posture" })).toBeInTheDocument();
    expect(screen.getByLabelText("edited override")).toHaveValue("SAME CURRENT");

    fireEvent.click(screen.getByRole("button", { name: /Base identity preamble/ }));
    expect(screen.getByLabelText("edited override")).toHaveValue("FIRST BLOCK DRAFT");
  });

  it("opens the Inspector for a prompt_block row", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(defaultPrompts()));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    const rows = await screen.findAllByTestId("prompt-block-row");

    fireEvent.click(within(rows[1]!).getByRole("button", { name: "inspect" }));

    const dialog = await screen.findByRole("dialog", { name: "Prompt block inspector" });
    expect(within(dialog).getByText("voice_and_posture")).toBeInTheDocument();
  });

  it("copies default and current prompt text through the clipboard API", async () => {
    const prompts = {
      blocks: [
        {
          ...defaultPrompts().blocks[0]!,
          current_text: "CURRENT PREAMBLE",
          current_text_kind: "stored_override",
          overridden: true,
          updated_at: 123,
        },
      ],
    };
    const writeText = mockClipboard();
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(prompts));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    await screen.findByRole("heading", { name: "Base identity preamble" });

    fireEvent.click(screen.getByRole("button", { name: "copy default text" }));
    fireEvent.click(screen.getByRole("button", { name: "copy current text" }));
    fireEvent.click(screen.getByRole("button", { name: "copy all" }));

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith("DEFAULT PREAMBLE");
      expect(writeText).toHaveBeenCalledWith("CURRENT PREAMBLE");
      expect(writeText).toHaveBeenCalledWith(assembledPrompt().text);
    });
  });

  it("searches the assembled prompt text and reports the match count", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(defaultPrompts()));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    await screen.findByRole("heading", { name: "Base identity preamble" });

    fireEvent.change(screen.getByLabelText("search assembled prompt text"), {
      target: { value: "borg_host_capabilities" },
    });

    expect(screen.getByText("2 matches")).toBeInTheDocument();
  });

  it("counts overlapping assembled prompt search matches", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(defaultPrompts()));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse({ text: "aaa", sections: ["overlap"] }));
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    await screen.findByRole("heading", { name: "Base identity preamble" });

    expect(screen.getByPlaceholderText("case-insensitive; overlaps count")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("search assembled prompt text"), {
      target: { value: "aa" },
    });

    expect(screen.getByText("2 matches")).toBeInTheDocument();
  });

  it("searches assembled prompt text case-insensitively", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(defaultPrompts()));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(
          jsonResponse({ text: "Alpha alpha ALPHA", sections: ["case_test"] }),
        );
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    await screen.findByRole("heading", { name: "Base identity preamble" });

    fireEvent.change(screen.getByLabelText("search assembled prompt text"), {
      target: { value: "alpha" },
    });

    expect(screen.getByText("3 matches")).toBeInTheDocument();
  });

  it("renders the rough token estimate", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(defaultPrompts()));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    await screen.findByRole("heading", { name: "Base identity preamble" });

    expect(screen.getByTestId("assembled-token-estimate")).toHaveTextContent(
      /approximate ~\d+ tokens \(chars\/4, rough\)/,
    );
  });

  it("renders the assembled section outline", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/prompts" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(defaultPrompts()));
      }
      if (path === "/api/prompts/assembled" && (init?.method ?? "GET") === "GET") {
        return Promise.resolve(jsonResponse(assembledPrompt()));
      }
      return Promise.resolve(jsonResponse({ error: { message: "unhandled" } }, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPromptsScreen();
    await screen.findByRole("heading", { name: "Base identity preamble" });

    const outline = screen.getByLabelText("assembled prompt sections");
    expect(
      within(outline).getByRole("button", { name: "base_identity_preamble" }),
    ).toBeInTheDocument();
    expect(within(outline).getByRole("button", { name: "voice_and_posture" })).toBeInTheDocument();
    expect(
      within(outline).getByRole("button", { name: "borg_host_capabilities" }),
    ).toBeInTheDocument();
  });
});
