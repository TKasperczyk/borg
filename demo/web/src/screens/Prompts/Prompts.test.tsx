import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

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

afterEach(() => {
  vi.unstubAllGlobals();
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

    render(<PromptsScreen />);

    await screen.findByText("Base identity preamble");

    const saveButtons = screen.getAllByRole("button", { name: "save" });
    expect(saveButtons.every((button) => (button as HTMLButtonElement).disabled)).toBe(true);
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

    render(<PromptsScreen />);
    await screen.findByText("Voice and posture");

    const textareas = screen.getAllByRole("textbox") as HTMLTextAreaElement[];
    const voiceTextarea = textareas[1]!;
    fireEvent.change(voiceTextarea, { target: { value: "CUSTOM VOICE" } });

    const saveButtons = screen.getAllByRole("button", { name: "save" });
    const enabledSave = saveButtons.find((button) => !(button as HTMLButtonElement).disabled);
    expect(enabledSave).toBeDefined();
    fireEvent.click(enabledSave as HTMLButtonElement);
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

    render(<PromptsScreen />);
    await screen.findByText("Voice and posture");

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

    render(<PromptsScreen />);

    await screen.findByText("Host capabilities");
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

    render(<PromptsScreen />);
    await screen.findByText("Host capabilities");

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "EDITED HOST CAPABILITIES" } });
    fireEvent.click(screen.getByRole("button", { name: "save" }));

    const dialog = screen.getByRole("dialog");
    expect(
      within(dialog).getByText(/Saving freezes the current live connector-composed/),
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

    render(<PromptsScreen />);
    await screen.findByText("Host capabilities");

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
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

    render(<PromptsScreen />);
    await screen.findByText("Base identity preamble");
    fireEvent.click(screen.getByRole("button", { name: "preview assembled prompt" }));

    expect(
      screen.getByText(
        /per-turn dynamic context \(retrieval, evidence ledger, commitments, current message\)/,
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("borg_host_capabilities")).toBeInTheDocument();
    expect(screen.getByText(/<borg_host_capabilities>/)).toBeInTheDocument();
  });
});
