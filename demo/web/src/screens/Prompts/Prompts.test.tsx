import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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
        overridden: false,
        updated_at: null,
      },
      {
        key: "voice_and_posture",
        label: "Voice and posture",
        description: "Speaking style.",
        default_text: "DEFAULT VOICE",
        current_text: "DEFAULT VOICE",
        overridden: false,
        updated_at: null,
      },
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
    const fetchMock = vi.fn((_request: RequestInfo | URL, _init?: RequestInit) =>
      Promise.resolve(jsonResponse(defaultPrompts())),
    );
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
      if (path === "/api/prompts/voice_and_posture" && init?.method === "PUT") {
        return Promise.resolve(
          jsonResponse({
            key: "voice_and_posture",
            label: "Voice and posture",
            description: "Speaking style.",
            default_text: "DEFAULT VOICE",
            current_text: "CUSTOM VOICE",
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
      if (path === "/api/prompts/voice_and_posture" && init?.method === "DELETE") {
        return Promise.resolve(
          jsonResponse({ ...overriddenPrompts.blocks[0]!, current_text: "DEFAULT VOICE", overridden: false, updated_at: null }),
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
});
