import { describe, expect, it } from "vitest";

import { validateAssistantOutput } from "./output-validator.js";

describe("validateAssistantOutput", () => {
  it("blocks unquoted role labels at any line start", () => {
    expect(validateAssistantOutput("Normal opening.\nHuman: Done.")).toMatchObject({
      ok: false,
      failure: {
        reason: "output_validator",
        kind: "role_label",
        line: 2,
        label: "Human",
      },
    });
  });

  it("allows role labels inside fenced code blocks and blockquotes", () => {
    expect(validateAssistantOutput("```text\nHuman: Done.\n```")).toEqual({ ok: true });
    expect(validateAssistantOutput("> Human: Done.")).toEqual({ ok: true });
  });

  it("allows inline mentions of role labels in explanatory prose", () => {
    expect(validateAssistantOutput("The string Human: is a transcript role label.")).toEqual({
      ok: true,
    });
  });

  it("blocks narrated non-generation and empty drafts", () => {
    expect(validateAssistantOutput("(no response)")).toMatchObject({
      ok: false,
      failure: {
        reason: "invalid_non_generation_text",
      },
    });
    expect(validateAssistantOutput("   ")).toMatchObject({
      ok: false,
      failure: {
        reason: "empty_finalizer",
      },
    });
  });

  it.each([
    "(not generating)",
    "(not generating.)",
    "[not generating]",
    "(not replying)",
    "[not replying]",
    "(no reply generated.)",
    "(no reply)",
    "[no reply]",
    "No reply generated.",
  ])("blocks v8 non-generation stand-in %s", (draft) => {
    expect(validateAssistantOutput(draft)).toMatchObject({
      ok: false,
      failure: {
        reason: "invalid_non_generation_text",
        kind: "non_generation_text",
      },
    });
  });
});
