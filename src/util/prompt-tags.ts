// Mechanical escaping for substrate-owned prompt tags. This only prevents
// untrusted content from forging borg_* prompt blocks; it does not interpret
// natural-language content.
export function escapeReservedBorgTags(content: string): string {
  return content.replace(/<(\/?)borg_/gi, "<$1-borg_");
}

// Mechanical XML text escaping for model-authored or operator-authored text
// wrapped inside substrate-owned prompt tags. This preserves the text content
// while preventing it from closing or forging surrounding prompt structure.
export function escapeXmlText(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

// Mechanical removal of tool-call serialization scaffolding that a model can
// bleed into a string argument value -- e.g. a finalizer `reason` whose stored
// text ends with `...real reason.</reason><parameter name="...">value`. This is
// substrate hygiene for the recall/prompt boundary (the same family as
// escapeReservedBorgTags and internal-id scrubbing), NOT an interpretation of
// natural-language content. The bleed is always a tail: the model completes the
// real value, then serializes the remainder of the tool call, so we truncate
// from the first scaffolding marker (plus an immediately preceding parameter
// close tag, e.g. `</reason>`) onward. It keys on structural tokens only
// (`<parameter>`/`<invoke>`/`<function_calls>`, namespaced or not), never on
// words, so it is language-agnostic. A generic close tag is only dropped when it
// is directly adjacent to such scaffolding, so ordinary prose is left intact.
const TOOL_CALL_SCAFFOLDING_BLEED =
  /(?:<\/[a-z_][\w.-]*>\s*)?<\/?(?:antml:)?(?:function_calls|invoke|parameter)\b[^>]*>[\s\S]*$/i;

export function stripToolCallScaffolding(content: string): string {
  return content.replace(TOOL_CALL_SCAFFOLDING_BLEED, "").trimEnd();
}
