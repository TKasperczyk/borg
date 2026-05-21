// Renders borg-tagged prompt sections and neutralizes forged borg tags in content.
import { escapeReservedBorgTags } from "../../../util/prompt-tags.js";

export { escapeReservedBorgTags } from "../../../util/prompt-tags.js";

export type TaggedPromptSection = {
  tag: string;
  content: string | null | undefined;
};

export function renderTaggedPromptSection(
  tag: string,
  content: string | null | undefined,
): string | null {
  if (content === null || content === undefined) {
    return null;
  }

  return [`<${tag}>`, escapeReservedBorgTags(content), `</${tag}>`].join("\n");
}

export function renderTaggedPromptBlock(
  preamble: string,
  sections: readonly TaggedPromptSection[],
): string | null {
  const rendered = sections
    .map((section) => renderTaggedPromptSection(section.tag, section.content))
    .filter((section): section is string => section !== null);

  if (rendered.length === 0) {
    return null;
  }

  return [preamble, ...rendered].join("\n\n");
}
