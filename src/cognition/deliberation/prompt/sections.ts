// Renders borg-tagged prompt sections and neutralizes forged borg tags in content.
export type TaggedPromptSection = {
  tag: string;
  content: string | null | undefined;
};

export function escapeReservedBorgTags(content: string): string {
  // Neutralize any borg-tag-looking content inside remembered text so a
  // retrieved record cannot close its enclosing block and forge a new one.
  return content.replace(/<(\/?)borg_/gi, "<$1-borg_");
}

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
