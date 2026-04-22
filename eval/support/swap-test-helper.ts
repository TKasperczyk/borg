const SUBSTRATE_BLOCK_TAGS = [
  "borg_held_preferences",
  "borg_commitment_records",
  "borg_retrieved_episodes",
  "borg_retrieved_semantic",
  "borg_current_period",
  "borg_recent_growth",
  "borg_open_questions",
  "borg_pending_corrections",
] as const;

export type SubstrateBlockTag = (typeof SUBSTRATE_BLOCK_TAGS)[number];

const SUBSTRATE_BLOCK_TAG_SET = new Set<string>(SUBSTRATE_BLOCK_TAGS);
const BORG_BLOCK_PATTERN = /<(borg_[a-z_]+)>[\s\S]*?<\/\1>/g;

export function extractSubstrateBlocks(systemPrompt: string): Map<string, string> {
  const blocks = new Map<string, string>();

  for (const match of systemPrompt.matchAll(BORG_BLOCK_PATTERN)) {
    const tag = match[1];
    const block = match[0];

    if (tag === undefined || block === undefined || !SUBSTRATE_BLOCK_TAG_SET.has(tag)) {
      continue;
    }

    blocks.set(tag, block);
  }

  return blocks;
}

export function compareSubstrateBlocks(
  left: ReadonlyMap<string, string>,
  right: ReadonlyMap<string, string>,
): {
  equal: boolean;
  differences: string[];
} {
  const differences: string[] = [];

  for (const tag of SUBSTRATE_BLOCK_TAGS) {
    const leftBlock = left.get(tag);
    const rightBlock = right.get(tag);

    if (leftBlock === undefined && rightBlock === undefined) {
      continue;
    }

    if (leftBlock === undefined) {
      differences.push(`${tag}: missing from left prompt`);
      continue;
    }

    if (rightBlock === undefined) {
      differences.push(`${tag}: missing from right prompt`);
      continue;
    }

    if (leftBlock !== rightBlock) {
      differences.push(`${tag}: block content differs`);
    }
  }

  return {
    equal: differences.length === 0,
    differences,
  };
}
