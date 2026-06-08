import {
  BASE_IDENTITY_PREAMBLE,
  EPISTEMIC_POSTURE_SECTION,
  IDENTITY_POSTURE_SECTION,
  SELF_ARCHITECTURE_SECTION,
  VOICE_AND_POSTURE_SECTION,
} from "./base-identity.js";
import { DEFAULT_HOST_CAPABILITIES_SECTION } from "./host-capability-contracts.js";

export const PROMPT_KEYS = [
  "base_identity_preamble",
  "self_architecture",
  "voice_and_posture",
  "epistemic_posture",
  "identity_posture",
  "host_capabilities",
] as const;

export type PromptKey = (typeof PROMPT_KEYS)[number];

export type PromptBlockSpec = {
  key: PromptKey;
  label: string;
  description: string;
  default: string;
};

export const PROMPT_BLOCKS: readonly PromptBlockSpec[] = [
  {
    key: "base_identity_preamble",
    label: "Base identity preamble",
    description:
      "The opening 'you are an AI being...' framing block. Sets borg's substrate-first stance.",
    default: BASE_IDENTITY_PREAMBLE,
  },
  {
    key: "self_architecture",
    label: "Self architecture",
    description:
      "How Sol's own mind works: the turn loop, global recall / contextual disclosure, and the offline dream/reflection cycle.",
    default: SELF_ARCHITECTURE_SECTION,
  },
  {
    key: "voice_and_posture",
    label: "Voice and posture",
    description:
      "Speaking style: prose by default, no service phrases, no reflexive clarifying questions.",
    default: VOICE_AND_POSTURE_SECTION,
  },
  {
    key: "epistemic_posture",
    label: "Epistemic posture",
    description:
      "Retrieved memory as ground truth; no fabricated specifics; honest 'I don't know' paths.",
    default: EPISTEMIC_POSTURE_SECTION,
  },
  {
    key: "identity_posture",
    label: "Identity posture",
    description:
      "First-person presence, no third-person narration, group-chat participation, attribution care.",
    default: IDENTITY_POSTURE_SECTION,
  },
  {
    key: "host_capabilities",
    label: "Host capabilities",
    description:
      "What the runtime can and cannot do (inputs, output channels, prohibited capabilities).",
    default: DEFAULT_HOST_CAPABILITIES_SECTION,
  },
];

const PROMPT_BLOCKS_BY_KEY = new Map<PromptKey, PromptBlockSpec>(
  PROMPT_BLOCKS.map((block) => [block.key, block]),
);

export function getPromptBlockSpec(key: PromptKey): PromptBlockSpec {
  const spec = PROMPT_BLOCKS_BY_KEY.get(key);
  if (spec === undefined) {
    throw new Error(`Unknown prompt key: ${String(key)}`);
  }
  return spec;
}

export function isPromptKey(value: unknown): value is PromptKey {
  return typeof value === "string" && PROMPT_BLOCKS_BY_KEY.has(value as PromptKey);
}
