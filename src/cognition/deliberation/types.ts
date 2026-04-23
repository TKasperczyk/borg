// Shared deliberation data shapes used by the orchestrator and extracted helpers.
import type { LLMClient } from "../../llm/index.js";
import type { CommitmentRecord, EntityRepository } from "../../memory/commitments/index.js";
import type {
  AutobiographicalPeriod,
  GoalRecord,
  GrowthMarker,
  OpenQuestion,
  TraitRecord,
  ValueRecord,
} from "../../memory/self/index.js";
import type { SocialProfile } from "../../memory/social/index.js";
import type { SkillSelectionResult } from "../../memory/procedural/index.js";
import type { ReviewQueueItem } from "../../memory/semantic/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type {
  RetrievedEpisode,
  RetrievedSemantic,
  RetrievalSearchOptions,
} from "../../retrieval/index.js";
import type { ToolDispatcher } from "../../tools/index.js";
import type { SessionId } from "../../util/ids.js";
import type { ToolLoopCallRecord } from "../action/index.js";
import type { AutonomyTriggerContext } from "../autonomy-trigger.js";
import type { RecencyMessage } from "../recency/index.js";
import type { PerceptionResult } from "../types.js";

export type TurnStakes = "low" | "medium" | "high";

export type SelfSnapshot = {
  values: ValueRecord[];
  goals: GoalRecord[];
  traits: TraitRecord[];
  /**
   * The being's current autobiographical period (label + narrative). Phase
   * F wires this into the deliberator prompt so the being has a glimpse of
   * its own arc rather than values/goals/traits alone. Null when no period
   * has been opened yet.
   */
  currentPeriod?: AutobiographicalPeriod | null;
  /**
   * Recent growth markers -- what the being has newly learned or noticed
   * about itself. Surfaced as a thin "Recent learning" section so the
   * being doesn't keep rediscovering the same ground every session.
   */
  recentGrowthMarkers?: readonly GrowthMarker[];
};

export type DeliberationContext = {
  sessionId: SessionId;
  audience?: string;
  userMessage: string;
  userEntryId?: string;
  autonomyTrigger?: AutonomyTriggerContext | null;
  perception: PerceptionResult;
  retrievalResult: RetrievedEpisode[];
  /**
   * Semantic-band retrieval for this query: graph walks across supports/
   * contradicts/is_a relations from matched semantic nodes. Previously
   * attached per-episode with the same value duplicated; Phase C lifted
   * it out so it can be rendered once regardless of episode count and
   * retrieved independently of episode hits.
   */
  retrievedSemantic?: RetrievedSemantic | null;
  contradictionPresent?: boolean;
  applicableCommitments?: readonly CommitmentRecord[];
  openQuestionsContext?: readonly OpenQuestion[];
  pendingCorrectionsContext?: readonly ReviewQueueItem[];
  selectedSkill?: SkillSelectionResult | null;
  entityRepository?: EntityRepository;
  workingMemory: WorkingMemory;
  selfSnapshot: SelfSnapshot;
  /**
   * Social band: the profile of the person the being is talking to, when
   * audience is known. Phase F wires a thin summary (trust, interactions,
   * last contact) into the prompt so the being has relational context
   * rather than treating every audience as a cold first contact.
   */
  audienceProfile?: SocialProfile | null;
  /**
   * Recent dialogue from this session's stream, pre-compiled as LLM-ready
   * messages. If omitted, the deliberator behaves as it did pre-Phase-A:
   * the LLM sees only the current user message. Passing a window restores
   * the being's visibility into its own just-completed turns.
   */
  recencyMessages?: readonly RecencyMessage[];
  options?: {
    stakes?: TurnStakes;
    maxThinkingTokens?: number;
  };
  reRetrieve?: (query: string, options?: RetrievalSearchOptions) => Promise<RetrievedEpisode[]>;
};

export type DeliberationUsage = {
  input_tokens: number;
  output_tokens: number;
  stop_reason: string | null;
};

export type DeliberationResult = {
  path: "system_1" | "system_2";
  response: string;
  thoughts: string[];
  tool_calls: ToolLoopCallRecord[];
  usage: DeliberationUsage;
  decision_reason: string;
  retrievedEpisodes: RetrievedEpisode[];
  thoughtsPersisted: boolean;
};

export type DeliberatorOptions = {
  llmClient: LLMClient;
  toolDispatcher: ToolDispatcher;
  cognitionModel: string;
  backgroundModel: string;
};
