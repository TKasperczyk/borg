import type { ActionRepository } from "../../memory/actions/index.js";
import type { CommitmentRecord, CommitmentRepository } from "../../memory/commitments/index.js";
import type { EntityRepository } from "../../memory/commitments/index.js";
import type {
  RelationalSlotRepository,
} from "../../memory/relational-slots/index.js";
import type {
  GoalRecord,
  GoalsRepository,
  OpenQuestion,
  OpenQuestionsRepository,
} from "../../memory/self/index.js";
import type { ReviewQueueItem } from "../../memory/semantic/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type { EvidenceItem, RetrievedEpisode, RetrievedSemantic } from "../../retrieval/index.js";
import type { StreamEntry, StreamReader } from "../../stream/index.js";
import type { EntityId, SessionId } from "../../util/ids.js";
import type { FrameAnomalyClassification } from "../frame-anomaly/index.js";
import type { ActiveParticipant } from "../participants.js";

export type ActionLedgerRepository = Pick<ActionRepository, "list"> &
  Partial<Pick<ActionRepository, "findSimilarDescriptionPairs">>;
export type CommitmentLedgerRepository = Pick<CommitmentRepository, "list">;
export type GoalLedgerRepository = Pick<GoalsRepository, "list">;

export type EvidenceLedgerBuilderOptions = {
  createStreamReader: (sessionId: SessionId) => StreamReader;
  relationalSlotRepository: Pick<RelationalSlotRepository, "list">;
  actionRepository: ActionLedgerRepository;
  commitmentRepository?: CommitmentLedgerRepository;
  goalsRepository?: GoalLedgerRepository;
  openQuestionsRepository?: Pick<OpenQuestionsRepository, "findByHandles">;
  currentSessionTranscriptTokenBudget: number;
  actionThreadRenderLimit?: number;
  actionThreadSimilarityThreshold?: number;
  actionThreadSourceRecordLimit?: number;
  entityRepository?: Pick<EntityRepository, "get">;
};

export type EvidenceLedgerBuildInput = {
  sessionId: SessionId;
  turnId?: string;
  audienceEntityId: EntityId | null;
  currentUserMessage: string;
  currentUserEntry?: StreamEntry;
  workingMemory: WorkingMemory;
  applicableCommitments: readonly CommitmentRecord[];
  retrievedEvidence: readonly EvidenceItem[];
  retrievedEpisodes: readonly RetrievedEpisode[];
  retrievedSemantic?: RetrievedSemantic | null;
  openQuestions: readonly OpenQuestion[];
  pendingCorrections: readonly ReviewQueueItem[];
  frameAnomaly?: FrameAnomalyClassification | null;
  activeParticipants?: readonly ActiveParticipant[];
};
