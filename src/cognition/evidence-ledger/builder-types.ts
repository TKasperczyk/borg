import type { ActionRepository } from "../../memory/actions/index.js";
import type { CommitmentRecord, CommitmentRepository } from "../../memory/commitments/index.js";
import type { EntityRepository } from "../../memory/commitments/index.js";
import type { RelationalSlotRepository } from "../../memory/relational-slots/index.js";
import type {
  GoalRecord,
  GoalsRepository,
  OpenQuestion,
  OpenQuestionsRepository,
} from "../../memory/self/index.js";
import type {
  OpenCommitmentReconciliationStatus,
  ReviewQueueItem,
} from "../../memory/review-queue/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type { EvidenceItem, RetrievedEpisode, RetrievedSemantic } from "../../retrieval/index.js";
import type { StreamEntry, StreamReader } from "../../stream/index.js";
import type { AttachmentRepository } from "../../attachments/index.js";
import type { CrossSessionSelfActivityRow } from "../../memory/activity/index.js";
import type { ObservedEventIntrospectionRow } from "../../memory/observed-events/index.js";
import type { SelfDecisionIntrospectionRow } from "../../memory/self-decisions/index.js";
import type { SharedStateEntry } from "../../memory/shared-state/index.js";
import type { EntityId, SessionId } from "../../util/ids.js";
import type { ActualFrameAnomalyClassification } from "../frame-anomaly/index.js";
import type { ActiveParticipant } from "../participants.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import type { AutobiographicalRecallResult } from "../autobiographical-recall.js";

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
  attachmentRepository?: Pick<AttachmentRepository, "get">;
  maxImagesPerLedger?: number;
  maxLedgerImageBytes?: number;
  imageRenderMaxDimension?: number;
  tracer?: TurnTracer;
};

export type EvidenceLedgerBuildInput = {
  sessionId: SessionId;
  turnId?: string;
  nowMs?: number;
  audienceEntityId: EntityId | null;
  currentUserMessage: string;
  currentUserEntry?: StreamEntry;
  currentUserEntries?: readonly StreamEntry[];
  workingMemory: WorkingMemory;
  applicableCommitments: readonly CommitmentRecord[];
  retrievedEvidence: readonly EvidenceItem[];
  retrievedEpisodes: readonly RetrievedEpisode[];
  retrievedSemantic?: RetrievedSemantic | null;
  openQuestions: readonly OpenQuestion[];
  pendingCorrections: readonly ReviewQueueItem[];
  pendingCommitmentReviews?: readonly OpenCommitmentReconciliationStatus[];
  frameAnomaly?: ActualFrameAnomalyClassification | null;
  activeParticipants?: readonly ActiveParticipant[];
  crossSessionSelfActivity?: readonly CrossSessionSelfActivityRow[];
  selfDecisionIntrospection?: readonly SelfDecisionIntrospectionRow[];
  observedEventIntrospection?: readonly ObservedEventIntrospectionRow[];
  autobiographicalRecall?: AutobiographicalRecallResult | null;
  sharedStateRecall?: readonly SharedStateEntry[];
};
