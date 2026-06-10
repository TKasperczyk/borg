import {
  getAttachmentMetadata,
  getCommitments,
  getCreatorDirectives,
  getCorrectionReviews,
  getDreamAudit,
  getIdentity,
  getMemoryBand,
  getPrompts,
  getReviews,
  getSemanticEdge,
  getSemanticNode,
  getSessions,
  getSharedState,
  getState,
  getStream,
  getTurns,
} from "../../api/client";
import type {
  AttachmentMetadataResponse,
  AutobiographicalPeriod,
  CommitmentItem,
  CreatorDirectiveItem,
  EpisodeMemoryItem,
  GrowthMarker,
  IdentityGoal,
  IdentityTrait,
  IdentityValue,
  MaintenanceAuditRow,
  OpenQuestion,
  PromptBlockView,
  RelationalMemoryItem,
  ReviewRow,
  SemanticMemoryEdge,
  SemanticMemoryNode,
  SessionRecord,
  SharedStateEntry,
  StreamEntry,
} from "../../api/types";
import type { GovernanceTabId, RouteId } from "../../routes";
import { isRecord } from "../../screens/screen-utils";
import { resolveObjectType, type ObjectType } from "./inspector-id";

export type InspectorTab =
  | "summary"
  | "evidence"
  | "relationships"
  | "timeline"
  | "actions"
  | "raw";

export const INSPECTOR_TABS: readonly InspectorTab[] = [
  "summary",
  "evidence",
  "relationships",
  "timeline",
  "actions",
  "raw",
];

export type InspectorReliability = "direct" | "in_list" | "needs_backend";

export type InspectorFetchContext = {
  sessionId: string;
  audience: string | null;
};

export type RelatedObjectRef = {
  type: ObjectType;
  id: string;
  fieldLabel: string;
};

export type SourceRoute =
  | RouteId
  | {
      route: RouteId;
      governanceTab?: GovernanceTabId;
    };

export type ObjectModel = {
  label: string;
  reliability: InspectorReliability;
  sourceRoute: SourceRoute | null;
  tabs: readonly InspectorTab[];
  fetch: (id: string, ctx: InspectorFetchContext) => Promise<unknown | null>;
  pivots: (obj: unknown) => RelatedObjectRef[];
};

const SUMMARY_RAW_TABS: readonly InspectorTab[] = ["summary", "raw"];
const BASIC_LIST_TABS: readonly InspectorTab[] = ["summary", "relationships", "timeline", "raw"];
const INSPECTOR_FANOUT_CONCURRENCY = 4;
const CORRECTABLE_TABS: readonly InspectorTab[] = [
  "summary",
  "evidence",
  "relationships",
  "timeline",
  "actions",
  "raw",
];

const EMPTY_PIVOTS = () => [];

function needsBackend(label: string, sourceRoute: SourceRoute | null = null): ObjectModel {
  return {
    label,
    reliability: "needs_backend",
    sourceRoute,
    tabs: SUMMARY_RAW_TABS,
    fetch: async () => null,
    pivots: EMPTY_PIVOTS,
  };
}

function uniqueStrings(values: readonly string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];

  for (const value of values) {
    if (value.length === 0 || seen.has(value)) {
      continue;
    }
    seen.add(value);
    result.push(value);
  }

  return result;
}

async function boundedAllSettled<T, R>(
  items: readonly T[],
  worker: (item: T) => Promise<R>,
  concurrency = INSPECTOR_FANOUT_CONCURRENCY,
): Promise<PromiseSettledResult<R>[]> {
  const results = new Array<PromiseSettledResult<R>>(items.length);
  let nextIndex = 0;

  async function runWorker(): Promise<void> {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      const item = items[index] as T;
      try {
        results[index] = { status: "fulfilled", value: await worker(item) };
      } catch (reason) {
        results[index] = { status: "rejected", reason };
      }
    }
  }

  const workerCount = Math.min(Math.max(1, concurrency), items.length);
  await Promise.all(Array.from({ length: workerCount }, () => runWorker()));
  return results;
}

function fulfilledValues<T>(results: readonly PromiseSettledResult<T>[]): T[] {
  return results.flatMap((result) => (result.status === "fulfilled" ? [result.value] : []));
}

function stringValue(record: Record<string, unknown>, key: string): string | null {
  const value = record[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function numberValue(record: Record<string, unknown>, key: string): number | null {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringArrayValue(record: Record<string, unknown>, key: string): string[] {
  const value = record[key];
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && item.length > 0)
    : [];
}

function addRef(
  refs: RelatedObjectRef[],
  type: ObjectType,
  id: string | null,
  fieldLabel: string,
): void {
  if (id !== null && id.length > 0) {
    refs.push({ type, id, fieldLabel });
  }
}

function addRefs(
  refs: RelatedObjectRef[],
  type: ObjectType,
  ids: readonly string[],
  fieldLabel: string,
): void {
  for (const id of ids) {
    addRef(refs, type, id, fieldLabel);
  }
}

function addTypedId(refs: RelatedObjectRef[], id: string | null, fieldLabel: string): void {
  if (id === null) {
    return;
  }

  const type = resolveObjectType(id);
  if (type !== null) {
    refs.push({ type, id, fieldLabel });
  }
}

function addTypedIds(refs: RelatedObjectRef[], ids: readonly string[], fieldLabel: string): void {
  for (const id of ids) {
    addTypedId(refs, id, fieldLabel);
  }
}

function streamEntryPivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj)) {
    return [];
  }

  const refs: RelatedObjectRef[] = [];
  addRef(refs, "turn", stringValue(obj, "turn_id"), "turn_id");
  addRef(refs, "session", stringValue(obj, "session_id"), "session_id");
  addRef(refs, "entity", stringValue(obj, "sender_entity_id"), "sender_entity_id");
  addRef(refs, "entity", stringValue(obj, "reply_target_entity_id"), "reply_target_entity_id");

  const responseTo = obj.response_to;
  if (isRecord(responseTo)) {
    addRefs(
      refs,
      "stream_entry",
      stringArrayValue(responseTo, "source_entry_ids"),
      "response_to.source_entry_ids",
    );
  }

  const content = obj.content;
  if (isRecord(content)) {
    addRef(refs, "attachment", stringValue(content, "attachment_id"), "content.attachment_id");
    addRef(
      refs,
      "image_perception",
      stringValue(content, "perception_id"),
      "content.perception_id",
    );
  }

  return refs;
}

function attachmentPivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj) || !isRecord(obj.attachment)) {
    return [];
  }

  const attachment = obj.attachment;
  const refs: RelatedObjectRef[] = [];
  addRef(
    refs,
    "image_perception",
    stringValue(attachment, "perception_id"),
    "attachment.perception_id",
  );
  addRef(
    refs,
    "stream_entry",
    stringValue(attachment, "parent_entry_id"),
    "attachment.parent_entry_id",
  );
  addRef(
    refs,
    "stream_entry",
    stringValue(attachment, "stream_entry_id"),
    "attachment.stream_entry_id",
  );
  addRef(refs, "turn", stringValue(attachment, "parent_turn_id"), "attachment.parent_turn_id");
  return refs;
}

function episodePivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj)) {
    return [];
  }

  const refs: RelatedObjectRef[] = [];
  addRefs(refs, "stream_entry", stringArrayValue(obj, "source_stream_ids"), "source_stream_ids");
  const lineage = obj.lineage;
  if (isRecord(lineage)) {
    addRefs(refs, "episode", stringArrayValue(lineage, "derived_from"), "lineage.derived_from");
    addRefs(refs, "episode", stringArrayValue(lineage, "supersedes"), "lineage.supersedes");
  }
  return refs;
}

function semanticNodePivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj)) {
    return [];
  }

  const refs: RelatedObjectRef[] = [];
  addRefs(refs, "episode", stringArrayValue(obj, "source_episode_ids"), "source_episode_ids");
  return refs;
}

function semanticEdgePivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj)) {
    return [];
  }

  const refs: RelatedObjectRef[] = [];
  addRef(refs, "semantic_node", stringValue(obj, "from_node_id"), "from_node_id");
  addRef(refs, "semantic_node", stringValue(obj, "to_node_id"), "to_node_id");
  addRefs(refs, "episode", stringArrayValue(obj, "evidence_episode_ids"), "evidence_episode_ids");
  addRef(
    refs,
    "semantic_edge",
    stringValue(obj, "invalidated_by_edge_id"),
    "invalidated_by_edge_id",
  );
  const reviewId = numberValue(obj, "invalidated_by_review_id");
  if (reviewId !== null) {
    addRef(refs, "review", String(reviewId), "invalidated_by_review_id");
  }
  return refs;
}

function identityEvidencePivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj)) {
    return [];
  }

  const refs: RelatedObjectRef[] = [];
  addRefs(refs, "episode", stringArrayValue(obj, "evidence_episode_ids"), "evidence_episode_ids");
  return refs;
}

function goalPivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj)) {
    return [];
  }

  const refs: RelatedObjectRef[] = [];
  addRef(refs, "goal", stringValue(obj, "goal_id"), "goal_id");
  return refs;
}

function autobiographicalPeriodPivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj)) {
    return [];
  }

  const refs: RelatedObjectRef[] = [];
  addRefs(refs, "episode", stringArrayValue(obj, "key_episode_ids"), "key_episode_ids");
  return refs;
}

function commitmentPivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj)) {
    return [];
  }

  const refs: RelatedObjectRef[] = [];
  addRefs(
    refs,
    "stream_entry",
    stringArrayValue(obj, "source_stream_entry_ids"),
    "source_stream_entry_ids",
  );
  addRef(refs, "commitment", stringValue(obj, "superseded_by_id"), "superseded_by_id");
  addRef(
    refs,
    "shared_state_entry",
    stringValue(obj, "canonicalized_by_artifact_entry_id"),
    "canonicalized_by_artifact_entry_id",
  );
  return refs;
}

function creatorDirectivePivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj)) {
    return [];
  }

  const refs: RelatedObjectRef[] = [];
  addRef(refs, "session", stringValue(obj, "source_session_id"), "source_session_id");
  addRefs(
    refs,
    "stream_entry",
    stringArrayValue(obj, "authorization_stream_entry_ids"),
    "authorization_stream_entry_ids",
  );
  addRefs(
    refs,
    "stream_entry",
    stringArrayValue(obj, "content_source_stream_entry_ids"),
    "content_source_stream_entry_ids",
  );
  addRefs(
    refs,
    "entity",
    stringArrayValue(obj, "activation_allowed_entity_ids"),
    "activation_allowed_entity_ids",
  );
  addRefs(
    refs,
    "entity",
    stringArrayValue(obj, "activation_excluded_entity_ids"),
    "activation_excluded_entity_ids",
  );
  addRef(refs, "entity", stringValue(obj, "subject_entity_id"), "subject_entity_id");
  addRef(refs, "creator_directive", stringValue(obj, "superseded_by_id"), "superseded_by_id");
  return refs;
}

function sharedStatePivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj)) {
    return [];
  }

  const refs: RelatedObjectRef[] = [];
  addRef(refs, "entity", stringValue(obj, "audience_entity_id"), "audience_entity_id");
  addRef(refs, "entity", stringValue(obj, "owner_entity_id"), "owner_entity_id");
  addRefs(
    refs,
    "stream_entry",
    stringArrayValue(obj, "provenance_stream_entry_ids"),
    "provenance_stream_entry_ids",
  );
  addRefs(
    refs,
    "stream_entry",
    stringArrayValue(obj, "last_updated_stream_entry_ids"),
    "last_updated_stream_entry_ids",
  );
  addRef(refs, "shared_state_entry", stringValue(obj, "superseded_by_id"), "superseded_by_id");

  const canonicalizes = obj.canonicalizes;
  if (isRecord(canonicalizes)) {
    addRefs(refs, "goal", stringArrayValue(canonicalizes, "goal_ids"), "canonicalizes.goal_ids");
    addRefs(
      refs,
      "commitment",
      stringArrayValue(canonicalizes, "commitment_ids"),
      "canonicalizes.commitment_ids",
    );
    addRefs(
      refs,
      "action_record",
      stringArrayValue(canonicalizes, "action_ids"),
      "canonicalizes.action_ids",
    );
    addRefs(
      refs,
      "open_question",
      stringArrayValue(canonicalizes, "open_question_ids"),
      "canonicalizes.open_question_ids",
    );
  }

  return refs;
}

function reviewPivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj) || !isRecord(obj.refs)) {
    return [];
  }

  const refs = obj.refs;
  const related: RelatedObjectRef[] = [];
  addRefs(related, "semantic_node", stringArrayValue(refs, "node_ids"), "refs.node_ids");
  addRef(related, "semantic_edge", stringValue(refs, "edge_id"), "refs.edge_id");
  addRefs(related, "episode", stringArrayValue(refs, "episode_ids"), "refs.episode_ids");
  addRefs(
    related,
    "creator_directive",
    stringArrayValue(refs, "directive_ids"),
    "refs.directive_ids",
  );
  addRefs(related, "commitment", stringArrayValue(refs, "commitment_ids"), "refs.commitment_ids");
  addTypedId(related, stringValue(refs, "target_id"), "refs.target_id");

  return related;
}

function sessionPivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj)) {
    return [];
  }

  const refs: RelatedObjectRef[] = [];
  addRef(refs, "entity", stringValue(obj, "audience_entity_id"), "audience_entity_id");
  addRef(refs, "turn", stringValue(obj, "last_turn_id"), "last_turn_id");
  return refs;
}

function relationalSlotPivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj)) {
    return [];
  }

  const refs: RelatedObjectRef[] = [];
  addRef(refs, "entity", stringValue(obj, "subject_entity_id"), "subject_entity_id");
  return refs;
}

function dreamAuditPivots(obj: unknown): RelatedObjectRef[] {
  if (!isRecord(obj)) {
    return [];
  }

  const refs: RelatedObjectRef[] = [];
  addRef(refs, "maintenance_run", stringValue(obj, "run_id"), "run_id");
  return refs;
}

async function findStreamEntry(id: string): Promise<StreamEntry | null> {
  const sessionIds = await sessionIdsForGlobalResolution();
  const responses = fulfilledValues(
    await boundedAllSettled(sessionIds, async (session) => getStream({ session, limit: 100 })),
  );
  for (const response of responses) {
    const entry = response.entries.find((item) => item.id === id);
    if (entry !== undefined) {
      return entry;
    }
  }
  return null;
}

async function findSession(id: string): Promise<SessionRecord | null> {
  const response = await getSessions();
  return response.sessions.find((session) => session.session_id === id) ?? null;
}

// Interim frontend global lookup. The proper long-term fix is the known
// backend gap: a unified object lookup endpoint that resolves ids directly.
async function sessionIdsForGlobalResolution(): Promise<string[]> {
  const response = await getSessions();
  return uniqueStrings(["default", ...response.sessions.map((session) => session.session_id)]);
}

async function findEpisode(id: string): Promise<EpisodeMemoryItem | null> {
  const detail = await getMemoryBand("episodic", { limit: 100 });
  return detail.band === "episodic" ? (detail.items.find((item) => item.id === id) ?? null) : null;
}

async function findSkill(id: string) {
  const detail = await getMemoryBand("procedural", { limit: 100 });
  return detail.band === "procedural"
    ? (detail.items.find((item) => item.id === id) ?? null)
    : null;
}

async function findRelationalSlot(id: string): Promise<RelationalMemoryItem | null> {
  const detail = await getMemoryBand("relational", { limit: 100 });
  return detail.band === "relational"
    ? (detail.items.find((item) => item.id === id) ?? null)
    : null;
}

async function findValue(id: string): Promise<IdentityValue | null> {
  const identity = await getIdentity();
  return identity.values.find((item) => item.id === id) ?? null;
}

async function findGoal(id: string): Promise<IdentityGoal | null> {
  const identity = await getIdentity();
  return identity.goals.find((item) => item.id === id) ?? null;
}

async function findTrait(id: string): Promise<IdentityTrait | null> {
  const identity = await getIdentity();
  return identity.traits.find((item) => item.id === id) ?? null;
}

async function findOpenQuestion(id: string): Promise<OpenQuestion | null> {
  const identity = await getIdentity();
  return identity.open_questions.find((item) => item.id === id) ?? null;
}

async function findGrowthMarker(id: string): Promise<GrowthMarker | null> {
  const identity = await getIdentity();
  return identity.growth_markers.find((item) => item.id === id) ?? null;
}

async function findAutobiographicalPeriod(id: string): Promise<AutobiographicalPeriod | null> {
  const identity = await getIdentity();
  return identity.periods.find((item) => item.id === id) ?? null;
}

async function findCommitment(id: string): Promise<CommitmentItem | null> {
  const response = await getCommitments({ state: "all", enforcement: "all" });
  return response.commitments.find((item) => item.id === id) ?? null;
}

async function findCreatorDirective(id: string): Promise<CreatorDirectiveItem | null> {
  const response = await getCreatorDirectives({ status: "all" });
  return response.directives.find((item) => item.id === id) ?? null;
}

async function findSharedStateEntry(id: string): Promise<SharedStateEntry | null> {
  const audienceLabels = await sharedStateAudienceLabelsForGlobalResolution();
  const sharedAudiences = fulfilledValues(
    await boundedAllSettled(audienceLabels, async (audience) => getSharedState(audience)),
  );
  for (const audience of sharedAudiences) {
    const entry = audience.entries.find((item) => item.id === id);
    if (entry !== undefined) {
      return entry;
    }
  }
  return null;
}

async function sharedStateAudienceLabelsForGlobalResolution(): Promise<string[]> {
  const [sessionsResult, stateResult] = await Promise.allSettled([getSessions(), getState()]);
  return uniqueStrings([
    "self",
    ...(stateResult.status === "fulfilled" ? stateResult.value.audiences : []),
    ...(sessionsResult.status === "fulfilled"
      ? sessionsResult.value.sessions.map((session) => session.audience_label)
      : []),
  ]);
}

async function findReview(id: string): Promise<ReviewRow | null> {
  const [reviews, correctionReviews] = await Promise.all([
    getReviews({ openOnly: false }),
    getCorrectionReviews(),
  ]);
  const rows = [...correctionReviews.rows, ...reviews.rows];
  return rows.find((row) => String(row.id) === id) ?? null;
}

async function findDreamAudit(id: string): Promise<MaintenanceAuditRow | null> {
  const numericId = Number(id);
  if (!Number.isFinite(numericId)) {
    return null;
  }

  const response = await getDreamAudit(100);
  return response.rows.find((row) => row.id === numericId) ?? null;
}

async function findPromptBlock(id: string): Promise<PromptBlockView | null> {
  const response = await getPrompts();
  return response.blocks.find((block) => block.key === id) ?? null;
}

async function findTurn(id: string) {
  const sessionIds = await sessionIdsForGlobalResolution();
  const responses = fulfilledValues(
    await boundedAllSettled(sessionIds, async (session) => getTurns({ session, limit: 100 })),
  );
  for (const response of responses) {
    const row = response.rows.find((item) => item.turn_id === id);
    if (row !== undefined) {
      return row;
    }
  }
  return null;
}

async function findImagePerception(
  id: string,
): Promise<AttachmentMetadataResponse["perception"] | null> {
  const sessionIds = await sessionIdsForGlobalResolution();
  const responses = fulfilledValues(
    await boundedAllSettled(sessionIds, async (session) => getStream({ session, limit: 100 })),
  );
  for (const response of responses) {
    for (const entry of response.entries) {
      const content = entry.content;
      if (!isRecord(content) || content.perception_id !== id) {
        continue;
      }
      const attachmentId = stringValue(content, "attachment_id");
      if (attachmentId === null) {
        continue;
      }
      try {
        const metadata = await getAttachmentMetadata(attachmentId);
        if (metadata.perception?.perception_id === id) {
          return metadata.perception;
        }
      } catch {
        // Ignore per-attachment lookup failures; another session entry may still resolve it.
      }
    }
  }
  return null;
}

export const objectRegistry = {
  stream_entry: {
    label: "Stream entry",
    reliability: "in_list",
    sourceRoute: "stream",
    tabs: BASIC_LIST_TABS,
    fetch: findStreamEntry,
    pivots: streamEntryPivots,
  },
  session: {
    label: "Session",
    reliability: "in_list",
    sourceRoute: "cognition",
    tabs: ["summary", "relationships", "timeline", "actions", "raw"],
    fetch: findSession,
    pivots: sessionPivots,
  },
  episode: {
    label: "Episode",
    reliability: "in_list",
    sourceRoute: "memory",
    tabs: CORRECTABLE_TABS,
    fetch: findEpisode,
    pivots: episodePivots,
  },
  goal: {
    label: "Goal",
    reliability: "in_list",
    sourceRoute: "identity",
    tabs: CORRECTABLE_TABS,
    fetch: findGoal,
    pivots: EMPTY_PIVOTS,
  },
  value: {
    label: "Value",
    reliability: "in_list",
    sourceRoute: "identity",
    tabs: CORRECTABLE_TABS,
    fetch: findValue,
    pivots: identityEvidencePivots,
  },
  trait: {
    label: "Trait",
    reliability: "in_list",
    sourceRoute: "identity",
    tabs: CORRECTABLE_TABS,
    fetch: findTrait,
    pivots: identityEvidencePivots,
  },
  autobiographical_period: {
    label: "Autobiographical period",
    reliability: "in_list",
    sourceRoute: "identity",
    tabs: BASIC_LIST_TABS,
    fetch: findAutobiographicalPeriod,
    pivots: autobiographicalPeriodPivots,
  },
  growth_marker: {
    label: "Growth marker",
    reliability: "in_list",
    sourceRoute: "identity",
    tabs: BASIC_LIST_TABS,
    fetch: findGrowthMarker,
    pivots: identityEvidencePivots,
  },
  open_question: {
    label: "Open question",
    reliability: "in_list",
    sourceRoute: "identity",
    tabs: CORRECTABLE_TABS,
    fetch: findOpenQuestion,
    pivots: goalPivots,
  },
  semantic_node: {
    label: "Semantic node",
    reliability: "direct",
    sourceRoute: "memory",
    tabs: CORRECTABLE_TABS,
    fetch: getSemanticNode,
    pivots: semanticNodePivots,
  },
  semantic_edge: {
    label: "Semantic edge",
    reliability: "direct",
    sourceRoute: "memory",
    tabs: CORRECTABLE_TABS,
    fetch: getSemanticEdge,
    pivots: semanticEdgePivots,
  },
  commitment: {
    label: "Commitment",
    reliability: "in_list",
    sourceRoute: { route: "governance", governanceTab: "commitments" },
    tabs: CORRECTABLE_TABS,
    fetch: findCommitment,
    pivots: commitmentPivots,
  },
  creator_directive: {
    label: "Creator directive",
    reliability: "in_list",
    sourceRoute: { route: "governance", governanceTab: "shared_state" },
    tabs: ["summary", "relationships", "timeline", "actions", "raw"],
    fetch: findCreatorDirective,
    pivots: creatorDirectivePivots,
  },
  entity: needsBackend("Entity", null),
  action_record: needsBackend("Action record", null),
  relational_slot: {
    label: "Relational slot",
    reliability: "in_list",
    sourceRoute: "memory",
    tabs: BASIC_LIST_TABS,
    fetch: findRelationalSlot,
    pivots: relationalSlotPivots,
  },
  shared_state_entry: {
    label: "Shared-state entry",
    reliability: "in_list",
    sourceRoute: { route: "governance", governanceTab: "shared_state" },
    tabs: ["summary", "relationships", "timeline", "raw"],
    fetch: findSharedStateEntry,
    pivots: sharedStatePivots,
  },
  consolidation_family: needsBackend("Consolidation family", "dream"),
  activity_event: needsBackend("Activity event", null),
  self_decision_event: needsBackend("Self decision event", "identity"),
  observed_event: needsBackend("Observed event", "stream"),
  scheduled_wake: needsBackend("Scheduled wake", "dream"),
  skill: {
    label: "Skill",
    reliability: "in_list",
    sourceRoute: "memory",
    tabs: BASIC_LIST_TABS,
    fetch: findSkill,
    pivots: semanticNodePivots,
  },
  procedural_evidence: needsBackend("Procedural evidence", "memory"),
  maintenance_run: needsBackend("Maintenance run", "dream"),
  executive_step: needsBackend("Executive step", null),
  attachment: {
    label: "Attachment",
    reliability: "direct",
    sourceRoute: "stream",
    tabs: BASIC_LIST_TABS,
    fetch: getAttachmentMetadata,
    pivots: attachmentPivots,
  },
  image_perception: {
    label: "Image perception",
    reliability: "in_list",
    sourceRoute: "stream",
    tabs: BASIC_LIST_TABS,
    fetch: findImagePerception,
    pivots: EMPTY_PIVOTS,
  },
  autonomy_wake: needsBackend("Autonomy wake", "dream"),
  turn: {
    label: "Turn evidence",
    reliability: "in_list",
    sourceRoute: "cognition",
    tabs: ["summary", "evidence", "timeline", "raw"],
    fetch: findTurn,
    pivots: EMPTY_PIVOTS,
  },
  review: {
    label: "Review",
    reliability: "in_list",
    sourceRoute: "review",
    tabs: ["summary", "relationships", "timeline", "actions", "raw"],
    fetch: findReview,
    pivots: reviewPivots,
  },
  dream_audit: {
    label: "Dream audit row",
    reliability: "in_list",
    sourceRoute: "dream",
    tabs: ["summary", "relationships", "timeline", "actions", "raw"],
    fetch: findDreamAudit,
    pivots: dreamAuditPivots,
  },
  prompt_block: {
    label: "Prompt block",
    reliability: "in_list",
    sourceRoute: "prompts",
    tabs: ["summary", "timeline", "actions", "raw"],
    fetch: findPromptBlock,
    pivots: EMPTY_PIVOTS,
  },
} satisfies Record<ObjectType, ObjectModel>;

export function isWhySupported(type: ObjectType): boolean {
  return (
    type === "episode" ||
    type === "semantic_node" ||
    type === "semantic_edge" ||
    type === "value" ||
    type === "goal" ||
    type === "trait" ||
    type === "commitment" ||
    type === "open_question"
  );
}
