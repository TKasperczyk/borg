export type BorgDefaultErrorCode =
  | "BORG_AUTH_ERROR"
  | "BORG_ATTACHMENT_ERROR"
  | "BORG_AUTONOMY_ERROR"
  | "BORG_BUDGET_EXCEEDED"
  | "BORG_COGNITION_ERROR"
  | "BORG_COMMITMENT_ERROR"
  | "BORG_CONFIG_ERROR"
  | "BORG_EMBEDDING_ERROR"
  | "BORG_LLM_ERROR"
  | "BORG_PROVENANCE_ERROR"
  | "BORG_RETRIEVAL_ERROR"
  | "BORG_SEMANTIC_ERROR"
  | "BORG_SESSION_BUSY"
  | "BORG_STORAGE_ERROR"
  | "BORG_STREAM_ERROR"
  | "BORG_TOOL_ERROR"
  | "BORG_WORKING_MEMORY_ERROR";

export type KnownBorgErrorCode =
  | BorgDefaultErrorCode
  | "ACTION_LIFECYCLE_TURN_COUNTER_INVALID"
  | "ACTION_RECORDS_EXISTING_TABLE_NON_EMPTY"
  | "ACTION_RECORD_NOT_FOUND"
  | "ACTION_RECORD_ROW_INVALID"
  | "AFFECTIVE_OUTPUT_INVALID"
  | "ATTACHMENT_BLOB_CORRUPTED"
  | "ATTACHMENT_BLOB_READ_FAILED"
  | "ATTACHMENT_DIMENSIONS_TOO_LARGE"
  | "ATTACHMENT_DIMENSIONS_UNREADABLE"
  | "ATTACHMENT_IMAGE_MALFORMED"
  | "ATTACHMENT_IMAGE_TOO_LARGE"
  | "ATTACHMENT_INACTIVE"
  | "ATTACHMENT_LEDGER_RENDER_BYTES_TOO_LARGE"
  | "ATTACHMENT_LEDGER_RENDER_DIMENSIONS_TOO_LARGE"
  | "ATTACHMENT_NOT_FOUND"
  | "ATTACHMENT_STREAM_ENTRY_UNLINKED"
  | "ATTACHMENT_TOO_MANY_IMAGES"
  | "ATTACHMENT_UNSUPPORTED_MEDIA_TYPE"
  | "AUTH_CREDENTIALS_MALFORMED"
  | "AUTH_NO_CREDENTIALS"
  | "AUTH_REFRESH_FAILED"
  | "AUTOBIOGRAPHICAL_PERIOD_ALREADY_EXISTS"
  | "AUTOBIOGRAPHICAL_PERIOD_NOT_FOUND"
  | "AUTONOMY_WAKE_ROW_INVALID"
  | "BORG_RETRIEVAL_PROJECTION_INVARIANT"
  | "CLI_NOT_FOUND"
  | "COMMITMENT_NOT_FOUND"
  | "COMMITMENT_ROW_INVALID"
  | "CONFIG_FILE_INVALID"
  | "CONSOLIDATOR_INVALID"
  | "CONSOLIDATOR_PLAN_INVALID"
  | "CORRECTION_PATCH_INVALID"
  | "CORRECTION_TARGET_UNSUPPORTED"
  | "DEFAULT_USER_REQUIRED"
  | "ENTITY_EXTRACTION_FAILED"
  | "ENTITY_FALLBACK_INVALID"
  | "ENTITY_NAME_REQUIRED"
  | "EPISODE_CURSOR_INVALID"
  | "EPISODE_DELETE_FAILED"
  | "EPISODE_INSERT_FAILED"
  | "EPISODE_INVALID"
  | "EPISODE_NOT_FOUND"
  | "EPISODE_PATCH_INVALID"
  | "EPISODE_ROW_INVALID"
  | "EPISODE_SOURCE_ANCHOR_REQUIRED"
  | "EPISODE_STATS_INVALID"
  | "EPISODE_STATS_MISSING"
  | "EPISODE_STATS_PATCH_INVALID"
  | "EPISODE_UPDATE_FAILED"
  | "EXECUTIVE_STEP_INVALID_TRANSITION"
  | "EXECUTIVE_STEP_NOT_FOUND"
  | "EXECUTIVE_STEP_OPEN_LIMIT"
  | "EXECUTIVE_STEP_WAIT_REQUIRES_DUE_AT"
  | "EXTRACTOR_OUTPUT_INVALID"
  | "EXTRACTOR_SOURCE_ID_INVALID"
  | "GOAL_NOT_FOUND"
  | "GOAL_PARENT_MISSING"
  | "GROUP_SENDER_REQUIRED"
  | "GROWTH_MARKER_INVALID"
  | "GROWTH_MARKER_NOT_FOUND"
  | "IDENTITY_CAS_MISMATCH"
  | "IDENTITY_EVENT_INSERT_FAILED"
  | "IDENTITY_EVENT_INVALID"
  | "IDENTITY_GUARD_CREATE_REJECTED"
  | "IDENTITY_REVIEW_REQUIRED"
  | "IMAGE_PERCEPTION_PAYLOAD_MISSING"
  | "IMAGE_PERCEPTION_ROW_INVALID"
  | "IMAGE_PERCEPTION_SCHEMA_INVALID"
  | "IMAGE_PERCEPTION_TOOL_MISSING"
  | "LANCEDB_SCHEMA_EVOLUTION_UNSUPPORTED"
  | "LANCEDB_SCHEMA_INVALID"
  | "LANCEDB_SCHEMA_MISMATCH"
  | "LLM_ATTACHMENT_RESOLVER_MISSING"
  | "LLM_STRUCTURED_OUTPUT_PARSE_FAILED"
  | "MAINTENANCE_AUDIT_INSERT_FAILED"
  | "MAINTENANCE_AUDIT_INVALID"
  | "MAINTENANCE_PROCESS_CADENCE_OVERLAP"
  | "MAINTENANCE_REVERSER_MISSING"
  | "MODE_DETECTION_FAILED"
  | "MODE_FALLBACK_INVALID"
  | "MOOD_ROW_INVALID"
  | "OFFLINE_BUDGET_EXCEEDED"
  | "OPEN_QUESTION_ALREADY_EXISTS"
  | "OPEN_QUESTION_INVALID"
  | "OPEN_QUESTION_INVALID_TRANSITION"
  | "OPEN_QUESTION_NOT_FOUND"
  | "OPEN_QUESTION_RESOLUTION_EVIDENCE_REQUIRED"
  | "OVERSEER_TARGET_FAILED"
  | "PROCEDURAL_CONTEXT_INVALID"
  | "PROCEDURAL_EVIDENCE_NOT_FOUND"
  | "PROCEDURAL_EVIDENCE_ROW_INVALID"
  | "PROCEDURAL_SKILL_SPLIT_INVALID"
  | "PROCEDURAL_SKILL_SPLIT_PARTS_INVALID"
  | "PROCEDURAL_SKILL_SPLIT_TARGETS_EMPTY"
  | "PROCEDURAL_SKILL_SPLIT_TARGETS_INCOMPLETE"
  | "PROCEDURAL_SKILL_SPLIT_TARGET_DUPLICATE"
  | "PROCEDURAL_SKILL_SPLIT_TARGET_UNKNOWN"
  | "PROCEDURAL_SYNTHESIZER_DEDUP_MISSING"
  | "PROCEDURAL_SYNTHESIZER_INVALID"
  | "PROVENANCE_REQUIRED"
  | "RECALL_STATE_INVALID"
  | "RECALL_STATE_ROW_INVALID"
  | "REFLECTOR_INVALID"
  | "REFLECTOR_INVALID_REF"
  | "REFLECTOR_OUTPUT_INVALID"
  | "REFLECTOR_PLAN_INVALID"
  | "RELATIONAL_SLOT_ROW_INVALID"
  | "REVIEW_QUEUE_ALREADY_RESOLVED"
  | "REVIEW_QUEUE_APPLYING_STATE_UNSUPPORTED"
  | "REVIEW_QUEUE_HANDLER_UNREGISTERED"
  | "REVIEW_QUEUE_INSERT_FAILED"
  | "REVIEW_QUEUE_INVALID"
  | "REVIEW_QUEUE_MALFORMED_PAIR_REFS"
  | "REVIEW_QUEUE_REPAIR_UNSUPPORTED"
  | "REVIEW_QUEUE_RESOLUTION_INVALID"
  | "REVIEW_QUEUE_RESOLUTION_IN_PROGRESS"
  | "REVIEW_QUEUE_RESOLUTION_RACE"
  | "REVIEW_QUEUE_TARGET_NOT_FOUND"
  | "REVIEW_QUEUE_WINNER_INVALID"
  | "REVIEW_QUEUE_WINNER_REQUIRED"
  | "REVIEW_RESOLVER_REPAIR_INVALID"
  | "RUMINATOR_INVALID"
  | "RUMINATOR_PLAN_INVALID"
  | "SELF_AUTOBIOGRAPHICAL_INVALID"
  | "SELF_NARRATOR_EMPTY_NARRATIVE"
  | "SELF_NARRATOR_INVALID"
  | "SELF_NARRATOR_INVALID_REF"
  | "SEMANTIC_BELIEF_DEPENDENCY_INSERT_FAILED"
  | "SEMANTIC_BELIEF_DEPENDENCY_INVALID"
  | "SEMANTIC_EDGE_AS_OF_INVALID"
  | "SEMANTIC_EDGE_CORRECTION_UNSUPPORTED"
  | "SEMANTIC_EDGE_DANGLING"
  | "SEMANTIC_EDGE_DUPLICATE"
  | "SEMANTIC_EDGE_FORGET_UNSUPPORTED"
  | "SEMANTIC_EDGE_ID_REQUIRED"
  | "SEMANTIC_EDGE_INVALID"
  | "SEMANTIC_EDGE_INVALIDATE_BEFORE_VALID_FROM"
  | "SEMANTIC_EDGE_INVALIDATION_TIME_INVALID"
  | "SEMANTIC_EDGE_NOT_FOUND"
  | "SEMANTIC_EXTRACTOR_INVALID"
  | "SEMANTIC_EXTRACTOR_INVALID_REF"
  | "SEMANTIC_EXTRACTOR_PLAN_INVALID"
  | "SEMANTIC_LIMIT_INVALID"
  | "SEMANTIC_NODE_CONFIDENCE_ADJUSTMENT_REQUIRES_TRANSACTION"
  | "SEMANTIC_NODE_CONFIDENCE_INVALID"
  | "SEMANTIC_NODE_CONFIDENCE_UPDATED_AT_INVALID"
  | "SEMANTIC_NODE_DELETE_FAILED"
  | "SEMANTIC_NODE_INSERT_FAILED"
  | "SEMANTIC_NODE_NOT_FOUND"
  | "SEMANTIC_NODE_RESTORE_FAILED"
  | "SEMANTIC_NODE_STATUS_TIME_INVALID"
  | "SEMANTIC_NODE_UPDATE_FAILED"
  | "SEMANTIC_NODE_VECTOR_SYNC_SOURCE_MISSING"
  | "SEMANTIC_NODE_VECTOR_SYNC_SOURCE_MISSING_CLEANUP_FAILED"
  | "SEMANTIC_NODE_VECTOR_SYNC_TARGET_MISSING"
  | "SEMANTIC_ROW_INVALID"
  | "SEMANTIC_WALK_INVALID"
  | "SESSION_LOCK_ACQUIRE_FAILED"
  | "SESSION_LOCK_INVALID_TIMEOUT"
  | "SESSION_TURN_BUSY"
  | "SHARED_STATE_ENTRY_NOT_FOUND"
  | "SHARED_STATE_INVALID_OPERATION"
  | "SHARED_STATE_ROW_INVALID"
  | "SHARED_STATE_SOURCE_NOT_TRUSTED"
  | "SHARED_STATE_STATE_KEY_REQUIRED"
  | "SKILL_CONTEXT_KEY_INVALID"
  | "SKILL_CONTEXT_STATS_RECORD_FAILED"
  | "SKILL_CONTEXT_STATS_ROW_INVALID"
  | "SKILL_DELETE_FAILED"
  | "SKILL_INSERT_FAILED"
  | "SKILL_NOT_FOUND"
  | "SKILL_REPLACE_FAILED"
  | "SKILL_ROW_INVALID"
  | "SKILL_SPLIT_ALREADY_APPLIED"
  | "SKILL_SPLIT_CLAIM_LOST"
  | "SKILL_SPLIT_CONTEXT_DUPLICATE"
  | "SKILL_SPLIT_CONTEXT_MISSING"
  | "SKILL_SPLIT_EMPTY"
  | "SKILL_SPLIT_FAILED"
  | "SKILL_SPLIT_TARGETS_EMPTY"
  | "SOCIAL_ROW_INVALID"
  | "STREAM_INDEX_UPDATE_FAILED"
  | "STREAM_INDEX_POISONED"
  | "STREAM_SERIALIZE_FAILED"
  | "STREAM_WATERMARK_INVALID_CURSOR"
  | "TOOL_ALREADY_REGISTERED"
  | "TRAIT_DECAY_INVALID"
  | "TRAIT_NOT_FOUND"
  | "VALUE_NOT_FOUND"
  | "WORKING_MEMORY_CLEAR_FAILED"
  | "WORKING_MEMORY_INVALID"
  | "WORKING_MEMORY_LOAD_FAILED"
  | "WORKING_MEMORY_PERSISTENCE_DISABLED"
  | "WORKING_MEMORY_SAVE_FAILED";

export type BorgErrorCode = KnownBorgErrorCode | (string & {});
export type AuthErrorCode = Extract<KnownBorgErrorCode, "BORG_AUTH_ERROR" | `AUTH_${string}`>;
export type AttachmentErrorCode = Extract<
  KnownBorgErrorCode,
  "BORG_ATTACHMENT_ERROR" | `ATTACHMENT_${string}` | `IMAGE_PERCEPTION_${string}`
>;
export type ConfigErrorCode = Extract<KnownBorgErrorCode, "BORG_CONFIG_ERROR" | `CONFIG_${string}`>;
export type StreamErrorCode = Extract<KnownBorgErrorCode, "BORG_STREAM_ERROR" | `STREAM_${string}`>;
export type LLMErrorCode = Extract<KnownBorgErrorCode, "BORG_LLM_ERROR" | `LLM_${string}`>;
export type StorageErrorCode = Extract<
  KnownBorgErrorCode,
  "BORG_STORAGE_ERROR" | `LANCEDB_${string}` | `${string}_ROW_INVALID`
>;

export type BorgErrorOptions = {
  cause?: unknown;
};

export type BorgTypedErrorOptions = BorgErrorOptions & {
  code?: BorgErrorCode;
};

export type BorgErrorJSON = {
  name: string;
  code: BorgErrorCode;
  message: string;
  cause?: unknown;
};

function serializeCause(cause: unknown): unknown {
  if (!(cause instanceof Error)) {
    return cause;
  }

  return {
    name: cause.name,
    message: cause.message,
  };
}

export function describeError(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }

  return String(error);
}

export abstract class BorgError extends Error {
  readonly code: BorgErrorCode;

  constructor(code: BorgErrorCode, message: string, options: BorgErrorOptions = {}) {
    super(message, { cause: options.cause });
    this.name = new.target.name;
    this.code = code;
  }

  toJSON(): BorgErrorJSON {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      cause: serializeCause(this.cause),
    };
  }
}

export class ConfigError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_CONFIG_ERROR", message, options);
  }
}

export class StreamError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_STREAM_ERROR", message, options);
  }
}

export class EmbeddingError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_EMBEDDING_ERROR", message, options);
  }
}

export class LLMError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_LLM_ERROR", message, options);
  }
}

export class RetrievalError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_RETRIEVAL_ERROR", message, options);
  }
}

export class StorageError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_STORAGE_ERROR", message, options);
  }
}

export class IdentityCasMismatchError extends StorageError {
  readonly recordType: string;
  readonly recordId: string;
  readonly expectedVersion: number;

  constructor(input: { recordType: string; recordId: string; expectedVersion: number }) {
    super(
      `Identity CAS mismatch for ${input.recordType} ${input.recordId} at version ${input.expectedVersion}`,
      {
        code: "IDENTITY_CAS_MISMATCH",
      },
    );
    this.recordType = input.recordType;
    this.recordId = input.recordId;
    this.expectedVersion = input.expectedVersion;
  }
}

export class CognitionError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_COGNITION_ERROR", message, options);
  }
}

export class WorkingMemoryError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_WORKING_MEMORY_ERROR", message, options);
  }
}

export class SemanticError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_SEMANTIC_ERROR", message, options);
  }
}

export class CommitmentError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_COMMITMENT_ERROR", message, options);
  }
}

export class ProvenanceError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_PROVENANCE_ERROR", message, options);
  }
}

export class BudgetExceededError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_BUDGET_EXCEEDED", message, options);
  }
}

export class AuthError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_AUTH_ERROR", message, options);
  }
}

export class ToolError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_TOOL_ERROR", message, options);
  }
}

export class AutonomyError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_AUTONOMY_ERROR", message, options);
  }
}

export class AttachmentError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_ATTACHMENT_ERROR", message, options);
  }
}

export class SessionBusyError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_SESSION_BUSY", message, options);
  }
}
