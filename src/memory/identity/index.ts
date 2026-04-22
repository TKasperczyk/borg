export { IdentityGuard, type IdentityGuardDecision, type IdentityGuardState } from "./guard.js";
export { identityMigrations } from "./migrations.js";
export {
  IdentityService,
  type IdentityServiceOptions,
  type IdentityUpdateOptions,
  type IdentityUpdateResult,
} from "./service.js";
export {
  IdentityEventRepository,
  type IdentityEventRepositoryOptions,
} from "./repository.js";
export {
  IDENTITY_RECORD_TYPES,
  identityEventSchema,
  identityRecordTypeSchema,
  type IdentityEvent,
  type IdentityRecordType,
} from "./types.js";
