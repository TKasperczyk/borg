// Identity governance: the audit, guard, and lifecycle layer over identity-
// bearing records. The records themselves live in memory/self (values,
// goals, traits, autobiographical, growth markers, open questions) and
// memory/commitments. IdentityService composes those repositories and
// routes writes through IdentityGuard + IdentityEventRepository so an
// "I changed my mind" mutation cannot silently overwrite established state.
// This is not a second memory band -- it is the governance over the data
// that lives in the bands.

export { IdentityGuard, type IdentityGuardDecision, type IdentityGuardState } from "./guard.js";
export { identityMigrations } from "./migrations.js";
export {
  IdentityService,
  type IdentityServiceOptions,
  type IdentityUpdateOptions,
  type IdentityUpdateResult,
} from "./service.js";
export { IdentityEventRepository, type IdentityEventRepositoryOptions } from "./repository.js";
export {
  IDENTITY_RECORD_TYPES,
  identityEventSchema,
  identityRecordTypeSchema,
  type IdentityEvent,
  type IdentityRecordType,
} from "./types.js";
