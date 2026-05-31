import type { Config } from "../config/index.js";
import {
  type CreatorDirective,
  type CreatorDirectiveRepository,
} from "../memory/creator-directives/index.js";
import {
  sessionIdSchema,
  type SessionRecord,
  type SessionSourceType,
  type SessionsRepository,
} from "../sessions/index.js";
import type { StreamEntry, StreamReader } from "../stream/index.js";
import { OUTBOUND_POST_TOOL_NAME } from "../tools/internal/outbound-post-name.js";
import type { Clock } from "../util/clock.js";
import { ToolError } from "../util/errors.js";
import { isPlainRecord } from "../util/guards.js";
import type { SessionId } from "../util/ids.js";

const AUTONOMOUS_OUTBOUND_CAP_SCAN_MAX_ENTRIES = 4_096;
const AUTONOMOUS_OUTBOUND_CAP_SCAN_MAX_BYTES = 8 * 1024 * 1024;

export const PROACTIVE_OUTBOUND_CREATOR_DIRECTIVE_TOPIC_TAG = "proactive_outbound";

export type AutonomousOutboundAuthorizationKind = "config" | "creator_directive";

export type AutonomousOutboundPromptTarget = {
  session_id: SessionId;
  source_type: SessionRecord["source_type"];
  audience_label: string;
  conversation_kind: SessionRecord["conversation_kind"];
  participation_policy: SessionRecord["participation_policy"];
  authorization: AutonomousOutboundAuthorizationKind;
};

export type AutonomousOutboundPromptContext = {
  maxPostsPerWindow: number;
  maxPostsPerTargetPerWindow: number;
  remainingPostsInWindow: number;
  windowMs: number;
  targets: AutonomousOutboundPromptTarget[];
};

export type AutonomousOutboundPolicyOptions = {
  config: Config["autonomy"]["proactiveOutbound"];
  sessionsRepository: Pick<SessionsRepository, "get" | "list">;
  creatorDirectiveRepository: Pick<CreatorDirectiveRepository, "list" | "listApplicable">;
  createStreamReader: (sessionId: SessionId) => StreamReader;
  transportSourceTypes?: readonly SessionSourceType[];
  clock: Clock;
};

type RecentAttemptCount = {
  count: number;
  scanCapReached: boolean;
};

type RecentAttemptCountInput = {
  sessionId: SessionId;
  targetSessionId?: SessionId;
  excludeTurnId?: string;
};

function directiveTargetsSessionAudience(
  directive: CreatorDirective,
  target: SessionRecord,
): boolean {
  return (
    target.audience_entity_id !== null &&
    directive.subject_kind === "entity" &&
    directive.subject_entity_id === target.audience_entity_id
  );
}

function autonomousOutboundToolCallTargetSessionId(entry: StreamEntry): SessionId | null {
  if (entry.kind !== "tool_call" || !isPlainRecord(entry.content)) {
    return null;
  }

  const input = entry.content.input;
  if (!isPlainRecord(input)) {
    return null;
  }

  const parsed = sessionIdSchema.safeParse(input.target_session_id);

  return parsed.success ? parsed.data : null;
}

function isAutonomousOutboundToolCall(
  entry: StreamEntry,
  targetSessionId?: SessionId,
  excludeTurnId?: string,
): boolean {
  if (entry.kind !== "tool_call" || !isPlainRecord(entry.content)) {
    return false;
  }

  if (excludeTurnId !== undefined && entry.turn_id === excludeTurnId) {
    return false;
  }

  if (targetSessionId !== undefined) {
    const entryTargetSessionId = autonomousOutboundToolCallTargetSessionId(entry);
    if (entryTargetSessionId !== targetSessionId) {
      return false;
    }
  }

  return (
    entry.content.tool_name === OUTBOUND_POST_TOOL_NAME &&
    (entry.content.origin === "autonomous" || entry.content.turn_origin === "autonomous")
  );
}

function promptTarget(
  session: SessionRecord,
  authorization: AutonomousOutboundAuthorizationKind,
): AutonomousOutboundPromptTarget {
  return {
    session_id: session.session_id,
    source_type: session.source_type,
    audience_label: session.audience_label,
    conversation_kind: session.conversation_kind,
    participation_policy: session.participation_policy,
    authorization,
  };
}

export class AutonomousOutboundPolicy {
  constructor(private readonly options: AutonomousOutboundPolicyOptions) {}

  promptContext(currentSessionId: SessionId): AutonomousOutboundPromptContext | null {
    if (!this.options.config.enabled) {
      return null;
    }

    const attempts = this.countRecentAttemptsForWakingSession({ sessionId: currentSessionId });
    if (attempts.scanCapReached) {
      return null;
    }

    const remainingPostsInWindow = Math.max(
      0,
      this.options.config.maxPostsPerWindow - attempts.count,
    );
    if (remainingPostsInWindow <= 0) {
      return null;
    }

    const targets = this.authorizedTargets(currentSessionId);
    if (targets.length === 0) {
      return null;
    }

    return {
      maxPostsPerWindow: this.options.config.maxPostsPerWindow,
      maxPostsPerTargetPerWindow: this.options.config.maxPostsPerTargetPerWindow,
      remainingPostsInWindow,
      windowMs: this.options.config.windowMs,
      targets,
    };
  }

  assertAuthorized(input: {
    currentSessionId: SessionId;
    targetSession: SessionRecord;
    currentTurnId?: string;
  }): void {
    if (!this.options.config.enabled) {
      throw new ToolError("Autonomous outbound messaging is disabled", {
        code: "AUTONOMOUS_OUTBOUND_DISABLED",
      });
    }

    if (!this.transportAuthorizes(input.targetSession)) {
      throw new ToolError("Autonomous outbound target has no wired transport connector", {
        code: "AUTONOMOUS_OUTBOUND_TARGET_NOT_TRANSPORTABLE",
      });
    }

    const authorization = this.authorizationForTarget(input.targetSession);
    if (authorization === null) {
      throw new ToolError("Autonomous outbound target is not structurally authorized", {
        code: "AUTONOMOUS_OUTBOUND_TARGET_NOT_AUTHORIZED",
      });
    }

    const attempts = this.countRecentAttemptsForWakingSession({
      sessionId: input.currentSessionId,
      excludeTurnId: input.currentTurnId,
    });
    if (attempts.scanCapReached) {
      throw new ToolError("Autonomous outbound cap could not be verified within scan limits", {
        code: "AUTONOMOUS_OUTBOUND_CAP_SCAN_LIMIT",
      });
    }

    if (attempts.count >= this.options.config.maxPostsPerWindow) {
      throw new ToolError("Autonomous outbound rolling cap exceeded", {
        code: "AUTONOMOUS_OUTBOUND_CAP_EXCEEDED",
      });
    }

    const targetAttempts = this.countRecentAttemptsForTarget({
      currentSessionId: input.currentSessionId,
      targetSessionId: input.targetSession.session_id,
      excludeTurnId: input.currentTurnId,
    });
    if (targetAttempts.scanCapReached) {
      throw new ToolError(
        "Autonomous outbound target cap could not be verified within scan limits",
        {
          code: "AUTONOMOUS_OUTBOUND_CAP_SCAN_LIMIT",
        },
      );
    }

    if (targetAttempts.count >= this.options.config.maxPostsPerTargetPerWindow) {
      throw new ToolError("Autonomous outbound target rolling cap exceeded", {
        code: "AUTONOMOUS_OUTBOUND_TARGET_CAP_EXCEEDED",
      });
    }
  }

  authorizationForTarget(target: SessionRecord): AutonomousOutboundAuthorizationKind | null {
    if (!this.transportAuthorizes(target)) {
      return null;
    }

    if (this.configAuthorizes(target)) {
      return "config";
    }

    if (this.creatorDirectiveAuthorizes(target)) {
      return "creator_directive";
    }

    return null;
  }

  private authorizedTargets(currentSessionId: SessionId): AutonomousOutboundPromptTarget[] {
    const targets: AutonomousOutboundPromptTarget[] = [];
    const sessions = this.options.sessionsRepository.list({
      status: "active",
      excludeSessionId: currentSessionId,
      limit: this.options.config.maxAuthorizedTargets,
    });

    for (const session of sessions) {
      const authorization = this.authorizationForTarget(session);

      if (authorization !== null && this.targetCapAllows(currentSessionId, session.session_id)) {
        targets.push(promptTarget(session, authorization));
      }
    }

    return targets;
  }

  private configAuthorizes(target: SessionRecord): boolean {
    return (
      this.options.config.allowByConfig.sessionIds.includes(target.session_id) ||
      this.options.config.allowByConfig.sourceTypes.includes(target.source_type)
    );
  }

  private transportAuthorizes(target: SessionRecord): boolean {
    return (this.options.transportSourceTypes ?? []).includes(target.source_type);
  }

  private creatorDirectiveAuthorizes(target: SessionRecord): boolean {
    if (!this.options.config.allowByCreatorDirective || target.audience_entity_id === null) {
      return false;
    }

    const directives = this.options.creatorDirectiveRepository.list({
      status: "active",
      kind: "routing_instruction",
      subjectKind: "entity",
      subjectEntityId: target.audience_entity_id,
      topicTag: PROACTIVE_OUTBOUND_CREATOR_DIRECTIVE_TOPIC_TAG,
    });
    if (directives.length === 0) {
      return false;
    }

    const applicableByDirectiveId = new Map(
      this.options.creatorDirectiveRepository
        .listApplicable({
          currentAudienceEntityId: target.audience_entity_id,
          currentSenderBorgRole: null,
          participantEntityIds: [target.audience_entity_id],
          sessionRole: target.audience_role,
        })
        .map((item) => [item.directive.id, item]),
    );

    return directives.some(
      (directive) =>
        directiveTargetsSessionAudience(directive, target) &&
        applicableByDirectiveId.get(directive.id)?.activation.active === true,
    );
  }

  private targetCapAllows(currentSessionId: SessionId, targetSessionId: SessionId): boolean {
    const attempts = this.countRecentAttemptsForTarget({ currentSessionId, targetSessionId });

    return (
      !attempts.scanCapReached && attempts.count < this.options.config.maxPostsPerTargetPerWindow
    );
  }

  private countRecentAttemptsForWakingSession(input: RecentAttemptCountInput): RecentAttemptCount {
    // Caps intentionally count autonomous outbound tool_call attempts, regardless of
    // delivery result. That keeps retry storms bounded without joining result rows.
    const cutoff = this.options.clock.now() - this.options.config.windowMs;
    const scan = this.options.createStreamReader(input.sessionId).scanReverse({
      maxEntries: AUTONOMOUS_OUTBOUND_CAP_SCAN_MAX_ENTRIES,
      maxBytes: AUTONOMOUS_OUTBOUND_CAP_SCAN_MAX_BYTES,
      budgetFilter: (entry) => entry.kind === "tool_call",
      filter: (entry) =>
        entry.timestamp >= cutoff &&
        isAutonomousOutboundToolCall(entry, input.targetSessionId, input.excludeTurnId),
    });

    return {
      count: scan.entries.length,
      scanCapReached: scan.capReached !== null,
    };
  }

  private countRecentAttemptsForTarget(input: {
    currentSessionId: SessionId;
    targetSessionId: SessionId;
    excludeTurnId?: string;
  }): RecentAttemptCount {
    // The per-target window cap scans prior attempts across all currently-active
    // sessions (plus the waking one). This is exhaustive today because nothing
    // mutates a session's status away from "active" -- there is no archival or
    // idle sweep. If one is ever added, prior attempts recorded in a since-idled
    // session would drop out of this scan and weaken the cap; at that point this
    // list filter must widen to include the relevant non-active statuses.
    const sessionIds = new Set<SessionId>([
      input.currentSessionId,
      ...this.options.sessionsRepository
        .list({ status: "active" })
        .map((session) => session.session_id),
    ]);
    let count = 0;

    for (const sessionId of sessionIds) {
      const attempts = this.countRecentAttemptsForWakingSession({
        sessionId,
        targetSessionId: input.targetSessionId,
        excludeTurnId: input.excludeTurnId,
      });

      count += attempts.count;
      if (attempts.scanCapReached) {
        return {
          count,
          scanCapReached: true,
        };
      }
    }

    return {
      count,
      scanCapReached: false,
    };
  }
}
