import { appendFileSync, mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "../config/index.js";
import {
  CreatorDirectiveRepository,
  creatorDirectiveMigrations,
} from "../memory/creator-directives/index.js";
import {
  SessionsRepository,
  sessionMigrations,
  type SessionSourceType,
} from "../sessions/index.js";
import { composeMigrations, openDatabase, type SqliteDatabase } from "../storage/sqlite/index.js";
import {
  getSessionStreamPath,
  StreamReader,
  StreamWriter,
  type StreamEntry,
} from "../stream/index.js";
import { ManualClock } from "../util/clock.js";
import { ToolError } from "../util/errors.js";
import {
  createEntityId,
  createSessionId,
  createStreamEntryId,
  type SessionId,
} from "../util/ids.js";

import {
  AutonomousOutboundPolicy,
  PROACTIVE_OUTBOUND_CREATOR_DIRECTIVE_TOPIC_TAG,
} from "./autonomous-policy.js";

function policyConfig(
  overrides: Partial<typeof DEFAULT_CONFIG.autonomy.proactiveOutbound> = {},
): typeof DEFAULT_CONFIG.autonomy.proactiveOutbound {
  return {
    ...DEFAULT_CONFIG.autonomy.proactiveOutbound,
    ...overrides,
    allowByConfig: {
      ...DEFAULT_CONFIG.autonomy.proactiveOutbound.allowByConfig,
      ...overrides.allowByConfig,
      sessionIds: [
        ...(overrides.allowByConfig?.sessionIds ??
          DEFAULT_CONFIG.autonomy.proactiveOutbound.allowByConfig.sessionIds),
      ],
      sourceTypes: [
        ...(overrides.allowByConfig?.sourceTypes ??
          DEFAULT_CONFIG.autonomy.proactiveOutbound.allowByConfig.sourceTypes),
      ],
    },
  };
}

function setup(): {
  tempDir: string;
  db: SqliteDatabase;
  clock: ManualClock;
  sessionsRepository: SessionsRepository;
  creatorDirectiveRepository: CreatorDirectiveRepository;
  createPolicy: (
    config: typeof DEFAULT_CONFIG.autonomy.proactiveOutbound,
    transportSourceTypes?: readonly SessionSourceType[],
  ) => AutonomousOutboundPolicy;
} {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-autonomous-outbound-"));
  const clock = new ManualClock(1_000);
  const db = openDatabase(join(tempDir, "borg.sqlite"), {
    migrations: composeMigrations(sessionMigrations, creatorDirectiveMigrations),
  });
  const sessionsRepository = new SessionsRepository({ db, clock });
  const creatorDirectiveRepository = new CreatorDirectiveRepository({ db, clock });

  return {
    tempDir,
    db,
    clock,
    sessionsRepository,
    creatorDirectiveRepository,
    createPolicy: (config, transportSourceTypes) =>
      new AutonomousOutboundPolicy({
        config,
        sessionsRepository,
        creatorDirectiveRepository,
        createStreamReader: (sessionId: SessionId) =>
          new StreamReader({
            dataDir: tempDir,
            sessionId,
          }),
        ...(transportSourceTypes === undefined ? {} : { transportSourceTypes }),
        clock,
      }),
  };
}

const LARGE_STREAM_FILLER = "x".repeat(256 * 1024);
const LARGE_STREAM_FILLER_ENTRY_COUNT = 40;

type RawStreamEntryInput = {
  timestamp: number;
  kind: StreamEntry["kind"];
  content: unknown;
  turn_id?: string;
};

function appendStreamEntries(
  dataDir: string,
  sessionId: SessionId,
  entries: readonly RawStreamEntryInput[],
): void {
  const streamPath = getSessionStreamPath(dataDir, sessionId);

  mkdirSync(dirname(streamPath), { recursive: true });
  appendFileSync(
    streamPath,
    entries
      .map((entry) =>
        JSON.stringify({
          id: createStreamEntryId(),
          timestamp: entry.timestamp,
          kind: entry.kind,
          content: entry.content,
          ...(entry.turn_id === undefined ? {} : { turn_id: entry.turn_id }),
          session_id: sessionId,
          compressed: false,
        }),
      )
      .map((line) => `${line}\n`)
      .join(""),
    { encoding: "utf8", flag: "a" },
  );
}

function largeFillerEntries(timestamp: number): RawStreamEntryInput[] {
  return Array.from({ length: LARGE_STREAM_FILLER_ENTRY_COUNT }, (_, index) => ({
    timestamp,
    kind: "internal_event",
    content: {
      event: "scan_filler",
      index,
      payload: LARGE_STREAM_FILLER,
    },
  }));
}

function outboundAttemptEntry(
  timestamp: number,
  targetSessionId: SessionId,
  callId: string,
): RawStreamEntryInput {
  return {
    timestamp,
    kind: "tool_call",
    content: {
      call_id: callId,
      tool_name: "tool.outbound.post",
      input: {
        target_session_id: targetSessionId,
        instruction: "Prior autonomous attempt.",
      },
      origin: "autonomous",
    },
  };
}

function expectToolErrorCode(action: () => void, code: string): void {
  let thrown: unknown;

  try {
    action();
  } catch (error) {
    thrown = error;
  }

  expect(thrown).toBeInstanceOf(ToolError);
  expect((thrown as ToolError).code).toBe(code);
}

describe("AutonomousOutboundPolicy", () => {
  const cleanups: Array<() => void> = [];

  afterEach(() => {
    while (cleanups.length > 0) {
      cleanups.pop()?.();
    }
  });

  it("is off by default", () => {
    const harness = setup();
    cleanups.push(() => {
      harness.db.close();
      rmSync(harness.tempDir, { recursive: true, force: true });
    });
    const currentSessionId = createSessionId();
    const targetSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "demo",
      label: "alice",
      audience_label: "Alice",
      conversation_kind: "demo",
    });
    const policy = harness.createPolicy(policyConfig(), ["demo"]);

    expect(policy.promptContext(currentSessionId)).toBeNull();
    expect(() =>
      policy.assertAuthorized({
        currentSessionId,
        targetSession,
      }),
    ).toThrow(/disabled/);
  });

  it("authorizes configured target sessions structurally", () => {
    const harness = setup();
    cleanups.push(() => {
      harness.db.close();
      rmSync(harness.tempDir, { recursive: true, force: true });
    });
    const currentSessionId = createSessionId();
    const targetSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "demo",
      label: "alice",
      audience_label: "Alice",
      conversation_kind: "demo",
    });
    const policy = harness.createPolicy(
      policyConfig({
        enabled: true,
        allowByConfig: {
          sessionIds: [targetSession.session_id],
          sourceTypes: [],
        },
      }),
      ["demo"],
    );

    expect(policy.authorizationForTarget(targetSession)).toBe("config");
    expect(policy.promptContext(currentSessionId)?.targets).toEqual([
      expect.objectContaining({
        session_id: targetSession.session_id,
        authorization: "config",
      }),
    ]);
    expect(() =>
      policy.assertAuthorized({
        currentSessionId,
        targetSession,
      }),
    ).not.toThrow();
  });

  it("omits and rejects authorized targets without a wired transport connector", () => {
    const harness = setup();
    cleanups.push(() => {
      harness.db.close();
      rmSync(harness.tempDir, { recursive: true, force: true });
    });
    const currentSessionId = createSessionId();
    const targetSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "slack",
      label: "alice-slack",
      audience_label: "Alice",
      conversation_kind: "channel",
    });
    const policy = harness.createPolicy(
      policyConfig({
        enabled: true,
        allowByConfig: {
          sessionIds: [targetSession.session_id],
          sourceTypes: [],
        },
      }),
      ["demo"],
    );

    expect(policy.authorizationForTarget(targetSession)).toBeNull();
    expect(policy.promptContext(currentSessionId)).toBeNull();
    expect(() =>
      policy.assertAuthorized({
        currentSessionId,
        targetSession,
      }),
    ).toThrow(/no wired transport/);
  });

  it("fails closed when no transport source types are provided", () => {
    const harness = setup();
    cleanups.push(() => {
      harness.db.close();
      rmSync(harness.tempDir, { recursive: true, force: true });
    });
    const currentSessionId = createSessionId();
    const targetSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "demo",
      label: "alice",
      audience_label: "Alice",
      conversation_kind: "demo",
    });
    const policy = harness.createPolicy(
      policyConfig({
        enabled: true,
        allowByConfig: {
          sessionIds: [targetSession.session_id],
          sourceTypes: [],
        },
      }),
    );

    expect(policy.authorizationForTarget(targetSession)).toBeNull();
    expect(policy.promptContext(currentSessionId)).toBeNull();
    expect(() =>
      policy.assertAuthorized({
        currentSessionId,
        targetSession,
      }),
    ).toThrow(/no wired transport/);
  });

  it("authorizes active routing directives by machine topic tag and target audience id", () => {
    const harness = setup();
    cleanups.push(() => {
      harness.db.close();
      rmSync(harness.tempDir, { recursive: true, force: true });
    });
    const currentSessionId = createSessionId();
    const creatorId = createEntityId();
    const aliceId = createEntityId();
    const targetSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "demo",
      label: "alice",
      audience_label: "Alice",
      audience_entity_id: aliceId,
      conversation_kind: "demo",
    });
    harness.creatorDirectiveRepository.queue({
      kind: "routing_instruction",
      createdByEntityId: creatorId,
      sourceSessionId: currentSessionId,
      authorizationStreamEntryIds: [createStreamEntryId()],
      contentSourceStreamEntryIds: [createStreamEntryId()],
      subjectKind: "entity",
      subjectEntityId: aliceId,
      operationalDirective: "Creator permits autonomous outreach to this audience.",
      disclosurePolicy: {
        content_scope: "subject_only",
        allowed_entity_ids: [],
        excluded_entity_ids: [],
        subject_may_know: true,
        mention_policy: "proactive",
        denied_audience_behavior: "omit",
        boundary_prompt: null,
        topic_tags: [PROACTIVE_OUTBOUND_CREATOR_DIRECTIVE_TOPIC_TAG],
      },
      priority: 10,
    });
    const policy = harness.createPolicy(
      policyConfig({
        enabled: true,
      }),
      ["demo"],
    );

    expect(policy.authorizationForTarget(targetSession)).toBe("creator_directive");
    expect(policy.promptContext(currentSessionId)?.targets).toEqual([
      expect.objectContaining({
        session_id: targetSession.session_id,
        authorization: "creator_directive",
      }),
    ]);
  });

  it("authorizes activation-active routing directives without requiring content disclosure", () => {
    const harness = setup();
    cleanups.push(() => {
      harness.db.close();
      rmSync(harness.tempDir, { recursive: true, force: true });
    });
    const currentSessionId = createSessionId();
    const creatorId = createEntityId();
    const aliceId = createEntityId();
    const targetSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "demo",
      label: "alice",
      audience_label: "Alice",
      audience_entity_id: aliceId,
      conversation_kind: "demo",
    });
    harness.creatorDirectiveRepository.queue({
      kind: "routing_instruction",
      createdByEntityId: creatorId,
      sourceSessionId: currentSessionId,
      authorizationStreamEntryIds: [createStreamEntryId()],
      contentSourceStreamEntryIds: [createStreamEntryId()],
      subjectKind: "entity",
      subjectEntityId: aliceId,
      operationalDirective: "Creator permits autonomous outreach to this audience.",
      disclosurePolicy: {
        content_scope: "operator_only",
        allowed_entity_ids: [],
        excluded_entity_ids: [],
        subject_may_know: null,
        mention_policy: "proactive",
        denied_audience_behavior: "omit",
        boundary_prompt: null,
        topic_tags: [PROACTIVE_OUTBOUND_CREATOR_DIRECTIVE_TOPIC_TAG],
      },
      activationPolicy: {
        scope: "allow_list",
        allowed_entity_ids: [aliceId],
        excluded_entity_ids: [],
      },
      priority: 10,
    });
    const policy = harness.createPolicy(
      policyConfig({
        enabled: true,
      }),
      ["demo"],
    );

    expect(policy.authorizationForTarget(targetSession)).toBe("creator_directive");
    expect(policy.promptContext(currentSessionId)?.targets).toEqual([
      expect.objectContaining({
        session_id: targetSession.session_id,
        authorization: "creator_directive",
      }),
    ]);
    expect(() =>
      policy.assertAuthorized({
        currentSessionId,
        targetSession,
      }),
    ).not.toThrow();
  });

  it("does not authorize routing directives that are not activation-active for the target", () => {
    const harness = setup();
    cleanups.push(() => {
      harness.db.close();
      rmSync(harness.tempDir, { recursive: true, force: true });
    });
    const currentSessionId = createSessionId();
    const creatorId = createEntityId();
    const aliceId = createEntityId();
    const bobId = createEntityId();
    const targetSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "demo",
      label: "alice",
      audience_label: "Alice",
      audience_entity_id: aliceId,
      conversation_kind: "demo",
    });
    harness.creatorDirectiveRepository.queue({
      kind: "routing_instruction",
      createdByEntityId: creatorId,
      sourceSessionId: currentSessionId,
      authorizationStreamEntryIds: [createStreamEntryId()],
      contentSourceStreamEntryIds: [createStreamEntryId()],
      subjectKind: "entity",
      subjectEntityId: aliceId,
      operationalDirective: "Creator permits autonomous outreach when the directive is active.",
      disclosurePolicy: {
        content_scope: "subject_only",
        allowed_entity_ids: [],
        excluded_entity_ids: [],
        subject_may_know: true,
        mention_policy: "proactive",
        denied_audience_behavior: "omit",
        boundary_prompt: null,
        topic_tags: [PROACTIVE_OUTBOUND_CREATOR_DIRECTIVE_TOPIC_TAG],
      },
      activationPolicy: {
        scope: "allow_list",
        allowed_entity_ids: [bobId],
        excluded_entity_ids: [],
      },
      priority: 10,
    });
    const policy = harness.createPolicy(
      policyConfig({
        enabled: true,
      }),
      ["demo"],
    );

    expect(policy.authorizationForTarget(targetSession)).toBeNull();
    expect(policy.promptContext(currentSessionId)).toBeNull();
    expect(() =>
      policy.assertAuthorized({
        currentSessionId,
        targetSession,
      }),
    ).toThrow(/not structurally authorized/);
  });

  it("omits prompt targets and rejects when the rolling outbound cap is exhausted", async () => {
    const harness = setup();
    cleanups.push(() => {
      harness.db.close();
      rmSync(harness.tempDir, { recursive: true, force: true });
    });
    const currentSessionId = createSessionId();
    const targetSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "demo",
      label: "alice",
      audience_label: "Alice",
      conversation_kind: "demo",
    });
    const writer = new StreamWriter({
      dataDir: harness.tempDir,
      sessionId: currentSessionId,
      clock: harness.clock,
    });
    try {
      await writer.append({
        kind: "tool_call",
        content: {
          call_id: "toolu_prior",
          tool_name: "tool.outbound.post",
          input: {
            target_session_id: targetSession.session_id,
            instruction: "Prior autonomous attempt.",
          },
          origin: "autonomous",
        },
      });
      await writer.append({
        kind: "tool_call",
        content: {
          call_id: "toolu_current",
          tool_name: "tool.outbound.post",
          input: {
            target_session_id: targetSession.session_id,
            instruction: "Current autonomous attempt.",
          },
          origin: "autonomous",
        },
      });
    } finally {
      writer.close();
    }
    const policy = harness.createPolicy(
      policyConfig({
        enabled: true,
        maxPostsPerWindow: 1,
        maxPostsPerTargetPerWindow: 2,
        allowByConfig: {
          sessionIds: [targetSession.session_id],
          sourceTypes: [],
        },
      }),
      ["demo"],
    );

    expect(policy.promptContext(currentSessionId)).toBeNull();
    expect(() =>
      policy.assertAuthorized({
        currentSessionId,
        targetSession,
      }),
    ).toThrow(/rolling cap/);
  });

  it("excludes the current tool call turn when checking autonomous caps", async () => {
    const harness = setup();
    cleanups.push(() => {
      harness.db.close();
      rmSync(harness.tempDir, { recursive: true, force: true });
    });
    const currentSessionId = createSessionId();
    const targetSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "demo",
      label: "alice",
      audience_label: "Alice",
      conversation_kind: "demo",
    });
    const writer = new StreamWriter({
      dataDir: harness.tempDir,
      sessionId: currentSessionId,
      clock: harness.clock,
    });
    try {
      await writer.append({
        kind: "tool_call",
        turn_id: "turn-current",
        content: {
          call_id: "toolu_current",
          tool_name: "tool.outbound.post",
          input: {
            target_session_id: targetSession.session_id,
            instruction: "Current autonomous attempt.",
          },
          origin: "autonomous",
        },
      });
    } finally {
      writer.close();
    }
    const policy = harness.createPolicy(
      policyConfig({
        enabled: true,
        maxPostsPerWindow: 1,
        maxPostsPerTargetPerWindow: 1,
        allowByConfig: {
          sessionIds: [targetSession.session_id],
          sourceTypes: [],
        },
      }),
      ["demo"],
    );

    expect(() =>
      policy.assertAuthorized({
        currentSessionId,
        targetSession,
        currentTurnId: "turn-current",
      }),
    ).not.toThrow();
  });

  it("omits and rejects a target after its per-target attempt cap is reached", async () => {
    const harness = setup();
    cleanups.push(() => {
      harness.db.close();
      rmSync(harness.tempDir, { recursive: true, force: true });
    });
    const currentSessionId = createSessionId();
    const otherWakeSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "autonomy",
      label: "autonomy",
      audience_label: "Borg",
      conversation_kind: "demo",
    });
    const targetSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "demo",
      label: "alice",
      audience_label: "Alice",
      conversation_kind: "demo",
    });
    const writer = new StreamWriter({
      dataDir: harness.tempDir,
      sessionId: otherWakeSession.session_id,
      clock: harness.clock,
    });
    try {
      await writer.append({
        kind: "tool_call",
        turn_id: "turn-prior",
        content: {
          call_id: "toolu_prior",
          tool_name: "tool.outbound.post",
          input: {
            target_session_id: targetSession.session_id,
            instruction: "Prior autonomous attempt.",
          },
          origin: "autonomous",
        },
      });
    } finally {
      writer.close();
    }
    const policy = harness.createPolicy(
      policyConfig({
        enabled: true,
        maxPostsPerWindow: 10,
        maxPostsPerTargetPerWindow: 1,
        allowByConfig: {
          sessionIds: [targetSession.session_id],
          sourceTypes: [],
        },
      }),
      ["demo"],
    );

    expect(policy.promptContext(currentSessionId)).toBeNull();
    expect(() =>
      policy.assertAuthorized({
        currentSessionId,
        targetSession,
      }),
    ).toThrow(/target rolling cap/);
  });

  it("counts waking-session attempts in a stream larger than the byte cap", () => {
    const harness = setup();
    cleanups.push(() => {
      harness.db.close();
      rmSync(harness.tempDir, { recursive: true, force: true });
    });
    harness.clock.set(10_000);
    const currentSessionId = createSessionId();
    const targetSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "demo",
      label: "alice",
      audience_label: "Alice",
      conversation_kind: "demo",
    });
    appendStreamEntries(harness.tempDir, currentSessionId, [
      ...largeFillerEntries(1_000),
      outboundAttemptEntry(9_400, targetSession.session_id, "toolu_recent_one"),
      outboundAttemptEntry(9_500, targetSession.session_id, "toolu_recent_two"),
    ]);
    const policy = harness.createPolicy(
      policyConfig({
        enabled: true,
        windowMs: 1_000,
        maxPostsPerWindow: 3,
        maxPostsPerTargetPerWindow: 10,
        allowByConfig: {
          sessionIds: [targetSession.session_id],
          sourceTypes: [],
        },
      }),
      ["demo"],
    );

    expect(policy.promptContext(currentSessionId)).toEqual(
      expect.objectContaining({
        remainingPostsInWindow: 1,
        targets: [
          expect.objectContaining({
            session_id: targetSession.session_id,
          }),
        ],
      }),
    );
    expect(() =>
      policy.assertAuthorized({
        currentSessionId,
        targetSession,
      }),
    ).not.toThrow();
  });

  it("counts per-target attempts in large active streams without hitting scan limits", () => {
    const harness = setup();
    cleanups.push(() => {
      harness.db.close();
      rmSync(harness.tempDir, { recursive: true, force: true });
    });
    harness.clock.set(10_000);
    const currentSessionId = createSessionId();
    const otherWakeSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "autonomy",
      label: "autonomy",
      audience_label: "Borg",
      conversation_kind: "demo",
    });
    const targetSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "demo",
      label: "alice",
      audience_label: "Alice",
      conversation_kind: "demo",
    });
    appendStreamEntries(harness.tempDir, otherWakeSession.session_id, [
      ...largeFillerEntries(1_000),
      outboundAttemptEntry(9_500, targetSession.session_id, "toolu_target_prior"),
    ]);
    const policy = harness.createPolicy(
      policyConfig({
        enabled: true,
        windowMs: 1_000,
        maxPostsPerWindow: 10,
        maxPostsPerTargetPerWindow: 1,
        allowByConfig: {
          sessionIds: [targetSession.session_id],
          sourceTypes: [],
        },
      }),
      ["demo"],
    );

    expectToolErrorCode(
      () =>
        policy.assertAuthorized({
          currentSessionId,
          targetSession,
        }),
      "AUTONOMOUS_OUTBOUND_TARGET_CAP_EXCEEDED",
    );
  });

  it("still fails closed when the byte cap is hit before cutoff-age entries", () => {
    const harness = setup();
    cleanups.push(() => {
      harness.db.close();
      rmSync(harness.tempDir, { recursive: true, force: true });
    });
    harness.clock.set(10_000);
    const currentSessionId = createSessionId();
    const targetSession = harness.sessionsRepository.ensure({
      session_id: createSessionId(),
      source_type: "demo",
      label: "alice",
      audience_label: "Alice",
      conversation_kind: "demo",
    });
    appendStreamEntries(harness.tempDir, currentSessionId, largeFillerEntries(9_500));
    const policy = harness.createPolicy(
      policyConfig({
        enabled: true,
        windowMs: 1_000,
        maxPostsPerWindow: 10,
        maxPostsPerTargetPerWindow: 10,
        allowByConfig: {
          sessionIds: [targetSession.session_id],
          sourceTypes: [],
        },
      }),
      ["demo"],
    );

    expectToolErrorCode(
      () =>
        policy.assertAuthorized({
          currentSessionId,
          targetSession,
        }),
      "AUTONOMOUS_OUTBOUND_CAP_SCAN_LIMIT",
    );
  });
});
