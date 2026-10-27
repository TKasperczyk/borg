import { z } from "zod";

import type {
  AppendBacklogTerminalResult,
  BacklogTerminalService,
  ChatResponseCatchUpRunInput,
  ChatResponseCatchUpRunner,
} from "../cognition/ingestion/index.js";
import type { BorgActivityFacade } from "../borg/public-facade.js";
import { buildInboxReplyActivityProjection } from "../memory/activity/inbox-reply-projection.js";
import type { EntityRepository } from "../memory/commitments/index.js";
import type { SessionsRepository } from "../sessions/index.js";
import type { Clock } from "../util/clock.js";
import { createIdHelpers, type EntityId } from "../util/ids.js";
import { jsonValueSchema } from "../util/json-value.js";
import {
  startObservatoryStage,
  endObservatoryStage,
  reportObservatoryObserverError,
  type ObservatoryObserver,
} from "./observatory/emitter.js";

const observatoryTurnIds = createIdHelpers<"ObservatoryTurnId">("turn");

export const teamsInboxTransportMetadataSchema = z
  .object({
    thread_id: z.string().min(1),
    sender: z
      .object({
        external_id: z.string().min(1),
        display_name: z.string().min(1),
        bot: z.boolean(),
      })
      .strict(),
    mentioned: z.boolean(),
    quotes_bot: z.boolean(),
  })
  .strict();


const storedMetadataSchema = z
  .record(z.string(), jsonValueSchema)
  .and(z.object({ teams_inbox: teamsInboxTransportMetadataSchema }));

const teamAgentObserveResponseSchema = z.discriminatedUnion("action", [
  z
    .object({
      action: z.literal("reply"),
      content: z.string().min(1),
      reason: z.string().optional(),
    })
    .strict(),
  z
    .object({
      action: z.literal("silent"),
      content: z.string().optional(),
      reason: z.string().optional(),
    })
    .strict(),
]);

export type TeamAgentTurnRunnerOptions = {
  tenant: string;
  baseUrl: string;
  apiToken: string;
  timeoutMs: number;
  staleMs: number;
  terminal: BacklogTerminalService;
  entityRepository: Pick<EntityRepository, "get" | "getSelf">;
  sessions: Pick<SessionsRepository, "get">;
  activity: Pick<BorgActivityFacade, "projectRepliedTurn">;
  clock: Clock;
  onGenerating?: (input: {
    sessionId: ChatResponseCatchUpRunInput["sessionId"];
    entryIds: ChatResponseCatchUpRunInput["inboundBatch"]["entryIds"];
  }) => void;
  fetchFn?: typeof fetch;
  observatory?: ObservatoryObserver;
  onAttempt?: (turnId: string, input: ChatResponseCatchUpRunInput) => void;
};

export class TeamAgentTurnRunner implements ChatResponseCatchUpRunner {
  private readonly fetchFn: typeof fetch;
  private readonly attempts = new Map<string, { count: number; expiresAt: number }>();

  constructor(private readonly options: TeamAgentTurnRunnerOptions) {
    this.fetchFn = options.fetchFn ?? fetch;
  }

  async run(input: ChatResponseCatchUpRunInput): Promise<void> {
    const turnId = observatoryTurnIds.create();
    let attempt = 1;
    try {
      input.onAttempt?.(turnId);
      this.options.onAttempt?.(turnId, input);
      if (this.options.observatory !== undefined) {
        const now = this.options.clock.now();
        for (const [key, value] of this.attempts) {
          if (value.expiresAt > now) break;
          this.attempts.delete(key);
        }
        const key = JSON.stringify([input.sessionId, input.inboundBatch.entryIds]);
        attempt = (this.attempts.get(key)?.count ?? 0) + 1;
        this.attempts.delete(key);
        this.attempts.set(key, { count: attempt, expiresAt: now + 60 * 60_000 });
        while (this.attempts.size > 5_000) {
          const oldest = this.attempts.keys().next().value;
          if (oldest === undefined) break;
          this.attempts.delete(oldest);
        }
      }
    } catch (error) {
      reportObservatoryObserverError("runner.attempt", error);
    }
    const common = {
      tenant: this.options.tenant,
      kind: "inbox.runner.call",
      turn_id: turnId,
      refs: { sidecar_session_id: input.sessionId, entry_ids: input.inboundBatch.entryIds },
    };
    const skipped = (sealedReason?: string): void => {
      const stage = startObservatoryStage(this.options.observatory, {
        ...common,
        data: { runs_full_turn: false, message_count: input.inboundBatch.entryIds.length, attempt },
      });
      endObservatoryStage(stage, {
        status: "skipped",
        data: {
          sealed_reason: sealedReason,
          terminal_kind: sealedReason === undefined ? undefined : "agent_observed",
        },
      });
    };
    const existing = this.options.terminal.findTerminalCoveringEntry({
      sessionId: input.sessionId,
      entryId: input.inboundBatch.entryIds[input.inboundBatch.entryIds.length - 1]!,
    });
    if (existing.status === "found") {
      skipped();
      return;
    }

    const batch = await this.options.terminal.hydrateBacklogBatch({
      sessionId: input.sessionId,
      entryIds: input.inboundBatch.entryIds,
      throughCursorInclusive: input.inboundBatch.throughCursorInclusive,
    });
    const metadata = batch.sourceEntries.map((entry) =>
      storedMetadataSchema.safeParse(entry.metadata),
    );
    if (metadata.some((parsed) => !parsed.success)) {
      await this.options.terminal.sealBacklogPrefix({
        sessionId: input.sessionId,
        sourceEntryIds: input.inboundBatch.entryIds,
        reason: "Legacy inbox backlog sealed because transport metadata is unavailable",
      });
      skipped("transport_metadata_unavailable");
      return;
    }
    const transportMetadata = metadata.map((parsed) => {
      if (!parsed.success) {
        throw new Error("validated Teams inbox metadata unexpectedly became invalid");
      }
      return parsed.data;
    });

    const staleBefore = this.options.clock.now() - this.options.staleMs;
    if (
      batch.sourceEntries.every((entry) => (entry.observed_at ?? entry.timestamp) < staleBefore)
    ) {
      await this.options.terminal.sealStaleBacklog({
        sessionId: input.sessionId,
        staleBefore,
      });
      skipped("stale");
      return;
    }

    const messages = batch.sourceEntries.map((entry, index) => {
      const entryMetadata = transportMetadata[index]!;
      const senderEntity =
        entry.sender_entity_id === null || entry.sender_entity_id === undefined
          ? null
          : this.options.entityRepository.get(entry.sender_entity_id);
      return {
        entry_id: entry.id,
        text: entry.content as string,
        sender: {
          ...entryMetadata.teams_inbox.sender,
          display_name:
            entryMetadata.teams_inbox.sender.display_name ||
            senderEntity?.canonical_name ||
            entryMetadata.teams_inbox.sender.external_id,
        },
        observed_at: new Date(entry.observed_at ?? entry.timestamp).toISOString(),
        mentioned: entryMetadata.teams_inbox.mentioned,
        quotes_bot: entryMetadata.teams_inbox.quotes_bot,
      };
    });
    const first = batch.sourceEntries[0]!;
    const firstMetadata = transportMetadata[0]!.teams_inbox;
    const conversation = first.conversation;
    const externalId = first.source_message_key?.source_external_id;
    if (conversation === undefined || externalId === undefined) {
      throw new Error("teams inbox entry is missing durable conversation identity");
    }

    const requestBody = {
      model: this.options.tenant,
      source: "inbox",
      thread_id: firstMetadata.thread_id,
      sidecar_session_id: input.sessionId,
      conversation: { ...conversation, external_id: externalId },
      messages,
    };
    const runsFullTurn =
      conversation.type === "personal" ||
      transportMetadata.some(
        (entryMetadata) =>
          entryMetadata.teams_inbox.mentioned || entryMetadata.teams_inbox.quotes_bot,
      );
    if (runsFullTurn) {
      try {
        this.options.onGenerating?.({
          sessionId: input.sessionId,
          entryIds: input.inboundBatch.entryIds,
        });
      } catch (error) {
        console.error("Team Agent inbox generating observer failed", error);
      }
    }

    const stage = startObservatoryStage(this.options.observatory, {
      ...common,
      refs: { ...common.refs, thread_id: firstMetadata.thread_id },
      conversation: requestBody.conversation,
      data: { runs_full_turn: runsFullTurn, message_count: messages.length, attempt },
    });
    let httpStatus: number | undefined;
    try {
      const response = await this.fetchFn(new URL("/v1/chat/observe", this.options.baseUrl), {
        method: "POST",
        redirect: "error",
        signal: AbortSignal.timeout(this.options.timeoutMs),
        headers: {
          authorization: `Bearer ${this.options.apiToken}`,
          "content-type": "application/json",
          "X-Observatory-Turn-Id": turnId,
        },
        body: JSON.stringify(requestBody),
      });
      httpStatus = response.status;

      if (response.status >= 400 && response.status < 500) {
        endObservatoryStage(stage, {
          status: "error",
          data: { http_status: httpStatus, action: "silent", terminal_kind: "agent_observed" },
        });
        await this.options.terminal.appendBacklogTerminal({
          sessionId: input.sessionId,
          sourceEntryIds: input.inboundBatch.entryIds,
          terminal: {
            kind: "agent_observed",
            reason: `Team Agent rejected inbox batch with HTTP ${response.status}`,
          },
        });
        return;
      }
      if (!response.ok) {
        throw new Error(`Team Agent observe failed with HTTP ${response.status}`);
      }

      let raw: unknown;
      try {
        raw = JSON.parse(await response.text()) as unknown;
      } catch (error) {
        throw new Error("Team Agent observe returned malformed JSON", { cause: error });
      }
      const parsed = teamAgentObserveResponseSchema.safeParse(raw);
      if (!parsed.success) {
        throw new Error("Team Agent observe returned an invalid 2xx response", {
          cause: parsed.error,
        });
      }

      const terminal =
        parsed.data.action === "reply"
          ? ({ kind: "agent_msg", content: parsed.data.content } as const)
          : ({
              kind: "agent_observed",
              reason: parsed.data.reason ?? "Team Agent stayed silent",
            } as const);
      endObservatoryStage(stage, {
        data: { http_status: httpStatus, action: parsed.data.action, terminal_kind: terminal.kind },
      });
      const appended = await this.options.terminal.appendBacklogTerminal({
        sessionId: input.sessionId,
        sourceEntryIds: input.inboundBatch.entryIds,
        terminal,
      });
      if (terminal.kind === "agent_msg") {
        this.projectReplyActivity(input.sessionId, appended);
      }
    } catch (error) {
      endObservatoryStage(stage, { status: "error", error, data: { http_status: httpStatus } });
      throw error;
    }
  }

  // The inbox path bypasses /memory/append-turn, which is where the sidecar otherwise records
  // the memory owner's own reply as a borg_replied activity event. Without this projection the
  // owner's Teams replies never reached recent_activity or the recall planner's owner rows.
  private projectReplyActivity(
    sessionId: ChatResponseCatchUpRunInput["sessionId"],
    appended: AppendBacklogTerminalResult,
  ): void {
    const warn = (reason: string, error?: unknown) =>
      console.warn("memory-sidecar: inbox reply activity not recorded", {
        tenant: this.options.tenant,
        reason,
        ...(error === undefined
          ? {}
          : { error_name: error instanceof Error ? error.name : typeof error }),
      });
    try {
      const projection = buildInboxReplyActivityProjection({
        session: this.options.sessions.get(sessionId),
        selfEntityId: this.options.entityRepository.getSelf()?.id ?? null,
        terminal: {
          id: appended.terminalEntry.id,
          sessionId: appended.terminalEntry.session_id,
          timestamp: appended.terminalEntry.timestamp,
        },
        senderEntityIds: appended.sourceEntries.flatMap((entry) =>
          entry.sender_entity_id === null || entry.sender_entity_id === undefined
            ? []
            : [entry.sender_entity_id as EntityId],
        ),
      });
      if (projection.kind === "skip") {
        warn(projection.reason);
        return;
      }
      this.options.activity.projectRepliedTurn(projection.input);
    } catch (error) {
      warn("projection_failed", error);
    }
  }
}
