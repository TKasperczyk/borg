import { z } from "zod";

import type { BacklogTerminalService } from "../cognition/ingestion/backlog-terminal.js";
import type { AgentDeliveryRepository } from "../cognition/ingestion/agent-deliveries.js";
import type {
  TaskEventService,
  StoredTaskEvent,
  TaskEventCatchUpRunner,
} from "../cognition/ingestion/task-events.js";
import type { BorgActivityFacade } from "../borg/public-facade.js";
import { buildInboxReplyActivityProjection } from "../memory/activity/inbox-reply-projection.js";
import type { EntityRepository } from "../memory/commitments/index.js";
import type { SessionsRepository } from "../sessions/index.js";
import type { StreamEntry } from "../stream/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { CognitionError } from "../util/errors.js";
import type { SessionId } from "../util/ids.js";
import { teamsInboxTransportMetadataSchema } from "./team-agent-turn-runner.js";

const responseSchema = z
  .object({
    action: z.literal("reply"),
    content: z.string().refine((content) => content.trim().length > 0),
  })
  .strict();

export type TeamAgentTaskEventRunnerOptions = {
  tenant: string;
  baseUrl: string;
  apiToken: string;
  timeoutMs: number;
  terminal: BacklogTerminalService;
  taskEvents: TaskEventService;
  deliveries: AgentDeliveryRepository;
  entityRepository: Pick<EntityRepository, "getSelf">;
  sessions: Pick<SessionsRepository, "get">;
  activity: Pick<BorgActivityFacade, "projectRepliedTurn">;
  tracer?: TurnTracer;
  fetchFn?: typeof fetch;
};

export class TeamAgentTaskEventRunner implements TaskEventCatchUpRunner {
  constructor(private readonly options: TeamAgentTaskEventRunnerOptions) {}

  async run(input: { sessionId: SessionId; taskEvent: StoredTaskEvent }): Promise<void> {
    const existing = this.options.taskEvents.findTerminal(input.sessionId, input.taskEvent);
    if (existing !== null) {
      await this.finish(existing);
      return;
    }
    const session = this.options.sessions.get(input.sessionId);
    if (session?.source_type !== "teams_inbox" || session.source_external_id === null) {
      throw new CognitionError("Task event session has no Teams conversation", {
        code: "TASK_EVENT_SESSION_INVALID",
      });
    }
    const { event, entry } = input.taskEvent;
    const requester = this.requester(input.sessionId, input.taskEvent);
    const response = await (this.options.fetchFn ?? fetch)(
      new URL("/v1/chat/task-result", this.options.baseUrl),
      {
        method: "POST",
        redirect: "error",
        signal: AbortSignal.timeout(this.options.timeoutMs),
        headers: {
          authorization: `Bearer ${this.options.apiToken}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: this.options.tenant,
          sidecar_session_id: input.sessionId,
          conversation: {
            external_id: session.source_external_id,
            type:
              session.conversation_kind === "dm"
                ? "personal"
                : session.conversation_kind === "thread"
                  ? "groupChat"
                  : "channel",
            name: session.label,
          },
          event: {
            event_id: event.event_id,
            event_entry_id: entry.id,
            task_id: event.task_id,
            task_version: event.task_version,
            kind: event.kind,
            occurred_at: event.occurred_at,
            outcome: event.outcome,
          },
          requester,
        }),
      },
    );
    const fallback = response.status >= 400 && response.status < 500;
    let content: string;
    if (fallback) {
      content = `Task ${event.task_id} finished: ${event.outcome.summary}`;
    } else {
      if (response.status !== 200) {
        throw new CognitionError(`Team Agent task-result failed with HTTP ${response.status}`, {
          code: "TASK_EVENT_HTTP_FAILED",
        });
      }
      const raw: unknown = await response.json();
      const parsed = responseSchema.safeParse(raw);
      if (!parsed.success) {
        throw new CognitionError("Team Agent task-result returned an invalid response", {
          code: "TASK_EVENT_RESPONSE_INVALID",
        });
      }
      content = parsed.data.content;
    }
    const terminal = await this.options.terminal.appendTaskEventTerminal({
      sessionId: input.sessionId,
      responseTo: {
        kind: "task_event",
        event_id: event.event_id,
        event_entry_id: entry.id,
        task_id: event.task_id,
        task_version: event.task_version,
      },
      content,
      audience: session.audience_entity_id ?? session.audience_label,
    });
    this.options.tracer?.emit("task_event.terminal_committed", {
      turnId: `task-event:${entry.id}`,
      session_id: input.sessionId,
      terminal_entry_id: terminal.id,
      event_entry_id: entry.id,
      fallback,
      http_status: response.status,
    });
    await this.finish(terminal);
  }

  async reconcile(sessionId: SessionId): Promise<void> {
    for (const terminal of this.options.taskEvents.listTerminals(sessionId)) {
      if (!this.options.deliveries.hasTerminal(sessionId, terminal.id)) await this.finish(terminal);
    }
  }

  private requester(
    sessionId: SessionId,
    { event }: StoredTaskEvent,
  ): {
    external_id: string;
    display_name: string;
  } | null {
    let requester: { external_id: string; display_name: string } | null = null;
    // Source handles are supplied by the task creator. Never infer a requester
    // from task prose, the latest turn, or a different conversation.
    for (const id of event.origin.source_entry_ids) {
      const source = this.options.taskEvents.readEntry(sessionId, id);
      if (source?.kind !== "user_msg") return null;
      const parsed = teamsInboxTransportMetadataSchema.safeParse(source.metadata?.teams_inbox);
      if (!parsed.success || parsed.data.sender.bot) return null;
      const sender = parsed.data.sender;
      if (requester !== null && requester.external_id !== sender.external_id) return null;
      requester = { external_id: sender.external_id, display_name: sender.display_name };
    }
    return requester;
  }

  private async finish(terminal: StreamEntry): Promise<void> {
    if (terminal.response_to?.kind !== "task_event" || typeof terminal.content !== "string") {
      throw new CognitionError("Invalid task event terminal", {
        code: "TASK_EVENT_TERMINAL_INVALID",
      });
    }
    if (this.options.deliveries.hasTerminal(terminal.session_id, terminal.id)) return;
    await this.options.terminal.ingestTaskEventTerminal(terminal);
    try {
      const projection = buildInboxReplyActivityProjection({
        session: this.options.sessions.get(terminal.session_id),
        selfEntityId: this.options.entityRepository.getSelf()?.id ?? null,
        terminal: {
          id: terminal.id,
          sessionId: terminal.session_id,
          timestamp: terminal.timestamp,
        },
        senderEntityIds: [],
      });
      if (projection.kind === "project") {
        this.options.activity.projectRepliedTurn(projection.input);
      } else {
        console.warn("memory-sidecar: task reply activity not recorded", {
          tenant: this.options.tenant,
          reason: projection.reason,
        });
      }
    } catch (error) {
      console.warn("memory-sidecar: task reply activity not recorded", {
        tenant: this.options.tenant,
        reason: "projection_failed",
        error_name: error instanceof Error ? error.name : typeof error,
      });
    }
    this.options.deliveries.create({
      sessionId: terminal.session_id,
      terminalEntryId: terminal.id,
      taskId: terminal.response_to.task_id,
      content: terminal.content,
      createdAt: terminal.timestamp,
    });
    this.options.tracer?.emit("task_event.delivery_created", {
      turnId: `task-event:${terminal.response_to.event_entry_id}`,
      session_id: terminal.session_id,
      terminal_entry_id: terminal.id,
    });
  }
}
