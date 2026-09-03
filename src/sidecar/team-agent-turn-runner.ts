import { z } from "zod";

import type {
  BacklogTerminalService,
  ChatResponseCatchUpRunInput,
  ChatResponseCatchUpRunner,
} from "../cognition/ingestion/index.js";
import type { EntityRepository } from "../memory/commitments/index.js";
import type { Clock } from "../util/clock.js";
import { jsonValueSchema } from "../util/json-value.js";

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

export type TeamsInboxTransportMetadata = z.infer<typeof teamsInboxTransportMetadataSchema>;

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
  entityRepository: Pick<EntityRepository, "get">;
  clock: Clock;
  fetchFn?: typeof fetch;
};

export class TeamAgentTurnRunner implements ChatResponseCatchUpRunner {
  private readonly fetchFn: typeof fetch;

  constructor(private readonly options: TeamAgentTurnRunnerOptions) {
    this.fetchFn = options.fetchFn ?? fetch;
  }

  async run(input: ChatResponseCatchUpRunInput): Promise<void> {
    const existing = this.options.terminal.findTerminalCoveringEntry({
      sessionId: input.sessionId,
      entryId: input.inboundBatch.entryIds[input.inboundBatch.entryIds.length - 1]!,
    });
    if (existing.status === "found") {
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

    const response = await this.fetchFn(new URL("/v1/chat/observe", this.options.baseUrl), {
      method: "POST",
      redirect: "error",
      signal: AbortSignal.timeout(this.options.timeoutMs),
      headers: {
        authorization: `Bearer ${this.options.apiToken}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: this.options.tenant,
        source: "inbox",
        thread_id: firstMetadata.thread_id,
        sidecar_session_id: input.sessionId,
        conversation: { ...conversation, external_id: externalId },
        messages,
      }),
    });

    if (response.status >= 400 && response.status < 500) {
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
    await this.options.terminal.appendBacklogTerminal({
      sessionId: input.sessionId,
      sourceEntryIds: input.inboundBatch.entryIds,
      terminal,
    });
  }
}
