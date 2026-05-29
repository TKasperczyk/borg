import type { SessionRecord } from "../sessions/index.js";
import type { StreamEntry, StreamWriter } from "../stream/index.js";
import type { Clock } from "../util/clock.js";
import { describeError } from "../util/errors.js";

import type { MessageConnectorRegistry } from "./connector-registry.js";
import type { OutboundDeliveryResult, OutboundMessage } from "./types.js";

export type OutboundDeliveryOptions = {
  connectorRegistry: MessageConnectorRegistry;
  createStreamWriter: (sessionId: SessionRecord["session_id"]) => StreamWriter;
  clock: Clock;
};

export type DeliverOutboundMessageInput = {
  session: SessionRecord;
  message: OutboundMessage;
  streamWriter?: Pick<StreamWriter, "append">;
};

export class OutboundDelivery {
  constructor(private readonly options: OutboundDeliveryOptions) {}

  private appendAgentMessage(
    writer: Pick<StreamWriter, "append">,
    input: DeliverOutboundMessageInput,
  ): Promise<StreamEntry> {
    return writer.append({
      kind: "agent_msg",
      content: input.message.content,
      ...(input.message.streamInput ?? {}),
    });
  }

  private async appendDeliveryEvent(
    input: {
      session: SessionRecord;
      streamEntry: StreamEntry;
      event: string;
      content: Record<string, unknown>;
    },
    streamWriter?: Pick<StreamWriter, "append">,
  ): Promise<void> {
    const append = async (writer: Pick<StreamWriter, "append">) => {
      await writer.append({
        kind: "internal_event",
        ...(input.streamEntry.turn_id === undefined
          ? {}
          : { turn_id: input.streamEntry.turn_id }),
        content: {
          event: input.event,
          session_id: input.session.session_id,
          source_type: input.session.source_type,
          outbound_stream_entry_id: input.streamEntry.id,
          ts: this.options.clock.now(),
          ...input.content,
        },
      });
    };

    if (streamWriter !== undefined) {
      await append(streamWriter);
      return;
    }

    const writer = this.options.createStreamWriter(input.session.session_id);
    try {
      await append(writer);
    } finally {
      writer.close();
    }
  }

  async deliver(input: DeliverOutboundMessageInput): Promise<OutboundDeliveryResult> {
    let streamEntry: StreamEntry;
    if (input.streamWriter === undefined) {
      const writer = this.options.createStreamWriter(input.session.session_id);
      try {
        streamEntry = await this.appendAgentMessage(writer, input);
      } finally {
        writer.close();
      }
    } else {
      streamEntry = await this.appendAgentMessage(input.streamWriter, input);
    }
    const connector = this.options.connectorRegistry.get(input.session.source_type);

    if (connector === null) {
      await this.appendDeliveryEvent(
        {
          session: input.session,
          streamEntry,
          event: "outbound_delivery.no_connector",
          content: {
            status: "composed_not_transported",
          },
        },
        input.streamWriter,
      );

      return {
        status: "composed_not_transported",
        streamEntry,
        sourceType: input.session.source_type,
      };
    }

    try {
      const result = await connector.deliver({
        session: input.session,
        message: input.message,
        streamEntry,
      });

      return {
        status: result.status,
        streamEntry,
        sourceType: input.session.source_type,
        ...(result.externalMessageId === undefined ? {} : { externalMessageId: result.externalMessageId }),
      };
    } catch (error) {
      const formatted = describeError(error);

      await this.appendDeliveryEvent(
        {
          session: input.session,
          streamEntry,
          event: "outbound_delivery.transport_failed",
          content: {
            status: "transport_failed",
            error: formatted,
          },
        },
        input.streamWriter,
      );

      return {
        status: "transport_failed",
        streamEntry,
        sourceType: input.session.source_type,
        error: formatted,
      };
    }
  }
}
