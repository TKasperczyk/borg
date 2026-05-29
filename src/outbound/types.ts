import type { SessionRecord, SessionSourceType } from "../sessions/index.js";
import type { StreamEntry, StreamEntryInput } from "../stream/index.js";

export type OutboundMessage = {
  content: string;
  streamInput?: Omit<StreamEntryInput, "kind" | "content">;
};

export type OutboundConnectorDeliverInput = {
  session: SessionRecord;
  message: OutboundMessage;
  streamEntry: StreamEntry;
};

export type OutboundConnectorDeliverResult = {
  status: "transported";
  externalMessageId?: string;
};

export type MessageConnector = {
  readonly sourceType: SessionSourceType;
  deliver(input: OutboundConnectorDeliverInput): Promise<OutboundConnectorDeliverResult>;
};

export type OutboundDeliveryStatus =
  | "transported"
  | "composed_not_transported"
  | "transport_failed";

export type OutboundDeliveryResult = {
  status: OutboundDeliveryStatus;
  streamEntry: StreamEntry;
  sourceType: SessionSourceType;
  externalMessageId?: string;
  error?: string;
};

export type OutboundDeliveryReceipt = {
  status: OutboundDeliveryStatus;
  streamEntryId: StreamEntry["id"];
  sourceType: SessionSourceType;
  externalMessageId?: string;
  error?: string;
};
