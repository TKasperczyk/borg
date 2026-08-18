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

// A connector reports failure by throwing, not by returning a status, so this union has exactly one
// member and always will. Reading that as "the other outcomes have never occurred" is wrong twice
// over: there are no other members to occur, and the states they would name are recorded anyway --
// `OutboundDelivery.deliver` turns a throw into `transport_failed` and a missing connector
// into `composed_not_transported`. Both of those also append an `outbound_delivery.*` stream event,
// while the success path appends none; the stream, not this field, is where delivery history lives.
//
// `transported` means the connector accepted the message, and nothing past that hop is observed
// here: whether the destination stored it, routed it, or dropped it is outside anything borg
// records. A delivery that failed after this point looks identical to one that arrived.
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
