import type { MessageConnector, OutboundConnectorDeliverResult } from "./types.js";

export class DemoMessageConnector implements MessageConnector {
  readonly sourceType = "demo" as const;

  async deliver(): Promise<OutboundConnectorDeliverResult> {
    return {
      status: "transported",
    };
  }
}
