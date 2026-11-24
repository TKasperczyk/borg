import type { StreamWriter } from "../../stream/index.js";
import type { SessionId } from "../../util/ids.js";
import {
  RECORD_CORRECTION_TOOL_NAME,
  recordCorrectionInputSchema,
  recordCorrectionResultSchema,
  type RecordCorrectionInput,
  type RecordCorrectionResult,
} from "../../memory/identity/annotations.js";
import type { RecordCorrectionService } from "../../memory/lifecycle-ops/correct-record.js";
import type { ToolDefinition } from "../dispatcher.js";
import type { RecordIdInput } from "../record-id-inputs.js";

export const RECORD_CORRECTION_ID_INPUTS = [
  { path: "target_id", kinds: ["episode", "semantic_node"] },
  { path: "evidence.*.id", kinds: ["episode", "stream_entry"] },
] as const satisfies readonly RecordIdInput[];

export function createRecordCorrectTool(options: {
  recordCorrections: Pick<RecordCorrectionService, "correct">;
  createStreamWriter: (sessionId: SessionId) => StreamWriter;
}): ToolDefinition<RecordCorrectionInput, RecordCorrectionResult> {
  return {
    name: RECORD_CORRECTION_TOOL_NAME,
    description:
      "Append a correction of up to 600 characters beside an episode or semantic node that states something false. The record's text stays unchanged. Cite existing episode or stream ids as own_read or reported, naming reported_by. An optional corrects_span must be quoted verbatim from the record.",
    menuSummary: "Append an evidenced correction to an episode or semantic node.",
    allowedOrigins: ["autonomous"],
    writeScope: "write",
    inputSchema: recordCorrectionInputSchema,
    outputSchema: recordCorrectionResultSchema,
    recordIdInputs: RECORD_CORRECTION_ID_INPUTS,
    async invoke(input, context) {
      const streamWriter = options.createStreamWriter(context.sessionId);
      try {
        return await options.recordCorrections.correct(input, {
          turnId: context.turnId,
          sessionId: context.sessionId,
          turnOrigin: context.turnOrigin ?? "autonomous",
          path: "tool",
          streamWriter,
        });
      } finally {
        streamWriter.close();
      }
    },
  };
}
