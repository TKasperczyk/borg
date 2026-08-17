import { describe, expect, it } from "vitest";

import {
  FINALIZER_TOOL_TRANSCRIPT_MAX_BYTES,
  FinalizerToolTranscriptCollector,
  parseFinalizerToolTranscript,
  prepareFinalizerToolTranscript,
} from "./finalizer-tool-transcript.js";
import { fingerprintCanonicalRequest } from "./request-fingerprint.js";

const requestBinding = fingerprintCanonicalRequest({ messages: [], model: "fake" });

function observeSuccess(
  collector: FinalizerToolTranscriptCollector,
  output: unknown = { found: true },
): void {
  collector.observe({
    ordinal: 1,
    iteration: 1,
    batchPosition: 1,
    callId: "toolu_transcript",
    toolName: "tool.test.read",
    rawArguments: { query: "🌌", limit: 3 },
    disposition: "dispatched",
    result: { ok: true, output },
    durationMs: 12,
  });
}

describe("FinalizerToolTranscriptCollector", () => {
  it("produces a versioned exact transcript with canonical argument identity", () => {
    const collector = new FinalizerToolTranscriptCollector();
    observeSuccess(collector);
    const snapshot = collector.finish({
      requestBinding,
      expectedEventCount: 1,
      sourceCompleted: true,
    });
    const prepared = prepareFinalizerToolTranscript({ snapshot });

    expect(snapshot.transcript).toMatchObject({
      schema_version: 1,
      request_binding: requestBinding,
      complete: true,
      event_count: 1,
      dispatched_count: 1,
      events: [
        {
          raw_arguments: { query: "🌌", limit: 3 },
          arguments_fingerprint: expect.objectContaining({ canonicalSha256: expect.any(String) }),
          result: { ok: true, output: { found: true } },
        },
      ],
    });
    expect(prepared.manifest).toMatchObject({
      status: "complete",
      replay_eligible: true,
      event_count: 1,
      dispatched_count: 1,
      relative_path: `finalizer-tool-transcripts/${prepared.manifest.canonical_sha256}`,
    });
    expect(prepared.pendingSidecar).not.toBeNull();
    expect(
      parseFinalizerToolTranscript(
        JSON.parse(Buffer.from(prepared.pendingSidecar!.bytes).toString("utf8")) as unknown,
      ),
    ).toEqual(snapshot.transcript);
  });

  it("marks observation and source completeness failures without throwing", () => {
    const collector = new FinalizerToolTranscriptCollector();
    expect(() =>
      collector.observe({
        ordinal: 1,
        iteration: 1,
        batchPosition: 1,
        callId: "toolu_bad_json",
        toolName: "tool.test.read",
        rawArguments: { invalid: undefined },
        disposition: "dispatched",
        result: { ok: false, error: "validation failed" },
        durationMs: 0,
      }),
    ).not.toThrow();
    const snapshot = collector.finish({
      requestBinding,
      expectedEventCount: null,
      sourceCompleted: false,
    });
    const prepared = prepareFinalizerToolTranscript({ snapshot });

    expect(snapshot.transcript).toMatchObject({
      complete: false,
      event_count: 1,
      incomplete_reasons: ["observation_failed", "source_incomplete"],
    });
    expect(parseFinalizerToolTranscript(snapshot.transcript)).toEqual(snapshot.transcript);
    expect(prepared).toMatchObject({
      manifest: { status: "incomplete", replay_eligible: false },
      pendingSidecar: null,
    });
  });

  it("rejects structurally inconsistent persisted transcript counts", () => {
    const collector = new FinalizerToolTranscriptCollector();
    observeSuccess(collector);
    const transcript = collector.finish({
      requestBinding,
      expectedEventCount: 1,
      sourceCompleted: true,
    }).transcript;

    expect(() => parseFinalizerToolTranscript({ ...transcript, event_count: 2 })).toThrow(
      /event count is inconsistent/,
    );
    expect(() =>
      parseFinalizerToolTranscript({
        ...transcript,
        events: [{ ...transcript.events[0]!, ordinal: 2 }],
      }),
    ).toThrow(/ordinals are inconsistent/);
  });

  it("never truncates a transcript that exceeds the 8 MiB hard cap", () => {
    const collector = new FinalizerToolTranscriptCollector();
    const payload = "x".repeat(FINALIZER_TOOL_TRANSCRIPT_MAX_BYTES);
    observeSuccess(collector, { payload });
    const snapshot = collector.finish({
      requestBinding,
      expectedEventCount: 1,
      sourceCompleted: true,
    });
    const prepared = prepareFinalizerToolTranscript({ snapshot });

    expect(prepared.manifest).toMatchObject({
      status: "omitted_oversized",
      replay_eligible: false,
      event_count: 1,
      relative_path: null,
    });
    expect(prepared.manifest.payload_bytes).toBeGreaterThan(FINALIZER_TOOL_TRANSCRIPT_MAX_BYTES);
    expect(prepared.pendingSidecar).toBeNull();
    const capturedResult = snapshot.transcript.events[0]?.result;
    if (capturedResult?.ok !== true) throw new Error("expected captured success payload");
    expect((capturedResult.output as { payload: string }).payload).toHaveLength(payload.length);
  });

  it("represents a complete zero-event loop without a sidecar", () => {
    const snapshot = new FinalizerToolTranscriptCollector().finish({
      requestBinding,
      expectedEventCount: 0,
      sourceCompleted: true,
    });
    expect(prepareFinalizerToolTranscript({ snapshot })).toEqual({
      manifest: {
        status: "none",
        event_count: 0,
        dispatched_count: 0,
        payload_bytes: 0,
        canonical_sha256: null,
        relative_path: null,
        request_binding: requestBinding,
        replay_eligible: true,
        incomplete_reasons: [],
      },
      pendingSidecar: null,
    });
  });
});
