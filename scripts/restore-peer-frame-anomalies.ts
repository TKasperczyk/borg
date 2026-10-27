/** Run against a stopped, migrated bank. JSONL annotations are the recovery journal. */
import { z } from "zod";
import { loadConfig } from "../src/config/index.js";
import { EntityRepository } from "../src/memory/commitments/repository.js";
import {
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromMetadata,
  unknownMemoryDisclosureLabel,
} from "../src/memory/common/disclosure-label.js";
import { memoryDisclosurePayloadFields } from "../src/memory/common/disclosure-serializers.js";
import { SourceStreamAudienceDisclosureResolver } from "../src/memory/common/source-stream-audience-disclosure.js";
import { SessionsRepository } from "../src/sessions/repository.js";
import {
  StreamEntryIndexRepository,
  StreamReader,
  StreamWriter,
  collectInactiveStreamEntryRefs,
  isQuarantinedUserEntryMarker,
  streamEntryIsActive,
  type StreamEntry,
} from "../src/stream/index.js";
import {
  FRAME_ANOMALY_ANNOTATION_EVENT,
  frameAnomalyAnnotationEventSchema,
} from "../src/stream/frame-anomaly.js";
import { quarantinedUserEntrySourceIds } from "../src/stream/turn-status.js";
import { parseSessionId } from "../src/util/ids.js";
import { openDataRepairDatabase, repairArgs, runRepairCli } from "./audited-data-repair.js";

const legacyClassificationSchema = z.object({
  kind: z.string(),
  confidence: z.number().min(0).max(1),
  rationale: z.string(),
  turn_id: z.string().optional(),
});

export type FrameAnomalyRestorationRow = {
  session_id: string;
  source_type: string;
  entry_id: string;
  quarantine_marker_ids: string[];
  status: "restore" | "kept_inactive" | "invalid_annotation_source";
};

export async function restorePeerFrameAnomalies(options: {
  dataDir: string;
  apply?: boolean;
  liveStoreConfirmed?: string;
}): Promise<FrameAnomalyRestorationRow[]> {
  const config = loadConfig({ dataDir: options.dataDir, env: {} });
  const db = openDataRepairDatabase(
    options.dataDir,
    options.apply !== true,
    options.liveStoreConfirmed,
  );
  const result: FrameAnomalyRestorationRow[] = [];
  try {
    const index = new StreamEntryIndexRepository({ db, dataDir: options.dataDir });
    const disclosure = new SourceStreamAudienceDisclosureResolver({
      dataDir: options.dataDir,
      entryIndex: index,
      sessionsRepository: new SessionsRepository({ db }),
      entityRepository: new EntityRepository({ db }),
    });
    // No session-name heuristics, audience gate, or bounded session-list cut.
    const sessions = db
      .prepare("SELECT session_id, source_type FROM sessions ORDER BY session_id")
      .all() as { session_id: string; source_type: string }[];
    for (const session of sessions) {
      if (!config.frameAnomaly.peerChannelSourceTypes.includes(session.source_type)) continue;
      const sessionId = parseSessionId(session.session_id);
      const entries: StreamEntry[] = [];
      for await (const entry of new StreamReader({ dataDir: options.dataDir, sessionId }).iterate())
        entries.push(entry);
      const inactive = collectInactiveStreamEntryRefs(entries);
      const otherInvalidations = collectInactiveStreamEntryRefs(
        entries.filter((entry) => !isQuarantinedUserEntryMarker(entry)),
      );
      const byId = new Map(entries.map((entry) => [entry.id as string, entry]));
      const markersBySource = new Map<string, StreamEntry[]>();
      for (const marker of entries) {
        for (const id of quarantinedUserEntrySourceIds(marker)) {
          const markers = markersBySource.get(id) ?? [];
          markers.push(marker);
          markersBySource.set(id, markers);
        }
      }
      const annotations: z.infer<typeof frameAnomalyAnnotationEventSchema>[] = [];
      for (const [id, markers] of markersBySource) {
        const source = byId.get(id);
        if (
          source === undefined ||
          source.session_id !== sessionId ||
          streamEntryIsActive(source, inactive)
        )
          continue;
        const row: FrameAnomalyRestorationRow = {
          ...session,
          entry_id: id,
          quarantine_marker_ids: markers.map((marker) => marker.id),
          status: "restore",
        };
        result.push(row);
        // Explicit aborts, aborted turn membership and any non-quarantine
        // invalidation retain priority. Restoring a strike never revives them.
        if (!streamEntryIsActive(source, otherInvalidations)) {
          row.status = "kept_inactive";
          continue;
        }
        const classifications = markers.map((marker) =>
          legacyClassificationSchema.safeParse(marker.content),
        );
        if (classifications.some((parsed) => !parsed.success)) {
          row.status = "invalid_annotation_source";
          continue;
        }
        const label =
          memoryDisclosureLabelFromMetadata(source.metadata?.disclosure_label) ??
          disclosure.resolveStreamEntryLabel(source);
        for (const marker of markers) {
          if (
            source.frame_anomalies?.some((annotation) =>
              annotation.restores_quarantine_marker_ids.includes(marker.id),
            )
          )
            continue;
          const classification = legacyClassificationSchema.parse(marker.content);
          const rationaleDisclosure = combineMemoryDisclosureLabels(
            quarantinedUserEntrySourceIds(marker).map((id) => {
              const citedSource = byId.get(id);
              return citedSource === undefined
                ? unknownMemoryDisclosureLabel()
                : (memoryDisclosureLabelFromMetadata(citedSource.metadata?.disclosure_label) ??
                    disclosure.resolveStreamEntryLabel(citedSource));
            }),
          );
          annotations.push(
            frameAnomalyAnnotationEventSchema.parse({
              event: FRAME_ANOMALY_ANNOTATION_EVENT,
              version: 1,
              source_stream_entry_id: source.id,
              source_session_id: sessionId,
              source_type: session.source_type,
              source_stream_entry_ids: quarantinedUserEntrySourceIds(marker),
              kind: classification.kind,
              confidence: classification.confidence,
              rationale: classification.rationale,
              classifier_model: null, // Legacy markers did not record the model; do not invent it.
              classifier_turn_id: classification.turn_id ?? marker.turn_id ?? null,
              classifier_source_event_id: marker.id,
              restores_quarantine_marker_ids: [marker.id],
              audience: source.audience ?? null,
              sender_entity_id: source.sender_entity_id,
              source_disclosure_label: memoryDisclosurePayloadFields(label).disclosure_label,
              ...memoryDisclosurePayloadFields(rationaleDisclosure),
            }),
          );
        }
      }
      if (options.apply === true) {
        const writer = new StreamWriter({ dataDir: options.dataDir, sessionId, entryIndex: index });
        try {
          if (annotations.length > 0)
            await writer.appendMany(
              annotations.map((content) => ({ kind: "internal_event", content })),
            );
          // Also repairs a crash after an annotation was fsync'd but before its
          // index projection committed. Re-running appends no duplicate events.
          // Replay applies only previously unapplied marker/source reversals;
          // old annotations never override a later index-only deactivation.
          await index.backfillSession(sessionId);
        } finally {
          writer.close();
        }
      }
    }
    return result;
  } finally {
    db.close();
  }
}

runRepairCli(import.meta.url, async () => {
  const args = repairArgs(process.argv.slice(2));
  const rows = await restorePeerFrameAnomalies(args);
  console.table(
    rows.map((row) => ({ ...row, quarantine_marker_ids: row.quarantine_marker_ids.join(",") })),
  );
  console.log(
    args.apply
      ? "Applied replayable peer annotations."
      : "Dry run: no files changed. Pass --apply to append restoration annotations.",
  );
});
