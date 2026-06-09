type ParticipantRosterRenderMember = {
  entity_id: string;
  display_name: string;
  known_relationships: readonly string[];
  audience_role: string;
  relationship_source: string | null;
  relationship_sources?: readonly string[];
};

type ParticipantRosterRenderSubject = {
  entity_id: string;
  display_name: string;
  known_relationships: readonly string[];
  relationship_source: string | null;
  relationship_sources?: readonly string[];
};

type ParticipantRosterRenderUncertain = {
  entity_id: string | null;
  display_name: string | null;
  known_relationships: readonly string[];
  reason: string;
  relationship_source: string | null;
  relationship_sources?: readonly string[];
};

export type ParticipantRosterForRendering = {
  participants: readonly ParticipantRosterRenderMember[];
  non_chat_subjects: readonly ParticipantRosterRenderSubject[];
  unknown_or_uncertain: readonly ParticipantRosterRenderUncertain[];
};

function renderRelationshipSummary(relationships: readonly string[]): string | null {
  if (relationships.length === 0) {
    return null;
  }

  return `relationships: ${relationships.join(", ")}`;
}

function renderSource(source: string | null): string | null {
  return source === null ? null : `source: ${source}`;
}

function renderParts(parts: readonly (string | null)[]): string {
  return parts.filter((part): part is string => part !== null).join("; ");
}

export function renderParticipantRoster(
  roster: ParticipantRosterForRendering | null | undefined,
): string | null {
  if (
    roster === null ||
    roster === undefined ||
    (roster.participants.length === 0 &&
      roster.non_chat_subjects.length === 0 &&
      roster.unknown_or_uncertain.length === 0)
  ) {
    return null;
  }

  const lines = ["Thread roster:"];

  for (const participant of roster.participants) {
    lines.push(
      `- ${participant.display_name} (${renderParts([
        `id: ${participant.entity_id}`,
        participant.audience_role,
        renderRelationshipSummary(participant.known_relationships),
        renderSource(participant.relationship_source),
      ])})`,
    );
  }

  if (roster.non_chat_subjects.length > 0) {
    lines.push("Non-chat subjects:");

    for (const subject of roster.non_chat_subjects) {
      lines.push(
        `- ${subject.display_name} (${renderParts([
          `id: ${subject.entity_id}`,
          renderRelationshipSummary(subject.known_relationships),
          renderSource(subject.relationship_source),
        ])})`,
      );
    }
  }

  if (roster.unknown_or_uncertain.length > 0) {
    lines.push("Unknown or uncertain:");

    for (const item of roster.unknown_or_uncertain) {
      lines.push(
        `- ${item.display_name ?? "unknown entity"} (${renderParts([
          item.entity_id === null ? null : `id: ${item.entity_id}`,
          item.reason,
          renderRelationshipSummary(item.known_relationships),
          renderSource(item.relationship_source),
        ])})`,
      );
    }
  }

  return lines.join("\n");
}
