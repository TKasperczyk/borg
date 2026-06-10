import { getCommitments, getSessions, getSharedState, getState } from "../../api/client";
import type { CommitmentItem, SharedStateEntry } from "../../api/types";

export type SharedStateAudienceEntries = {
  audience: string;
  entries: SharedStateEntry[];
};

export type DirectiveSupportData = {
  audienceDiscoveryTruncated: boolean;
  commitments: CommitmentItem[];
  sharedAudiences: SharedStateAudienceEntries[];
};

export const SESSION_AUDIENCE_DISCOVERY_CAP = 1000;

function uniqueStrings(values: readonly string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];

  for (const value of values) {
    if (value.length === 0 || seen.has(value)) {
      continue;
    }
    seen.add(value);
    result.push(value);
  }

  return result;
}

export async function loadDirectiveSupportData(
  sessionId: string,
  commitments?: readonly CommitmentItem[],
): Promise<DirectiveSupportData> {
  const [sessionsResponse, commitmentsResponse, stateResponse] = await Promise.all([
    getSessions(),
    commitments === undefined
      ? getCommitments({ state: "all" })
      : Promise.resolve({ commitments: [...commitments] }),
    getState({ session: sessionId }),
  ]);
  const audienceLabels = uniqueStrings([
    "self",
    ...stateResponse.audiences,
    ...sessionsResponse.sessions.map((session) => session.audience_label),
  ]);
  const sharedAudiences = await Promise.all(
    audienceLabels.map(async (audience) => {
      const response = await getSharedState(audience);
      return { audience: response.audience, entries: response.entries };
    }),
  );

  return {
    audienceDiscoveryTruncated: sessionsResponse.sessions.length === SESSION_AUDIENCE_DISCOVERY_CAP,
    commitments: commitmentsResponse.commitments,
    sharedAudiences,
  };
}
