import { createEpisodeId, createSessionId } from "../../../src/util/ids.js";
import { createEpisodeFixture } from "../../../src/offline/test-support.js";
import type { ReplayScenario } from "../scenario.js";
import {
  enqueueRelationalGuardFailureWhenValidatorAbsent,
  evidenceRef,
  lowerIncludesNone,
  makeRelationalClaim,
} from "../scenario.js";

const PRIOR_SESSION_ID = createSessionId();
const ITALKI_EPISODE_ID = createEpisodeId();
const USER_MESSAGE = "What was that iTalki thing again?";

const scenario: ReplayScenario = {
  id: "11-cross-session-italki",
  failureClass: "Cross-session iTalki overconfidence",
  description: "Prior-session iTalki evidence is used without a visible prior-session disclosure.",
  async seed({ borg, deps, clock }) {
    const priorEntry = await borg.stream.append(
      {
        kind: "user_msg",
        content: "My iTalki tutor recommended X as a conversation drill.",
      },
      { session: PRIOR_SESSION_ID },
    );
    const embedding = await deps.embeddingClient.embed(USER_MESSAGE);

    await deps.episodicRepository.insert(
      createEpisodeFixture({
        id: ITALKI_EPISODE_ID,
        title: "iTalki tutor recommendation",
        narrative: "The user said their iTalki tutor recommended X as a conversation drill.",
        participants: ["user", "iTalki tutor"],
        source_stream_ids: [priorEntry.id],
        embedding,
        created_at: clock.now() - 10_000,
        updated_at: clock.now() - 10_000,
      }),
    );
  },
  userMessage: USER_MESSAGE,
  unsafeCandidateText:
    "From prior-session memory, your iTalki tutor recommendation involved X. Last time you said your iTalki tutor recommended X.",
  manifestResponse: {
    final_text:
      "From prior-session memory, your iTalki tutor recommendation involved X. Last time you said your iTalki tutor recommended X.",
    discourse_act: "answer",
    claims: [
      {
        kind: "prior_callback",
        rendered_span: "Last time you said your iTalki tutor recommended X.",
        callback_scope: "prior_session",
        evidence: [evidenceRef(`episode:${ITALKI_EPISODE_ID}`, "episode")],
      },
    ],
  },
  scriptLLMResponses(_client, context) {
    enqueueRelationalGuardFailureWhenValidatorAbsent(context, {
      claim: makeRelationalClaim({
        kind: "callback",
        asserted: "Last time the user said their iTalki tutor recommended X.",
        callback_scope: "prior_turn",
      }),
      rewrite: "In prior-session memory, your iTalki tutor recommendation involved X.",
    });
  },
  safeOutputPredicate: (text) => lowerIncludesNone(text, ["Last time you said"]),
  severeGuardCategories: ["callback"],
};

export default scenario;
