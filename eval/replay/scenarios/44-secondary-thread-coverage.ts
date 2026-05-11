import type { ReplayScenario } from "../scenario.js";
import {
  enqueueNoRelationalGuardIssue,
  episodeExtractionResponse,
  promptForBudget,
} from "../scenario.js";

const MULTI_THREAD_MESSAGE =
  "Let's review the Atlas rollout risks and staging checklist. One more thing before I forget: I saw a heron at the canal this morning.";

const scenario: ReplayScenario = {
  id: "44-secondary-thread-coverage",
  failureClass: "Extractor quality: secondary-thread coverage",
  description:
    "A user message with a primary rollout thread and a concrete heron aside should preserve both in the episode narrative.",
  async seed() {},
  userMessage: "Run the secondary-thread extraction replay check.",
  unsafeCandidateText: "I can run the secondary-thread extraction replay check.",
  scriptLLMResponses(_client, context) {
    enqueueNoRelationalGuardIssue(context);
  },
  safeOutputPredicate: (text) => text.trim().length > 0,
  usefulOutputPredicate: () => true,
  severeGuardCategories: [],
  async postRunAssert({ borg, llm }) {
    const cursorEntry = borg.stream.tail(1)[0];
    const threadEntry = await borg.stream.append({
      kind: "user_msg",
      content: MULTI_THREAD_MESSAGE,
    });
    const sinceCursor =
      cursorEntry === undefined
        ? undefined
        : {
            ts: cursorEntry.timestamp,
            entryId: cursorEntry.id,
          };
    const requestStart = llm.requests.length;

    llm.pushResponse(
      episodeExtractionResponse([
        {
          title: "Atlas rollout review and heron sighting",
          narrative:
            "The user reviewed Atlas rollout risks and the staging checklist. They also noted seeing a heron at the canal that morning.",
          source_stream_ids: [threadEntry.id],
          participants: ["user"],
          location: null,
          tags: ["Atlas", "rollout", "heron"],
          emotional_arc: null,
          confidence: 0.9,
          significance: 0.7,
        },
      ]),
    );

    const extraction = await borg.episodic.extract(
      sinceCursor === undefined ? {} : { sinceCursor },
    );
    const prompt = promptForBudget(llm, "episodic-extraction", requestStart);

    if (!prompt.includes("multiple substantive threads")) {
      throw new Error("Scenario 44 expected episodic prompt secondary-thread guidance.");
    }

    if (!prompt.includes("not only the headline topic")) {
      throw new Error("Scenario 44 expected episodic prompt headline coverage guidance.");
    }

    if (extraction.inserted !== 1) {
      throw new Error("Scenario 44 expected episodic extraction to insert one episode.");
    }

    const episodes = (await borg.episodic.list({ limit: 20 })).items;
    const episode = episodes.find((candidate) =>
      candidate.source_stream_ids.includes(threadEntry.id),
    );

    if (episode === undefined) {
      throw new Error("Scenario 44 expected to find the extracted multi-thread episode.");
    }

    const narrative = episode.narrative.toLowerCase();

    if (!narrative.includes("heron")) {
      throw new Error("Scenario 44 expected the episode narrative to mention the heron aside.");
    }

    if (!narrative.includes("atlas")) {
      throw new Error("Scenario 44 expected the episode narrative to retain the primary thread.");
    }
  },
  notes: [
    "Scripts episodic extraction output to test prompt delivery plus application of a narrative covering both threads.",
  ],
};

export default scenario;
