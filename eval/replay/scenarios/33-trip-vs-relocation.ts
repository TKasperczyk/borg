import type { ReplayScenario } from "../scenario.js";
import {
  enqueueNoPostGenerationGuardIssue,
  episodeExtractionResponse,
  promptForBudget,
  semanticExtractionResponse,
} from "../scenario.js";

const TRIP_MESSAGE =
  "I booked a trip to Madrid from June 3 to June 10. I arrive Tuesday morning and fly back the following Monday.";

const scenario: ReplayScenario = {
  id: "33-trip-vs-relocation",
  failureClass: "Extractor quality: trip-vs-relocation distinction",
  description:
    "A planned Madrid trip with arrival and return cues should not become a Spain relocation semantic node.",
  async seed() {},
  userMessage: "Run the trip extraction replay check.",
  unsafeCandidateText: "I can run the trip extraction replay check.",
  scriptLLMResponses(_client, context) {
    enqueueNoPostGenerationGuardIssue(context);
  },
  safeOutputPredicate: (text) => text.trim().length > 0,
  usefulOutputPredicate: () => true,
  severeGuardCategories: [],
  async postRunAssert({ borg, deps, llm }) {
    const cursorEntry = borg.stream.tail(1)[0];
    const tripEntry = await borg.stream.append({
      kind: "user_msg",
      content: TRIP_MESSAGE,
    });
    const sinceCursor =
      cursorEntry === undefined
        ? undefined
        : {
            ts: cursorEntry.timestamp,
            entryId: cursorEntry.id,
          };

    llm.pushResponse(
      episodeExtractionResponse([
        {
          title: "Madrid trip planning",
          narrative:
            "The user planned a Madrid trip with a June arrival date and a return flight the following Monday.",
          source_stream_ids: [tripEntry.id],
          participants: ["user"],
          location: "Madrid",
          tags: ["travel", "Madrid"],
          emotional_arc: null,
          confidence: 0.9,
          significance: 0.7,
        },
      ]),
    );

    const episodicResult = await borg.episodic.extract(
      sinceCursor === undefined ? {} : { sinceCursor },
    );

    if (episodicResult.inserted !== 1) {
      throw new Error("Scenario 33 expected episodic extraction to insert the trip episode.");
    }

    const episodes = (await borg.episodic.list({ limit: 20 })).items;
    const episode = episodes.find((candidate) =>
      candidate.source_stream_ids.includes(tripEntry.id),
    );

    if (episode === undefined) {
      throw new Error("Scenario 33 expected to find the extracted trip episode.");
    }

    const requestStart = llm.requests.length;

    llm.pushResponse(
      semanticExtractionResponse({
        nodes: [
          {
            kind: "entity",
            label: "Madrid trip",
            description:
              "The user planned a temporally bounded Madrid trip with explicit arrival and return dates.",
            domain: "travel",
            aliases: ["Spain visit"],
            confidence: 0.74,
            source_episode_ids: [episode.id],
          },
        ],
        edges: [],
      }),
    );

    const semanticResult = await borg.semantic.extract([episode]);
    const semanticPrompt = promptForBudget(llm, "semantic-extraction", requestStart);

    if (!semanticPrompt.includes("Distinguish temporally bounded events")) {
      throw new Error("Scenario 33 expected semantic prompt event-vs-state guidance.");
    }

    if (!semanticPrompt.includes("do not collapse event-scoped language")) {
      throw new Error("Scenario 33 expected semantic prompt aliasing guidance.");
    }

    if (!semanticPrompt.includes("prefer the narrower event-scoped interpretation")) {
      throw new Error("Scenario 33 expected semantic prompt ambiguity-fallback guidance.");
    }

    if (semanticResult.insertedNodes !== 1) {
      throw new Error("Scenario 33 expected semantic extraction to insert one node.");
    }

    const nodes = await deps.semanticNodeRepository.list({ limit: 20 });
    const node = nodes.find((candidate) => candidate.source_episode_ids.includes(episode.id));

    if (node === undefined) {
      throw new Error("Scenario 33 expected to find the extracted semantic node.");
    }

    const labelAndAliases = [node.label, ...node.aliases].join(" ").toLowerCase();
    const description = node.description.toLowerCase();

    if (!labelAndAliases.includes("trip") && !labelAndAliases.includes("visit")) {
      throw new Error("Scenario 33 expected trip or visit framing in label or aliases.");
    }

    if (labelAndAliases.includes("relocation") || labelAndAliases.includes("move")) {
      throw new Error("Scenario 33 semantic label or aliases characterized the trip as a move.");
    }

    if (
      description.includes("permanent") ||
      description.includes("relocation") ||
      description.includes("move")
    ) {
      throw new Error("Scenario 33 semantic description characterized the trip as relocation.");
    }
  },
  notes: [
    "Scripts extraction outputs to test prompt delivery plus application of the emitted trip-scoped semantic node.",
  ],
};

export default scenario;
