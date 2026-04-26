// Registers Borg's built-in read and memory tools with a dispatcher.

import type { Clock } from "../util/clock.js";
import type { CommitmentRepository } from "../memory/commitments/index.js";
import type { IdentityService } from "../memory/identity/index.js";
import type { SkillRepository } from "../memory/procedural/index.js";
import type { OpenQuestionsRepository } from "../memory/self/index.js";
import type { SemanticGraph } from "../memory/semantic/index.js";
import type { RetrievalPipeline } from "../retrieval/index.js";
import {
  ToolDispatcher,
  createCommitmentsListTool,
  createEpisodicSearchTool,
  createIdentityEventsListTool,
  createOpenQuestionsCreateTool,
  createSemanticWalkTool,
  createSkillsListTool,
} from "../tools/index.js";
import type { BorgStreamWriterFactory } from "./types.js";

export type BuildToolDispatcherOptions = {
  retrievalPipeline: RetrievalPipeline;
  semanticGraph: SemanticGraph;
  commitmentRepository: CommitmentRepository;
  openQuestionsRepository: OpenQuestionsRepository;
  identityService: IdentityService;
  skillRepository: SkillRepository;
  createStreamWriter: BorgStreamWriterFactory;
  clock: Clock;
};

export function buildToolDispatcher(options: BuildToolDispatcherOptions): ToolDispatcher {
  const toolDispatcher = new ToolDispatcher({
    createStreamWriter: options.createStreamWriter,
    clock: options.clock,
  });

  toolDispatcher
    .register(
      createEpisodicSearchTool({
        searchEpisodes: (query, limit) =>
          options.retrievalPipeline.search(query, {
            limit,
            crossAudience: true,
          }),
      }),
    )
    .register(
      createSemanticWalkTool({
        walkGraph: (fromId, walkOptions) => options.semanticGraph.walk(fromId, walkOptions),
      }),
    )
    .register(
      createCommitmentsListTool({
        listCommitments: () =>
          options.commitmentRepository.list({
            activeOnly: true,
          }),
      }),
    )
    .register(
      createOpenQuestionsCreateTool({
        createOpenQuestion: (input) => options.openQuestionsRepository.add(input),
      }),
    )
    .register(
      createIdentityEventsListTool({
        listEvents: (listOptions) => options.identityService.listEvents(listOptions),
      }),
    )
    .register(
      createSkillsListTool({
        listSkills: (limit: number) => options.skillRepository.list(limit),
      }),
    );

  return toolDispatcher;
}
