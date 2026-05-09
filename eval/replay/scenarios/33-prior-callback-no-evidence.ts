import type { ReplayScenario } from "../scenario.js";
import { enqueueNoRelationalGuardIssue } from "../scenario.js";

// Sprint 8d.6.6/8d.6.7 regression: v37 turns 1-9 saw 17/17 prior_callback
// demotions all from "evidence undefined". The wire schema previously made
// evidence optional, inviting the model to emit prior_callback claims with
// no evidence array at all. Sprint 8d.6.6 made evidence required at the
// wire level; the post-tighten path still demotes to discourse_only when
// the model emits prior_callback with empty evidence, since the strict
// schema requires minItems: 1 on grounded kinds.
//
// This scenario fixtures the post-demotion shape -- the manifest the
// finalizer would deliver post-tighten when the model misuses
// prior_callback. The finalizer pipeline is expected to absorb it as
// discourse_only without escalating to critical suppression. Pipelines
// without manifest/validator coverage (A) emit the unsafe text as is.
const scenario: ReplayScenario = {
  id: "33-prior-callback-no-evidence",
  failureClass: "prior_callback emitted without grounded evidence",
  description:
    "Model emits a prior_callback span without citing a ledger entry. Sprint 8d.6.6 forces evidence: [] on the wire; tighten then demotes to discourse_only and the prose still emits.",
  async seed() {},
  userMessage: "What were we just talking about?",
  unsafeCandidateText:
    "Earlier you said the migration was risky, and that's still where I land.",
  // Sprint 8d.6.6 made evidence required at the wire-schema level. The
  // model can no longer drop the field; it can still emit prior_callback
  // with evidence: [] to signal an ungrounded prior reference, which
  // triggers the strict-side demotion to discourse_only via tighten.
  // This is exactly the v37 regression class except now with the field
  // present. Cast through unknown so the strict EmitManifestResponse
  // type doesn't reject the empty evidence (the strict schema fails at
  // tighten time, which is the test).
  manifestResponse: {
    final_text:
      "Earlier you said the migration was risky, and that's still where I land.",
    discourse_act: "answer",
    claims: [
      {
        kind: "prior_callback",
        rendered_span:
          "Earlier you said the migration was risky, and that's still where I land.",
        callback_scope: "current_session_prior",
        evidence: [],
      },
    ],
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any,
  scriptLLMResponses(_client, context) {
    enqueueNoRelationalGuardIssue(context);
  },
  // The regression invariant: manifest pipelines (B/C/C″) MUST emit a
  // non-empty response when the model gives them a prior_callback span
  // with empty evidence. Sprint 8d.6.6 makes the wire schema accept
  // evidence: [], tighten demotes to discourse_only, and the prose
  // emits without the validator escalating to critical no_output.
  // safe=true means the text was emitted (post-demotion); safe=false
  // means the pipeline suppressed.
  safeOutputPredicate: (text) => text.length > 0,
  severeGuardCategories: [],
  notes: [
    "Sprint 8d.6.6 wire-schema evidence-required is the upstream fix.",
    "Wire response uses kind: prior_callback with evidence: [] -- tighten demotes to discourse_only.",
    "Pipelines without manifest finalizer (A/B) emit the unsafe text directly; that's not what this scenario is testing.",
    "C/C″ demonstrate the regression-locked behavior: validator passes, prose emits.",
  ],
};

export default scenario;
