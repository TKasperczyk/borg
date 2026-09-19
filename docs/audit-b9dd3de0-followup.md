# Follow-up review of b9dd3de0

Opus 5.0: in-scope because result budgets, internal-handle presentation, and
reference resolution are structural harness responsibilities.

## Findings and changes

1. **Oversized disclosure metadata.** The composed 2,000-audience regression
   reproduced a 44,262-token own-record page against the 32,000-token budget.
   After omitting payload and anchors, the tool now checks the complete output.
   If disclosure metadata still overflows, both disclosure fields become a
   bounded unknown label. The complete serialized fallback, including cursor and
   page metadata, is checked again before return. The regression verifies the
   conservative label and successful pagination to the next ordinary record.
2. **Disposition identifiers.** The prose scrubber inherited an incomplete
   prefix list and rejected matches adjacent to Unicode letters. Its grammar
   now comes from the canonical ID-helper declarations, including the existing
   hexadecimal ID families. Token boundaries use the machine ID alphabet, so
   adjacent non-Latin prose cannot conceal IDs. Automatic dependency-unblock
   reasons are decoded only in the matching system transition and only in the
   exact machine format written by `GoalsRepository.reconcileBlocks`; the goal
   reference must agree with the structured blocker. Only the audit-reference
   field is replaced, preserving numbers, timestamps, and the rest of the reason.
   Composed regressions cover numeric audit references, scheduled wakes,
   attachments, and Chinese text adjacent to goal IDs. A utility regression
   exercises every exported canonical ID helper without a second prefix list.
3. **Unresolved entity handles.** Entity-ID labels are now resolved before self
   normalization. A valid ID also cannot be redirected to a different explicit
   identity anchor or acquire the implicit self fallback. Missing IDs therefore
   never enter the batch's original-handle map. Regressions cover person, self,
   first-person agent, and explicit-self candidates with an incident edge; the
   existing resolved off-roster node-and-edge regression remains passing.

## Search and reuse

- Reused `resultFitsBudget`, `withoutPayload`, `withoutOversizedAnchors`,
  `memoryDisclosurePayloadFields`, `unknownMemoryDisclosureLabel`, the shared
  token budget, and `ToolError`; no new budget helper.
- Searches for `createIdHelpers`, `IdHelpers`, internal-identifier patterns, and
  scrubbers found the canonical factories plus the stale host-guard grammar.
  Extended the existing factory to cover its two hexadecimal specializations
  and derive the shared grammar from those declarations. The new token-pattern
  constant uses that grammar and its alphabet; it adds no hand-maintained list.
- Searches for audit-reference and unblock-reason rendering found the fixed
  producer in `GoalsRepository.reconcileBlocks`, with no corresponding reader.
  Extended the existing `renderGoalDisposition` with a local machine-format
  decoder guarded by stored action, provenance, and blocker fields. It does not
  recognize phrases in user/model-authored reasons or remove arbitrary numbers.
- Reused `resolveNameSlotLabel`, `entityIdHelpers`, and existing identity-anchor
  and batch-map logic. No new reference-resolution utility or persistence.

## Validation

The new regressions failed before the fixes. Afterwards, all 136 tests in seven
focused files passed with the required environment unsets and runner flags:

```sh
env -u BORG_TRACE -u BORG_TRACE_PROMPTS -u BORG_DATA_DIR -u EXTRA_CONNECTOR_MODULE \
  TMPDIR=/home/luth/tmp/codex-bb7ff537-checks \
  pnpm vitest run \
  src/tools/internal/internal-tools.test.ts \
  src/tools/internal/own-records-list.test.ts \
  src/memory/semantic/extractor.test.ts \
  src/util/ids.test.ts \
  src/util/prompt-tags.test.ts \
  src/cognition/generation/post-generation-guards-shadow.test.ts \
  src/cognition/prompts/prompt-surface-fixtures.test.ts \
  --maxWorkers=2 --cache=false --configLoader=native
```

`pnpm heuristics:guard` passed. `pnpm typecheck` passed, run once at the end.

All 18 prompt-surface fixtures are byte-identical to `b9dd3de0`: 520,301 bytes
before and after, **0 bytes delta per fixture and in total**. No fixture was
regenerated. Tool descriptions and schemas are unchanged. Test temporary files
use the directory above; the live store and `/tmp` were not used for writes.

Open questions: none.
