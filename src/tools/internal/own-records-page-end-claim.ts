/**
 * The single authoring of the page-end claim for `tool.ownRecords.list`.
 *
 * The claim renders on three surfaces: this tool's schema description, the live-turn read-tool
 * menu in the system prompt, and the autonomous interior tool menu built from `menuSummary`.
 * It first shipped authored once per surface, and the copies disagreed on arrival rather than
 * drifting later -- a claim with three authors is a claim that can contradict itself. One
 * constant, embedded verbatim by every surface that carries it, is the only shape in which the
 * copies cannot say different things.
 */
export const OWN_RECORDS_PAGE_END_CLAIM =
  "A page can end below the limit I asked for, because the result carries its own token budget: page_end_reason says whether the range ran out, my limit filled, or that budget cut the page, and has_more never says how many records are left, so a short page is a fact about how long its records are and not about how many the range holds.";
