import { describe, expect, it } from "vitest";

import { createIdentityEventsListForCognitionTool } from "../tools/index.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";

import {
  AUTONOMY_TRIGGER_CONTEXT_MAX_CHARS,
  formatAutonomyTriggerContext,
  renderAutonomyTriggerPresentation,
} from "./autonomy-trigger.js";

const BASE = {
  source_name: "goal_followup_due",
  source_type: "trigger",
  event_id: "goal_aaaaaaaaaaaaaaaa:1787050000000:1785840551552:deadline",
  sort_ts: 1_787_050_000_000,
} as const;

describe("formatAutonomyTriggerContext epoch annotation", () => {
  it("counts only whole question ids and texts in the retained payload", () => {
    const context = {
      ...BASE,
      payload: {
        open_question_id: "oq_primary",
        question: '主題"\n'.repeat(4000),
        last_touched: 1000,
        additional_questions: [
          { open_question_id: "oq_short", question: "短い問い", last_touched: 1000 },
        ],
      },
    };
    const whole = renderAutonomyTriggerPresentation(context, 100_000);
    expect(whole.shownOpenQuestionIds).toEqual(["oq_primary", "oq_short"]);
    const excerpted = renderAutonomyTriggerPresentation(context, 4_000);
    expect(excerpted.shownOpenQuestionIds).not.toContain("oq_primary");
    expect(excerpted.shownOpenQuestionIds).toContain("oq_short");
    expect(excerpted.text.length).toBeLessThanOrEqual(4000);
    const dropped = renderAutonomyTriggerPresentation(
      {
        ...context,
        payload: {
          ...context.payload,
          disclosure: "x".repeat(40_000),
        },
      },
      500,
    );
    expect(dropped.shownOpenQuestionIds).toEqual([]);
  });

  it("adds a calendar sibling for payload timestamps without touching the raw field", () => {
    const rendered = formatAutonomyTriggerContext({
      ...BASE,
      payload: { target_at: 1_787_050_000_000, last_progress_ts: null, priority: 10 },
    });

    expect(rendered).toContain('"target_at": 1787050000000');
    expect(rendered).toContain('"target_at_iso": "2026-08-18T10:46:40.000Z"');
    expect(rendered).toContain('"last_progress_ts": null');
    expect(rendered).not.toContain("last_progress_ts_iso");
    expect(rendered).not.toContain("priority_iso");
  });

  it("annotates nested objects and the secondary goal batch", () => {
    const rendered = formatAutonomyTriggerContext({
      ...BASE,
      payload: {
        selected_goal: { id: "goal_aaaaaaaaaaaaaaaa", created_at: 1_785_840_551_552 },
        secondary_due_goals: [{ goal_id: "goal_bbbbbbbbbbbbbbbb", sort_ts: 1_786_556_400_000 }],
      },
    });

    expect(rendered).toContain('"created_at_iso": "2026-08-04T10:49:11.552Z"');
    expect(rendered).toContain('"sort_ts_iso": "2026-08-12T17:40:00.000Z"');
  });

  it("names the identity-event log's domain whenever the payload carries that reader", () => {
    const rendered = formatAutonomyTriggerContext({
      ...BASE,
      source_name: "scheduled_reflection",
      payload: {
        interval_ms: 14_400_000,
        recent_identity_events: [
          { id: 1, record_type: "trait", record_id: "trt_aaaaaaaaaaaaaaaa", action: "decay" },
        ],
      },
    });

    expect(rendered).toContain("not every write a record received");
    expect(rendered).toContain("record_version that write produced");
    expect(rendered).toContain("not evidence that the record did not change");
    expect(rendered.indexOf("recent_identity_events: this is the log")).toBeGreaterThan(
      rendered.indexOf('"record_id": "trt_aaaaaaaaaaaaaaaa"'),
    );
  });

  it("still names the domain when the reader returned nothing, and stays silent when it did not run", () => {
    const empty = formatAutonomyTriggerContext({
      ...BASE,
      source_name: "scheduled_reflection",
      payload: { interval_ms: 14_400_000, recent_identity_events: [] },
    });
    const absent = formatAutonomyTriggerContext({
      ...BASE,
      payload: { interval_ms: 14_400_000 },
    });

    expect(empty).toContain("not every write a record received");
    expect(absent).not.toContain("not every write a record received");
  });

  it.each([false, true])(
    "preserves selected event metadata under pressure (oversized metadata=%s)",
    async (oversizedMetadata) => {
      const tool = createIdentityEventsListForCognitionTool({
        listEvents: () =>
          Array.from({ length: 10 }, (_, index) => ({
            id: index + 1,
            record_type: "goal" as const,
            record_id: `goal_${String(index).padStart(16, "a")}`,
            action: "update",
            old_value: {
              progress_notes: `OLD_${index}_${"o".repeat(110_000)}`,
              record_version: index + 1,
            },
            new_value: {
              progress_notes: `${"n".repeat(115_000)}_NEW_${index}`,
              record_version: index + 2,
            },
            reason: oversizedMetadata
              ? `Reason ${index}: ${"r".repeat(6_000)}`
              : `Scheduled identity maintenance ${index}.`,
            provenance: { kind: "offline" as const, process: "test-reflector" },
            review_item_id: null,
            overwrite_without_review: false,
            ts: 1_787_050_000_000 + index,
          })),
      });
      const output = await tool.invoke(
        { limit: 10 },
        { sessionId: DEFAULT_SESSION_ID, origin: "autonomous" },
      );
      const rendered = formatAutonomyTriggerContext({
        ...BASE,
        source_name: "scheduled_reflection",
        payload: {
          interval_ms: 14_400_000,
          recent_identity_events: output.events,
          prior_self_thought: { text: "thought".repeat(2_000), updated_at: BASE.sort_ts },
          oversized_structural_tail: "z".repeat(100_000),
        },
      });

      expect(rendered.length).toBeLessThanOrEqual(AUTONOMY_TRIGGER_CONTEXT_MAX_CHARS);
      expect(rendered.match(/excerpt_notice:/g)).toHaveLength(1);
      expect(rendered).toContain("mechanically bounded to 32000 chars");
      expect(rendered).toContain("old-to-new change excerpt is bounded to 1500 chars");
      expect(rendered).toContain('"record_type": "goal"');
      expect(rendered).toContain('"excerpt_exact": false');
      expect(rendered).toContain("not every write a record received");
      expect(rendered).not.toContain("o".repeat(10_000));
      expect(rendered).not.toContain("n".repeat(10_000));

      const payloadStart = rendered.indexOf("payload:\n") + "payload:\n".length;
      const payloadEnd = rendered.indexOf("\nnote on recent_identity_events:");
      const payload = JSON.parse(rendered.slice(payloadStart, payloadEnd));
      const selected = payload.recent_identity_events as typeof output.events;
      if (oversizedMetadata) {
        expect(selected.length).toBeGreaterThan(0);
        expect(selected.length).toBeLessThan(output.events.length);
      } else {
        expect(selected).toHaveLength(output.events.length);
      }
      expect(rendered).toContain(
        `recent_identity_events_omitted: ${output.events.length - selected.length}`,
      );
      for (const [index, event] of selected.entries()) {
        const { change: _change, ...metadata } = output.events[index]!;
        expect(event).toMatchObject(metadata);
      }
    },
  );

  it("names what an old sort_ts on a dormant-question wake does and does not mean", () => {
    const rendered = formatAutonomyTriggerContext({
      source_name: "open_question_dormant",
      source_type: "trigger",
      event_id: "oq_aaaaaaaaaaaaaaaa:1787050000000",
      sort_ts: 1_787_050_000_000,
      payload: {
        open_question_id: "oq_aaaaaaaaaaaaaaaa",
        question: "What is the right autonomy cadence?",
        urgency: 0.6,
        last_touched: 1_787_050_000_000,
        unresolved_rumination_ticks: 0,
        last_ruminated_at: null,
      },
    });

    expect(rendered).toContain("cannot wake me a second time");
    expect(rendered).toContain("never a count of how often it has already woken me");
    expect(rendered).toContain("leaves the pair unlatched and can return");
    expect(rendered).toContain("updates its rumination count and time");
    expect(rendered.indexOf("note on this dormant-question wake")).toBeGreaterThan(
      rendered.indexOf('"last_ruminated_at": null'),
    );
  });

  describe("an open question's text under the context cap", () => {
    const DORMANT = {
      source_name: "open_question_dormant",
      source_type: "trigger",
      event_id: "oq_aaaaaaaaaaaaaaaa:1787050000000",
      sort_ts: 1_787_050_000_000,
    } as const;
    const relatedEpisodes = (offset: number) =>
      Array.from({ length: 5 }, (_, index) => ({
        id: `ep_${String(offset + index).padStart(16, "a")}`,
        title: `Epizod ${offset + index}: ${"tytuł ".repeat(80)}`,
        narrative: "narracja epizodu ".repeat(400),
        citation_chain: Array.from({ length: 12 }, (_, entry) => ({
          id: `strm_${String((offset + index) * 12 + entry).padStart(16, "b")}`,
          content: "wpis strumienia ".repeat(200),
          disclosure: "disclosure_class=public",
        })),
      }));
    const parsePayload = (rendered: string) =>
      JSON.parse(
        rendered.slice(
          rendered.indexOf("payload:\n") + "payload:\n".length,
          rendered.indexOf("\nnote on this dormant-question wake"),
        ),
      );

    it("keeps every presented question whole while its related episodes absorb the cut", () => {
      const primary = `Czy przypisałem „to słowo” właściwej osobie — ${"i skąd wziąłem tę atrybucję, ".repeat(5)}koniec?`;
      const additional = `この質問は最後まで読めるか。${"どの欄を比べれば決まるのか。".repeat(80)}終わり`;
      const rendered = formatAutonomyTriggerContext({
        ...DORMANT,
        payload: {
          open_question_id: "oq_aaaaaaaaaaaaaaaa",
          question: primary,
          urgency: 0.65,
          last_touched: 1_787_050_000_000,
          unresolved_rumination_ticks: 0,
          last_ruminated_at: null,
          related_episodes: relatedEpisodes(0),
          additional_questions: [
            {
              open_question_id: "oq_bbbbbbbbbbbbbbbb",
              question: additional,
              urgency: 0.6,
              last_touched: 1_787_050_100_000,
              unresolved_rumination_ticks: 0,
              last_ruminated_at: null,
              related_episodes: relatedEpisodes(5),
            },
          ],
        },
      });

      expect(rendered.length).toBeLessThanOrEqual(AUTONOMY_TRIGGER_CONTEXT_MAX_CHARS);
      expect(rendered).not.toContain("payload_fields_omitted");
      const payload = parsePayload(rendered);
      expect(payload.question).toBe(primary);
      expect(payload.additional_questions[0].question).toBe(additional);
      const episodes = [
        ...payload.related_episodes,
        ...payload.additional_questions[0].related_episodes,
      ];
      expect(episodes).toHaveLength(10);
      for (const episode of episodes) {
        expect(episode.narrative).toContain("[ELIDED]");
        expect(episode.citation_chain).toHaveLength(12);
        expect(episode.citation_chain[0].disclosure).toBe("disclosure_class=public");
      }
    });

    it("drops trailing related episodes before excerpting questions that fit beside whole labels", () => {
      const label = (disclosure_class: string, audience: string[]) => ({
        disclosure: `disclosure_class=${disclosure_class} origin_audience=${audience.join(",")} private-to=${audience.join(",")}; I can use this internally; I do not disclose it to the current audience unless authorized`,
        disclosure_label: {
          disclosure_class,
          origin_audience_entity_ids: audience,
          private_to_entity_ids: audience,
          public_to_entity_ids: [],
        },
      });
      // The shape tool.episodic.search returns: every episode and citation carries both labels.
      const labelledEpisodes = (offset: number, citations: number[]) =>
        citations.map((count, index) => ({
          id: `ep_${String(offset + index).padStart(16, "c")}`,
          title: `Wątek ${offset + index}: ${"tytuł ".repeat(20)}`,
          narrative: "narracja epizodu ".repeat(100),
          participants: Array.from({ length: 7 }, (_, n) => `ent_${String(n).padStart(16, "d")}`),
          tags: Array.from({ length: 8 }, (_, n) => `znacznik ${n}`),
          source_stream_ids: Array.from(
            { length: 5 },
            (_, n) => `strm_${String((offset + index) * 10 + n).padStart(16, "e")}`,
          ),
          start_time: 1_787_050_000_000,
          end_time: 1_787_050_300_000,
          ...label("relationship_private", ["ent_ffffffffffffffff"]),
          score: 0.5 - (offset + index) / 100,
          score_breakdown: {
            similarity: 0,
            decayed_salience: 0.5498351895770041,
            time_relevance: 0,
          },
          citation_chain: Array.from({ length: count }, (_, entry) => ({
            id: `strm_${String((offset + index) * 10 + entry).padStart(16, "g")}`,
            kind: "user_msg",
            timestamp: 1_787_050_000_000,
            content: "wpis strumienia ".repeat(30),
            ...label("relationship_private", ["ent_ffffffffffffffff"]),
          })),
        }));
      const primary = `Is a durable cursor what separates a process that recovers from backlog from one that stays stuck? ${"Which field would tell the two apart before I lean on it? ".repeat(19)}End.`;
      const additional = `Czy ta twardość jest wiernością zobowiązaniu, ${"czy własną formą elegancji, za którą się chowam, ".repeat(5)}koniec?`;
      const payload = {
        open_question_id: "oq_aaaaaaaaaaaaaaaa",
        question: primary,
        urgency: 0.3,
        last_touched: 1_787_050_000_000,
        unresolved_rumination_ticks: 0,
        last_ruminated_at: null,
        ...label("relationship_private", ["ent_hhhhhhhhhhhhhhhh", "ent_iiiiiiiiiiiiiiii"]),
        additional_questions: [
          {
            open_question_id: "oq_bbbbbbbbbbbbbbbb",
            question: additional,
            urgency: 0.45,
            last_touched: 1_787_050_100_000,
            unresolved_rumination_ticks: 1,
            last_ruminated_at: 1_787_050_100_004,
            ...label("self_private", []),
            related_episodes: labelledEpisodes(5, [3, 1, 1, 3, 1]),
          },
        ],
        related_episodes: labelledEpisodes(0, [3, 2, 3, 1, 1]),
        prior_self_thought: {
          text: "poprzednia myśl ".repeat(150),
          updated_at: 1_787_050_200_000,
          self_entity_id: "ent_jjjjjjjjjjjjjjjj",
          ...label("self_private", []),
        },
      };
      const rendered = formatAutonomyTriggerContext({ ...DORMANT, payload });

      expect(rendered.length).toBeLessThanOrEqual(AUTONOMY_TRIGGER_CONTEXT_MAX_CHARS);
      expect(rendered).not.toContain("payload_fields_omitted");
      const parsed = parsePayload(rendered);
      expect(parsed.question).toBe(primary);
      expect(parsed.open_question_id).toBe("oq_aaaaaaaaaaaaaaaa");
      expect(parsed.additional_questions[0].question).toBe(additional);
      expect(parsed.additional_questions[0].open_question_id).toBe("oq_bbbbbbbbbbbbbbbb");
      expect(parsed.prior_self_thought.disclosure_label).toEqual(
        payload.prior_self_thought.disclosure_label,
      );
      // The last question's lowest-ranked episodes go first; what stays keeps its order and labels.
      const scores = (episodes: Array<{ score: number }>) => episodes.map((entry) => entry.score);
      const kept = parsed.additional_questions[0].related_episodes;
      expect(kept.length).toBeLessThan(5);
      expect(scores(kept)).toEqual(
        scores(payload.additional_questions[0]!.related_episodes).slice(0, kept.length),
      );
      expect(scores(parsed.related_episodes)).toEqual(scores(payload.related_episodes));
      for (const episode of [...parsed.related_episodes, ...kept]) {
        expect(episode.disclosure_label).toEqual(payload.related_episodes[0]!.disclosure_label);
      }
    });

    it("keeps a presented question's id whole when the shared budget is shorter than it", () => {
      const rendered = formatAutonomyTriggerContext({
        ...DORMANT,
        payload: {
          open_question_id: "oq_aaaaaaaaaaaaaaaa",
          question: "Który znacznik rozstrzyga?",
          last_touched: 1_787_050_000_000,
          tags: Array.from({ length: 1_400 }, (_, n) => `znacznik numer ${n} `.repeat(3)),
          // Dropping it would not let its id fit whole, so it stays, excerpted.
          related_episodes: [{ id: "ep_cccccccccccccccc" }],
          additional_questions: [
            {
              open_question_id: "oq_bbbbbbbbbbbbbbbb",
              question: "どの欄が決めるのか?",
              related_episodes: [],
            },
          ],
        },
      });

      expect(rendered.length).toBeLessThanOrEqual(AUTONOMY_TRIGGER_CONTEXT_MAX_CHARS);
      const parsed = parsePayload(rendered);
      expect(parsed.related_episodes[0].id).toContain("[ELIDED]");
      expect(parsed.open_question_id).toBe("oq_aaaaaaaaaaaaaaaa");
      expect(parsed.additional_questions[0].open_question_id).toBe("oq_bbbbbbbbbbbbbbbb");
      expect(parsed.additional_questions[0].question).toBe("どの欄が決めるのか?");
    });

    it("still excerpts a question field on a record that is not an open question", () => {
      const question = "Is this a goal note rather than an open question? ".repeat(10);
      const rendered = formatAutonomyTriggerContext({
        ...BASE,
        payload: {
          goal_id: "goal_aaaaaaaaaaaaaaaa",
          question,
          related_episodes: relatedEpisodes(0),
        },
      });

      expect(rendered.length).toBeLessThanOrEqual(AUTONOMY_TRIGGER_CONTEXT_MAX_CHARS);
      expect(rendered).not.toContain(question);
      expect(rendered).toMatch(/"question": "Is this a goal note[^"]*\[ELIDED\]/);
    });

    it("excerpts a question that cannot fit whole before dropping any field", () => {
      const question = "Czy to pytanie się zmieści? ".repeat(1_500);
      const rendered = formatAutonomyTriggerContext({
        ...DORMANT,
        payload: {
          open_question_id: "oq_aaaaaaaaaaaaaaaa",
          question,
          urgency: 0.65,
          last_touched: 1_787_050_000_000,
          unresolved_rumination_ticks: 0,
          last_ruminated_at: null,
          disclosure: "disclosure_class=self_private",
          related_episodes: relatedEpisodes(0),
        },
      });

      expect(question.length).toBeGreaterThan(AUTONOMY_TRIGGER_CONTEXT_MAX_CHARS);
      expect(rendered.length).toBeLessThanOrEqual(AUTONOMY_TRIGGER_CONTEXT_MAX_CHARS);
      expect(rendered).not.toContain("payload_fields_omitted");
      const payload = parsePayload(rendered);
      expect(payload.question).toContain("[ELIDED]");
      expect(payload.question.startsWith("Czy to pytanie się zmieści?")).toBe(true);
      expect(payload.disclosure).toBe("disclosure_class=self_private");
      expect(payload.related_episodes).toHaveLength(5);
    });

    it("omits related episodes, counted, before any field under a cap smaller than their labels", () => {
      const label = {
        disclosure:
          "disclosure_class=relationship_private origin_audience=ent_kkkkkkkkkkkkkkkk,ent_llllllllllllllll private-to=ent_kkkkkkkkkkkkkkkk,ent_llllllllllllllll; I can use this internally; I do not disclose it to the current audience unless authorized",
        disclosure_label: {
          disclosure_class: "relationship_private",
          origin_audience_entity_ids: ["ent_kkkkkkkkkkkkkkkk", "ent_llllllllllllllll"],
          private_to_entity_ids: ["ent_kkkkkkkkkkkkkkkk", "ent_llllllllllllllll"],
          public_to_entity_ids: [],
        },
      };
      const labelledEpisodes = (offset: number) =>
        Array.from({ length: 5 }, (_, index) => ({
          id: `ep_${String(offset + index).padStart(16, "m")}`,
          narrative: "narracja epizodu ".repeat(100),
          ...label,
          citation_chain: [{ id: `strm_${String(offset + index).padStart(16, "n")}`, ...label }],
        }));
      const additional = `この問いは残るか。${"どの欄が決めるのか。".repeat(500)}終わり`;
      const rendered = formatAutonomyTriggerContext(
        {
          ...DORMANT,
          payload: {
            open_question_id: "oq_aaaaaaaaaaaaaaaa",
            question: "Czy to pytanie zostanie na stronie?",
            last_touched: 1_787_050_000_000,
            ...label,
            related_episodes: labelledEpisodes(0),
            additional_questions: [
              {
                open_question_id: "oq_bbbbbbbbbbbbbbbb",
                question: additional,
                last_touched: 1_787_050_100_000,
                ...label,
                related_episodes: labelledEpisodes(5),
              },
            ],
          },
        },
        4_000,
      );

      expect(rendered.length).toBeLessThanOrEqual(4_000);
      expect(rendered).toContain("mechanically bounded to 4000 chars");
      expect(rendered).toContain("related_episodes_omitted: 10.");
      expect(rendered).not.toContain("payload_fields_omitted");
      const payload = parsePayload(rendered);
      expect(payload.question).toBe("Czy to pytanie zostanie na stronie?");
      expect(payload.related_episodes).toEqual([]);
      expect(payload.additional_questions[0].open_question_id).toBe("oq_bbbbbbbbbbbbbbbb");
      expect(payload.additional_questions[0].question.startsWith("この問いは残るか。")).toBe(true);
      expect(payload.additional_questions[0].question).toContain("[ELIDED]");
      expect(payload.additional_questions[0].disclosure_label).toEqual(label.disclosure_label);
    });

    it("drops trailing related episodes, counted, before their evidence ids would be cut", () => {
      const label = {
        disclosure:
          "disclosure_class=relationship_private origin_audience=ent_ssssssssssssssss private-to=ent_ssssssssssssssss; I can use this internally; I do not disclose it to the current audience unless authorized",
        disclosure_label: {
          disclosure_class: "relationship_private",
          origin_audience_entity_ids: ["ent_ssssssssssssssss"],
          private_to_entity_ids: ["ent_ssssssssssssssss"],
          public_to_entity_ids: [],
        },
      };
      const streamId = (n: number) => `strm_${String(n).padStart(16, "r")}`;
      // Every episode and citation carries whole labels, as tool.episodic.search returns them.
      const labelledEpisodes = (offset: number) =>
        Array.from({ length: 6 }, (_, index) => ({
          id: `ep_${String(offset + index).padStart(16, "p")}`,
          title: `Odcinek ${offset + index}: ${"tytuł ".repeat(20)}`,
          narrative: "narracja epizodu ".repeat(60),
          participants: ["Tom", `ent_${String(offset + index).padStart(16, "q")}`],
          source_stream_ids: [streamId((offset + index) * 2), streamId((offset + index) * 2 + 1)],
          ...label,
          citation_chain: [0, 1].map((entry) => ({
            id: streamId((offset + index) * 2 + entry),
            kind: "user_msg",
            content: "wpis strumienia ".repeat(30),
            ...label,
          })),
        }));
      const payload = {
        open_question_id: "oq_aaaaaaaaaaaaaaaa",
        question: "Czy wcześniejsze ostrzeżenie było już tą samą diagnozą?",
        last_touched: 1_787_050_000_000,
        ...label,
        related_episodes: labelledEpisodes(0),
        additional_questions: [
          {
            open_question_id: "oq_bbbbbbbbbbbbbbbb",
            question: "この沈黙は選択だったのか、それとも切れた配線だったのか。",
            last_touched: 1_787_050_100_000,
            ...label,
            related_episodes: labelledEpisodes(6),
          },
        ],
      };
      const handles = (episode: {
        id: string;
        source_stream_ids: string[];
        citation_chain: Array<{ id: string }>;
      }) => [episode.id, ...episode.source_stream_ids, ...episode.citation_chain.map((c) => c.id)];

      let partial = 0;
      for (let cap = 4_000; cap <= AUTONOMY_TRIGGER_CONTEXT_MAX_CHARS; cap += 1_000) {
        const rendered = formatAutonomyTriggerContext({ ...DORMANT, payload }, cap);
        expect(rendered.length).toBeLessThanOrEqual(cap);
        expect(rendered).not.toContain("payload_fields_omitted");
        const parsed = parsePayload(rendered);
        expect(parsed.question).toBe(payload.question);
        expect(parsed.additional_questions[0].question).toBe(
          payload.additional_questions[0]!.question,
        );
        const primary = parsed.related_episodes;
        const additional = parsed.additional_questions[0].related_episodes;
        const omitted = Number(/related_episodes_omitted: (\d+)\./.exec(rendered)?.[1] ?? 0);
        expect(primary.length + additional.length + omitted).toBe(12);
        if (additional.length > 0) {
          expect(primary).toHaveLength(6);
        }
        // What stays is the highest-ranked prefix of each list, every evidence id whole.
        expect(primary.map(handles)).toEqual(
          payload.related_episodes.slice(0, primary.length).map(handles),
        );
        expect(additional.map(handles)).toEqual(
          payload.additional_questions[0]!.related_episodes.slice(0, additional.length).map(handles),
        );
        if (omitted > 0 && omitted < 12) {
          partial += 1;
        }
      }
      expect(partial).toBeGreaterThan(0);
    });
  });

  it("stays silent on dormancy for a wake that carries no open question", () => {
    const rendered = formatAutonomyTriggerContext({
      ...BASE,
      payload: { goal_id: "goal_aaaaaaaaaaaaaaaa", last_touched: 1_787_050_000_000 },
    });

    expect(rendered).not.toContain("note on this dormant-question wake");
  });

  it("leaves an existing sibling and an unrepresentable instant alone", () => {
    const rendered = formatAutonomyTriggerContext({
      ...BASE,
      payload: {
        target_at: 1_787_050_000_000,
        target_at_iso: "already supplied",
        due_at: 9e15,
      },
    });

    expect(rendered).toContain('"target_at_iso": "already supplied"');
    expect(rendered).not.toContain('"target_at_iso": "2026-08-18T10:46:40.000Z"');
    expect(rendered).not.toContain("due_at_iso");
  });
});
