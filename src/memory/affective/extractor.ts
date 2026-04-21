import { z } from "zod";

import type { LLMClient } from "../../llm/index.js";
import { LLMError } from "../../util/errors.js";
import { tokenizeText } from "../../util/text/tokenize.js";

import { affectiveSignalSchema, type AffectiveSignal, type DominantEmotion } from "./types.js";

const POSITIVE_WORDS = new Set([
  "appreciate",
  "awesome",
  "better",
  "calm",
  "celebrate",
  "clear",
  "confident",
  "curious",
  "delight",
  "easy",
  "enjoy",
  "excited",
  "fantastic",
  "fixed",
  "glad",
  "good",
  "great",
  "happy",
  "helpful",
  "hopeful",
  "improved",
  "inspired",
  "interesting",
  "joy",
  "kind",
  "love",
  "nice",
  "optimistic",
  "peaceful",
  "pleased",
  "proud",
  "relieved",
  "resolved",
  "safe",
  "satisfied",
  "solid",
  "stable",
  "supportive",
  "thrilled",
  "win",
  "wonderful",
  "works",
]);

const NEGATIVE_WORDS = new Set([
  "angry",
  "annoyed",
  "anxious",
  "ashamed",
  "awful",
  "bad",
  "blocked",
  "broken",
  "confused",
  "crash",
  "depressed",
  "disappointed",
  "dread",
  "error",
  "fail",
  "failing",
  "failed",
  "fear",
  "frustrated",
  "furious",
  "grief",
  "guilty",
  "hard",
  "hate",
  "hurt",
  "lost",
  "mess",
  "mistake",
  "nervous",
  "panic",
  "problem",
  "regret",
  "rough",
  "sad",
  "scared",
  "stuck",
  "terrible",
  "tired",
  "unclear",
  "upset",
  "worried",
  "wrong",
]);

const EMOTION_LEXICONS: Record<DominantEmotion, Set<string>> = {
  joy: new Set(["happy", "joy", "glad", "pleased", "proud", "thrilled", "delight"]),
  sadness: new Set(["sad", "grief", "hurt", "lonely", "down", "depressed"]),
  fear: new Set(["afraid", "anxious", "fear", "nervous", "panic", "scared", "worried"]),
  anger: new Set(["angry", "annoyed", "furious", "mad", "rage", "upset"]),
  surprise: new Set(["suddenly", "surprised", "unexpected", "wow"]),
  curiosity: new Set(["curious", "explore", "interesting", "wonder", "why"]),
  neutral: new Set(),
};

const affectiveFallbackSchema = affectiveSignalSchema;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function countMatches(tokens: ReadonlySet<string>, lexicon: ReadonlySet<string>): number {
  let matches = 0;

  for (const token of tokens) {
    if (lexicon.has(token)) {
      matches += 1;
    }
  }

  return matches;
}

function dominantEmotionFromCounts(counts: Record<DominantEmotion, number>): DominantEmotion {
  let best: DominantEmotion = "neutral";
  let bestScore = 0;

  for (const [emotion, score] of Object.entries(counts) as Array<[DominantEmotion, number]>) {
    if (score > bestScore) {
      best = emotion;
      bestScore = score;
    }
  }

  return bestScore === 0 ? "neutral" : best;
}

function heuristicAnalysis(text: string): { signal: AffectiveSignal; confidence: number } {
  const tokens = tokenizeText(text);
  const positiveCount = countMatches(tokens, POSITIVE_WORDS);
  const negativeCount = countMatches(tokens, NEGATIVE_WORDS);
  const emotionCounts = Object.fromEntries(
    Object.entries(EMOTION_LEXICONS).map(([emotion, lexicon]) => [
      emotion,
      countMatches(tokens, lexicon),
    ]),
  ) as Record<DominantEmotion, number>;
  const dominantEmotion = dominantEmotionFromCounts(emotionCounts);
  const sentimentTotal = positiveCount + negativeCount;
  const rawValence =
    sentimentTotal === 0 ? 0 : (positiveCount - negativeCount) / Math.max(1, sentimentTotal);
  const exclamationBoost = Math.min(0.4, (text.match(/!/g) ?? []).length * 0.08);
  const ellipsisPenalty = Math.min(0.2, (text.match(/\.{3,}/g) ?? []).length * 0.05);
  const letters = [...text].filter((char) => /[a-z]/i.test(char));
  const uppercaseLetters = letters.filter((char) => /[A-Z]/.test(char)).length;
  const capsRatio = letters.length === 0 ? 0 : uppercaseLetters / letters.length;
  const arousal = clamp(
    0.05 +
      exclamationBoost +
      Math.min(0.3, capsRatio * 0.5) +
      (dominantEmotion === "anger" || dominantEmotion === "fear" || dominantEmotion === "surprise"
        ? 0.12
        : dominantEmotion === "joy" || dominantEmotion === "curiosity"
          ? 0.08
          : 0) -
      ellipsisPenalty,
    0,
    1,
  );
  const confidence = clamp(sentimentTotal / 3 + exclamationBoost + capsRatio * 0.5, 0, 1);

  return {
    signal: affectiveSignalSchema.parse({
      valence: clamp(rawValence, -1, 1),
      arousal,
      dominant_emotion: dominantEmotion,
    }),
    confidence,
  };
}

function parseFallbackResponse(text: string): AffectiveSignal {
  let raw: unknown;

  try {
    raw = JSON.parse(text) as unknown;
  } catch (error) {
    throw new LLMError("Affective extractor returned non-JSON output", {
      cause: error,
      code: "AFFECTIVE_OUTPUT_INVALID",
    });
  }

  return affectiveFallbackSchema.parse(raw);
}

export type AffectiveExtractorOptions = {
  llmClient?: LLMClient;
  model?: string;
  useLlmFallback?: boolean;
};

export class AffectiveExtractor {
  constructor(private readonly options: AffectiveExtractorOptions = {}) {}

  async analyze(text: string, recentHistory: readonly string[] = []): Promise<AffectiveSignal> {
    const heuristic = heuristicAnalysis(text);
    const tokenCount = tokenizeText(text).size;

    if (
      this.options.useLlmFallback !== true ||
      this.options.llmClient === undefined ||
      this.options.model === undefined ||
      tokenCount <= 40 ||
      heuristic.confidence >= 0.25
    ) {
      return heuristic.signal;
    }

    const response = await this.options.llmClient.complete({
      model: this.options.model,
      system:
        'Infer a grounded affective signal from text. Return strict JSON with {"valence":-1..1,"arousal":0..1,"dominant_emotion":"joy|sadness|fear|anger|surprise|curiosity|neutral|null"} and no surrounding prose.',
      messages: [
        {
          role: "user",
          content: JSON.stringify({
            text,
            recent_history: recentHistory.slice(-3),
          }),
        },
      ],
      max_tokens: 120,
      budget: "perception-affective",
    });

    return parseFallbackResponse(response.text);
  }
}

export function analyzeAffectiveSignalHeuristically(text: string): AffectiveSignal {
  return heuristicAnalysis(text).signal;
}
