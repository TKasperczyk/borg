# Borg Assessor

The assessor is a conversational evaluation harness. It drives a fresh Borg instance through
multi-turn scenarios, reads Borg's trace summaries, and produces a markdown verdict report.

This is separate from `eval/metrics/`: eval metrics are score-producing modules for bounded
single-shot checks, while the assessor is a scenario driver that lets a Claude assessor decide
what to ask next and when enough evidence exists.

## Run

```sh
pnpm assess
pnpm assess --scenario recall
pnpm assess --scenario recall --out /tmp/borg-assessor.md --keep
pnpm assess --mock
```

`--mock` uses a deterministic fake assessor and fake Borg LLM, so it is suitable for CI smoke
checks. Real runs use `claude-sonnet-4-6` for the assessor and leave Borg under test on its
configured model slots. Real mode is selected automatically when `ANTHROPIC_API_KEY`,
`ANTHROPIC_AUTH_TOKEN`, or Claude OAuth credentials are present; otherwise the CLI falls back to
mock mode. Use `--real` or `--mock` to force a mode.

Useful limits:

```sh
pnpm assess --max-turns 8 --max-llm-calls 20
```

Each scenario gets isolated storage and trace files:

- data dir: `/tmp/borg-assessor-<runId>-<scenarioName>`
- trace: `/tmp/borg-assessor-<runId>-<scenarioName>.trace.jsonl`

By default those files are removed when the scenario finishes. Use `--keep` when debugging.

## Scenarios

Scenario files live in `assessor/scenarios/` and export a `Scenario`:

- `name`
- `description`
- `systemPrompt`
- `maxTurns`
- optional `borgConfigOverrides`
- optional `traceAssertions`
- optional `mockConversation`

Add the scenario to `assessor/scenarios/index.ts` so the CLI and smoke tests discover it.

Trace assertions are intentionally simple. They check for tool calls, trace events, response
patterns, stream entries, or the harness-level autonomous wake check. They supplement the
assessor's judgment; they do not replace the conversational verdict.

## Autonomy

The `autonomous-wake` scenario uses clock injection with `ManualClock`. The assessor still only
gets `chat_with_borg` and `read_trace`; the runner advances the clock and ticks autonomy in an
independent trace assertion so the LLM tool surface stays small and deterministic.

## CI

The workflow stub at `.github/workflows/assessor.yml` runs:

```sh
pnpm assess --mock
```

Full real-API assessor runs are operator-triggered and intentionally not part of regular CI.

