# Borg demo server contract — facts gathered from demo/server/src (for wiring the mock later)

## REST
- GET /api/state?session= → { active_session, audiences[], counts{turns, commitments, open_qs, open_reviews, dream_audit_rows}, current_mood, runtime{model, embedding{model,dims}}, version }
- GET /api/sessions → { sessions[] } ; POST /api/sessions/operator ; POST /api/sessions/:id/participation {policy, reason}
- GET /api/stream?session&kind&audience ; GET /api/turns?session&cursor → rows{turn_id, started_at, audience, outcome, suppression_reason}, next_cursor
  - outcome: "emitted" | "failed" | "observed" | SuppressionOutcomeClass (from classifySuppressionReason)
  - stream entry kinds incl: user_msg, agent_msg, agent_observed, agent_suppressed, dream_report
- GET /api/turns/:id/ledger → cached evidence ledger
- POST /api/turn — send a message (multipart for attachments)
- GET /api/memory/bands?session → 8 bands: episodic(what happened), semantic(what Borg believes), procedural(how Borg solves things), affective(mood and trajectory), self(values/goals/traits/narrative; stats: values,goals,traits,open_questions,growth_markers,periods), commitments(scoped promises and boundaries; active/revoked), social(per-entity trust and history), relational(evidence-backed relationship facts). Each: {id,n,name,desc,count,stats[{k,v}]}
- GET /api/memory/bands/:id → browse items; affective returns {current, history} mood
- GET /api/semantic/graph?limit → { nodes[{id,label,display_label,status: active|contested|contradicted|quarantined, kind, edge_count}], edges[{id,source,target,type,weight}], total_nodes, total_edges, rendered }
- GET /api/semantic/nodes/:id, /api/semantic/edges/:id
- GET /api/identity → { values, goals, traits, open_questions, growth_markers, periods, open_question_events }
- POST /api/identity/values | goals | growth-markers ; PATCH /api/identity/goals/:id {action: abandon{reason} | bump{delta}} ; PATCH /api/identity/open-questions/:id
- GET /api/commitments?state= → mapCommitment: { id, text, type, kind, enforcement_class, critical_domain, state, priority, directive_family, audience, made_to, about, committed_by, source, created_at, expires_at, expired_at, revoked_at, revoked_reason, superseded_by_id, last_reinforced_at }
- POST /api/commitments {type,kind,directive,priority,audience,made_to,about,expires_at,directive_family} (enforcementClass forced "advisory") ; POST /api/commitments/:id/revoke {reason}
- GET /api/creator-directives?status=active|revoked|superseded|all → { id, kind, text(=operational_directive ?? canonical_fact), canonical_fact, operational_directive, activation_scope, activation_allowed/excluded_entity_ids, content_scope, mention_policy, status, subject_kind, subject_entity_id/name, priority, superseded_by_id, revoked_reason, created_at, updated_at }
- POST /api/creator-directives/:id/revoke {reason} ; /:id/supersede {replacement_id}

## Reviews
- REVIEW_KINDS: contradiction, duplicate, new_insight, misattribution, temporal_drift, identity_inconsistency, correction, belief_revision, skill_split, creator_directive_reconciliation, commitment_reconciliation
- REVIEW_RESOLUTIONS (generic PATCH /api/reviews/:id {action, note?, winner_node_id?}): keep_both, supersede, invalidate, dismiss, accept, reject, keep, weaken, archive_node, invalidate_edge
- creator_directive_reconciliation MUST go through POST /api/reviews/:id/creator-directive-reconciliation: {action:"supersede", survivor_id, reason?} | {action:"keep", reason?} (409 if patched generically)
- belief_revision rows only allow "dismiss" via PATCH /api/dream/review/:id (apply happens via belief-reviser)
- Correction flow: GET /api/correction/reviews ; GET /api/correction/:id/why ; POST /api/correction/:id/forget ; POST /api/correction/:id/correct {patch, reason} ; POST /api/correction/semantic-edges/:id/invalidate {at?, reason?} ; PATCH /api/correction/reviews/:id {action: accept|reject, note}
- Review row: { id, kind, refs, reason, created_at, resolved_at, resolution }

## Dream
- GET /api/dream/state → { processes[{name, description, last_run_at, last_status, last_audit_id, budget, enabled}], pending_extraction_episodes, schedule[], dream_reports[], audit_rows[{id,run_id,process,action,targets,reversal,applied_at,reverted_at,reverted_by}], belief_revision_rows[], scheduler{enabled, light_interval_ms, heavy_interval_ms, optimize_storage, light_processes, heavy_processes, process_budgets} }
- POST /api/dream/plan {processes?[]} → plan (cached by plan_id) ; POST /api/dream/apply {processes?, plan_id?}
- GET /api/dream/audit ; POST /api/dream/audit/:id/revert
- 13 OFFLINE_PROCESS_NAMES + description: consolidator "merge redundant episodes"; reflector "episodes to semantic insights"; semantic-extractor "extract graph facts"; curator "salience, heat, archive, decay"; overseer "flag substrate issues"; associator "link related memory records"; review-resolver "process review queue items"; creator-directive-reconciler "reconcile redundant or conflicting creator directives"; ruminator "open-question rumination"; self-narrator "autobiography and growth markers"; procedural-synthesizer "skill abstractions"; belief-reviser "invalidate, weaken, contradict"; commitment-reconciler "reconcile redundant or conflicting commitments"

## Prompts / settings
- GET /api/prompts → {blocks: prompts.list()} ; GET /api/prompts/assembled (previewAssembledFraming) ; PUT /api/prompts/:key {text ≤50k} ; DELETE /api/prompts/:key (reset to default). Keys = PROMPT_KEYS enum.
- POST /api/entities/creator {name} ; POST /api/entities/:id/borg-role {role} ; POST /api/entities {name, kind}
- POST /api/admin/reset {confirm:"RESET"}

## WS /api/live
- client → {type: subscribe|unsubscribe, session_id} ; subscribe_global/unsubscribe_global (default global on). 64-frame/60s ring buffer replayed on subscribe.
- frames (all have ts):
  - turn:phase:started|completed|failed {event, data{turn_id, session_id, phase, duration_ms,…}}
  - turn:phase:detail {turn_id, phase?, event, summary}  (summary = "k=v k=v" ≤200ch)
  - turn:token {turn_id, phase: "delib"|"final", chunk_text, sequence}
  - turn:token:flush {turn_id, phase, full_text}
  - turn:delib_path {turn_id, path: system_1|system_2}
  - turn:final_attempt {turn_id, attempt} (commitment-guard regeneration)
  - evidence_ledger:built {turn_id, ledger}
  - turn:terminal {event, data} — TurnTerminalOutcome: reflected | suppressed_closure | suppressed_generation_gate | suppressed_action | aborted | error
  - stream:append {session_id, entries[]}
  - maintenance:tick {cadence: light|heavy|manual, status, processes[], changed, changes, errors, pending_extraction_episodes, run_id, duration_ms, reason}
  - dream:process:started|completed {process, run_id, phase: plan|apply, duration_ms, errors, candidates_accepted}
  - borg:reset

## Cognition pipeline (turn phases, ARCHITECTURE.md)
pre-turn catch-up + audience resolution → perception (mode/entities/affect/temporal) → frame-anomaly → extraction (directives/corrections/goals/actions) → retrieval + evidence ledger (+ shared-state compile) → deliberation (system_1 fast / system_2 deliberate; tokens phase "delib") → finalization (exactly one of EmitAnswer | EmitObserve | EmitNoOutput | EmitSelfReport; tokens phase "final") → guards (commitment/closure/internal-id/safety; may regenerate) → reflection (mood, social, working memory)
- Deliberate silence = EmitNoOutput → persisted agent_suppressed marker w/ finalizer-no-output reason. agent_observed = active observation. These are successful turns.
- Mood = { valence, arousal, dominant emotion } + history; trace taxonomy phases: turn_phase, perception, working_memory, retrieval, deliberation, tools, commitments, extraction, discourse, reflection, review, ingestion, offline, maintenance, session
