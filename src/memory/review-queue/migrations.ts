export const REVIEW_QUEUE_BASELINE_SQL = `
  CREATE TABLE review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    refs TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    resolved_at INTEGER NULL,
    resolution TEXT NULL
  );
  CREATE INDEX review_queue_belief_revision_target_idx
    ON review_queue (
      json_extract(refs, '$.target_type'),
      json_extract(refs, '$.target_id'),
      created_at DESC,
      id DESC
    )
    WHERE kind = 'belief_revision'
      AND resolved_at IS NULL;
  CREATE INDEX review_queue_kind_idx
    ON review_queue(kind);
  CREATE INDEX review_queue_open_idx
    ON review_queue(resolved_at);
`;
