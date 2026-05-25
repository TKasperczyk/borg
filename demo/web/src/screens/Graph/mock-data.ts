// mocked for v1 -- real-scale wiring deferred
export type GraphNode = {
  id: string;
  label: string;
  kind: "entity" | "concept" | "proposition";
  x: number;
  y: number;
  hot: number;
};

export type GraphEdge = {
  from: string;
  to: string;
  rel: string;
  state?: "contradicted";
};

export const GRAPH_NODES: GraphNode[] = [
  { id: "n_alice", label: "alice", kind: "entity", x: 500, y: 300, hot: 0.85 },
  { id: "n_maya", label: "maya", kind: "entity", x: 700, y: 180, hot: 0.62 },
  { id: "n_tom", label: "tom", kind: "entity", x: 300, y: 180, hot: 0.92 },
  { id: "n_borg", label: "borg", kind: "entity", x: 500, y: 460, hot: 0.78 },
  { id: "n_book", label: "attention book", kind: "concept", x: 870, y: 250, hot: 0.51 },
  { id: "n_focus", label: "focus / attention", kind: "concept", x: 920, y: 380, hot: 0.34 },
  { id: "n_pg", label: "pgvector", kind: "concept", x: 170, y: 340, hot: 0.78 },
  { id: "n_hnsw", label: "hnsw", kind: "concept", x: 80, y: 440, hot: 0.65 },
  { id: "n_ivf", label: "ivfflat", kind: "concept", x: 70, y: 260, hot: 0.2 },
  { id: "n_priv", label: "audience privacy", kind: "proposition", x: 290, y: 70, hot: 0.71 },
  { id: "n_fam", label: "fam:home", kind: "entity", x: 500, y: 80, hot: 0.46 },
  { id: "n_skill", label: "skill:debug-pgvector", kind: "concept", x: 150, y: 160, hot: 0.55 },
  { id: "n_close", label: "closure-pressure rule", kind: "proposition", x: 730, y: 470, hot: 0.42 }
];

export const GRAPH_EDGES: GraphEdge[] = [
  { from: "n_alice", to: "n_maya", rel: "knows" },
  { from: "n_alice", to: "n_tom", rel: "knows" },
  { from: "n_alice", to: "n_borg", rel: "talks_to" },
  { from: "n_tom", to: "n_borg", rel: "talks_to" },
  { from: "n_maya", to: "n_book", rel: "recommended" },
  { from: "n_alice", to: "n_book", rel: "reading" },
  { from: "n_book", to: "n_focus", rel: "category" },
  { from: "n_alice", to: "n_pg", rel: "works_on" },
  { from: "n_pg", to: "n_hnsw", rel: "indexed_by" },
  { from: "n_pg", to: "n_ivf", rel: "indexed_by", state: "contradicted" },
  { from: "n_hnsw", to: "n_ivf", rel: "supersedes" },
  { from: "n_skill", to: "n_pg", rel: "applies_to" },
  { from: "n_priv", to: "n_tom", rel: "scoped_to" },
  { from: "n_priv", to: "n_borg", rel: "constrains" },
  { from: "n_fam", to: "n_tom", rel: "includes" },
  { from: "n_fam", to: "n_alice", rel: "includes" },
  { from: "n_close", to: "n_borg", rel: "constrains" }
];

export const SELECTED_NODE = "n_alice";
