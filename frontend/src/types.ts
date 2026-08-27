export type Understanding =
  | "unassessed"
  | "solid"
  | "partial"
  | "misconceived"
  | "absent";

export type NodeKind = "kingdom" | "paper" | "concept" | "stone";

export type EdgeKind =
  | "prerequisite"
  | "similarity"
  | "branch"
  | "member"
  | "appears_in";

export interface GraphNode {
  id: string;
  kind: NodeKind;
  label: string;
  kingdom_id: string | null;
  understanding: Understanding | null;
  misconception_count: number;
  degree: number;
  paper_id: string | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  kind: EdgeKind;
  weight: number;
}

export interface GraphView {
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated: boolean;
}

export interface Kingdom {
  id: string;
  label: string;
  summary: string | null;
  paper_ids: string[];
}

export interface Misconception {
  id: string;
  concept_label: string;
  belief: string;
  correction: string;
  severity: "minor" | "moderate" | "blocking";
  status: "open" | "addressed" | "recurring";
  times_observed: number;
}
