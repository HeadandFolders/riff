import type { GraphView, Kingdom, Misconception } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function json<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} — ${path}`);
  }
  return (await response.json()) as T;
}

export function fetchGraph(kingdomId?: string | null): Promise<GraphView> {
  const query = kingdomId ? `?kingdom_id=${encodeURIComponent(kingdomId)}` : "";
  return json<GraphView>(`/graph${query}`);
}

export function fetchKingdoms(): Promise<Kingdom[]> {
  return json<Kingdom[]>("/kingdoms");
}

export function fetchMisconceptions(): Promise<Misconception[]> {
  return json<Misconception[]>("/misconceptions");
}
