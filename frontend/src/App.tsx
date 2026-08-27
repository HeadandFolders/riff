import { useEffect, useMemo, useState } from "react";

import { fetchGraph, fetchKingdoms, fetchMisconceptions } from "./api";
import { KingdomMap } from "./KingdomMap";
import type { GraphNode, GraphView, Kingdom, Misconception } from "./types";

export default function App() {
  const [graph, setGraph] = useState<GraphView | null>(null);
  const [kingdoms, setKingdoms] = useState<Kingdom[]>([]);
  const [misconceptions, setMisconceptions] = useState<Misconception[]>([]);
  const [kingdomId, setKingdomId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function load(kingdom: string | null) {
    setLoading(true);
    setError(null);
    try {
      const [graphView, kingdomList, misconceptionList] = await Promise.all([
        fetchGraph(kingdom),
        fetchKingdoms(),
        fetchMisconceptions(),
      ]);
      setGraph(graphView);
      setKingdoms(kingdomList);
      setMisconceptions(misconceptionList);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load(kingdomId);
  }, [kingdomId]);

  const selectedMisconceptions = useMemo(() => {
    if (!selected || selected.kind !== "concept") return [];
    const label = selected.label.toLowerCase();
    return misconceptions.filter((m) => m.concept_label.toLowerCase() === label);
  }, [selected, misconceptions]);

  const counts = useMemo(() => {
    const nodes = graph?.nodes ?? [];
    return {
      papers: nodes.filter((n) => n.kind === "paper").length,
      concepts: nodes.filter((n) => n.kind === "concept").length,
      misconceived: nodes.filter((n) => n.understanding === "misconceived").length,
    };
  }, [graph]);

  return (
    <div className="app">
      <header>
        <div className="brand">
          <strong>riff</strong>
          <span>research kingdoms</span>
        </div>

        <div className="controls">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search ideas…"
            spellCheck={false}
          />
          <select
            value={kingdomId ?? ""}
            onChange={(event) => setKingdomId(event.target.value || null)}
          >
            <option value="">All kingdoms</option>
            {kingdoms.map((kingdom) => (
              <option key={kingdom.id} value={kingdom.id}>
                {kingdom.label}
              </option>
            ))}
          </select>
          <button onClick={() => void load(kingdomId)}>Refresh</button>
        </div>

        <div className="counts">
          <span>{counts.papers} castles</span>
          <span>{counts.concepts} ideas</span>
          {counts.misconceived > 0 && (
            <span className="warn">{counts.misconceived} misconceptions</span>
          )}
        </div>
      </header>

      <main>
        {error && (
          <div className="notice error">
            <strong>Could not reach the API.</strong> {error}
            <p>Start the backend with <code>uvicorn app.main:app --port 8080</code>.</p>
          </div>
        )}

        {!error && loading && !graph && <div className="notice">Loading the map…</div>}

        {!error && graph && graph.nodes.length === 0 && (
          <div className="notice">
            <strong>No ideas mapped yet.</strong>
            <p>Add a paper to the queue and read a section — stones and concepts appear here as you go.</p>
          </div>
        )}

        {!error && graph && graph.nodes.length > 0 && (
          <KingdomMap
            data={graph}
            query={query}
            selectedId={selected?.id ?? null}
            onSelect={setSelected}
          />
        )}

        {selected && (
          <aside>
            <div className="panel-head">
              <span className="kind">{selected.kind}</span>
              <button onClick={() => setSelected(null)}>Close</button>
            </div>
            <h2>{selected.label}</h2>

            {selected.kind === "concept" && (
              <p className="verdict" data-level={selected.understanding ?? "unassessed"}>
                {selected.understanding ?? "unassessed"}
              </p>
            )}

            <dl>
              <dt>Connections</dt>
              <dd>{selected.degree}</dd>
              {selected.kingdom_id && (
                <>
                  <dt>Kingdom</dt>
                  <dd>
                    {kingdoms.find((k) => k.id === selected.kingdom_id)?.label ??
                      selected.kingdom_id}
                  </dd>
                </>
              )}
            </dl>

            {selectedMisconceptions.length > 0 && (
              <section>
                <h3>Misconceptions on record</h3>
                {selectedMisconceptions.map((item) => (
                  <div key={item.id} className="misconception">
                    <p className="belief">You believed: {item.belief}</p>
                    <p className="correction">{item.correction}</p>
                    <p className="meta">
                      {item.severity} · seen {item.times_observed}×
                      {item.status === "recurring" ? " · recurring" : ""}
                    </p>
                  </div>
                ))}
              </section>
            )}
          </aside>
        )}
      </main>

      {graph?.truncated && (
        <footer>Map truncated at the node limit — filter by kingdom to see detail.</footer>
      )}
    </div>
  );
}
