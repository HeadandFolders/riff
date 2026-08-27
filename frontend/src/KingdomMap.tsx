import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";

import type { EdgeKind, GraphNode, GraphView, Understanding } from "./types";

interface SimNode extends GraphNode, SimulationNodeDatum {}

interface SimLink extends SimulationLinkDatum<SimNode> {
  kind: EdgeKind;
}

const UNDERSTANDING_COLOR: Record<Understanding, string> = {
  misconceived: "#e5484d",
  partial: "#f5a623",
  solid: "#30a46c",
  unassessed: "#6b7280",
  absent: "#4b5563",
};

const PAPER_COLOR = "#5b8def";
const KINGDOM_COLOR = "#8b5cf6";

function radius(node: SimNode): number {
  if (node.kind === "kingdom") return 30;
  if (node.kind === "paper") return Math.min(8 + node.degree * 0.7, 18);
  return Math.min(4 + node.degree * 0.5, 11);
}

function fill(node: SimNode): string {
  if (node.kind === "kingdom") return KINGDOM_COLOR;
  if (node.kind === "paper") return PAPER_COLOR;
  return UNDERSTANDING_COLOR[node.understanding ?? "unassessed"];
}

interface Props {
  data: GraphView;
  query: string;
  selectedId: string | null;
  onSelect: (node: GraphNode | null) => void;
}

export function KingdomMap({ data, query, selectedId, onSelect }: Props) {
  const wrapper = useRef<HTMLDivElement>(null);
  const simulation = useRef<Simulation<SimNode, SimLink> | null>(null);
  const frame = useRef<number | null>(null);
  const dragging = useRef<SimNode | null>(null);
  const panning = useRef<{ x: number; y: number } | null>(null);

  const [size, setSize] = useState({ width: 900, height: 620 });
  const [view, setView] = useState({ x: 0, y: 0, k: 1 });
  const [hovered, setHovered] = useState<string | null>(null);
  const [, bump] = useState(0);

  const { nodes, links } = useMemo(() => {
    const simNodes: SimNode[] = data.nodes.map((node) => ({ ...node }));
    const byId = new Map(simNodes.map((node) => [node.id, node]));
    const simLinks: SimLink[] = data.edges
      .filter((edge) => byId.has(edge.source) && byId.has(edge.target))
      .map((edge) => ({
        source: byId.get(edge.source)!,
        target: byId.get(edge.target)!,
        kind: edge.kind,
      }));
    return { nodes: simNodes, links: simLinks };
  }, [data]);

  useEffect(() => {
    const element = wrapper.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({
        width: Math.max(entry.contentRect.width, 320),
        height: Math.max(entry.contentRect.height, 360),
      });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const sim = forceSimulation<SimNode, SimLink>(nodes)
      .force(
        "link",
        forceLink<SimNode, SimLink>(links)
          .id((node) => node.id)
          .distance((link) => (link.kind === "member" ? 60 : 90))
          .strength((link) => (link.kind === "appears_in" ? 0.35 : 0.12)),
      )
      .force("charge", forceManyBody<SimNode>().strength((node) => (node.kind === "kingdom" ? -900 : -160)))
      .force("center", forceCenter(size.width / 2, size.height / 2))
      .force("collide", forceCollide<SimNode>().radius((node) => radius(node) + 4))
      .alphaDecay(0.035);

    sim.on("tick", () => {
      if (frame.current !== null) return;
      frame.current = requestAnimationFrame(() => {
        frame.current = null;
        bump((value) => value + 1);
      });
    });

    simulation.current = sim;
    return () => {
      sim.stop();
      if (frame.current !== null) cancelAnimationFrame(frame.current);
      frame.current = null;
      simulation.current = null;
    };
  }, [nodes, links, size.width, size.height]);

  const needle = query.trim().toLowerCase();
  const matches = (node: SimNode) =>
    needle.length === 0 || node.label.toLowerCase().includes(needle);

  function toWorld(event: ReactMouseEvent): { x: number; y: number } {
    const rect = (event.currentTarget as SVGSVGElement).getBoundingClientRect();
    return {
      x: (event.clientX - rect.left - view.x) / view.k,
      y: (event.clientY - rect.top - view.y) / view.k,
    };
  }

  function onWheel(event: ReactWheelEvent<SVGSVGElement>) {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const next = Math.min(Math.max(view.k * (event.deltaY < 0 ? 1.12 : 0.89), 0.25), 4);
    setView({
      k: next,
      x: px - ((px - view.x) / view.k) * next,
      y: py - ((py - view.y) / view.k) * next,
    });
  }

  function onMouseMove(event: ReactMouseEvent<SVGSVGElement>) {
    if (dragging.current) {
      const point = toWorld(event);
      dragging.current.fx = point.x;
      dragging.current.fy = point.y;
      simulation.current?.alpha(0.3).restart();
      return;
    }
    if (panning.current) {
      setView((current) => ({
        ...current,
        x: current.x + event.clientX - panning.current!.x,
        y: current.y + event.clientY - panning.current!.y,
      }));
      panning.current = { x: event.clientX, y: event.clientY };
    }
  }

  function release() {
    if (dragging.current) {
      dragging.current.fx = null;
      dragging.current.fy = null;
      dragging.current = null;
    }
    panning.current = null;
  }

  const labelVisible = (node: SimNode) =>
    node.kind === "kingdom" ||
    node.id === selectedId ||
    node.id === hovered ||
    (node.kind === "paper" && view.k > 0.85) ||
    (node.kind === "concept" && view.k > 1.5) ||
    (needle.length > 0 && matches(node));

  return (
    <div className="map-wrapper" ref={wrapper}>
      <svg
        width={size.width}
        height={size.height}
        onWheel={onWheel}
        onMouseDown={(event) => {
          panning.current = { x: event.clientX, y: event.clientY };
        }}
        onMouseMove={onMouseMove}
        onMouseUp={release}
        onMouseLeave={release}
        onDoubleClick={() => setView({ x: 0, y: 0, k: 1 })}
      >
        <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
          {links.map((link, index) => {
            const source = link.source as SimNode;
            const target = link.target as SimNode;
            const dim = needle.length > 0 && !matches(source) && !matches(target);
            return (
              <line
                key={index}
                x1={source.x ?? 0}
                y1={source.y ?? 0}
                x2={target.x ?? 0}
                y2={target.y ?? 0}
                stroke={link.kind === "prerequisite" ? "#7c8798" : "#39404d"}
                strokeWidth={link.kind === "member" ? 0.6 : 1}
                strokeOpacity={dim ? 0.06 : 0.5}
                strokeDasharray={link.kind === "similarity" ? "3 3" : undefined}
              />
            );
          })}

          {nodes.map((node) => {
            const dim = !matches(node);
            const r = radius(node);
            return (
              <g
                key={node.id}
                transform={`translate(${node.x ?? 0},${node.y ?? 0})`}
                opacity={dim ? 0.15 : 1}
                onMouseEnter={() => setHovered(node.id)}
                onMouseLeave={() => setHovered(null)}
                onMouseDown={(event) => {
                  event.stopPropagation();
                  dragging.current = node;
                }}
                onClick={(event) => {
                  event.stopPropagation();
                  onSelect(node);
                }}
                style={{ cursor: "pointer" }}
              >
                <circle
                  r={r}
                  fill={fill(node)}
                  fillOpacity={node.kind === "kingdom" ? 0.12 : 0.9}
                  stroke={node.id === selectedId ? "#ffffff" : fill(node)}
                  strokeWidth={node.id === selectedId ? 2 : node.kind === "kingdom" ? 1.5 : 0}
                />
                {node.kind === "concept" && node.misconception_count > 0 && (
                  <circle r={r + 3.5} fill="none" stroke={UNDERSTANDING_COLOR.misconceived} strokeWidth={1} />
                )}
                {labelVisible(node) && (
                  <text
                    x={r + 5}
                    y={4}
                    fontSize={node.kind === "kingdom" ? 13 : 11}
                    fill={node.kind === "kingdom" ? "#c4b5fd" : "#c9d1d9"}
                  >
                    {node.label.length > 46 ? `${node.label.slice(0, 46)}…` : node.label}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>

      <div className="map-legend">
        <span><i style={{ background: UNDERSTANDING_COLOR.solid }} />solid</span>
        <span><i style={{ background: UNDERSTANDING_COLOR.partial }} />partial</span>
        <span><i style={{ background: UNDERSTANDING_COLOR.misconceived }} />misconception</span>
        <span><i style={{ background: UNDERSTANDING_COLOR.unassessed }} />unassessed</span>
        <span><i style={{ background: PAPER_COLOR }} />castle</span>
        <span><i style={{ background: KINGDOM_COLOR }} />kingdom</span>
      </div>
    </div>
  );
}
