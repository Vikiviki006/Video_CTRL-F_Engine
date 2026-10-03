import React, { useState } from "react";
import {
  Play,
  Clapperboard,
  ChevronDown,
  ChevronRight,
  Network,
  Info,
} from "lucide-react";
import {
  GraphSearchResult,
  mediaUrl,
  generateClip,
  RelatedNode,
  RelationshipEvidence,
} from "../api";

interface Props {
  result: GraphSearchResult;
  index: number;
  onJump: (timestamp: number, videoId: string) => void;
}

function formatTs(sec: number) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function pct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

/** Human-readable explanation of why an edge exists, from its recorded basis. */
function describeBasis(basis: string | null): string {
  switch (basis) {
    case "deterministic_containment":
      return "structural containment";
    case "timestamp_ordering":
      return "measured timestamp ordering";
    case "measured_cosine_similarity":
      return "measured cosine similarity between frame embeddings";
    default:
      return basis ? basis.replace(/^extractor:/, "extractor: ") : "unspecified basis";
  }
}

const SCORE_TERMS: Array<{ key: string; label: string; color: string }> = [
  { key: "vector", label: "Vector", color: "bg-slate-400" },
  { key: "graph", label: "Graph", color: "bg-sky-500" },
  { key: "temporal", label: "Temporal", color: "bg-amber-500" },
  { key: "confidence", label: "Confidence", color: "bg-emerald-500" },
];

function ScoreBar({ result }: { result: GraphSearchResult }) {
  const entries = SCORE_TERMS.map((t) => ({
    ...t,
    value: result.contributions?.[t.key] ?? 0,
  }));
  const total = entries.reduce((sum, e) => sum + e.value, 0);

  return (
    <div>
      {/* The bar segments are the per-term contributions to final_score.
          They are read straight from the API's `contributions` field, which
          sums back to final_score, so this is a rendering of measured values
          and not a second opinion formed in the browser. */}
      <div className="flex h-2 w-full overflow-hidden rounded-full bg-slate-800">
        {total > 0 &&
          entries.map((e) => (
            <div
              key={e.key}
              className={e.color}
              style={{ width: `${(e.value / total) * 100}%` }}
              title={`${e.label}: ${pct(e.value)}`}
            />
          ))}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-slate-400">
        {entries.map((e) => (
          <span key={e.key} className="flex items-center gap-1">
            <span className={`inline-block h-2 w-2 rounded-full ${e.color}`} />
            {e.label} {pct(e.value)}
          </span>
        ))}
      </div>
    </div>
  );
}

function RelatedNodeList({ nodes }: { nodes: RelatedNode[] }) {
  if (nodes.length === 0) {
    return (
      <p className="text-[11px] text-slate-500">
        No related nodes were reached. With no concept extractor configured, the
        graph only contains SCENE and FRAME nodes, so a result reached purely by
        vector similarity has no related-node chain to show.
      </p>
    );
  }
  return (
    <ul className="space-y-1">
      {nodes.map((n) => (
        <li key={n.id} className="flex items-baseline gap-2 text-[11px]">
          <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[9px] text-slate-300">
            {n.node_type}
          </span>
          <span className="text-slate-200">{n.label || n.entity_key}</span>
          {n.timestamp_seconds != null && (
            <span className="text-slate-500">@{formatTs(n.timestamp_seconds)}</span>
          )}
          <span className="text-slate-500">depth {n.depth ?? "?"}</span>
        </li>
      ))}
    </ul>
  );
}

function RelationshipList({ edges }: { edges: RelationshipEvidence[] }) {
  if (edges.length === 0) {
    return (
      <p className="text-[11px] text-slate-500">
        No edge was traversed to reach this result, so it is a seed: ranked by
        vector similarity alone.
      </p>
    );
  }
  return (
    <ul className="space-y-1">
      {edges.map((e, i) => (
        <li key={`${e.source_node_id}-${e.target_node_id}-${i}`} className="text-[11px]">
          <span className="font-mono text-sky-400">{e.relationship}</span>{" "}
          <span className="text-slate-500">
            weight {e.weight.toFixed(3)} · confidence {e.confidence.toFixed(3)}
          </span>
          <div className="flex items-start gap-1 text-[10px] text-slate-500">
            <Info className="mt-0.5 h-2.5 w-2.5 shrink-0" />
            basis: {describeBasis(e.basis)}
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function GraphResultCard({ result, index, onJump }: Props) {
  const [open, setOpen] = useState(false);
  const [clipLoading, setClipLoading] = useState(false);
  const [clipUrl, setClipUrl] = useState<string | null>(result.clip_url);

  const handleClip = async () => {
    if (clipUrl) {
      window.open(mediaUrl(clipUrl), "_blank");
      return;
    }
    setClipLoading(true);
    try {
      const res = await generateClip(result.frame_id);
      setClipUrl(res.clip_url);
      window.open(mediaUrl(res.clip_url), "_blank");
    } catch {
      // ignore - clip generation is best-effort
    } finally {
      setClipLoading(false);
    }
  };

  return (
    <div className="p-3 rounded-xl bg-slate-900/70 border border-slate-800 hover:border-slate-600 transition">
      <div className="flex gap-3">
        <div className="text-slate-500 text-sm font-mono w-6 shrink-0 pt-1">
          {String(index + 1).padStart(2, "0")}
        </div>
        {result.thumbnail_url && (
          <img
            src={mediaUrl(result.thumbnail_url)}
            alt=""
            className="w-28 h-16 object-cover rounded-lg bg-slate-800 shrink-0"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        )}
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium truncate">
            {result.video_name || "—"}
          </p>
          <p className="text-xs text-slate-400 mt-0.5">
            {formatTs(result.timestamp_seconds)}
          </p>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 mt-1 text-[11px]">
            <span className="text-emerald-400 font-medium">
              final {pct(result.final_score)}
            </span>
            <span className="text-slate-500">vector {result.vector_score.toFixed(3)}</span>
            <span className="text-sky-400">graph {result.graph_score.toFixed(3)}</span>
            <span className="text-amber-400">
              temporal {result.temporal_score.toFixed(3)}
            </span>
            {result.is_seed && (
              <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[9px] text-slate-400">
                seed
              </span>
            )}
          </div>

          <div className="mt-2">
            <ScoreBar result={result} />
          </div>

          <div className="flex gap-2 mt-2 items-center">
            {result.video_id && (
              <button
                onClick={() => onJump(result.timestamp_seconds, result.video_id!)}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-brand-600/80 hover:bg-brand-500 transition"
              >
                <Play className="h-3 w-3" /> Jump
              </button>
            )}
            <button
              onClick={handleClip}
              disabled={clipLoading}
              className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 transition disabled:opacity-50"
            >
              <Clapperboard className="h-3 w-3" />
              {clipLoading ? "…" : "Clip"}
            </button>
            <button
              onClick={() => setOpen((v) => !v)}
              className="flex items-center gap-1 text-xs px-2 py-1 rounded text-slate-400 hover:text-sky-400 transition"
            >
              {open ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              )}
              Why this result?
            </button>
          </div>
        </div>
      </div>

      {open && (
        <div className="mt-3 ml-9 grid gap-3 md:grid-cols-2 border-t border-slate-800 pt-3">
          <div>
            <p className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5 flex items-center gap-1">
              <Network className="h-3 w-3" /> Related nodes
              <span className="text-slate-600">({result.related_nodes.length})</span>
            </p>
            <RelatedNodeList nodes={result.related_nodes} />
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">
              Relationship chain
              <span className="text-slate-600">({result.relationships.length})</span>
            </p>
            <RelationshipList edges={result.relationships} />
          </div>
        </div>
      )}
    </div>
  );
}
