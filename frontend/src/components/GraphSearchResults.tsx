import React from "react";
import { AlertTriangle, GitBranch } from "lucide-react";
import { GraphSearchResponse, GraphSearchResult } from "../api";
import GraphResultCard from "./GraphResultCard";

interface Props {
  response: GraphSearchResponse | null;
  error: string | null;
  loading: boolean;
  onJump: (timestamp: number, videoId: string) => void;
}

/**
 * Renders graph-aware search results, including the explainability breakdown.
 *
 * The diagnostics row is deliberate: when `graph_available` is false the user
 * is told so explicitly, rather than being shown plain results that look
 * identical to a graph run that simply found nothing. Those are different
 * situations and conflating them would misrepresent what the system did.
 */
export default function GraphSearchResults({
  response,
  error,
  loading,
  onJump,
}: Props) {
  if (loading) {
    return (
      <p className="text-sm text-slate-500">Searching with graph re-ranking…</p>
    );
  }

  if (error) {
    return (
      <p className="text-sm text-amber-400 bg-amber-400/10 px-3 py-2 rounded-lg flex items-center gap-2">
        <AlertTriangle className="h-4 w-4 shrink-0" />
        {error}
      </p>
    );
  }

  if (!response || response.results.length === 0) return null;

  const stage = response.stage_latency_ms || {};
  const stageLine = Object.entries(stage)
    .map(([k, v]) => `${k.replace(/_/g, " ")} ${v.toFixed(0)}ms`)
    .join(" · ");

  return (
    <div className="space-y-2">
      <div className="flex flex-col gap-1 rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <GitBranch className="h-3.5 w-3.5" />
            Graph Results ({response.results.length})
          </h3>
          <span className="text-xs text-slate-500">
            {response.latency_ms.toFixed(0)} ms
          </span>
        </div>

        <p className="text-[11px] text-slate-500">
          {response.graph_available ? (
            <>
              depth {response.graph_depth} · {response.graph_expanded_nodes} nodes ·{" "}
              {response.graph_traversed_edges} edges traversed ·{" "}
              {response.candidate_pool_size} candidates
              {response.graph_truncated_nodes && " · node cap reached"}
              {response.graph_truncated_depth && " · depth cap reached"}
            </>
          ) : (
            <>
              No graph content available for these results — this video has no
              relational graph yet, so ranking fell back to vector similarity
              alone. This is not the same as a graph run that found nothing.
            </>
          )}
        </p>

        {stageLine && (
          <p className="text-[10px] text-slate-600 font-mono">{stageLine}</p>
        )}
      </div>

      {response.results.map((r: GraphSearchResult, i: number) => (
        <GraphResultCard
          key={r.frame_id}
          result={r}
          index={i}
          onJump={onJump}
        />
      ))}
    </div>
  );
}
