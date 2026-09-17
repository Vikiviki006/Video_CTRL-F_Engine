import React from "react";
import { SearchResult } from "../api";
import ResultCard from "./ResultCard";

interface Props {
  results: SearchResult[];
  latency: number | null;
  onJump: (timestamp: number, videoId: string) => void;
}

export default function SearchResults({ results, latency, onJump }: Props) {
  if (results.length === 0) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
          Results ({results.length})
        </h3>
        {latency != null && (
          <span className="text-xs text-slate-500">{latency.toFixed(0)} ms</span>
        )}
      </div>
      {results.map((r, i) => (
        <ResultCard key={r.frame_id} result={r} index={i} onJump={onJump} />
      ))}
    </div>
  );
}
