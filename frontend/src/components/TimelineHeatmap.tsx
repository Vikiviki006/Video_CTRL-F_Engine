import React from "react";
import { SearchResult } from "../api";

interface Props {
  results: SearchResult[];
  duration: number | null;
  currentTime?: number;
  onSeek: (t: number) => void;
}

export default function TimelineHeatmap({ results, duration, currentTime, onSeek }: Props) {
  if (!duration || duration <= 0 || results.length === 0) return null;

  return (
    <div className="mt-3">
      <div className="flex justify-between text-xs text-slate-500 mb-1">
        <span>0:00</span>
        <span>
          {Math.floor(duration / 60)}:{(Math.floor(duration) % 60).toString().padStart(2, "0")}
        </span>
      </div>
      <div className="relative h-8 bg-slate-900 rounded-lg overflow-hidden border border-slate-800">
        {/* Heat markers */}
        {results.map((r) => {
          const left = (r.timestamp_seconds / duration) * 100;
          const intensity = Math.min(1, Math.max(0.3, r.score));
          return (
            <button
              key={r.frame_id}
              title={`${r.timestamp_seconds.toFixed(1)}s · ${(r.score * 100).toFixed(0)}%`}
              onClick={() => onSeek(r.timestamp_seconds)}
              className="absolute top-1 bottom-1 w-1.5 rounded-sm hover:brightness-125 transition"
              style={{
                left: `${left}%`,
                backgroundColor: `rgba(14, 165, 233, ${intensity})`,
              }}
            />
          );
        })}
        {/* Playhead */}
        {currentTime != null && (
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white/80 pointer-events-none"
            style={{ left: `${(currentTime / duration) * 100}%` }}
          />
        )}
      </div>
    </div>
  );
}
