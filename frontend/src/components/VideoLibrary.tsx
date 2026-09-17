import React from "react";
import { Video } from "../api";
import { Clock, AlertCircle, CheckCircle, Loader2 } from "lucide-react";

interface Props {
  videos: Video[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRefresh: () => void;
}

function statusIcon(status: Video["status"]) {
  switch (status) {
    case "ready":
      return <CheckCircle className="h-4 w-4 text-emerald-400" />;
    case "processing":
    case "queued":
      return <Loader2 className="h-4 w-4 text-amber-400 animate-spin" />;
    case "failed":
      return <AlertCircle className="h-4 w-4 text-red-400" />;
  }
}

function formatDuration(sec: number | null) {
  if (sec == null) return "—";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function VideoLibrary({ videos, selectedId, onSelect, onRefresh }: Props) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
          Library
        </h2>
        <button
          onClick={onRefresh}
          className="text-xs text-brand-500 hover:text-brand-400"
        >
          Refresh
        </button>
      </div>

      {videos.length === 0 && (
        <p className="text-sm text-slate-500 py-4 text-center">No videos yet</p>
      )}

      {videos.map((v) => (
        <button
          key={v.id}
          onClick={() => onSelect(v.id)}
          className={`
            w-full text-left px-3 py-2.5 rounded-lg transition flex items-start gap-2
            ${selectedId === v.id ? "bg-brand-500/20 border border-brand-500/40" : "bg-slate-900/60 hover:bg-slate-800 border border-transparent"}
          `}
        >
          <div className="mt-0.5">{statusIcon(v.status)}</div>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium truncate">{v.original_filename}</p>
            <div className="flex items-center gap-2 text-xs text-slate-500 mt-0.5">
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {formatDuration(v.duration_seconds)}
              </span>
              <span className="capitalize">{v.status}</span>
            </div>
            {v.error_message && (
              <p className="text-xs text-red-400 mt-1 truncate">{v.error_message}</p>
            )}
          </div>
        </button>
      ))}
    </div>
  );
}
