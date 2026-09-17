import React, { useState } from "react";
import { Play, Clapperboard } from "lucide-react";
import { SearchResult, mediaUrl, generateClip } from "../api";

interface Props {
  result: SearchResult;
  index: number;
  onJump: (timestamp: number, videoId: string) => void;
}

function formatTs(sec: number) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function ResultCard({ result, index, onJump }: Props) {
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
      // ignore
    } finally {
      setClipLoading(false);
    }
  };

  return (
    <div className="flex gap-3 p-3 rounded-xl bg-slate-900/70 border border-slate-800 hover:border-slate-600 transition">
      <div className="text-slate-500 text-sm font-mono w-6 shrink-0 pt-1">
        {String(index + 1).padStart(2, "0")}
      </div>
      <img
        src={mediaUrl(result.thumbnail_url)}
        alt=""
        className="w-28 h-16 object-cover rounded-lg bg-slate-800 shrink-0"
        onError={(e) => {
          (e.target as HTMLImageElement).style.display = "none";
        }}
      />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium truncate">{result.video_name}</p>
        <p className="text-xs text-slate-400 mt-0.5">{formatTs(result.timestamp_seconds)}</p>
        <p className="text-xs text-emerald-400 mt-0.5">
          Similarity {(result.score * 100).toFixed(1)}%
        </p>
        <div className="flex gap-2 mt-2">
          <button
            onClick={() => onJump(result.timestamp_seconds, result.video_id)}
            className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-brand-600/80 hover:bg-brand-500 transition"
          >
            <Play className="h-3 w-3" /> Jump
          </button>
          <button
            onClick={handleClip}
            disabled={clipLoading}
            className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 transition disabled:opacity-50"
          >
            <Clapperboard className="h-3 w-3" />
            {clipLoading ? "…" : "Clip"}
          </button>
        </div>
      </div>
    </div>
  );
}
