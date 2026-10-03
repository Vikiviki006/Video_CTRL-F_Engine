import React, { useState, useEffect, useCallback } from "react";
import { GitBranch, Layers } from "lucide-react";
import VideoUploader from "./components/VideoUploader";
import VideoLibrary from "./components/VideoLibrary";
import SearchBar from "./components/SearchBar";
import VideoPlayer from "./components/VideoPlayer";
import SearchResults from "./components/SearchResults";
import GraphSearchResults from "./components/GraphSearchResults";
import TimelineHeatmap from "./components/TimelineHeatmap";
import { getVideos, Video } from "./api";
import { useVideoSearch } from "./hooks/useVideoSearch";
import { useGraphSearch } from "./hooks/useGraphSearch";

type SearchMode = "vector" | "graph";

export default function App() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [seekTo, setSeekTo] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);

  // Search mode is explicit rather than automatic. Both paths are available
  // at all times so the effect of graph re-ranking can be observed directly on
  // the same query, and so the vector-only path keeps behaving exactly as it
  // always has.
  const [mode, setMode] = useState<SearchMode>("vector");
  const [graphDepth, setGraphDepth] = useState(2);

  const vectorSearch = useVideoSearch();
  const graphSearch = useGraphSearch(graphDepth);

  const { results, loading, error } =
    mode === "vector" ? vectorSearch : graphSearch;

  const refresh = useCallback(async () => {
    try {
      const data = await getVideos();
      setVideos(data.videos);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [refresh]);

  const selected = videos.find((v) => v.id === selectedId) || null;
  const videoUrl = selected ? `/media/videos/${selected.filename}` : null;

  const handleJump = (ts: number, videoId: string) => {
    if (videoId !== selectedId) setSelectedId(videoId);
    setSeekTo(ts);
    // reset so same timestamp can be jumped again later
    setTimeout(() => setSeekTo(null), 100);
  };

  const runSearch = (q: string) => {
    if (mode === "vector") {
      vectorSearch.doSearch(q, selectedId);
    } else {
      graphSearch.doSearch(q, selectedId);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold tracking-tight">
              VideoCtrl-F
            </h1>
            <p className="text-xs text-slate-400">
              Semantic Video Search · SigLIP-2 · pgvector · Relational Graph
            </p>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left sidebar */}
        <aside className="lg:col-span-3 space-y-6">
          <VideoUploader
            onUploaded={(v) => {
              setVideos((prev) => [v, ...prev]);
              setSelectedId(v.id);
            }}
          />
          <VideoLibrary
            videos={videos}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onRefresh={refresh}
          />
        </aside>

        {/* Main content */}
        <section className="lg:col-span-9 space-y-4">
          <SearchBar
            onSearch={runSearch}
            loading={loading}
            disabled={!selected || selected.status !== "ready"}
          >
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <div className="inline-flex rounded-lg overflow-hidden border border-slate-700">
                <button
                  onClick={() => setMode("vector")}
                  className={`flex items-center gap-1 px-3 py-1 text-xs transition ${
                    mode === "vector"
                      ? "bg-slate-700 text-slate-100"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Layers className="h-3 w-3" /> Vector only
                </button>
                <button
                  onClick={() => setMode("graph")}
                  className={`flex items-center gap-1 px-3 py-1 text-xs transition ${
                    mode === "graph"
                      ? "bg-brand-600 text-white"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <GitBranch className="h-3 w-3" /> Vector + Graph
                </button>
              </div>

              {mode === "graph" && (
                <label className="flex items-center gap-2 text-xs text-slate-400">
                  Depth
                  <select
                    value={graphDepth}
                    onChange={(e) => setGraphDepth(Number(e.target.value))}
                    className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200"
                  >
                    {[0, 1, 2, 3].map((d) => (
                      <option key={d} value={d}>
                        {d}
                      </option>
                    ))}
                  </select>
                  <span className="text-slate-600">
                    (0 = vector only, the evaluation baseline)
                  </span>
                </label>
              )}
            </div>
          </SearchBar>

          {error && (
            <p className="text-sm text-amber-400 bg-amber-400/10 px-3 py-2 rounded-lg">
              {error}
            </p>
          )}

          <VideoPlayer
            videoUrl={videoUrl}
            seekTo={seekTo}
            onTimeUpdate={setCurrentTime}
          />

          <TimelineHeatmap
            results={results.map((r) => ({
              frame_id: r.frame_id,
              timestamp_seconds: r.timestamp_seconds,
              score: "final_score" in r ? r.final_score : r.score,
            }))}
            duration={selected?.duration_seconds ?? null}
            currentTime={currentTime}
            onSeek={(t) => setSeekTo(t)}
          />

          {mode === "vector" ? (
            <SearchResults
              results={vectorSearch.results}
              latency={vectorSearch.latency}
              onJump={handleJump}
            />
          ) : (
            <GraphSearchResults
              response={graphSearch.response}
              error={graphSearch.error}
              loading={graphSearch.loading}
              onJump={handleJump}
            />
          )}
        </section>
      </main>
    </div>
  );
}
