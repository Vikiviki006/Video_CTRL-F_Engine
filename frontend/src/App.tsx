import React, { useState, useEffect, useCallback } from "react";
import VideoUploader from "./components/VideoUploader";
import VideoLibrary from "./components/VideoLibrary";
import SearchBar from "./components/SearchBar";
import VideoPlayer from "./components/VideoPlayer";
import SearchResults from "./components/SearchResults";
import TimelineHeatmap from "./components/TimelineHeatmap";
import { getVideos, Video } from "./api";
import { useVideoSearch } from "./hooks/useVideoSearch";

export default function App() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [seekTo, setSeekTo] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);

  const { results, loading, error, latency, doSearch } = useVideoSearch();

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
              Semantic Video Search · SigLIP-2 · pgvector
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
            onSearch={(q) => doSearch(q, selectedId)}
            loading={loading}
            disabled={!selected || selected.status !== "ready"}
          />

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
            results={results.filter((r) => r.video_id === selectedId)}
            duration={selected?.duration_seconds ?? null}
            currentTime={currentTime}
            onSeek={(t) => setSeekTo(t)}
          />

          <SearchResults
            results={results}
            latency={latency}
            onJump={handleJump}
          />
        </section>
      </main>
    </div>
  );
}
