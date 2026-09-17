import React, { useCallback, useState, useRef } from "react";
import { Upload, Film } from "lucide-react";
import { uploadVideo, Video } from "../api";

interface Props {
  onUploaded: (video: Video) => void;
}

export default function VideoUploader({ onUploaded }: Props) {
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      setProgress(0);
      try {
        const video = await uploadVideo(file, setProgress);
        onUploaded(video);
        setProgress(null);
      } catch (e: any) {
        setError(e?.response?.data?.detail || "Upload failed");
        setProgress(null);
      }
    },
    [onUploaded]
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="w-full">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition
          ${dragging ? "border-brand-500 bg-brand-500/10" : "border-slate-700 hover:border-slate-500"}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".mp4,.mov,.mkv,.webm,.avi,video/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />
        <Film className="mx-auto mb-3 h-10 w-10 text-slate-400" />
        <p className="text-slate-300 font-medium">
          Drag & drop a video or <span className="text-brand-500">choose file</span>
        </p>
        <p className="text-sm text-slate-500 mt-1">
          MP4, MOV, MKV, WebM, AVI · max 2 GB
        </p>
      </div>

      {progress !== null && (
        <div className="mt-3">
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-brand-500 transition-all duration-200"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-slate-400 mt-1 text-center">{progress}%</p>
        </div>
      )}

      {error && (
        <p className="mt-2 text-sm text-red-400 text-center">{error}</p>
      )}
    </div>
  );
}
