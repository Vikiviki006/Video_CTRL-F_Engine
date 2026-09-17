import React, { useRef, useEffect } from "react";
import { mediaUrl } from "../api";

interface Props {
  videoUrl: string | null;
  seekTo: number | null;
  onTimeUpdate?: (t: number) => void;
}

export default function VideoPlayer({ videoUrl, seekTo, onTimeUpdate }: Props) {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (seekTo != null && ref.current) {
      ref.current.currentTime = seekTo;
      ref.current.play().catch(() => {});
    }
  }, [seekTo]);

  if (!videoUrl) {
    return (
      <div className="aspect-video bg-slate-900 rounded-xl flex items-center justify-center text-slate-500">
        Select a video to play
      </div>
    );
  }

  return (
    <video
      ref={ref}
      src={mediaUrl(videoUrl)}
      controls
      className="w-full aspect-video rounded-xl bg-black"
      onTimeUpdate={() => {
        if (ref.current && onTimeUpdate) onTimeUpdate(ref.current.currentTime);
      }}
    />
  );
}
