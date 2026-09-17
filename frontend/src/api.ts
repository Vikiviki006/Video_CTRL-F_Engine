import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const client = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
});

export interface Video {
  id: string;
  filename: string;
  original_filename: string;
  duration_seconds: number | null;
  fps: number | null;
  frame_count: number | null;
  status: "queued" | "processing" | "ready" | "failed";
  error_message: string | null;
  created_at: string;
}

export interface SearchResult {
  frame_id: string;
  video_id: string;
  video_name: string;
  timestamp_seconds: number;
  score: number;
  thumbnail_url: string;
  clip_url: string | null;
  group_id: number | null;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total: number;
  latency_ms: number;
}

export async function uploadVideo(
  file: File,
  onProgress?: (pct: number) => void
): Promise<Video> {
  const form = new FormData();
  form.append("file", file);
  const res = await client.post<Video>("/api/v1/videos", form, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (e.total && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    },
  });
  return res.data;
}

export async function getVideos(): Promise<{ videos: Video[]; total: number }> {
  const res = await client.get("/api/v1/videos");
  return res.data;
}

export async function getVideo(id: string): Promise<Video> {
  const res = await client.get(`/api/v1/videos/${id}`);
  return res.data;
}

export async function search(
  query: string,
  videoId?: string | null,
  topK = 10
): Promise<SearchResponse> {
  const res = await client.post<SearchResponse>("/api/v1/search", {
    query,
    video_id: videoId || null,
    top_k: topK,
    min_score: 0.0,
  });
  return res.data;
}

export async function generateClip(frameId: string): Promise<{ clip_url: string }> {
  const res = await client.post(`/api/v1/search/clip/${frameId}`);
  return res.data;
}

export function mediaUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}
