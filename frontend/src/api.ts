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

/* ------------------------------------------------------------------ *
 * Graph-aware search (Phase 2)
 *
 * Everything below is ADDITIVE. `search()`, `SearchResult` and
 * `SearchResponse` above are unchanged, so the existing plain-vector UI keeps
 * working exactly as before and no existing client can be affected.
 * ------------------------------------------------------------------ */

/** A node the backend's bounded traversal reached from a result frame. */
export interface RelatedNode {
  id: string;
  node_type: string;
  label: string | null;
  entity_key: string | null;
  frame_id: string | null;
  timestamp_seconds: number | null;
  confidence: number;
  depth: number | null;
}

/**
 * One traversed edge, including the `basis` that justifies it
 * ("deterministic_containment" | "timestamp_ordering" |
 *  "measured_cosine_similarity" | "extractor:<name>").
 */
export interface RelationshipEvidence {
  relationship: string;
  source_node_id: string;
  target_node_id: string;
  weight: number;
  confidence: number;
  basis: string | null;
}

export interface GraphSearchResult {
  frame_id: string;
  video_id: string | null;
  video_name: string | null;
  timestamp_seconds: number;
  thumbnail_url: string | null;
  clip_url: string | null;

  vector_score: number;
  graph_score: number;
  temporal_score: number;
  confidence_score: number;
  final_score: number;

  /** Additive breakdown of `final_score`; sums back to it. */
  contributions: Record<string, number>;
  related_nodes: RelatedNode[];
  relationships: RelationshipEvidence[];

  is_seed: boolean;
  graph_depth_reached: number | null;
}

export interface GraphSearchResponse {
  query: string;
  results: GraphSearchResult[];
  total: number;
  latency_ms: number;

  /* Diagnostics, so the UI can distinguish a thin graph from a broken one. */
  graph_available: boolean;
  graph_depth: number;
  graph_expanded_nodes: number;
  graph_traversed_edges: number;
  graph_truncated_depth: boolean;
  graph_truncated_nodes: boolean;
  candidate_pool_size: number;
  stage_latency_ms: Record<string, number>;
}

export interface GraphNode {
  id: string;
  video_id: string;
  node_type: string;
  entity_key: string;
  label: string | null;
  description: string | null;
  timestamp_start: number | null;
  timestamp_end: number | null;
  frame_id: string | null;
  confidence: number;
  metadata: Record<string, unknown> | null;
}

export interface GraphEdge {
  id: string;
  source_node_id: string;
  target_node_id: string;
  relationship: string;
  weight: number;
  confidence: number;
  timestamp_start: number | null;
  timestamp_end: number | null;
  metadata: Record<string, unknown> | null;
}

export interface VideoGraphResponse {
  video_id: string;
  /** "pending" while indexing, "ready" once indexed, "error" if it failed. */
  status: string;
  node_count: number;
  edge_count: number;
  node_type_counts: Record<string, number>;
  relationship_counts: Record<string, number>;
  truncated: boolean;
  nodes: GraphNode[];
  edges: GraphEdge[];
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
    min_score: -1.0,
  });
  return res.data;
}

/**
 * Graph-aware search. A *separate route*, not a flag on `/search`, so the
 * existing response contract is untouched. `graphDepth = 0` reproduces the
 * vector-only ranking, which is the evaluation baseline.
 */
export async function searchGraph(
  query: string,
  videoId?: string | null,
  topK = 10,
  graphDepth = 2
): Promise<GraphSearchResponse> {
  const res = await client.post<GraphSearchResponse>("/api/v1/search/graph", {
    query,
    video_id: videoId || null,
    top_k: topK,
    graph_depth: graphDepth,
    min_score: -1.0,
  });
  return res.data;
}

/** Fetch a video's whole relational graph. */
export async function getVideoGraph(videoId: string): Promise<VideoGraphResponse> {
  const res = await client.get<VideoGraphResponse>(`/api/v1/videos/${videoId}/graph`);
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
