import { useState, useCallback } from "react";
import { searchGraph, GraphSearchResponse } from "../api";

/**
 * Graph-aware search hook.
 *
 * Deliberately parallel to `useVideoSearch` rather than a replacement: the
 * plain vector path stays exactly as it was, and the caller chooses which one
 * to run. Both hooks expose the same shape (`results`, `loading`, `error`,
 * `latency`, `doSearch`) so switching is a one-line change in `App.tsx`.
 */
export function useGraphSearch(graphDepth = 2) {
  const [response, setResponse] = useState<GraphSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [lastQuery, setLastQuery] = useState("");

  const doSearch = useCallback(
    async (query: string, videoId?: string | null) => {
      if (!query || query.trim().length < 2) {
        setError("Query must be at least 2 characters");
        return;
      }
      setLoading(true);
      setError(null);
      setLastQuery(query);
      try {
        const res = await searchGraph(query, videoId, 10, graphDepth);
        setResponse(res);
        setLatency(res.latency_ms);
        if (res.results.length === 0) {
          setError("No matching moments found");
        }
      } catch (e: any) {
        setError(e?.response?.data?.detail || "Graph search failed");
        setResponse(null);
      } finally {
        setLoading(false);
      }
    },
    [graphDepth]
  );

  return {
    response,
    results: response?.results ?? [],
    loading,
    error,
    latency,
    lastQuery,
    doSearch,
  };
}
