import { useState, useCallback } from "react";
import { search, SearchResponse, SearchResult } from "../api";

export function useVideoSearch() {
  const [results, setResults] = useState<SearchResult[]>([]);
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
        const res: SearchResponse = await search(query, videoId);
        setResults(res.results);
        setLatency(res.latency_ms);
        if (res.results.length === 0) {
          setError("No matching moments found");
        }
      } catch (e: any) {
        setError(e?.response?.data?.detail || "Search failed");
        setResults([]);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return { results, loading, error, latency, lastQuery, doSearch, setResults };
}
