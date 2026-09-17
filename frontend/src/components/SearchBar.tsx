import React, { useState } from "react";
import { Search } from "lucide-react";

interface Props {
  onSearch: (query: string) => void;
  loading?: boolean;
  disabled?: boolean;
}

export default function SearchBar({ onSearch, loading, disabled }: Props) {
  const [query, setQuery] = useState("");

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim().length >= 2) onSearch(query.trim());
  };

  return (
    <form onSubmit={submit} className="w-full">
      <div className="relative">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={disabled || loading}
          placeholder='Find something in your video… e.g. "person drawing a diagram on a whiteboard"'
          className="
            w-full pl-12 pr-28 py-3.5 rounded-xl
            bg-slate-900 border border-slate-700
            text-slate-100 placeholder:text-slate-500
            focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent
            disabled:opacity-50
          "
        />
        <button
          type="submit"
          disabled={disabled || loading || query.trim().length < 2}
          className="
            absolute right-2 top-1/2 -translate-y-1/2
            px-4 py-1.5 rounded-lg bg-brand-600 hover:bg-brand-500
            text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed
            transition
          "
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </div>
    </form>
  );
}
