"use client";

import React, { useEffect, useState, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import SideNavBar from "@/components/SideNavBar";
import ScriptToggle from "@/components/ScriptToggle";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import { searchIdioms } from "@/lib/api";
import type { SearchResult, Script } from "@/lib/types";

interface ApiError {
  message: string;
}

function ResultsContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const query = searchParams.get("q") || "";

  const [inputQuery, setInputQuery] = useState(query);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeScript, setActiveScript] = useState<Script>("devanagari");

  useEffect(() => {
    setInputQuery(query);
    setLoading(true);

    searchIdioms(query)
      .then((res) => {
        setResults(res.results);
        setError(null);
      })
      .catch((e: ApiError) => {
        setError(e.message || "Failed to fetch results. Is the backend running?");
      })
      .finally(() => setLoading(false));
  }, [query]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputQuery.trim()) {
      router.push(`/results?q=${encodeURIComponent(inputQuery.trim())}`);
    }
  };

  const primaryResult = results[0];
  const relatedResults = results.slice(1);

  return (
    <div className="flex flex-1 max-w-[1280px] mx-auto w-full min-h-screen">
      <SideNavBar />

      <main className="flex-1 w-full lg:ml-64 px-4 md:px-12 py-8 md:py-12 flex flex-col gap-8">
        {/* Top bar with Search refinement & Script Switcher */}
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4 bg-white/80 backdrop-blur-md p-4 rounded-2xl border border-[#f7e0b5] shadow-sm">
          <form onSubmit={handleSearch} className="flex-1 relative">
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Devanagari Konkani मध्ये शोधा..."
              className="w-full pl-10 pr-24 py-2.5 bg-[#fff8f2] border border-[#f7e0b5] rounded-xl text-sm font-['Hanken_Grotesk'] text-[#251a01] focus:border-[#2a6865] focus:outline-none placeholder:text-[#8d7167]"
            />
            <span className="material-symbols-outlined absolute left-3 top-2.5 text-[#8d7167] text-[20px]">
              search
            </span>
            <button
              type="submit"
              className="absolute right-1.5 top-1.5 px-4 py-1.5 bg-[#2a6865] text-white rounded-lg text-xs font-semibold hover:bg-[#316e6b] transition-colors"
            >
              Search
            </button>
          </form>

          <div className="flex items-center justify-between md:justify-end gap-3">
            <span className="text-xs text-[#8d7167] font-semibold uppercase">Script</span>
            <ScriptToggle value={activeScript} onChange={setActiveScript} />
          </div>
        </div>

        {/* Query overview header */}
        <div className="flex items-baseline justify-between border-b border-[#f7e0b5] pb-4">
          <div>
            <h2 className="font-['EB_Garamond'] text-3xl font-bold text-[#2a6865]">
              Meaning & Context Analysis
            </h2>
            <p className="font-['Hanken_Grotesk'] text-sm text-[#594139] mt-1">
              Query: <strong className="text-[#973100] font-semibold">&ldquo;{query}&rdquo;</strong> —{" "}
              {loading ? "Searching..." : `${results.length} matched expression${results.length === 1 ? "" : "s"}`}
            </p>
          </div>
          <Link
            href="/explore"
            className="text-xs font-semibold text-[#973100] hover:underline uppercase tracking-wider"
          >
            ← Browse All
          </Link>
        </div>

        {error ? (
          <div className="bg-white rounded-3xl p-12 text-center border border-[#f7e0b5]">
            <span className="material-symbols-outlined text-5xl text-[#ba1a1a] mb-4">
              error
            </span>
            <h3 className="font-['EB_Garamond'] text-2xl font-bold text-[#251a01] mb-2">
              Something went wrong
            </h3>
            <p className="text-sm text-[#594139] max-w-md mx-auto mb-6">
              {error}
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-3 bg-[#973100] text-white rounded-xl text-sm font-semibold hover:bg-[#c04000] inline-block"
            >
              Try Again
            </button>
          </div>
        ) : loading ? (
          <div className="py-20 flex flex-col items-center justify-center text-[#8d7167] gap-3">
            <span className="material-symbols-outlined text-4xl animate-spin">progress_activity</span>
            <p className="text-sm font-['Hanken_Grotesk']">Searching Konkan Vani linguistic engine...</p>
          </div>
        ) : primaryResult ? (
          <div className="flex flex-col gap-8">
            {/* ── Primary Hero Result Card ────────────────────────────── */}
            <section className="bg-white rounded-3xl p-6 md:p-10 border border-[#f7e0b5] shadow-lg relative overflow-hidden">
              {/* Match indicator pill */}
              <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
                <ConfidenceBadge
                  matchType={primaryResult.match_type}
                  confidence={primaryResult.confidence}
                />
                <span className="text-xs text-[#8d7167] font-medium bg-[#fff2de] px-3 py-1 rounded-full border border-[#f7e0b5]">
                  Source: {primaryResult.idiom.source}
                </span>
              </div>

              {/* Main Expression */}
              <div className="mb-8">
                <h1 className="font-['EB_Garamond'] text-4xl md:text-5xl font-bold text-[#973100] tracking-tight leading-tight">
                  {activeScript === "roman"
                    ? primaryResult.idiom.romanized_text
                    : primaryResult.idiom.konkani_text}
                </h1>
                
                {/* Secondary transliteration */}
                <p className="font-['Hanken_Grotesk'] text-lg md:text-xl text-[#2a6865] mt-2 italic">
                  {activeScript === "roman"
                    ? primaryResult.idiom.konkani_text
                    : primaryResult.idiom.romanized_text}
                </p>
              </div>

              {/* ── Marathi Meaning (Primary) ──────────────────────────── */}
              <div className="mb-6 p-6 bg-[#f0f9ff] rounded-2xl border border-[#bfdbfe]">
                <div className="flex items-center gap-2 text-[#1e40af] font-semibold text-sm mb-3">
                  <span className="material-symbols-outlined text-[20px]">translate</span>
                  <span className="text-sm font-semibold uppercase tracking-wider">मराठी अर्थ</span>
                </div>
                <p className="font-['EB_Garamond'] text-xl md:text-2xl text-[#1e3a5f] leading-relaxed">
                  {primaryResult.idiom.marathi_meaning}
                </p>
              </div>

              {/* ── English Meaning ─────────────────────────────────────── */}
              <div className="mb-6 p-6 bg-[#f0fdf4] rounded-2xl border border-[#bbf7d0]">
                <div className="flex items-center gap-2 text-[#166534] font-semibold text-sm mb-3">
                  <span className="material-symbols-outlined text-[20px]">language</span>
                  <span className="text-sm font-semibold uppercase tracking-wider">English Meaning</span>
                </div>
                <p className="text-lg md:text-xl text-[#14532d] font-medium leading-relaxed">
                  {primaryResult.idiom.english_meaning}
                </p>
              </div>

              {/* Grid of Figurative, Literal, Example */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                {/* Figurative/Semantic Meaning */}
                <div className="bg-[#fff8f2] p-5 rounded-2xl border border-[#f7e0b5]">
                  <div className="flex items-center gap-2 text-[#973100] font-semibold text-sm mb-2">
                    <span className="material-symbols-outlined text-[18px]">psychology</span>
                    <span>Semantic Meaning</span>
                  </div>
                  <p className="text-sm md:text-base text-[#251a01] leading-relaxed">
                    {primaryResult.idiom.figurative_meaning}
                  </p>
                </div>

                {/* Literal Meaning */}
                <div className="bg-[#fff8f2] p-5 rounded-2xl border border-[#f7e0b5]">
                  <div className="flex items-center gap-2 text-[#2a6865] font-semibold text-sm mb-2">
                    <span className="material-symbols-outlined text-[18px]">menu_book</span>
                    <span>Literal Translation</span>
                  </div>
                  <p className="text-sm md:text-base text-[#251a01] leading-relaxed">
                    &ldquo;{primaryResult.idiom.literal_meaning}&rdquo;
                  </p>
                </div>

                {/* Category */}
                <div className="bg-[#fff8f2] p-5 rounded-2xl border border-[#f7e0b5]">
                  <div className="flex items-center gap-2 text-[#345d29] font-semibold text-sm mb-2">
                    <span className="material-symbols-outlined text-[18px]">category</span>
                    <span>Category</span>
                  </div>
                  <p className="text-sm md:text-base text-[#251a01] font-medium leading-relaxed">
                    {primaryResult.idiom.category || "Uncategorized"}
                  </p>
                </div>
              </div>

              {/* Example Sentence in context */}
              {primaryResult.idiom.example_sentence && (
                <div className="p-6 bg-[#fff2de]/80 rounded-2xl border border-[#f7e0b5] mb-6">
                  <div className="flex items-center gap-2 text-[#594139] text-xs font-semibold uppercase tracking-wider mb-2">
                    <span className="material-symbols-outlined text-[18px] text-[#973100]">format_quote</span>
                    <span>उदाहरण — Example Sentence</span>
                  </div>
                  <p className="font-['EB_Garamond'] text-xl md:text-2xl text-[#251a01] italic">
                    &ldquo;{primaryResult.idiom.example_sentence}&rdquo;
                  </p>
                </div>
              )}

              {/* Cultural Context */}
              {primaryResult.idiom.cultural_context && (
                <div className="p-5 bg-white rounded-2xl border border-[#f7e0b5]/80 text-xs text-[#594139] leading-relaxed">
                  <strong className="text-[#2a6865] font-semibold">Cultural Lore & Usage: </strong>
                  {primaryResult.idiom.cultural_context}
                </div>
              )}
            </section>

            {/* ── Related / Alternate Matches ──────────────────────────── */}
            {relatedResults.length > 0 && (
              <section className="mt-4">
                <h3 className="font-['EB_Garamond'] text-2xl font-bold text-[#2a6865] mb-4">
                  Related Proverbs in Corpus
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {relatedResults.map((result) => (
                    <div
                      key={result.idiom.id}
                      className="bg-white p-5 rounded-2xl border border-[#f7e0b5] flex flex-col justify-between hover:border-[#2a6865] transition-all"
                    >
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <ConfidenceBadge
                            matchType={result.match_type}
                            confidence={result.confidence}
                          />
                        </div>
                        <h4 className="font-['EB_Garamond'] text-xl font-bold text-[#2a6865]">
                          {activeScript === "roman"
                            ? result.idiom.romanized_text
                            : result.idiom.konkani_text}
                        </h4>
                        <p className="text-xs text-[#8d7167] italic mt-0.5">
                          {activeScript === "roman"
                            ? result.idiom.konkani_text
                            : result.idiom.romanized_text}
                        </p>
                        
                        {/* Marathi meaning in related results */}
                        <p className="text-xs text-[#1e40af] mt-2 bg-[#f0f9ff] px-2 py-1 rounded">
                          <strong>मराठी: </strong>{result.idiom.marathi_meaning}
                        </p>
                        
                        {/* English meaning in related results */}
                        <p className="text-xs text-[#166534] mt-1 bg-[#f0fdf4] px-2 py-1 rounded">
                          <strong>English: </strong>{result.idiom.english_meaning}
                        </p>
                        
                        <p className="text-xs text-[#251a01] mt-2">
                          <strong>Semantic: </strong>
                          {result.idiom.figurative_meaning}
                        </p>
                      </div>

                      <div className="mt-4 pt-3 border-t border-[#f7e0b5] flex justify-end">
                        <Link
                          href={`/results?q=${encodeURIComponent(result.idiom.romanized_text)}`}
                          className="text-xs font-semibold text-[#973100] hover:underline"
                        >
                          Explore This Idiom →
                        </Link>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>
        ) : (
          <div className="bg-white rounded-3xl p-12 text-center border border-[#f7e0b5]">
            <span className="material-symbols-outlined text-5xl text-[#8d7167] mb-4">
              sentiment_dissatisfied
            </span>
            <h3 className="font-['EB_Garamond'] text-2xl font-bold text-[#251a01] mb-2">
              No direct matches found for &ldquo;{query}&rdquo;
            </h3>
            <p className="text-sm text-[#594139] max-w-md mx-auto mb-6">
              Try searching with alternative Roman spelling (e.g. <em>Haat dakhvun</em>, <em>Udak pionn</em>) or explore our curated collection.
            </p>
            <Link
              href="/explore"
              className="px-6 py-3 bg-[#973100] text-white rounded-xl text-sm font-semibold hover:bg-[#c04000] inline-block"
            >
              Browse All Idioms
            </Link>
          </div>
        )}
      </main>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center">Loading search results...</div>}>
      <ResultsContent />
    </Suspense>
  );
}
