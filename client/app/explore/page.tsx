"use client";

import React, { useEffect, useState } from "react";
import SideNavBar from "@/components/SideNavBar";
import ScriptToggle from "@/components/ScriptToggle";
import IdiomCard from "@/components/IdiomCard";
import { browseIdioms } from "@/lib/api";
import type { Idiom, Script } from "@/lib/types";

export default function ExplorePage() {
  const [idioms, setIdioms] = useState<Idiom[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeScript, setActiveScript] = useState<Script>("devanagari");
  const [categoryFilter, setCategoryFilter] = useState("all");

  const categories = [
    { id: "all", label: "All Categories" },
    { id: "peace & resolution", label: "Peace & Resolution" },
    { id: "character", label: "Character & Traits" },
    { id: "livelihood", label: "Livelihood & Sea" },
    { id: "speech & manners", label: "Speech & Manners" },
    { id: "wisdom", label: "Maritime Wisdom" },
    { id: "pride", label: "Pride & Humility" },
  ];

  useEffect(() => {
    setLoading(true);
    // TODO: Wire this to live backend API:
    // fetch(`/api/idioms?category=${categoryFilter}`)
    browseIdioms(1, 50, undefined, categoryFilter)
      .then((res) => {
        setIdioms(res.idioms);
        setError(null);
      })
      .catch(() => {
        setError("Failed to load idioms. Is the backend running?");
      })
      .finally(() => setLoading(false));
  }, [categoryFilter]);

  return (
    <div className="flex flex-1 max-w-[1280px] mx-auto w-full min-h-screen">
      <SideNavBar />

      <main className="flex-1 w-full lg:ml-64 px-4 md:px-12 py-8 md:py-12 flex flex-col gap-8">
        {/* Header & Script Switcher */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white/80 backdrop-blur-md p-6 rounded-3xl border border-[#f7e0b5] shadow-sm">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-[#ffecc9] text-[#973100] rounded-full text-xs font-semibold uppercase tracking-wider mb-2">
              <span className="material-symbols-outlined text-[14px]">history_edu</span>
              <span>Linguistic Heritage Library</span>
            </div>
            <h1 className="font-['EB_Garamond'] text-3xl md:text-4xl font-bold text-[#2a6865]">
              Explore Konkani Proverbs & Idioms
            </h1>
            <p className="font-['Hanken_Grotesk'] text-sm text-[#594139] mt-1">
              Browse through curated coastal expressions across Devanagari, Kannada, and Roman scripts.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-[#8d7167] font-semibold uppercase">Script</span>
            <ScriptToggle value={activeScript} onChange={setActiveScript} />
          </div>
        </div>

        {/* Category Filters Bar */}
        <div className="flex items-center gap-2 overflow-x-auto pb-2 scrollbar-none">
          {categories.map((cat) => (
            <button
              key={cat.id}
              onClick={() => setCategoryFilter(cat.id)}
              className={`px-4 py-2 rounded-full text-xs font-semibold whitespace-nowrap transition-all ${
                categoryFilter === cat.id
                  ? "bg-[#2a6865] text-white shadow-sm"
                  : "bg-white text-[#594139] border border-[#f7e0b5] hover:bg-[#fff2de] hover:text-[#973100]"
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>

        {/* Idiom Cards Grid */}
        {error ? (
          <div className="bg-white rounded-3xl p-12 text-center border border-[#f7e0b5]">
            <span className="material-symbols-outlined text-5xl text-[#ba1a1a] mb-4">error</span>
            <h3 className="font-['EB_Garamond'] text-2xl font-bold text-[#251a01] mb-2">
              Could not load idioms
            </h3>
            <p className="text-sm text-[#594139] max-w-md mx-auto mb-6">{error}</p>
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-3 bg-[#973100] text-white rounded-xl text-sm font-semibold hover:bg-[#c04000] inline-block"
            >
              Retry
            </button>
          </div>
        ) : loading ? (
          <div className="py-20 flex flex-col items-center justify-center text-[#8d7167] gap-3">
            <span className="material-symbols-outlined text-4xl animate-spin">progress_activity</span>
            <p className="text-sm font-['Hanken_Grotesk']">Loading idiom repository...</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {idioms.map((idiom) => (
              <IdiomCard
                key={idiom.id}
                idiom={idiom}
                activeScript={activeScript}
              />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
