"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import SideNavBar from "@/components/SideNavBar";
import ScriptToggle from "@/components/ScriptToggle";
import IdiomCard from "@/components/IdiomCard";
import { searchIdioms } from "@/lib/api";
import type { Idiom, Script } from "@/lib/types";

export default function HomePage() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState("");
  const [activeScript, setActiveScript] = useState<Script>("devanagari");
  const [featuredIdioms, setFeaturedIdioms] = useState<Idiom[]>([]);

  useEffect(() => {
    searchIdioms("")
      .then((res) => setFeaturedIdioms(res.results.slice(0, 4).map((r) => r.idiom)))
      .catch(() => {});
  }, []);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      router.push(`/results?q=${encodeURIComponent(searchQuery.trim())}`);
    }
  };

  const sampleQueries = [
    { label: "हात दाखवून अयाक", q: "हात दाखवून अयाक" },
    { label: "Haat dakhvun ayaak", q: "Haat dakhvun" },
    { label: "उदक पिऊन विसर", q: "Udak pionn visor" },
    { label: "मोड्डे मारप", q: "Modde marap" },
  ];

  return (
    <div className="flex flex-1 max-w-[1280px] mx-auto w-full min-h-screen">
      {/* Navigation */}
      <SideNavBar />

      <main className="flex-1 w-full lg:ml-64 px-4 md:px-12 py-8 md:py-12 flex flex-col gap-12">
        {/* Top Control Bar: Script Switcher */}
        <div className="flex justify-between items-center bg-white/70 backdrop-blur-md p-4 rounded-2xl border border-[#f7e0b5] shadow-sm">
          <div>
            <span className="text-xs uppercase tracking-wider font-semibold text-[#8d7167]">
              Script Display
            </span>
          </div>
          <ScriptToggle value={activeScript} onChange={setActiveScript} />
        </div>

        {/* ── Hero Search Section ──────────────────────────────────────── */}
        <section className="flex flex-col items-center text-center max-w-3xl mx-auto w-full pt-4 md:pt-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-[#b1eeea]/60 border border-[#2a6865]/30 rounded-full text-xs font-semibold text-[#084f4d] mb-4">
            <span className="material-symbols-outlined text-[16px]">translate</span>
            <span>Multiscript NLP Engine: Devanagari • Kannada • Romi</span>
          </div>

          <h2 className="font-['EB_Garamond'] text-4xl md:text-5xl lg:text-6xl font-bold text-[#2a6865] mb-4 tracking-tight leading-tight">
            Discover the Wisdom of the Coast
          </h2>
          
          <p className="font-['Hanken_Grotesk'] text-base md:text-lg text-[#594139] mb-8 max-w-2xl leading-relaxed">
            Search through authentic Konkani idioms, proverbs (<em className="font-serif">mhonni</em>), and coastal sayings — translated, contextualized, and phonetically decoded.
          </p>

          {/* Search Form */}
          <form onSubmit={handleSearch} className="relative w-full max-w-2xl group mb-4">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <span className="material-symbols-outlined text-[#2a6865] text-2xl opacity-70">
                search
              </span>
            </div>
            
            <input
              id="search-input"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-12 pr-32 py-4 md:py-5 bg-white border-2 border-[#f7e0b5] rounded-2xl font-['Hanken_Grotesk'] text-base md:text-lg text-[#251a01] focus:border-[#2a6865] focus:ring-4 focus:ring-[#2a6865]/10 shadow-md outline-none placeholder:text-[#8d7167]"
              placeholder="Type in Devanagari, Kannada, or Roman spelling (e.g. Haat dakhvun...)"
              type="text"
              autoComplete="off"
              spellCheck={false}
            />

            <button
              type="submit"
              className="absolute inset-y-2 right-2 px-6 bg-[#973100] text-white rounded-xl font-['Hanken_Grotesk'] text-sm font-semibold hover:bg-[#c04000] active:scale-95 transition-all shadow-sm flex items-center gap-1.5"
            >
              <span>Explore</span>
              <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
            </button>
          </form>

          {/* Quick example query chips */}
          <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-[#594139]">
            <span className="font-medium text-[#8d7167]">Try searching:</span>
            {sampleQueries.map((item) => (
              <button
                key={item.label}
                type="button"
                onClick={() => {
                  setSearchQuery(item.q);
                  router.push(`/results?q=${encodeURIComponent(item.q)}`);
                }}
                className="px-3 py-1 bg-white border border-[#f7e0b5] rounded-full hover:bg-[#ffecc9] hover:border-[#973100] transition-all font-medium text-[#251a01]"
              >
                {item.label}
              </button>
            ))}
          </div>
        </section>

        {/* ── Proverb of the Day & Featured Collection ────────────────── */}
        <section className="mt-4">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="font-['EB_Garamond'] text-2xl md:text-3xl font-semibold text-[#251a01]">
                Featured Proverbs & Sayings
              </h3>
              <p className="font-['Hanken_Grotesk'] text-sm text-[#594139] mt-0.5">
                Curated linguistic treasures from Goan and coastal heritage
              </p>
            </div>
            <Link
              href="/explore"
              className="hidden sm:inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-[#973100] hover:text-[#c04000]"
            >
              <span>View All 100k+ Items</span>
              <span className="material-symbols-outlined text-[16px]">chevron_right</span>
            </Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {featuredIdioms.map((idiom, index) => (
              <IdiomCard
                key={idiom.id}
                idiom={idiom}
                activeScript={activeScript}
                isFeatured={index === 0}
              />
            ))}
          </div>
        </section>

        {/* ── Cultural Heritage Info ───────────────────────────────────── */}
        <section className="bg-[#ffecc9]/60 rounded-3xl p-8 border border-[#f7e0b5] grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-2">
            <span className="material-symbols-outlined text-[#973100] text-3xl">menu_book</span>
            <h4 className="font-['EB_Garamond'] text-xl font-semibold text-[#251a01]">
              Multi-Script Preservation
            </h4>
            <p className="text-xs text-[#594139] leading-relaxed">
              Konkani is uniquely spoken across five scripts. Konkan Vani bridges Devanagari and Romi through phonetically normalized indexing.
            </p>
          </div>

          <div className="space-y-2">
            <span className="material-symbols-outlined text-[#2a6865] text-3xl">psychology</span>
            <h4 className="font-['EB_Garamond'] text-xl font-semibold text-[#251a01]">
              Semantic & Phonetic NLP
            </h4>
            <p className="text-xs text-[#594139] leading-relaxed">
              Find exact idioms even when typed with informal English Romanization variants (e.g., <em>aa→a, ee→i, oo→u, v→w</em>).
            </p>
          </div>

          <div className="space-y-2">
            <span className="material-symbols-outlined text-[#345d29] text-3xl">waves</span>
            <h4 className="font-['EB_Garamond'] text-xl font-semibold text-[#251a01]">
              Coastal Lore & Context
            </h4>
            <p className="text-xs text-[#594139] leading-relaxed">
              Beyond literal translations, explore maritime, agricultural, and cultural parables preserved by generations of coastal Konkani speakers.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}
