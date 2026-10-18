"use client";

import React from "react";
import Link from "next/link";
import type { ProvisionalReading, WordGloss } from "@/lib/types";

const METHOD_LABEL: Record<WordGloss["method"], string> = {
  exact: "dictionary",
  suffix: "inflected form",
  phonetic: "phonetic",
  fuzzy: "spelling variant",
  embedding: "approximate",
  none: "unrecognised",
};

function methodStyle(method: WordGloss["method"]) {
  switch (method) {
    case "exact":
      return "bg-[#cbfbb7] text-[#28501e] border-[#345d29]/20";
    case "suffix":
    case "phonetic":
      return "bg-[#b1eeea] text-[#084f4d] border-[#2a6865]/20";
    case "fuzzy":
      return "bg-[#e0e7ff] text-[#3730a3] border-[#6366f1]/20";
    case "embedding":
      return "bg-[#ffecc9] text-[#973100] border-[#e1bfb4]";
    default:
      return "bg-[#f3f4f6] text-[#6b7280] border-[#d1d5db]";
  }
}

interface Props {
  reading: ProvisionalReading;
  query: string;
  activeScript: string;
}

export default function ProvisionalMeaning({ reading, query, activeScript }: Props) {
  const coveragePercent = Math.round((reading.coverage?.ratio ?? 0) * 100);
  // The composed confidence is intentionally NOT shown as a headline number:
  // for a non-attested reading it would read as a retrieval confidence.
  // It stays in the API payload and in the composed caveats instead.
  void reading.confidence;

  return (
    <div className="flex flex-col gap-8">
      {/* ── Header card ─────────────────────────────────────────────── */}
      <section className="bg-white rounded-3xl p-6 md:p-10 border border-[#f7e0b5] shadow-lg">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div className="flex flex-wrap items-center gap-2">
            <div
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-[#ffecc9] text-[#973100] border border-[#e1bfb4]"
              title="Composed word by word — this proverb is not in the corpus, so the reading is not an attested meaning"
            >
              <span className="material-symbols-outlined text-[15px]">construction</span>
              <span>Provisional Reading</span>
            </div>
            <div
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-[#fff2de] text-[#8d7167] border border-[#f7e0b5]"
              title="Share of the phrase the word glossary could gloss; the rest is reported as unrecognised"
            >
              <span>
                {reading.coverage?.glossed ?? 0}/{reading.coverage?.total ?? 0} words glossed
              </span>
            </div>
          </div>
          <span className="text-xs text-[#8d7167] font-medium bg-[#fff2de] px-3 py-1 rounded-full border border-[#f7e0b5]">
            Not found in corpus · built from word glosses
          </span>
        </div>

        <h1 className="font-['EB_Garamond'] text-4xl md:text-5xl font-bold text-[#973100] tracking-tight leading-tight">
          {query}
        </h1>
        <p className="font-['Hanken_Grotesk'] text-sm text-[#594139] mt-2">
          {reading.coverage?.glossed ?? 0} of {reading.coverage?.total ?? 0} words
          recognised ({coveragePercent}%). Every word is glossed below, and the
          meaning is composed from those glosses rather than looked up.
        </p>

        {/* Literal gloss sequence */}
        <div className="mt-6 p-4 bg-[#fff8f2] rounded-2xl border border-[#f7e0b5]">
          <p className="text-xs font-semibold text-[#973100] uppercase tracking-wider mb-2">
            Word-for-word gloss
          </p>
          <p className="font-['EB_Garamond'] text-xl md:text-2xl text-[#251a01] leading-relaxed">
            {reading.literal_meaning || "—"}
          </p>
        </div>

        {/* Marathi */}
        <div className="mt-6 p-6 bg-[#f0f9ff] rounded-2xl border border-[#bfdbfe]">
          <div className="flex items-center gap-2 text-[#1e40af] font-semibold text-sm mb-3">
            <span className="material-symbols-outlined text-[20px]">translate</span>
            <span className="text-sm font-semibold uppercase tracking-wider">
              मराठी अर्थ (संभाव्य)
            </span>
          </div>
          <p className="font-['EB_Garamond'] text-lg md:text-xl text-[#1e3a5f] leading-relaxed">
            {reading.marathi_meaning}
          </p>
        </div>

        {/* English */}
        <div className="mt-6 p-6 bg-[#f0fdf4] rounded-2xl border border-[#bbf7d0]">
          <div className="flex items-center gap-2 text-[#166534] font-semibold text-sm mb-3">
            <span className="material-symbols-outlined text-[20px]">language</span>
            <span className="text-sm font-semibold uppercase tracking-wider">
              Composed meaning
            </span>
          </div>
          <p className="text-lg md:text-xl text-[#14532d] font-medium leading-relaxed">
            {reading.english_meaning}
          </p>
        </div>

        {/* Full reasoning */}
        <div className="mt-6 p-6 bg-[#fff2de]/70 rounded-2xl border border-[#f7e0b5]">
          <div className="flex items-center gap-2 text-[#594139] text-xs font-semibold uppercase tracking-wider mb-2">
            <span className="material-symbols-outlined text-[18px] text-[#973100]">
              account_tree
            </span>
            <span>How this was composed</span>
          </div>
          <p className="text-sm md:text-base text-[#251a01] leading-relaxed">
            {reading.figurative_meaning}
          </p>
        </div>

        {reading.themes?.length > 0 && (
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold text-[#8d7167] uppercase tracking-wider">
              Imagery
            </span>
            {reading.themes.map((t) => (
              <span
                key={t}
                className="px-3 py-1 bg-[#fff2de] text-[#973100] border border-[#f7e0b5] rounded-full text-xs font-semibold capitalize"
              >
                {t}
              </span>
            ))}
            {reading.negated && (
              <span className="px-3 py-1 bg-[#fee2e2] text-[#991b1b] border border-[#fca5a5] rounded-full text-xs font-semibold">
                negation detected
              </span>
            )}
          </div>
        )}

        {/* Coverage bar */}
        <div className="mt-6">
          <div className="flex items-center justify-between text-xs text-[#8d7167] mb-1.5">
            <span className="font-semibold uppercase tracking-wider">Gloss coverage</span>
            <span>{coveragePercent}%</span>
          </div>
          <div className="h-2 rounded-full bg-[#f7e0b5]/60 overflow-hidden">
            <div
              className="h-full rounded-full bg-[#2a6865] transition-all"
              style={{ width: `${Math.max(2, coveragePercent)}%` }}
            />
          </div>
        </div>
      </section>

      {/* ── Word table ──────────────────────────────────────────────── */}
      <section>
        <h3 className="font-['EB_Garamond'] text-2xl font-bold text-[#2a6865] mb-1">
          Word-by-word analysis
        </h3>
        <p className="text-xs text-[#8d7167] mb-4">
          Each word is looked up as written, then by inflected stem, phonetic
          equivalent and spelling variant. Words still unknown are marked, never
          guessed silently.
        </p>
        <div className="overflow-x-auto rounded-2xl border border-[#f7e0b5] bg-white">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="bg-[#fff8f2] text-[#594139] text-xs uppercase tracking-wider">
                <th className="text-left px-4 py-3 font-semibold">Word</th>
                <th className="text-left px-4 py-3 font-semibold">Gloss</th>
                <th className="text-left px-4 py-3 font-semibold">मराठी</th>
                <th className="text-left px-4 py-3 font-semibold">Found by</th>
                <th className="text-left px-4 py-3 font-semibold">Type</th>
              </tr>
            </thead>
            <tbody>
              {reading.word_glosses?.map((w, i) => (
                <tr
                  key={`${w.token}-${i}`}
                  className="border-t border-[#f7e0b5]/60 align-top"
                >
                  <td className="px-4 py-3 font-['EB_Garamond'] text-base text-[#251a01] whitespace-nowrap">
                    {/* the word as typed, plus the glossary headword it resolved to */}
                    <span>{w.token}</span>
                    {w.devanagari && w.devanagari !== w.token && (
                      <span className="block text-[11px] text-[#8d7167]">
                        → {w.devanagari}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-[#14532d]">
                    {w.known ? (
                      <>
                        {w.method === "embedding" && <span className="text-[#973100]">≈ </span>}
                        {w.english}
                      </>
                    ) : (
                      <span className="text-[#8d7167] italic">unknown — left untranslated</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-[#1e3a5f]">
                    {w.known ? w.marathi : "—"}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-block px-2 py-0.5 rounded-full text-[11px] font-semibold border ${methodStyle(
                        w.method,
                      )}`}
                    >
                      {METHOD_LABEL[w.method]}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-[#594139] capitalize">
                    {w.pos ? w.pos.replace("?", " (guess)") : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Closest known idioms ────────────────────────────────────── */}
      {reading.closest_known?.length > 0 && (
        <section>
          <h3 className="font-['EB_Garamond'] text-2xl font-bold text-[#2a6865] mb-1">
            Closest expressions in the corpus
          </h3>
          <p className="text-xs text-[#8d7167] mb-4">
            These scored below the matching threshold, so they are shown for
            comparison — not as the meaning of your query.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {reading.closest_known.map((n) => (
              <div
                key={n.id}
                className="bg-white p-5 rounded-2xl border border-[#f7e0b5] hover:border-[#2a6865] transition-all"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-[#8d7167]">
                    similarity {Math.round(n.similarity * 100)}%
                  </span>
                  <span className="text-[11px] px-2 py-0.5 rounded-full bg-[#fff2de] text-[#973100] border border-[#f7e0b5]">
                    {n.category || "uncategorised"}
                  </span>
                </div>
                <h4 className="font-['EB_Garamond'] text-lg font-bold text-[#2a6865]">
                  {n.konkani_text}
                </h4>
                <p className="text-xs text-[#8d7167] italic mt-0.5">{n.romanized_text}</p>
                <p className="text-xs text-[#14532d] mt-2 bg-[#f0fdf4] px-2 py-1 rounded">
                  {n.english_meaning}
                </p>
                <Link
                  href={`/results?q=${encodeURIComponent(n.romanized_text)}`}
                  className="mt-3 inline-block text-xs font-semibold text-[#973100] hover:underline"
                >
                  Open this idiom →
                </Link>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Notes / caveats ────────────────────────────────────────── */}
      <section className="bg-[#fff8f2] rounded-2xl border border-[#f7e0b5] p-5">
        <p className="text-xs font-semibold text-[#973100] uppercase tracking-wider mb-2">
          What this answer is not
        </p>
        <ul className="list-disc pl-5 text-xs text-[#594139] space-y-1">
          {(reading.notes ?? []).map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
        <div className="mt-4 flex flex-wrap gap-3">
          <Link
            href="/contribute"
            className="px-5 py-2.5 bg-[#973100] text-white rounded-xl text-sm font-semibold hover:bg-[#c04000]"
          >
            Add this proverb to the corpus
          </Link>
          <Link
            href="/explore"
            className="px-5 py-2.5 bg-white text-[#2a6865] border border-[#f7e0b5] rounded-xl text-sm font-semibold hover:border-[#2a6865]"
          >
            Browse attested idioms
          </Link>
        </div>
      </section>
    </div>
  );
}
