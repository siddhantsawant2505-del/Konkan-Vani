"use client";

import React from "react";
import Link from "next/link";
import type { Idiom, Script } from "@/lib/types";

interface IdiomCardProps {
  idiom: Idiom;
  activeScript?: Script;
  isFeatured?: boolean;
}

export default function IdiomCard({
  idiom,
  activeScript = "devanagari",
  isFeatured = false,
}: IdiomCardProps) {
  const displayText =
    activeScript === "roman"
      ? idiom.romanized_text
      : idiom.konkani_text;

  const secondaryText =
    activeScript === "roman"
      ? idiom.konkani_text
      : idiom.romanized_text;

  return (
    <article
      className={`bg-white rounded-2xl p-6 md:p-8 flex flex-col justify-between border transition-all duration-300 hover:shadow-xl hover:-translate-y-0.5 ${
        isFeatured
          ? "border-l-4 border-l-[#c04000] border-t-[#f7e0b5] border-r-[#f7e0b5] border-b-[#f7e0b5] bg-gradient-to-br from-white to-[#fffcf8]"
          : "border-[#f7e0b5] hover:border-[#2a6865]/40"
      }`}
    >
      <div>
        {/* Top metadata tags */}
        <div className="flex items-center justify-between gap-2 mb-4">
          <span className="px-3 py-1 bg-[#fff2de] text-[#973100] border border-[#f7e0b5] rounded-full text-xs font-semibold uppercase tracking-wider">
            {idiom.category ?? idiom.script}
          </span>
          <span className="text-xs text-[#8d7167] font-medium italic">
            {idiom.source}
          </span>
        </div>

        {/* Idiom headline in EB Garamond */}
        <Link href={`/results?q=${encodeURIComponent(idiom.romanized_text)}`} className="group">
          <h3 className="font-['EB_Garamond'] text-2xl md:text-3xl font-semibold text-[#2a6865] group-hover:text-[#973100] transition-colors leading-snug">
            {displayText}
          </h3>
        </Link>

        {/* Secondary script transliteration */}
        <p className="font-['Hanken_Grotesk'] text-sm text-[#8d7167] mt-1 italic">
          {secondaryText}
        </p>

        {/* Marathi Meaning */}
        <div className="mt-4 p-3 bg-[#f0f9ff] rounded-xl border border-[#bfdbfe]">
          <p className="text-xs font-semibold text-[#1e40af] uppercase tracking-wider mb-1">
            मराठी अर्थ
          </p>
          <p className="font-['EB_Garamond'] text-base text-[#1e3a5f] leading-relaxed">
            {idiom.marathi_meaning}
          </p>
        </div>

        {/* English Meaning */}
        <div className="mt-3 p-3 bg-[#f0fdf4] rounded-xl border border-[#bbf7d0]">
          <p className="text-xs font-semibold text-[#166534] uppercase tracking-wider mb-1">
            English Meaning
          </p>
          <p className="text-sm text-[#14532d] font-medium leading-relaxed">
            {idiom.english_meaning}
          </p>
        </div>

        {/* Figurative & Literal Meaning */}
        <div className="mt-4 space-y-2">
          <p className="font-['Hanken_Grotesk'] text-[#251a01] text-base leading-relaxed">
            <strong className="text-[#973100] font-semibold">Semantic: </strong>
            {idiom.figurative_meaning}
          </p>
          <p className="font-['Hanken_Grotesk'] text-sm text-[#594139]">
            <span className="font-medium text-[#8d7167]">Literal: </span>
            {idiom.literal_meaning}
          </p>
        </div>

        {/* Example Sentence */}
        {idiom.example_sentence && (
          <div className="mt-4 p-3 bg-[#fff8f2] rounded-xl border border-[#f7e0b5]">
            <p className="text-xs font-semibold text-[#973100] uppercase tracking-wider mb-1">
              उदाहरण
            </p>
            <p className="font-['EB_Garamond'] text-sm text-[#251a01] italic leading-relaxed">
              &ldquo;{idiom.example_sentence}&rdquo;
            </p>
          </div>
        )}
      </div>

      {/* Footer link to detail page */}
      <div className="mt-6 pt-4 border-t border-[#f7e0b5]/60 flex items-center justify-between">
        <Link
          href={`/results?q=${encodeURIComponent(idiom.romanized_text)}`}
          className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-[#2a6865] hover:text-[#973100] transition-colors"
        >
          <span>View Context & Analysis</span>
          <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
        </Link>
      </div>
    </article>
  );
}
