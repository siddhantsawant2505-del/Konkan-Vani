"use client";

import React from "react";
import type { MatchType } from "@/lib/types";

interface ConfidenceBadgeProps {
  matchType: MatchType;
  confidence: number;
  className?: string;
}

export default function ConfidenceBadge({
  matchType,
  confidence,
  className = "",
}: ConfidenceBadgeProps) {
  const percentage = Math.round(confidence * 100);

  if (matchType === "phonetic") {
    return (
      <div
        className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-[#b1eeea] text-[#084f4d] border border-[#2a6865]/20 ${className}`}
        title={`Phonetic normalized match (${percentage}% confidence)`}
      >
        <span className="material-symbols-outlined text-[15px]">record_voice_over</span>
        <span>Phonetic Match ({percentage}%)</span>
      </div>
    );
  }

  if (matchType === "semantic") {
    return (
      <div
        className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-[#cbfbb7] text-[#28501e] border border-[#345d29]/20 ${className}`}
        title={`Semantic embedding match (${percentage}% confidence)`}
      >
        <span className="material-symbols-outlined text-[15px]">psychology</span>
        <span>Semantic Match ({percentage}%)</span>
      </div>
    );
  }

  return (
    <div
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-[#ffecc9] text-[#973100] border border-[#e1bfb4] ${className}`}
      title={`Direct match (${percentage}% confidence)`}
    >
      <span className="material-symbols-outlined text-[15px]">check_circle</span>
      <span>Exact Match ({percentage}%)</span>
    </div>
  );
}
