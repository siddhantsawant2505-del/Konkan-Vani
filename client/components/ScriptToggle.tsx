"use client";

import React from "react";
import type { Script } from "@/lib/types";

interface ScriptToggleProps {
  value: Script;
  onChange: (script: Script) => void;
  className?: string;
}

export default function ScriptToggle({ value, onChange, className = "" }: ScriptToggleProps) {
  return (
    <div
      className={`inline-flex items-center p-1 bg-[#ffecc9] border border-[#e1bfb4] rounded-full shadow-inner ${className}`}
      role="group"
      aria-label="Toggle display script"
    >
      <button
        type="button"
        onClick={() => onChange("devanagari")}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold tracking-wide transition-all ${
          value === "devanagari"
            ? "bg-[#973100] text-white shadow-sm"
            : "text-[#594139] hover:text-[#973100] hover:bg-[#fff2de]"
        }`}
        aria-pressed={value === "devanagari"}
      >
        <span className="font-['EB_Garamond'] text-sm leading-none">अ</span>
        <span>Devanagari</span>
      </button>
      
      <button
        type="button"
        onClick={() => onChange("kannada")}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold tracking-wide transition-all ${
          value === "kannada"
            ? "bg-[#973100] text-white shadow-sm"
            : "text-[#594139] hover:text-[#973100] hover:bg-[#fff2de]"
        }`}
        aria-pressed={value === "kannada"}
      >
        <span className="font-['EB_Garamond'] text-sm leading-none">ಕ</span>
        <span>Kannada</span>
      </button>

      <button
        type="button"
        onClick={() => onChange("roman")}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold tracking-wide transition-all ${
          value === "roman"
            ? "bg-[#973100] text-white shadow-sm"
            : "text-[#594139] hover:text-[#973100] hover:bg-[#fff2de]"
        }`}
        aria-pressed={value === "roman"}
      >
        <span className="font-['EB_Garamond'] text-sm leading-none italic font-bold">R</span>
        <span>Roman (Romi)</span>
      </button>
    </div>
  );
}
