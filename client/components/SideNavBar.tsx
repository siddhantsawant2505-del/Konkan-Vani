"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function SideNavBar() {
  const pathname = usePathname();
  const [itemCount, setItemCount] = useState<number | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/stats`)
      .then((res) => res.json())
      .then((data) => {
        if (typeof data.count === "number") {
          setItemCount(data.count);
        }
      })
      .catch(() => {});
  }, []);

  const navItems = [
    { label: "Home", href: "/", icon: "home" },
    { label: "Explore", href: "/explore", icon: "grid_view" },
    { label: "Contribute", href: "/contribute", icon: "add_circle" },
  ];

  return (
    <>
      {/* ── SideNavBar (Desktop) ─────────────────────────────────────────── */}
      <aside className="hidden lg:flex flex-col h-screen py-8 gap-4 w-64 fixed left-0 top-0 bg-[#fff2de] shadow-xl z-40 border-r border-[#f7e0b5]">
        <div className="px-6 mb-8 flex flex-col items-start">
          <Link href="/" className="group flex flex-col">
            <h1 className="font-['EB_Garamond'] text-2xl font-bold text-[#973100] tracking-tight group-hover:text-[#a93700] transition-colors">
              Konkan Vani
            </h1>
            <p className="font-['Hanken_Grotesk'] text-xs text-[#594139] mt-0.5 tracking-wider uppercase">
              Preserving Coastal Wisdom
            </p>
          </Link>
        </div>

        <nav className="flex-1 flex flex-col gap-1.5 px-4">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-4 py-3 rounded-full text-sm font-semibold tracking-wider uppercase transition-all ${
                  isActive
                    ? "bg-[#b1eeea] text-[#084f4d] shadow-sm font-bold"
                    : "text-[#594139] hover:bg-[#ffecc9] hover:text-[#973100]"
                }`}
              >
                <span className="material-symbols-outlined text-[20px]">{item.icon}</span>
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="px-6 mb-6">
          <Link
            href="/contribute"
            className="w-full bg-[#973100] text-white py-3 rounded-full font-['Hanken_Grotesk'] text-sm font-semibold text-center block hover:bg-[#a93700] transition-all shadow-md active:scale-95"
          >
            Share an Idiom
          </Link>
        </div>

        <div className="mt-auto flex flex-col gap-2 px-4 border-t border-[#f7e0b5] pt-6">
          <div className="px-3 py-2 bg-[#fff8f2] rounded-xl border border-[#f7e0b5] text-xs text-[#594139]">
            <p className="font-semibold text-[#973100]">NLP Corpus</p>
            <p className="text-[11px] text-[#8d7167] mt-0.5">
              {itemCount != null ? `${itemCount.toLocaleString()} curated idioms` : "Loading..."}
            </p>
          </div>
        </div>
      </aside>

      {/* ── TopNav (Mobile) ──────────────────────────────────────────────── */}
      <nav className="lg:hidden bg-[#fff8f2] sticky top-0 z-50 shadow-sm w-full border-b border-[#f7e0b5]">
        <div className="flex justify-between items-center px-4 py-3 max-w-[1280px] mx-auto">
          <Link href="/" className="font-['EB_Garamond'] text-xl font-bold text-[#973100]">
            Konkan Vani
          </Link>
          <div className="flex items-center gap-2">
            <Link
              href="/explore"
              className={`p-2 rounded-full ${pathname === "/explore" ? "bg-[#ffecc9] text-[#973100]" : "text-[#594139]"}`}
              aria-label="Explore"
            >
              <span className="material-symbols-outlined">grid_view</span>
            </Link>
            <Link
              href="/contribute"
              className={`p-2 rounded-full ${pathname === "/contribute" ? "bg-[#ffecc9] text-[#973100]" : "text-[#594139]"}`}
              aria-label="Contribute"
            >
              <span className="material-symbols-outlined">add_circle</span>
            </Link>
          </div>
        </div>
      </nav>
    </>
  );
}
