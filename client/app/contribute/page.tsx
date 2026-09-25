"use client";

import React, { useState } from "react";
import SideNavBar from "@/components/SideNavBar";
import { contributeIdiom } from "@/lib/api";

export default function ContributePage() {
  const [submitted, setSubmitted] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    konkani_text: "",
    romanized_text: "",
    script: "Devanagari",
    marathi_meaning: "",
    english_meaning: "",
    figurative_meaning: "",
    literal_meaning: "",
    example_sentence: "",
    category: "Character & Integrity",
    cultural_context: "",
    contributor_name: "",
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setSubmitError(null);
    try {
      const result = await contributeIdiom(formData);
      if (result.status === "accepted") {
        setSubmitted(true);
      } else {
        setSubmitError(result.message || "Submission failed. Please try again.");
      }
    } catch {
      setSubmitError("Failed to submit. Is the backend running?");
    } finally {
      setSubmitting(false);
    }
  };

  const categories = [
    "Character & Integrity",
    "Peace & Resolution",
    "Speech & Manners",
    "Wisdom & Life Lessons",
    "Livelihood & Sea",
    "Pride & Humility",
  ];

  return (
    <div className="flex flex-1 max-w-[1280px] mx-auto w-full min-h-screen">
      <SideNavBar />

      <main className="flex-1 w-full lg:ml-64 px-4 md:px-12 py-8 md:py-12 flex flex-col gap-8">
        <div className="bg-white/80 backdrop-blur-md p-6 md:p-8 rounded-3xl border border-[#f7e0b5] shadow-sm">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-[#ffecc9] text-[#973100] rounded-full text-xs font-semibold uppercase tracking-wider mb-2">
            <span className="material-symbols-outlined text-[14px]">volunteer_activism</span>
            <span>Community Preservation</span>
          </div>
          <h1 className="font-['EB_Garamond'] text-3xl md:text-4xl font-bold text-[#2a6865]">
            Share a Konkani Idiom or Proverb
          </h1>
          <p className="font-['Hanken_Grotesk'] text-sm text-[#594139] mt-1 max-w-2xl">
            Help preserve Konkani oral traditions, coastal folklore, and family sayings for future generations.
            Please provide the idiom in <strong>Devanagari Konkani</strong> along with its Marathi and English meanings.
          </p>
        </div>

        {submitted ? (
          <div className="bg-white rounded-3xl p-12 text-center border border-[#f7e0b5] shadow-md">
            <span className="material-symbols-outlined text-5xl text-[#345d29] mb-4">
              check_circle
            </span>
            <h2 className="font-['EB_Garamond'] text-3xl font-bold text-[#2a6865] mb-2">
              Dev Borem Korum! (Thank You!)
            </h2>
            <p className="text-sm text-[#594139] max-w-md mx-auto mb-6">
              Your contribution has been recorded and will be vetted by our linguistic review pipeline before inclusion into the main corpus.
            </p>
            <button
              onClick={() => {
                setSubmitted(false);
                setFormData({
                  konkani_text: "",
                  romanized_text: "",
                  script: "Devanagari",
                  marathi_meaning: "",
                  english_meaning: "",
                  figurative_meaning: "",
                  literal_meaning: "",
                  example_sentence: "",
                  category: "Character & Integrity",
                  cultural_context: "",
                  contributor_name: "",
                });
              }}
              className="px-6 py-2.5 bg-[#973100] text-white rounded-xl text-xs font-semibold hover:bg-[#c04000]"
            >
              Submit Another Idiom
            </button>
          </div>
        ) : (
          <form
            onSubmit={handleSubmit}
            className="bg-white rounded-3xl p-6 md:p-10 border border-[#f7e0b5] shadow-sm flex flex-col gap-6"
          >
            {/* Primary: Devanagari Konkani Text */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-[#594139] mb-2">
                  Konkani Text (Devanagari) *
                </label>
                <input
                  type="text"
                  name="konkani_text"
                  required
                  value={formData.konkani_text}
                  onChange={handleChange}
                  placeholder="e.g. हात दाखवून अयाक"
                  className="w-full px-4 py-3 bg-[#fff8f2] border border-[#f7e0b5] rounded-xl text-lg font-['EB_Garamond'] text-[#251a01] focus:border-[#2a6865] focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-[#594139] mb-2">
                  Romanized / Romi Spelling *
                </label>
                <input
                  type="text"
                  name="romanized_text"
                  required
                  value={formData.romanized_text}
                  onChange={handleChange}
                  placeholder="e.g. Haat dakhvun ayaak"
                  className="w-full px-4 py-3 bg-[#fff8f2] border border-[#f7e0b5] rounded-xl text-sm font-['Hanken_Grotesk'] text-[#251a01] focus:border-[#2a6865] focus:outline-none"
                />
              </div>
            </div>

            {/* Marathi + English Meanings */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-[#1e40af] mb-2">
                  मराठी अर्थ (Marathi Meaning) *
                </label>
                <textarea
                  name="marathi_meaning"
                  required
                  rows={3}
                  value={formData.marathi_meaning}
                  onChange={handleChange}
                  placeholder="सोप्या मराठीत या म्हणचा अर्थ सांगा..."
                  className="w-full px-4 py-3 bg-[#f0f9ff] border border-[#bfdbfe] rounded-xl text-sm text-[#1e3a5f] focus:border-[#2a6865] focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-[#166534] mb-2">
                  English Meaning *
                </label>
                <textarea
                  name="english_meaning"
                  required
                  rows={3}
                  value={formData.english_meaning}
                  onChange={handleChange}
                  placeholder="Explain the idiom in simple English..."
                  className="w-full px-4 py-3 bg-[#f0fdf4] border border-[#bbf7d0] rounded-xl text-sm text-[#14532d] focus:border-[#2a6865] focus:outline-none"
                />
              </div>
            </div>

            {/* Figurative + Literal */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-[#594139] mb-2">
                  Semantic / Figurative Meaning *
                </label>
                <textarea
                  name="figurative_meaning"
                  required
                  rows={3}
                  value={formData.figurative_meaning}
                  onChange={handleChange}
                  placeholder="What does this idiom actually convey in everyday conversation..."
                  className="w-full px-4 py-3 bg-[#fff8f2] border border-[#f7e0b5] rounded-xl text-sm text-[#251a01] focus:border-[#2a6865] focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-[#594139] mb-2">
                  Literal Translation *
                </label>
                <textarea
                  name="literal_meaning"
                  required
                  rows={3}
                  value={formData.literal_meaning}
                  onChange={handleChange}
                  placeholder="Word-for-word translation of the phrase..."
                  className="w-full px-4 py-3 bg-[#fff8f2] border border-[#f7e0b5] rounded-xl text-sm text-[#251a01] focus:border-[#2a6865] focus:outline-none"
                />
              </div>
            </div>

            {/* Category + Example */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-[#594139] mb-2">
                  Category
                </label>
                <select
                  name="category"
                  value={formData.category}
                  onChange={handleChange}
                  className="w-full px-4 py-3 bg-[#fff8f2] border border-[#f7e0b5] rounded-xl text-sm text-[#251a01] focus:border-[#2a6865] focus:outline-none"
                >
                  {categories.map((cat) => (
                    <option key={cat} value={cat}>{cat}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-[#594139] mb-2">
                  Example Sentence (Devanagari)
                </label>
                <input
                  type="text"
                  name="example_sentence"
                  value={formData.example_sentence}
                  onChange={handleChange}
                  placeholder="e.g. त्याने तुला काम सांगलं आणि त्याने हात दाखवून आयक केलं."
                  className="w-full px-4 py-3 bg-[#fff8f2] border border-[#f7e0b5] rounded-xl text-sm font-['EB_Garamond'] text-[#251a01] focus:border-[#2a6865] focus:outline-none"
                />
              </div>
            </div>

            {/* Cultural Context */}
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-[#594139] mb-2">
                Cultural Context / Background
              </label>
              <textarea
                name="cultural_context"
                rows={2}
                value={formData.cultural_context}
                onChange={handleChange}
                placeholder="Where does this idiom come from? What cultural traditions or history does it relate to?"
                className="w-full px-4 py-3 bg-[#fff8f2] border border-[#f7e0b5] rounded-xl text-sm text-[#251a01] focus:border-[#2a6865] focus:outline-none"
              />
            </div>

            {/* Contributor Name */}
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-[#594139] mb-2">
                Your Name (optional)
              </label>
              <input
                type="text"
                name="contributor_name"
                value={formData.contributor_name}
                onChange={handleChange}
                placeholder="Name for attribution"
                className="w-full px-4 py-3 bg-[#fff8f2] border border-[#f7e0b5] rounded-xl text-sm text-[#251a01] focus:border-[#2a6865] focus:outline-none"
              />
            </div>

            <div className="pt-4 border-t border-[#f7e0b5] flex items-center justify-end gap-4">
              {submitError && (
                <p className="text-sm text-[#ba1a1a] font-medium mr-auto">
                  {submitError}
                </p>
              )}
              <button
                type="submit"
                disabled={submitting}
                className="px-8 py-3.5 bg-[#973100] text-white rounded-xl font-['Hanken_Grotesk'] text-sm font-semibold hover:bg-[#c04000] shadow-sm transition-all disabled:opacity-50"
              >
                {submitting ? "Submitting..." : "Submit Idiom for Review"}
              </button>
            </div>
          </form>
        )}
      </main>
    </div>
  );
}
