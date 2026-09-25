// ─── Konkan Vani — Shared TypeScript Types ───────────────────────────────────
// Schema: Devanagari Konkani input → Marathi meaning + English meaning + figurative/literal

export type Script = "devanagari" | "kannada" | "roman";

export interface Idiom {
  id: string;
  konkani_text: string;           // Primary: Devanagari Konkani idiom
  romanized_text: string;         // Romi Konkani (Latin script) for phonetic matching
  script: string;                 // Primary script: "Devanagari"
  marathi_meaning: string;        // Simple Marathi explanation of the idiom
  english_meaning: string;        // English explanation of the idiom
  figurative_meaning: string;     // Semantic/conceptual meaning (what it conveys in culture)
  literal_meaning: string;        // Word-for-word translation
  example_sentence: string;       // Usage example in Devanagari
  category: string;               // Thematic category
  cultural_context: string;       // Cultural background/history
  source: string;                 // Source dataset or collection
  phonetic_key?: string;          // Normalized phonetic key for search
}

export type MatchType = "exact" | "phonetic" | "semantic" | "fuzzy";

export interface SearchResult {
  idiom: Idiom;
  match_type: MatchType;
  confidence: number; // 0–1
  matched_on: string; // which field matched
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total: number;
}

export interface BrowseResponse {
  idioms: Idiom[];
  total: number;
  page: number;
  page_size: number;
}
