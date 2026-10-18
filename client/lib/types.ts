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

export type MatchType =
  | "exact"
  | "phonetic"
  | "semantic"
  | "fuzzy"
  // Word-by-word reading of an idiom that is not in the corpus.
  | "compositional";

/** How one word of an unknown proverb was glossed by the backend. */
export interface WordGloss {
  token: string;
  devanagari: string;
  roman: string;
  english: string;
  marathi: string;
  pos: string;
  tag: string;
  method: "exact" | "suffix" | "phonetic" | "fuzzy" | "embedding" | "none";
  similarity: number;
  /** True for cross-lingual guesses that are not dictionary entries. */
  approximate: boolean;
  known: boolean;
}

/** Nearest corpus entry, shown even when too weak to count as a match. */
export interface ClosestKnown {
  id: string;
  konkani_text: string;
  romanized_text: string;
  english_meaning: string;
  figurative_meaning: string;
  category: string;
  similarity: number;
}

/** Composed reading of a proverb that the corpus does not contain. */
export interface ProvisionalReading {
  match_type: string;
  confidence: number;
  literal_meaning: string;
  english_meaning: string;
  figurative_meaning: string;
  marathi_meaning: string;
  coverage: { glossed: number; total: number; ratio: number };
  themes: string[];
  negated: boolean;
  word_glosses: WordGloss[];
  closest_known: ClosestKnown[];
  notes: string[];
}

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
  /** Present only when nothing matched, so the UI can explain the proverb. */
  provisional?: ProvisionalReading | null;
}

export interface BrowseResponse {
  idioms: Idiom[];
  total: number;
  page: number;
  page_size: number;
}
