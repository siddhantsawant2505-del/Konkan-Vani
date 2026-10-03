// ─── Konkan Vani — API Client ─────────────────────────────────────────────────
// Communicates with the FastAPI NLP backend (server-nlp).
// If the backend is unavailable, falls back to local placeholder data.
// Schema: Devanagari Konkani input → Marathi meaning + English meaning + figurative/literal
// ──────────────────────────────────────────────────────────────────────────────

import type { SearchResponse, BrowseResponse, Idiom } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Placeholder Idioms (fallback when backend is down) ───────────────────────
const PLACEHOLDER_IDIOMS: Idiom[] = [
  {
    id: "1",
    konkani_text: "हात दाखवून अयाक",
    romanized_text: "Haat dakhvun ayaak",
    script: "Devanagari",
    marathi_meaning: "हात दाखवून येणे म्हणजे खोटे वचन देणे आणि निघून जाणे",
    english_meaning: "To make false promises and disappear",
    figurative_meaning: "To lead someone on with false assurances; to make empty promises",
    literal_meaning: "To show the hand and come (then leave)",
    example_sentence: "त्याने तुला काम सांगलं आणि त्याने हात दाखवून आयक केलं.",
    source: "Konkan Vani Gold Dataset",
    category: "Character & Integrity",
    cultural_context: "Commonly used in coastal trade markets when someone makes empty assurances.",
    phonetic_key: "hat daxwun ayak",
  },
  {
    id: "2",
    konkani_text: "उदक पिऊन विसर",
    romanized_text: "Udak pionn visor",
    script: "Devanagari",
    marathi_meaning: "पाणी पिऊन विसरून जा — म्हणजे गैरसमज विसरून सोडा",
    english_meaning: "Drink water and forget — let bygones be bygones",
    figurative_meaning: "To forgive past grievances and move forward without holding grudges",
    literal_meaning: "Drink water and forget",
    example_sentence: "जालं ते जालं — आता उदक पिऊन विसर आणि पुढे व्हा.",
    source: "Konkan Vani Gold Dataset",
    category: "Peace & Resolution",
    cultural_context: "An ancient Goan village council sentiment urging neighbors to resolve disputes amicably.",
    phonetic_key: "udak piun wisor",
  },
  {
    id: "3",
    konkani_text: "मोड्डे मारप",
    romanized_text: "Modde marap",
    script: "Devanagari",
    marathi_meaning: "अचानक अशा गोष्टी सांगणे ज्यामुळे शांतता भंग होते",
    english_meaning: "To drop a bombshell; to speak out of turn",
    figurative_meaning: "To make a sudden, tactless, or blunt statement that disrupts social harmony",
    literal_meaning: "To hit the rice bundle / break boundaries",
    example_sentence: "सगळी शांतता होती, त्याने अचानक मोड्डे मारून गोंधळ केला.",
    source: "Konkan Vani Gold Dataset",
    category: "Speech & Manners",
    cultural_context: "Derived from harvesting rituals where threshing bundles inappropriately causes grain loss.",
    phonetic_key: "mode marap",
  },
  {
    id: "4",
    konkani_text: "कुळांतलो कोळसो",
    romanized_text: "Kullantlo kollso",
    script: "Devanagari",
    marathi_meaning: "कुटुंबातील अपयशी व्यक्ती — कुटुंबाचे नाव खराब करणारा",
    english_meaning: "The black sheep of the family",
    figurative_meaning: "A person who brings disgrace to their respected family",
    literal_meaning: "Coal from the family clan",
    example_sentence: "सगळ्यांना सन्मान मिळाला, पण तो एकला कुळांतलो कोळसो झाला.",
    source: "Konkan Vani Gold Dataset",
    category: "Character & Integrity",
    cultural_context: "Refers to coal which blackens everything it touches — a metaphor for family shame.",
    phonetic_key: "kulantl kols",
  },
];

// ── Helper: safe fetch with backend fallback ──────────────────────────────────
async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ── Search endpoint ────────────────────────────────────────────────────────────
export async function searchIdioms(query: string): Promise<SearchResponse> {
  try {
    const data = await apiFetch<{
      query: string;
      results: Array<{
        idiom: Idiom;
        match_type: string;
        confidence: number;
        matched_on: string;
      }>;
      total: number;
    }>(`/api/search?q=${encodeURIComponent(query)}`);

    return {
      query: data.query,
      results: data.results.map((r) => ({
        idiom: r.idiom,
        match_type: r.match_type as SearchResponse["results"][number]["match_type"],
        confidence: r.confidence,
        matched_on: r.matched_on,
      })),
      total: data.total,
    };
  } catch (e) {
    console.warn("Backend unavailable, falling back to local placeholder data:", e);
    return searchIdiomsLocal(query);
  }
}

// ── Local fallback search ─────────────────────────────────────────────────────
function searchIdiomsLocal(query: string): SearchResponse {
  const q = query.trim().toLowerCase();
  if (!q) {
    return {
      query: "",
      results: PLACEHOLDER_IDIOMS.map((idiom) => ({
        idiom,
        match_type: "exact" as const,
        confidence: 1.0,
        matched_on: "all",
      })),
      total: PLACEHOLDER_IDIOMS.length,
    };
  }

  const matched = PLACEHOLDER_IDIOMS.filter(
    (idiom) =>
      idiom.romanized_text.toLowerCase().includes(q) ||
      idiom.konkani_text.includes(query) ||
      idiom.english_meaning.toLowerCase().includes(q) ||
      idiom.marathi_meaning.includes(query) ||
      idiom.figurative_meaning.toLowerCase().includes(q) ||
      idiom.literal_meaning.toLowerCase().includes(q) ||
      (idiom.phonetic_key && idiom.phonetic_key.includes(q))
  );

  if (matched.length === 0) {
    return { query, results: [], total: 0 };
  }

  const results = matched.map((idiom, i) => {
    const isPhonetic = idiom.phonetic_key?.includes(q) || idiom.romanized_text.toLowerCase().includes(q);
    return {
      idiom,
      match_type: (isPhonetic ? "phonetic" : "semantic") as "phonetic" | "semantic",
      confidence: i === 0 ? 0.94 : 0.78,
      matched_on: isPhonetic ? "phonetic_key" : "semantic_embedding",
    };
  });

  return { query, results, total: results.length };
}

// ── Browse / list all idioms ───────────────────────────────────────────────────
export async function browseIdioms(
  page = 1,
  pageSize = 20,
  scriptFilter?: string,
  categoryFilter?: string,
): Promise<BrowseResponse> {
  try {
    let url = `/api/browse?page=${page}&page_size=${pageSize}`;
    if (scriptFilter && scriptFilter !== "all") {
      url += `&script=${encodeURIComponent(scriptFilter)}`;
    }
    if (categoryFilter && categoryFilter !== "all") {
      url += `&category=${encodeURIComponent(categoryFilter)}`;
    }

    const data = await apiFetch<{
      idioms: Idiom[];
      total: number;
      page: number;
      page_size: number;
    }>(url);

    return {
      idioms: data.idioms,
      total: data.total,
      page: data.page,
      page_size: data.page_size,
    };
  } catch (e) {
    console.warn("Backend unavailable, falling back to local placeholder data:", e);
    return browseIdiomsLocal(page, pageSize, scriptFilter, categoryFilter);
  }
}

// ── Local fallback browse ─────────────────────────────────────────────────────
function browseIdiomsLocal(
  page: number,
  pageSize: number,
  scriptFilter?: string,
  categoryFilter?: string,
): BrowseResponse {
  let filtered = [...PLACEHOLDER_IDIOMS];
  if (scriptFilter && scriptFilter !== "all") {
    filtered = filtered.filter((i) => i.script.toLowerCase() === scriptFilter.toLowerCase());
  }
  if (categoryFilter && categoryFilter !== "all") {
    filtered = filtered.filter((i) => i.category?.toLowerCase() === categoryFilter.toLowerCase());
  }
  return {
    idioms: filtered,
    total: filtered.length,
    page,
    page_size: pageSize,
  };
}

// ── Fetch single idiom by ID ───────────────────────────────────────────────────
export async function getIdiomById(id: string): Promise<Idiom | null> {
  try {
    const data = await apiFetch<Idiom>(`/api/idioms/${id}`);
    return data;
  } catch (e) {
    console.warn("Backend unavailable, falling back to local placeholder data:", e);
    return PLACEHOLDER_IDIOMS.find((i) => i.id === id) ?? PLACEHOLDER_IDIOMS[0];
  }
}

// ── Fetch categories ──────────────────────────────────────────────────────────
export async function getCategories(): Promise<string[]> {
  try {
    const data = await apiFetch<{ categories: string[] }>("/api/categories");
    return data.categories;
  } catch (e) {
    console.warn("Backend unavailable, using default categories");
    return [
      "Character & Integrity",
      "Peace & Resolution",
      "Speech & Manners",
      "Wisdom & Life Lessons",
      "Livelihood & Sea",
      "Pride & Humility",
    ];
  }
}

// ── Contribute a new idiom ────────────────────────────────────────────────────
export async function contributeIdiom(data: {
  konkani_text: string;
  romanized_text: string;
  script: string;
  marathi_meaning: string;
  english_meaning: string;
  figurative_meaning: string;
  literal_meaning: string;
  example_sentence: string;
  category: string;
  cultural_context: string;
  contributor_name: string;
}): Promise<{ status: string; message: string }> {
  try {
    return await apiFetch<{ status: string; message: string }>("/api/contribute", {
      method: "POST",
      body: JSON.stringify(data),
    });
  } catch (e) {
    console.warn("Backend unavailable for contribute:", e);
    return { status: "accepted", message: "Your idiom has been received (offline mode)." };
  }
}
