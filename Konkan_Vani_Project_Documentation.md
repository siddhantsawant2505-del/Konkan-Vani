# Konkan Vani — Complete Project Documentation

> **Purpose**: This document explains the entire Konkan Vani project — what it is, why it was built, how it works end-to-end, and what every significant line of code does. Intended for project presentation and evaluation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why It Was Built](#2-why-it-was-built)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Data Pipeline](#5-data-pipeline)
6. [The Custom NLP Model](#6-the-custom-nlp-model)
7. [Backend — File-by-File Code Explanation](#7-backend--file-by-file-code-explanation)
8. [Frontend — File-by-File Code Explanation](#8-frontend--file-by-file-code-explanation)
9. [Query Flow: End-to-End Request Lifecycle](#9-query-flow-end-to-end-request-lifecycle)
10. [Benchmark Results](#10-benchmark-results)
11. [How to Run the Project](#11-how-to-run-the-project)

---

## 1. Project Overview

**Konkan Vani** ("Voice of Konkan") is an NLP-powered web application that lets users search for, translate, and understand Konkani idioms and proverbs (*mhonni*). Konkani is a low-resource language spoken along India's western coast, primarily in Goa, and is uniquely written in **five different scripts** (Devanagari, Roman/Romi, Kannada, etc.), making standard NLP tools almost useless for it.

The system accepts input in **any script or spelling variant** and returns:
- The idiom's **literal meaning** (word-for-word translation)
- Its **figurative/semantic meaning** (what it actually implies culturally)
- An **English** and **Marathi** translation
- A **cultural context** explaining when and why it's used
- A **confidence score** for how certain the system is about the match

**Dataset**: 5,200 Konkani idioms and proverbs, each annotated with both Devanagari and Roman scripts, four meaning fields, category, and cultural context.

---

## 2. Why It Was Built

### The Problem

Konkani is endangered from a linguistic data perspective:

- **No standard script**: Speakers write the same sentence in Devanagari, Romi, Kannada, or Persian depending on region and religion.
- **No existing NLP tools**: Tokenizers, POS taggers, and pre-trained models trained on Hindi or Marathi fail on Konkani because of different phonology and vocabulary.
- **Oral tradition at risk**: Proverbs and idioms are passed down orally. Younger generations do not know them; there is no searchable digital corpus.

### The Solution

Konkan Vani solves this with a **4-stage matching pipeline**:

1. **Phonetic normalization** — collapses spelling variants (`aa→a`, `v→w`, etc.) so `"Haat dakhvun"` and `"Hat dakhwun"` find the same idiom.
2. **Fuzzy string matching** — handles typos and partial queries via rapidfuzz.
3. **Custom semantic embedding model** — fine-tuned on the Konkani dataset so the model understands the domain deeply.
4. **Compositional fallback** — for completely unknown idioms, glosses each word from a lexicon and composes a provisional meaning rather than returning a blank "not found" screen.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER (Browser)                          │
│              Next.js 15 Frontend  (port 3000)               │
│   client/app/page.tsx  →  client/app/results/page.tsx       │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP (fetch)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI NLP Server  (port 8000)                │
│         server-nlp/app/main.py                              │
│                                                             │
│  Routers:                                                   │
│   POST /api/resolve   →  resolve.py   →  query_pipeline.py  │
│   GET  /api/search    →  search.py    →  query_pipeline.py  │
│   GET  /api/browse    →  search.py    →  data_loader.py     │
│   GET  /health        →  health.py   →  data_loader.py     │
│                                                             │
│  Core NLP Modules:                                          │
│   model_registry.py   (singleton encoder — konkan-vani-v1) │
│   query_pipeline.py   (4-stage matching logic)              │
│   phonetic_normalize.py (aa→a, v→w normalizer)             │
│   word_analysis.py    (per-token embedding analysis)        │
│   compose.py          (provisional word-by-word reading)    │
│   lexicon.py          (word gloss dictionary)               │
│                                                             │
│  Data Layer:                                                │
│   data_loader.py      (IdiomStore singleton)                │
│   data/processed/     (CSV + FAISS index + embeddings)      │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   idiom_embeddings.npy  konkani.index   idioms_with_
   (5200×768 float32)    (FAISS index)   phonetic_keys.csv
```

### Data Flow Summary

1. User types a query in the browser.
2. Next.js calls `GET /api/search?q=<query>` on the FastAPI server.
3. FastAPI runs the query through a 4-stage pipeline.
4. Returns JSON: a list of matched idioms with confidence scores.
5. If nothing matched: returns a `provisional` word-by-word reading.
6. Frontend displays results with confidence badges and script toggles.

---

## 4. Technology Stack

| Layer | Technology | Why Chosen |
|---|---|---|
| Frontend | **Next.js 15** (React, TypeScript) | Server/client routing, fast builds, Tailwind CSS |
| Styling | **Tailwind CSS** | Utility-first — rapid custom UI without a framework |
| Backend API | **FastAPI** (Python) | Async, typed, auto-generates API docs |
| ML Model | **sentence-transformers/LaBSE** fine-tuned | Multilingual, works across Devanagari+Roman scripts |
| Vector Search | **FAISS** (IndexFlatIP) | Sub-millisecond nearest-neighbor search over 5200 vectors |
| Fuzzy Matching | **rapidfuzz** | C-backed fuzzy string matching |
| Data | **pandas** + **numpy** | Fast in-memory data processing |
| Training | **PyTorch** + **CUDA** | GPU-accelerated fine-tuning |
| Loss Function | **MultipleNegativesRankingLoss** | Contrastive: pulls idiom closer to its own meaning |

---

## 5. Data Pipeline

The data pipeline has **3 build steps** that must be run in order before the server starts:

### Step 1: Dataset (`data/processed/idioms_with_phonetic_keys.csv`)
- 5,200 rows, each representing one Konkani idiom/proverb.
- Columns: `konkani_text`, `romanized_text`, `literal_meaning`, `figurative_meaning`, `english_meaning`, `marathi_meaning`, `example_sentence`, `category`, `cultural_context`, `phonetic_key`.
- `phonetic_key` is the normalized romanized form used for fuzzy search.

### Step 2: Embeddings (`data/idiom_embeddings.npy`)
- Run by: `python server-nlp/scripts/build_embeddings.py`
- Each row is encoded as `konkani_text + " " + romanized_text` using the custom model.
- Output: 5200×768 float32 numpy array (15.6 MB).

### Step 3: FAISS Index (`data/processed/index/konkani.index`)
- Run by: `python -X utf8 src/build_index.py`
- Loads embeddings, L2-normalizes them, builds a FAISS `IndexFlatIP`.
- Inner product on normalized vectors = cosine similarity.
- Output: 15.2 MB FAISS binary + `konkani_meta.json` (metadata parallel to the index).

---

## 6. The Custom NLP Model

### What Is LaBSE?

**LaBSE** (Language-agnostic BERT Sentence Encoder) is a multilingual transformer model from Google. It maps sentences from 109 languages into a shared 768-dimensional vector space so that semantically similar sentences from *different languages* end up close to each other.

This is critical for Konkani: a query in Devanagari (`हात दाखवून अयाक`) and its Romanized equivalent (`Haat dakhvun ayaak`) need to produce similar vectors.

### Why Fine-Tune?

The generic LaBSE has never seen Konkani data. On our dataset, it achieved only **51.4% Recall@1** — meaning it failed to find the correct idiom 49% of the time for exact queries. This is because it treats Konkani proverbs as generic multilingual text, not as a specialized domain with fixed expressions.

### Fine-Tuning Approach: MultipleNegativesRankingLoss (MNRL)

For each training pair `(idiom_text, idiom_meaning)`:
- The model is asked: **given this idiom text, which meaning in the batch belongs to it?**
- All *other* meanings in the same batch are treated as hard negatives.
- This forces the model to learn that `"हात दाखवून अयाक"` is more similar to `"To make false promises and disappear"` than to any other idiom's meaning.

### Training Details

| Parameter | Value |
|---|---|
| Base Model | `sentence-transformers/LaBSE` |
| Training Pairs | 8,320 (idiom text ↔ combined meaning string) |
| Test Samples | 1,040 |
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Warmup Steps | 10% of total steps |
| GPU | NVIDIA RTX 3050 6GB (CUDA) |
| Mixed Precision | fp16 (AMP) |
| Training Time | 126 minutes |

### Results

| Metric | Generic LaBSE | Fine-Tuned (konkan-vani-encoder-v1) | Gain |
|---|---|---|---|
| Recall@1 | 51.4% | **99.8%** | +94.4% |
| Recall@5 | 78.8% | **99.9%** | +26.9% |
| MRR@10 | 0.628 | **0.998** | +59.1% |
| Avg Positive Similarity | 0.437 | **0.940** | +115.3% |

The fine-tuned model saved to: `models/konkan-vani-encoder-v1/`

---

## 7. Backend — File-by-File Code Explanation

---

### `server-nlp/scripts/train_model.py`
*Fine-tunes the LaBSE model on our Konkani dataset.*

| Lines | Code | What It Does |
|---|---|---|
| 37–42 | `SEED = 42; random.seed(); np.random.seed(); torch.manual_seed(); torch.cuda.manual_seed_all()` | Sets the same random seed everywhere so results are reproducible — running training twice gives the same model. |
| 122 | `device = "cuda" if torch.cuda.is_available() else "cpu"` | Picks GPU if available, otherwise CPU. GPU training is ~10× faster. |
| 136–141 | `df.sample(frac=1.0).reset_index(); split_idx = int(0.8 * len(df))` | Shuffles the dataset deterministically, then splits 80% for training and 20% for testing. Test split is never seen during training. |
| 150–158 | `meaning = english + " — " + figurative + " (" + marathi + ")"; InputExample(texts=[konk, meaning]); InputExample(texts=[rom, meaning])` | Creates two training pairs per idiom — Devanagari form and Roman form both paired with the combined meaning. This teaches the model both scripts simultaneously. |
| 177 | `train_loss = MultipleNegativesRankingLoss(model)` | The loss function. For a batch of N pairs, creates N² contrasts — every idiom competes against every other idiom's meaning as a negative. |
| 203 | `use_amp=True if device == "cuda" else False` | Enables automatic mixed precision (fp16). Reduces VRAM usage from ~12GB to ~5.5GB, making the 6GB RTX 3050 feasible. |
| 218 | `model.save(OUTPUT_MODEL_DIR)` | Saves the complete model (weights + tokenizer + config) to disk so the server can load it without re-training. |
| 77–100 | `sim_matrix = query_embs @ doc_embs.T; rank = np.where(sorted_indices == i)[0][0] + 1` | Evaluation: computes cosine similarity between every query and every document (N×N matrix), finds the rank of the correct answer, computes MRR and Recall@k. |

---

### `server-nlp/scripts/build_embeddings.py`
*Encodes all 5,200 idioms as vectors and saves to disk.*

| Lines | Code | What It Does |
|---|---|---|
| 24–28 | `_SERVER_NLP = ...; sys.path.insert(0, _SERVER_NLP)` | Adds `server-nlp/` to Python's path so `app.core.model_registry` can be imported from a script in a different directory. |
| 45–48 | `texts = (df["konkani_text"] + " " + df["romanized_text"]).tolist()` | Concatenates both scripts per idiom. The embedding captures phonetic patterns from both Devanagari and Roman simultaneously. |
| 53–54 | `model = model_registry.get_model()` | Loads the **same** model the runtime server uses. Critical: if different models are used for corpus and queries, similarity scores are meaningless. |
| 59–64 | `model.encode(texts, normalize_embeddings=True)` | Encodes all 5,200 texts. `normalize_embeddings=True` makes each vector unit-length so inner product = cosine similarity. |
| 70 | `np.save(output_path, embeddings)` | Saves the embeddings as a numpy binary file (fast load, compact format). |

---

### `src/build_index.py`
*Builds the FAISS index from the saved embeddings.*

| Lines | Code | What It Does |
|---|---|---|
| 73–76 | `assert embeddings.shape[0] == len(df)` | Verifies row count matches embedding count. If CSV was updated without rebuilding embeddings, fails with a clear error message. |
| 100–102 | `faiss_norms = np.linalg.norm(...); normalized = embeddings / faiss_norms` | L2-normalizes embeddings to unit length. FAISS `IndexFlatIP` computes inner products; on unit vectors, inner product = cosine similarity. |
| 110–111 | `index = faiss.IndexFlatIP(dim); index.add(normalized)` | Creates a flat (exact, no approximation) FAISS index in inner-product space and adds all 5,200 vectors. |
| 121–122 | `faiss.write_index(index, index_path)` | Saves the binary FAISS index. Can be loaded in milliseconds at server startup. |
| 126–127 | `json.dump(metadata, f)` | Saves a parallel JSON file — for each vector position, stores the full idiom metadata so results can be looked up by index. |

---

### `server-nlp/app/core/model_registry.py`
*Single shared encoder — loaded once, used everywhere.*

| Lines | Code | What It Does |
|---|---|---|
| 21–22 | `_MODEL = None; _LOCK = threading.Lock()` | Module-level global starts as `None`. Lock prevents two threads from both trying to initialize the model at the same time. |
| 25–28 | `_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))` | Computes project root by going 3 dirs up from `server-nlp/app/core/`. Makes path resolution work regardless of where Python is invoked from. |
| 39–41 | `if _MODEL is None: with _LOCK: if _MODEL is None:` | Double-checked locking pattern. First check (cheap, no lock) avoids lock contention after model is loaded. Inner check (inside lock) prevents duplicate initialization. |
| 44–48 | `if os.path.isdir(_CUSTOM_MODEL_PATH): _MODEL = SentenceTransformer(_CUSTOM_MODEL_PATH)` | Loads the fine-tuned model if checkpoint directory exists. Falls back to generic LaBSE if missing (e.g., fresh clone before training). |
| 64–66 | `kwargs.setdefault("normalize_embeddings", True); kwargs.setdefault("convert_to_numpy", True)` | Public encoding API always returns unit-normalized numpy arrays. Every caller gets consistent behavior without having to remember these flags. |

---

### `server-nlp/app/data_loader.py`
*Loads all data into memory at startup. Lives as a singleton called `store`.*

| Lines | Code | What It Does |
|---|---|---|
| 17 | `_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))` | Goes 2 dirs up from `server-nlp/app/` to find the project root where `data/` lives. |
| 30–31 | `self.df = pd.read_csv(_CSV_PATH); self.count = len(self.df)` | Loads all 5,200 idioms into a pandas DataFrame in memory. |
| 43–46 | `self._roman_lower = df["romanized_text"].str.lower().tolist(); self._konkani_raw = df["konkani_text"].tolist()` | Pre-caches data as plain Python lists. Scanning a flat list is ~300ms faster than calling `get_row()` 5,200 times (avoids pandas overhead per row). |
| 49–53 | `assert len(self.df) == len(self.embeddings)` | Sanity check at startup. If CSV was updated but embeddings weren't rebuilt, this fails immediately with a clear error rather than silently giving wrong results. |
| 83 | `assert self.faiss_index.ntotal == len(self.df)` | Same check for FAISS index — ensures index and CSV are in sync. |
| 107–108 | `q = query_emb.reshape(1, -1).astype(np.float32); scores, indices = self.faiss_index.search(q, top_k)` | FAISS requires shape `(1, dim)` and `float32` dtype. Reshapes and casts before passing to FAISS. |
| 116–118 | `sims = self.embeddings @ query_emb.astype(np.float32); top_indices = np.argsort(sims)[::-1][:top_k]` | Brute-force fallback if FAISS not available. Matrix multiply all embeddings vs query, sort descending. |
| 164 | `store = IdiomStore()` | Singleton created when this module is first imported. All modules that import `store` share the same object — no duplicate loading. |

---

### `server-nlp/app/core/phonetic_normalize.py`
*Collapses spelling variants into a canonical form.*

| Lines | Code | What It Does |
|---|---|---|
| 18 | `re.sub(r'[^\u0900-\u097F\s]', '', text)` | For Devanagari: keeps only Devanagari Unicode block (U+0900–U+097F) and whitespace. Strips punctuation and symbols. |
| 31–38 | `norm = re.sub(r'aa','a'); re.sub(r'ee','i'); re.sub(r'oo','u'); re.sub(r'ph','f'); re.sub(r'v','w'); re.sub(r'zh','j')` | Six phonetic substitution rules for Romanized Konkani. `"Haat dakhvun"` and `"Hat dakhwun"` both become `"hat daxwun"` — same phonetic key, same search result. |
| 38 | `re.sub(r'[^a-z\s]', '', norm)` | Strips anything not a Latin letter or space. Numbers, hyphens, apostrophes all removed. |

---

### `server-nlp/app/core/query_pipeline.py`
*The core matching logic — 4 stages, most specific first.*

| Lines | Code | What It Does |
|---|---|---|
| 38–42 | `FUZZY_THRESHOLD=75; SEMANTIC_THRESHOLD=0.63; SEMANTIC_MIN_OVERLAP=0.34` | Thresholds tuned by measuring false positives on real data. 0.63 sits in the gap where unrelated sentences scored ≤0.61 and real matches score ≥0.70. |
| 52–55 | `_OVERLAP_STOPWORDS = {"mhanje", "ani", "na", ...}` | Konkani function words with no topical meaning. Excluded when measuring word overlap so "ani" (and) doesn't create false matches. |
| 58–70 | `content_keys(text)` | Extracts content words: finds all word tokens, skips stopwords, phonetically normalizes each, returns a set. Works for both Devanagari and Roman text. |
| 73–85 | `lexical_overlap(query, candidate)` | What fraction of the query's content words appear in the candidate. `len(q & c) / len(q)` — set intersection divided by query size. Used to reject "topically similar but different proverb" false positives from FAISS. |
| 98–121 | `fuzzy_match()` — Devanagari branch | If input is Devanagari, compares normalized text against `konkani_text` column. `token_sort_ratio` sorts words alphabetically first, making it robust to word-order variation. |
| 123–148 | `fuzzy_match()` — Roman branch | Compares normalized phonetic key against all 5,200 stored phonetic keys using rapidfuzz. `extractOne` returns the single best match above `score_cutoff`. |
| 158–186 | `semantic_match()` | Encodes query with custom model, fetches top 3 FAISS results, checks both score threshold AND word overlap. Only returns a match if the candidate shares ≥34% content words with the query. |
| 204–256 | `resolve_idiom()` | Public resolver: tries Stage 1 (fuzzy) → Stage 2 (semantic) → Stage 3 (AI fallback, disabled by default) → Stage 4 (compositional). Stops at first success. Always attaches `word_analysis` to every response. |
| 265–363 | `search_idioms()` | Multi-result search: substring scan → phonetic fuzzy → semantic FAISS. Uses `seen_ids` set to deduplicate. Sorts by confidence descending before returning. |

---

### `server-nlp/app/core/word_analysis.py`
*Analyzes each word in the query using the shared model.*

| Lines | Code | What It Does |
|---|---|---|
| 43–44 | `_vocab = None; _vocab_emb = None` | Module-level cache. Vocabulary embeddings are computed once (on first request) and reused for all subsequent requests. |
| 50–58 | `_extract_vocab()` | Collects every unique word from all 5,200 idioms in both scripts. Words shorter than 2 characters are skipped. Returns sorted list. |
| 69–76 | `if list(npz["vocab"]) == vocab:` | Cache validation: saved vocab must exactly match current vocab. If dataset was updated, cache is invalidated and rebuilt. |
| 81 | `emb = encode(vocab, batch_size=256)` | Encodes all vocabulary words in batches of 256. Batch size avoids out-of-memory errors on long vocabulary lists. |
| 113–114 | `tok_embs = encode(tokens); sims = tok_embs @ _vocab_emb.T` | Matrix multiply: `(T tokens) × (V vocab)` → `(T × V)` similarity matrix. Every token is compared to every vocabulary word in one operation. |
| 118–122 | `order = np.argsort(row)[::-1][:top_k]` | For each token, sort vocab by similarity descending and take top-k. Returns the closest dataset words to each input word. |
| 131–176 | `resolve_words()` | Conservative gate: requires at least 2 content words with exact vocab hits AND majority strong matches. Prevents ordinary sentences from being treated as idioms. |

---

### `server-nlp/app/core/lexicon.py`
*Konkani word dictionary with 6-level layered lookup.*

The lexicon CSV (`data/lexicon/konkani_lexicon.csv`) is a manually curated dictionary of Konkani words with English, Marathi, POS (part-of-speech), and semantic tag annotations.

| Lookup Level | Code | What It Does |
|---|---|---|
| 1. **Exact** | Dictionary lookup `O(1)` | Token is literally a headword (Devanagari or Roman or its phonetic key). |
| 2. **Suffix** | Strip `_DEV_SUFFIXES` / `_ROM_SUFFIXES`, re-lookup | Handles inflected forms: `"पाण्यात"` → strip `"ात"` → `"पाणी"` (water). |
| 3. **Phonetic** | `phonetic_normalize()` then lookup | Handles `v/w`, `aa/a`, `ee/i`, `oo/u` variants. |
| 4. **Fuzzy** | `rapidfuzz`, cutoff 84 | Near-miss spellings of Roman words. High cutoff (84) because short words have few unique spellings. |
| 5. **Embedding** | LaBSE cross-lingual cosine, threshold 0.70 | For Devanagari tokens not in the dictionary. Compares the token's LaBSE vector with English gloss vectors of nearby entries. Marked "approximate." Roman tokens skip this (Recall@1 = 0.06 for Roman, too noisy). |
| 6. **None** | Returns `known=False` | Token not recognized. Shown verbatim in brackets `[word]` in the UI. |

---

### `server-nlp/app/core/compose.py`
*Produces a provisional meaning for unknown idioms.*

| Lines | Code | What It Does |
|---|---|---|
| 104–113 | `_dedupe_reduplication()` | Konkani repeats words for emphasis (`"मोठो मोठो"` = "very big"). Collapses consecutive identical words, sets `repeated=True` flag for the explanation text. |
| 158–177 | `_merge_compounds()` | Merges Konkani verb compounds. Participle + light verb (`"धुवून घेवप"`) becomes one entry: "wash (for oneself)". Stops the light verb from being shown as a separate action. |
| 288 | `action = verbs[-1] if verbs else None` | Konkani is verb-final — the main verb comes last. Takes the last recognized verb as the primary action word. |
| 290–329 | Phrase composition | Builds the English explanation: `"Provisional reading (X/Y words recognised): to {action} {things} — a saying about {theme}."` |
| 414 | `confidence = round(min(COMPOSITION_CEILING, base * (0.55 + 0.45 * coverage)), 4)` | Caps confidence at 0.72. A compositionally-derived reading is never as trustworthy as a corpus match — ceiling prevents the UI from showing 95% confidence for a guess. |
| 419 | `"provisional": True` | Flags the response so the frontend knows to show a "provisional" warning banner instead of treating it as an authoritative corpus match. |

---

### `server-nlp/app/main.py`
*FastAPI application entry point.*

| Lines | Code | What It Does |
|---|---|---|
| 12–14 | `_SERVER_DIR = os.path.abspath(...); sys.path.insert(0, _SERVER_DIR)` | Adds `server-nlp/` to Python path so all `app.*` imports resolve regardless of where `uvicorn` is launched from. |
| 20–24 | `app = FastAPI(title="Konkan Vani NLP Server", version="1.0.0")` | Creates the FastAPI app. The title and version appear in the auto-generated API docs at `/docs`. |
| 27–39 | `app.add_middleware(CORSMiddleware, allow_origins=[...], allow_methods=["*"])` | CORS middleware. Without this, the browser refuses to let the Next.js frontend (port 3000) call the FastAPI server (port 8000) — browsers block cross-origin requests by default. |
| 42–44 | `app.include_router(health_router); app.include_router(resolve_router); app.include_router(search_router)` | Mounts all three routers into the app. Each router handles its own URL prefix and endpoints. |

---

### `server-nlp/app/routers/health.py`
*Simple health-check endpoint.*

| Lines | Code | What It Does |
|---|---|---|
| 20–25 | `@router.get("/health"); return {"status": "ok", "idioms_loaded": store.count}` | Returns server status and how many idioms are loaded. Used by monitoring and by the frontend to detect if the backend is running. |

---

### `server-nlp/app/routers/resolve.py`
*POST /api/resolve — single-idiom deep resolver.*

| Lines | Code | What It Does |
|---|---|---|
| 23–24 | `class ResolveRequest(BaseModel): text: str` | Pydantic model for request validation. FastAPI automatically validates the POST body and returns 422 if `text` is missing. |
| 33–55 | `class ResolveResponse(BaseModel): matched: bool; match_type: str \| None = None; provisional: bool \| None = None; ...` | All fields are optional (`\| None = None`) so the same schema works for matched, compositional, and not-found responses. |
| 58–61 | `@router.post("/resolve"); result = resolve_idiom(req.text.strip(), store); return result` | Three lines — all business logic is in `query_pipeline.resolve_idiom()`. The router strips whitespace and delegates. |

---

### `server-nlp/app/routers/search.py`
*All search, browse, stats, and contribute endpoints.*

| Lines | Code | What It Does |
|---|---|---|
| 160–217 | `GET /api/search` | Calls `search_idioms()` for corpus matches. If no results, calls `compose_idiom_meaning()` and puts the word-by-word reading in the `provisional` field. Frontend checks for `provisional` and shows `ProvisionalMeaning` instead of "not found." |
| 220–246 | `GET /api/browse` | Iterates all idioms, applies optional script/category filters, slices to the requested page. Straightforward pagination for 5,200 rows. |
| 249–267 | `GET /api/stats` | Counts idioms by source and script. Returns total count and category list. |
| 270–273 | `GET /api/categories` | Returns sorted list of all category names. Used by the frontend filter dropdowns. |
| 276–281 | `GET /api/idioms/{idx}` | Fetches one idiom by row index. Returns 404 HTTPException if index is out of range. |
| 284–290 | `POST /api/contribute` | Accepts user-submitted idioms. Currently acknowledges receipt; in production would write to a review queue. |

---

## 8. Frontend — File-by-File Code Explanation

---

### `client/lib/types.ts`
*TypeScript interfaces shared across the entire frontend.*

| Lines | Code | What It Does |
|---|---|---|
| 4 | `export type Script = "devanagari" \| "kannada" \| "roman"` | Union type — only these three strings are valid. TypeScript enforces this everywhere `Script` is used; any typo is a compile error. |
| 6–20 | `export interface Idiom { id, konkani_text, romanized_text, marathi_meaning, english_meaning, ... }` | Shape of one idiom from the API. Every field explicitly typed — TypeScript catches API/UI mismatches at compile time. |
| 22–28 | `export type MatchType = "exact" \| "phonetic" \| "semantic" \| "fuzzy" \| "compositional"` | Every search result carries a match type. This drives the color and label of the confidence badge in the UI. |
| 58–71 | `export interface ProvisionalReading { coverage: {glossed, total, ratio}; themes; word_glosses; closest_known; ... }` | Shape of a compositional (word-by-word) reading. `coverage` tells the UI how many words were recognized. `closest_known` shows similar corpus entries even when no match was found. |

---

### `client/lib/api.ts`
*All HTTP communication with the FastAPI backend.*

| Lines | Code | What It Does |
|---|---|---|
| 14 | `const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"` | Backend URL from env var. In production, set `NEXT_PUBLIC_API_URL` to the deployed backend URL. Falls back to `localhost:8000` for development. |
| 81–94 | `async function apiFetch<T>(path, init?)` | Typed fetch wrapper. Generic `<T>` means `apiFetch<Idiom>(...)` returns `Promise<Idiom>`. Throws on non-2xx responses. |
| 97–126 | `export async function searchIdioms(query)` | Calls `GET /api/search`. If the backend is down, silently falls back to 4 hardcoded placeholder idioms so the UI always shows something. |
| 129–170 | `function searchIdiomsLocal(query)` | Offline fallback: simple substring filter over 4 hardcoded idioms. Used only when backend is unavailable. |
| 173–205 | `export async function browseIdioms(page, pageSize, scriptFilter, categoryFilter)` | Calls `GET /api/browse` with optional filters. Builds the URL dynamically, excluding undefined filters. |
| 230–238 | `export async function getIdiomById(id)` | Fetches a single idiom. Falls back to searching the local placeholder array if backend is down. |
| 259–281 | `export async function contributeIdiom(data)` | POSTs a new idiom to the backend. Falls back to an accepted message in offline mode. |

---

### `client/app/page.tsx`
*The home page — search input and featured idioms.*

| Lines | Code | What It Does |
|---|---|---|
| 1 | `"use client"` | Marks this as a Client Component in Next.js. Required because it uses `useState`, `useEffect`, and `useRouter` — browser-only APIs. |
| 14–16 | `useState(""); useState<Script>("devanagari"); useState<Idiom[]>([])` | Three state values: current search query, active script for display (Devanagari/Roman), and the 4 featured idioms in the homepage grid. |
| 18–22 | `useEffect(() => { searchIdioms("").then(...setFeaturedIdioms(res.results.slice(0, 4))) }, [])` | On page load, fetches 4 random idioms for the featured grid. The `catch(() => {})` silently ignores backend errors — grid just stays empty. |
| 24–29 | `handleSearch(e) { e.preventDefault(); router.push("/results?q=...") }` | Form submit handler. `preventDefault()` stops full-page reload. `encodeURIComponent` makes Devanagari text URL-safe. |
| 31–36 | `const sampleQueries = [...]` | Pre-defined example queries shown as clickable chips. Clicking one sets the query and immediately navigates to results. |
| 136–145 | `featuredIdioms.map((idiom) => <IdiomCard idiom={idiom} activeScript={activeScript} />)` | Renders the featured idioms grid. `activeScript` prop controls whether to display Devanagari or Roman text in each card. |

---

### `client/app/results/page.tsx`
*The results page — displays matched idioms or provisional readings.*

| Lines | Code | What It Does |
|---|---|---|
| 340–346 | `<Suspense fallback={...}><ResultsContent /></Suspense>` | Wraps content in React Suspense. `useSearchParams()` requires Suspense in Next.js — without it, the page crashes during server-side rendering. |
| 20 | `const query = searchParams.get("q") \|\| ""` | Reads the search query from the URL (`?q=...`). When user navigates to `/results?q=Haat+dakhvun`, this extracts `"Haat dakhvun"`. |
| 23–26 | `useState<SearchResult[]>([]); useState<ProvisionalReading \| null>(null)` | `results` holds corpus matches. `provisional` holds the word-by-word reading — only set when no corpus match found. |
| 31–45 | `useEffect(() => { searchIdioms(query).then(res => { setResults(res.results); setProvisional(res.provisional ?? null) }) }, [query])` | Re-fetches whenever the URL query changes. `finally` always clears loading state — spinner never gets stuck. |
| 54–55 | `const primaryResult = results[0]; const relatedResults = results.slice(1)` | First result gets the hero card (large, detailed). All others go into the smaller "Related Proverbs" grid below. |
| 308–313 | `provisional ? <ProvisionalMeaning reading={provisional} /> : <div>Nothing to analyse</div>` | Three-way render: (1) corpus matches → hero card + related grid; (2) no match but compositional reading → `ProvisionalMeaning`; (3) empty input or no letters → "nothing to analyse" message. |
| 140–246 | Hero card section | Shows the primary match with confidence badge, Devanagari/Roman toggled text, Marathi meaning, English meaning, figurative/literal/category grid, example sentence, and cultural context. |

---

## 9. Query Flow: End-to-End Request Lifecycle

Here is exactly what happens when a user types `"Haat dakhvun ayaak"` and clicks Search:

```
1. BROWSER (page.tsx)
   User types → state: "Haat dakhvun ayaak"
   Clicks "Explore" → handleSearch() fires
   e.preventDefault() → stops page reload
   router.push("/results?q=Haat%20dakhvun%20ayaak")

2. NEXT.JS ROUTER
   Renders /results/page.tsx
   useSearchParams().get("q") → "Haat dakhvun ayaak"
   useEffect detects query change → setLoading(true)
   Calls searchIdioms("Haat dakhvun ayaak")

3. API CLIENT (api.ts)
   apiFetch() → GET http://localhost:8000/api/search?q=Haat%20dakhvun%20ayaak

4. FASTAPI ROUTER (search.py)
   search(q="Haat dakhvun ayaak", top_k=10)
   Calls search_idioms("Haat dakhvun ayaak", top_k=10)

5. QUERY PIPELINE (query_pipeline.py)
   
   STEP A — Substring scan:
     "haat dakhvun ayaak" in any romanized_text? → No exact hit
   
   STEP B — Phonetic fuzzy:
     phonetic_normalize("Haat dakhvun ayaak") → "hat daxwun ayak"
     rapidfuzz.process.extract("hat daxwun ayak", 5200 phonetic keys)
     → Row 42: phonetic_key="hat daxwun ayak" scores 100/100
     → Match found: "हात दाखवून अयाक", confidence=1.0, match_type="phonetic"
   
   STEP C — Semantic FAISS (skipped — phonetic already matched)
   
   Returns: [{ idiom_data, match_type: "phonetic", confidence: 1.0 }]

6. SEARCH ROUTER (search.py)
   Wraps in SearchResultOut + IdiomOut models
   provisional = None (results exist)
   Returns JSON response

7. API CLIENT (api.ts)
   Maps JSON → TypeScript SearchResponse type
   Returns to ResultsContent component

8. RESULTS PAGE (results/page.tsx)
   setResults([result]) → primaryResult = result
   setProvisional(null)
   setLoading(false)
   
   Renders hero card:
   - Title: "हात दाखवून अयाक" (Devanagari) or "Haat dakhvun ayaak" (Roman)
   - ConfidenceBadge: "Phonetic Match · 100%"
   - Marathi: "हात दाखवून येणे म्हणजे खोटे वचन देणे..."
   - English: "To make false promises and disappear"
   - Figurative: "To lead someone on with false assurances"
   - Category: "Character & Integrity"
   - Cultural context: "Commonly used in coastal trade markets..."
```

**Total latency**: ~50–100 ms (model pre-loaded in memory after first warm-up).

---

## 10. Benchmark Results

From `models/benchmark_report.json` (generated by `train_model.py`):

| Metric | Baseline LaBSE | Fine-Tuned Model | Gain |
|---|---|---|---|
| **Recall@1** | 51.4% | **99.8%** | +94.4% |
| **Recall@5** | 78.8% | **99.9%** | +26.9% |
| **Recall@10** | 86.4% | **99.9%** | — |
| **MRR@10** | 0.628 | **0.998** | +59.1% |
| **Avg Positive Similarity** | 0.437 | **0.940** | +115.3% |

**Training config**: 8,320 pairs · 3 epochs · batch size 16 · RTX 3050 6GB · 126 minutes

### What These Metrics Mean

- **Recall@1**: Given a query idiom, is the correct meaning ranked #1 out of 1,040 test idioms? Our model gets this right **99.8%** of the time.
- **MRR@10**: Mean Reciprocal Rank — average of 1/rank for the correct answer within top-10. A value of 0.998 means the correct answer is almost always ranked #1.
- **Avg Positive Similarity**: Cosine similarity between a query and its own correct meaning. 0.940 vs 0.437 baseline — the model has deeply learned that Konkani idioms belong near their meanings in vector space.

---

## 11. How to Run the Project

### Prerequisites
- Python 3.12, Node.js 20+, NVIDIA GPU (optional — training only)

```bash
pip install sentence-transformers faiss-cpu rapidfuzz fastapi uvicorn pandas numpy torch
```

### Step 1: Build the dataset (if not already present)
```bash
py -X utf8 src/rebuild_dataset.py
```

### Step 2: Build embeddings
```bash
py server-nlp/scripts/build_embeddings.py
```

### Step 3: Build FAISS index
```bash
py -X utf8 src/build_index.py
```

### Step 4: Start the backend
```bash
cd server-nlp
py -m uvicorn app.main:app --reload --port 8000
```

### Step 5: Start the frontend
```bash
cd client
npm install
npm run dev
# Opens at http://localhost:3000
```

### Step 6 (Optional): Re-train the model
```bash
py server-nlp/scripts/train_model.py
# After training, re-run Steps 2 and 3 to rebuild index with new weights
```

---

> *This document covers all key source files. The project is a working end-to-end pipeline: browser UI → Next.js → FastAPI → fine-tuned LaBSE → FAISS vector search → 5,200-idiom Konkani corpus.*
