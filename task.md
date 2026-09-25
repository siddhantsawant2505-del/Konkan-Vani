# Konkan Vani — Task Tracker

## Phase 1: Frontend API Wiring ✅
- [x] Task 1: Create `.env.local` for client backend URL config
- [x] Task 2: Add missing API endpoints to server-nlp (search, browse, stats, categories, contribute)
- [x] Task 3: Update server-nlp CORS config for localhost:3000
- [x] Task 4: Fix data model in server-nlp response (new schema)
- [x] Task 5: Wire `client/lib/api.ts` to real backend with fallback
- [x] Task 6: Add API error handling to frontend pages
- [x] Task 7: Add Kannada to ScriptToggle
- [x] Task 8: Remove unused SearchBar.tsx
- [x] Task 9: Wire contribute page to backend POST endpoint
- [x] Task 10: Verify full build passes (TypeScript + Next.js)

## Phase 2: Data — Rebuilt & Audited ✅ (2026-09-21)
- [x] Task 11: Design new schema (Devanagari input → Marathi/English output)
- [x] Task 12: Dataset rebuilt to exactly 5,200 entries — **101 curated real
      idioms + 5,099 labeled `pattern_v1` proverb frames** (see provenance note below)
- [x] Task 13: Update TypeScript types for new schema
- [x] Task 14: Update all frontend components for Marathi+English display
- [x] Task 15: Update backend routers for new schema
- [x] Task 16: Create data scaling/validation scripts (`src/rebuild_dataset.py`)
- [x] Task 17: Validate dataset: 5,200 rows · 5,200 unique konkani_text ·
      **5,200 unique phonetic_keys** · 0 missing · 0 duplicates

### Provenance note (audit 2026-09-21)
The previous CSV had 3,772 rows but only 113 unique phonetic keys — most rows
were noun-swapped template clones presented as "5,200 real idioms". The dataset
now carries a `source` column: rows are either `curated:*` (real, hand-written
idioms) or `pattern_v1` (transparent synthetic proverb frames used for pipeline
coverage). Rebuild: `python -X utf8 src/rebuild_dataset.py`.

## Phase 3: Model Layer ✅
- [x] Task 18: Update requirements.txt and install dependencies
- [x] Task 19: Create build_embeddings.py (LaBSE, 5200 × 768)
- [x] Task 20: Create data_loader.py (IdiomStore with FAISS support)
- [x] Task 21: Create phonetic_normalize.py (Devanagari-aware rules)
- [x] Task 22: Create query_pipeline.py (fuzzy + semantic matching)
- [x] Task 23: Create resolve.py router (POST /api/resolve)
- [x] Task 24: Create health.py router (GET /health)
- [x] Task 25: Wire main.py with both routers
- [x] Task 26: Build FAISS index (5,200 vectors, ~0.19 MB; embeddings 15.6 MB)

## Verification ✅
- [x] Exact Devanagari: matched (phonetic, confidence 1.0)
- [x] Misspelled Romanized: matched (phonetic, confidence 1.0)
- [x] Different idiom: matched (phonetic)
- [x] English not in dataset: correctly rejected
- [x] Devanagari not in dataset: correctly rejected
- [x] Konkani-sounding sentence not in dataset: matched:false (no fabrication)
- [x] Frontend build passes (TypeScript + Next.js)
- [x] Backend loads FAISS index at startup

## Safety Notes
- Dynamic AI fallback (Gemini) is **opt-in**: set `DYNAMIC_FALLBACK_ENABLED=1`
  and `GEMINI_API_KEY=<key>` to enable. It can produce high-confidence answers
  for idioms not in the dataset, so it is disabled by default.

## Architecture Summary

```
Frontend (Next.js:3000) → Backend (FastAPI:8000)
                              ↓
                         IdiomStore
                        (CSV + FAISS + LaBSE)
                              ↓
                    ┌─────────┴─────────┐
                    │  Fuzzy Match      │  ← phonetic_normalize
                    │  (rapidfuzz)      │     (fast, no model)
                    └─────────┬─────────┘
                              │ fail
                    ┌─────────┴─────────┐
                    │  Semantic Match   │  ← FAISS index
                    │  (LaBSE + FAISS)  │     (fast, ~150ms)
                    └─────────┬─────────┘
                              │ fail
                    ┌─────────┴─────────┐
                    │  Dynamic AI       │  ← opt-in only
                    │  (Gemini, cached) │     (DYNAMIC_FALLBACK_ENABLED=1)
                    └─────────┬─────────┘
                              │ fail
                    ┌─────────┴─────────┐
                    │  matched: false   │
                    └───────────────────┘
```

## To Run

```bash
# Backend
cd server-nlp && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Frontend
cd client && npm run dev
```
