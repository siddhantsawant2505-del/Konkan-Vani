"""
Konkan Vani — FastAPI Backend
===============================
Serves the NLP search engine to the Next.js frontend.

Endpoints:
    GET  /                  — Health check
    GET  /api/search?q=...  — Hybrid search (semantic + phonetic)
    GET  /api/browse        — Paginated browse of all idioms
    GET  /api/stats         — Dataset statistics
    GET  /api/idiom/{id}    — Get a specific idiom by index

Usage:
    py -X utf8 -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
"""

import os
import sys
import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.hybrid_search import get_search_engine
from src.model.search_index import get_search_index

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Konkan Vani API",
    description="Konkani idiom and proverb search engine",
    version="0.1.0",
)

# CORS — allow the Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup: load index on first request (lazy)
# ---------------------------------------------------------------------------

_index_loaded = False


def _ensure_index():
    """Load the FAISS index if not yet loaded."""
    global _index_loaded
    if _index_loaded:
        return
    index = get_search_index()
    try:
        index.load()
        _index_loaded = True
        logger.info(f"Index loaded: {index.size} vectors")
    except FileNotFoundError:
        logger.warning("No index found. Run src/build_index.py first.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def health_check():
    """Health check endpoint."""
    _ensure_index()
    index = get_search_index()
    return {
        "status": "ok",
        "service": "Konkan Vani API",
        "index_loaded": index._built,
        "total_entries": index.size,
    }


@app.get("/api/search")
async def search(
    q: str = Query(..., description="Search query (Konkani text, any script)"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results"),
    semantic_weight: Optional[float] = Query(
        None, ge=0.0, le=1.0,
        description="Override adaptive semantic/phonetic weight (0=phonetic, 1=semantic)"
    ),
):
    """
    Hybrid search for Konkani idioms/proverbs.
    Combines semantic similarity (embeddings) with phonetic fuzzy matching.
    """
    _ensure_index()

    engine = get_search_engine()
    results = engine.search(q, top_k=top_k, semantic_weight=semantic_weight)

    return {
        "query": q,
        "results": results,
        "total_found": len(results),
        "search_type": "hybrid" if results else "none",
    }


@app.get("/api/browse")
async def browse(
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(50, ge=1, le=200, description="Page size"),
    script: Optional[str] = Query(None, description="Filter by script: Devanagari, Kannada, Roman"),
    source: Optional[str] = Query(None, description="Filter by source dataset"),
    type: Optional[str] = Query(None, description="Filter by type"),
):
    """Browse all idioms with pagination and optional filters."""
    _ensure_index()

    index = get_search_index()
    meta = index.metadata

    # Apply filters
    filtered = meta
    if script:
        filtered = [m for m in filtered if m.get("script", "").lower() == script.lower()]
    if source:
        filtered = [m for m in filtered if m.get("source_dataset", "").lower() == source.lower()]
    if type:
        filtered = [m for m in filtered if m.get("type", "").lower() == type.lower()]

    total = len(filtered)
    page = filtered[offset : offset + limit]

    return {
        "items": page,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total,
    }


@app.get("/api/stats")
async def stats():
    """Dataset statistics."""
    _ensure_index()

    engine = get_search_engine()
    index_stats = engine.get_statistics()

    # Compute breakdown stats from metadata
    index = get_search_index()
    meta = index.metadata

    by_script: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    by_type: Dict[str, int] = {}

    for m in meta:
        s = m.get("script", "Unknown")
        by_script[s] = by_script.get(s, 0) + 1
        src = m.get("source_dataset", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        t = m.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    return {
        **index_stats,
        "by_script": by_script,
        "by_source": by_source,
        "by_type": by_type,
    }


@app.get("/api/idiom/{idx}")
async def get_idiom(idx: int):
    """Get a specific idiom entry by its index in the dataset."""
    _ensure_index()

    index = get_search_index()
    if idx < 0 or idx >= index.size:
        raise HTTPException(status_code=404, detail=f"Index {idx} out of range (0..{index.size-1})")

    return index.get_metadata(idx)
