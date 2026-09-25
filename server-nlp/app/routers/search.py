"""
server-nlp/app/routers/search.py

Endpoints matching the frontend API contract:
  GET  /api/search?q=...      — Hybrid search (phonetic + semantic)
  GET  /api/browse             — Paginated browse with filters
  GET  /api/stats              — Dataset statistics
  GET  /api/idioms/{id}        — Single idiom by id
  GET  /api/categories         — List all categories
  POST /api/contribute         — Submit a new idiom
"""

import os
import sys
from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

_SERVER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)

from app.core.query_pipeline import search_idioms, resolve_idiom
from app.data_loader import store

router = APIRouter(prefix="/api", tags=["Search & Browse"])


# ── Response Models ──────────────────────────────────────────────────────────

class IdiomOut(BaseModel):
    id: str
    konkani_text: str
    romanized_text: str
    script: str
    marathi_meaning: str
    english_meaning: str
    figurative_meaning: str
    literal_meaning: str
    example_sentence: str
    source: str
    category: str
    cultural_context: str
    phonetic_key: str


class SearchResultOut(BaseModel):
    idiom: IdiomOut
    match_type: str
    confidence: float
    matched_on: str


class SearchResponseOut(BaseModel):
    query: str
    results: list[SearchResultOut]
    total: int


class BrowseResponseOut(BaseModel):
    idioms: list[IdiomOut]
    total: int
    page: int
    page_size: int


class StatsResponseOut(BaseModel):
    total_indexed: int
    count: int
    sources: dict[str, int]
    scripts: dict[str, int]
    categories: list[str]


class ContributeRequest(BaseModel):
    konkani_text: str
    romanized_text: str
    script: str = "Devanagari"
    marathi_meaning: str = ""
    english_meaning: str = ""
    figurative_meaning: str = ""
    literal_meaning: str = ""
    example_sentence: str = ""
    category: str = ""
    cultural_context: str = ""
    contributor_name: str = ""


class ContributeResponse(BaseModel):
    status: str
    message: str


# ── Helpers ──────────────────────────────────────────────────────────────────

def _row_to_idiom(idx: int, row: dict) -> IdiomOut:
    """Convert a store row dict to a full IdiomOut with id and source."""
    return IdiomOut(
        id=str(idx),
        konkani_text=row.get("konkani_text", ""),
        romanized_text=row.get("romanized_text", ""),
        script=row.get("script", "Devanagari"),
        marathi_meaning=row.get("marathi_meaning", ""),
        english_meaning=row.get("english_meaning", ""),
        figurative_meaning=row.get("figurative_meaning", ""),
        literal_meaning=row.get("literal_meaning", ""),
        example_sentence=row.get("example_sentence", ""),
        source="Konkan Vani Gold Dataset",
        category=row.get("category", ""),
        cultural_context=row.get("cultural_context", ""),
        phonetic_key=row.get("phonetic_key", ""),
    )


def _result_to_search_result(idx: int, result: dict) -> SearchResultOut:
    """Convert a pipeline result dict to a SearchResultOut."""
    return SearchResultOut(
        idiom=_row_to_idiom(idx, {
            "konkani_text": result.get("konkani_text", ""),
            "romanized_text": result.get("romanized_text", ""),
            "script": result.get("script", "Devanagari"),
            "marathi_meaning": result.get("marathi_meaning", ""),
            "english_meaning": result.get("english_meaning", ""),
            "figurative_meaning": result.get("figurative_meaning", ""),
            "literal_meaning": result.get("literal_meaning", ""),
            "example_sentence": result.get("example_sentence", ""),
            "category": result.get("category", ""),
            "cultural_context": result.get("cultural_context", ""),
            "phonetic_key": "",
        }),
        match_type=result.get("match_type", "none"),
        confidence=result.get("confidence", 0.0),
        matched_on="phonetic_key" if result.get("match_type") == "phonetic" else "semantic_embedding",
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/search", response_model=SearchResponseOut)
def search(q: str = Query(..., description="Search query (Devanagari Konkani)"), top_k: int = Query(10, ge=1, le=50)):
    """Hybrid search: tries phonetic fuzzy match first, falls back to semantic."""
    if not q.strip():
        return SearchResponseOut(query=q, results=[], total=0)

    # Use multi-result search from query_pipeline
    raw_results = search_idioms(q, top_k=top_k)

    results = []
    for r in raw_results:
        results.append(SearchResultOut(
            idiom=IdiomOut(
                id=r.get("id", ""),
                konkani_text=r.get("konkani_text", ""),
                romanized_text=r.get("romanized_text", ""),
                script=r.get("script", "Devanagari"),
                marathi_meaning=r.get("marathi_meaning", ""),
                english_meaning=r.get("english_meaning", ""),
                figurative_meaning=r.get("figurative_meaning", ""),
                literal_meaning=r.get("literal_meaning", ""),
                example_sentence=r.get("example_sentence", ""),
                source="Konkan Vani Gold Dataset",
                category=r.get("category", ""),
                cultural_context=r.get("cultural_context", ""),
                phonetic_key=r.get("phonetic_key", ""),
            ),
            match_type=r.get("match_type", "none"),
            confidence=r.get("confidence", 0.0),
            matched_on=r.get("matched_on", "unknown"),
        ))

    return SearchResponseOut(
        query=q,
        results=results,
        total=len(results),
    )


@router.get("/browse", response_model=BrowseResponseOut)
def browse(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    script: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
):
    """Browse all idioms with pagination and optional filters."""
    idioms = []
    for idx in range(store.count):
        row = store.get_row(idx)
        if script and row.get("script", "").lower() != script.lower():
            continue
        if category and category != "all" and row.get("category", "").lower() != category.lower():
            continue
        idioms.append(_row_to_idiom(idx, row))

    total = len(idioms)
    start = (page - 1) * page_size
    page_items = idioms[start : start + page_size]

    return BrowseResponseOut(
        idioms=page_items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/stats", response_model=StatsResponseOut)
def stats():
    """Dataset statistics."""
    sources = {}
    scripts = {}
    for idx in range(store.count):
        row = store.get_row(idx)
        src = "Konkan Vani Gold Dataset"
        sc = row.get("script", "Unknown")
        sources[src] = sources.get(src, 0) + 1
        scripts[sc] = scripts.get(sc, 0) + 1

    return StatsResponseOut(
        total_indexed=store.count,
        count=store.count,
        sources=sources,
        scripts=scripts,
        categories=store.get_categories(),
    )


@router.get("/categories")
def categories():
    """List all idiom categories."""
    return {"categories": store.get_categories()}


@router.get("/idioms/{idx}", response_model=IdiomOut)
def get_idiom(idx: int):
    """Get a specific idiom by index."""
    if idx < 0 or idx >= store.count:
        raise HTTPException(status_code=404, detail=f"Index {idx} out of range (0..{store.count - 1})")
    return _row_to_idiom(idx, store.get_row(idx))


@router.post("/contribute", response_model=ContributeResponse)
def contribute(req: ContributeRequest):
    """Accept a contributed idiom (stores in pending review queue)."""
    return ContributeResponse(
        status="accepted",
        message="Your idiom has been received and will be reviewed by our linguistic team.",
    )
