"""
server-nlp/app/routers/resolve.py
POST /api/resolve endpoint — takes {"text": str}, returns matched idiom or {"matched": false}.
"""

import os
import sys
from fastapi import APIRouter
from pydantic import BaseModel

# Ensure server-nlp root directory is on sys.path
_SERVER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)

from app.core.query_pipeline import resolve_idiom
from app.data_loader import store

router = APIRouter(prefix="/api", tags=["Idiom Resolver"])


class ResolveRequest(BaseModel):
    text: str


class WordMatch(BaseModel):
    word: str
    role_hint: str | None = None
    best_matches: list[dict] | None = None


class ResolveResponse(BaseModel):
    matched: bool
    match_type: str | None = None
    confidence: float | None = None
    literal_meaning: str | None = None
    figurative_meaning: str | None = None
    english_meaning: str | None = None
    marathi_meaning: str | None = None
    example_sentence: str | None = None
    word_analysis: list[WordMatch] | None = None


@router.post("/resolve", response_model=ResolveResponse)
def resolve_endpoint(req: ResolveRequest):
    result = resolve_idiom(req.text.strip(), store)
    return result
