"""
server-nlp/app/routers/health.py
GET /health endpoint — returns {"status": "ok", "idioms_loaded": <count>}.
"""

import os
import sys
from fastapi import APIRouter

# Ensure server-nlp root directory is on sys.path
_SERVER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)

from app.data_loader import store

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_endpoint():
    return {
        "status": "ok",
        "idioms_loaded": store.count,
    }
