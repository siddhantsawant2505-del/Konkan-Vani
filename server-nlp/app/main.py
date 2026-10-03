"""
server-nlp/app/main.py
FastAPI application wiring health and resolve routers.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add server-nlp directory to sys.path
_SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from app.routers.health import router as health_router
from app.routers.resolve import router as resolve_router
from app.routers.search import router as search_router

app = FastAPI(
    title="Konkan Vani NLP Server",
    description="NLP Model Layer for Konkani idiom & proverb meaning extraction",
    version="1.0.0",
)

# Enable CORS for Next.js frontend & local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(resolve_router)
app.include_router(search_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
