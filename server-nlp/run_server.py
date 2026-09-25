import os
import sys
import uvicorn

# Ensure server-nlp root directory is on sys.path
_SERVER_ROOT = os.path.abspath(os.path.dirname(__file__))
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)

from app.main import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Konkan Vani NLP Server on port {port}...")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
