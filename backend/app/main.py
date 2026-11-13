from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.documents import router as documents_router
from app.core.config import OLLAMA_BASE_URL
import httpx

app = FastAPI(
    title="CivicLens API",
    description="Privacy-first document intelligence platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)

@app.get("/")
async def root():
    return {
        "application": "CivicLens",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "CivicLens API"
    }

@app.get("/health/ollama")
async def ollama_health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{OLLAMA_BASE_URL}/api/tags"
            )
            response.raise_for_status()
            models = response.json().get("models", [])

        return {
            "status": "connected",
            "models": [
                model.get("name") for model in models
            ]
        }
    except Exception:
        return {
            "status": "disconnected",
            "message": "Ollama is not reachable"
        }
