"""
FastAPI app entrypoint for the ISL Translation System backend.

Run locally with (note: run from INSIDE backend/, not the project root,
since the internal imports below are relative to `app` being a top-level
package):
    cd backend
    uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.websocket.stream import router as websocket_router

app = FastAPI(title="ISL Translation System")

# Permissive CORS for local development -- the Next.js frontend runs on a
# different port (e.g. localhost:3000) than the backend (localhost:8000),
# so the browser needs explicit permission to connect across origins.
# TODO before any real deployment: restrict allow_origins to the actual
# frontend domain instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websocket_router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
