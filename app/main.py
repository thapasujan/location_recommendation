from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional
import os

from .recommender import load_data, recommend_places_knn, location_dict, places_lookup

# ---------- Config ----------
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
LOC_CSV = os.getenv("LOCATION_CSV", os.path.join(DATA_DIR, "location_dataset.csv"))
PLC_CSV = os.getenv("PLACES_CSV",   os.path.join(DATA_DIR, "places_dataset.csv"))

app = FastAPI(title="Location Recommender API", version="1.0.0")

# Open CORS for convenience; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class RecommendRequest(BaseModel):
    query: str = Field(..., description="Main location name e.g. 'Thamel'")
    k: int = Field(2, ge=1, le=50, description="How many nearest places to return")
    candidate_source: Literal["desc", "names"] = Field("desc", description="Use places from descriptions or from names themselves")


# ---------- Startup ----------
@app.on_event("startup")
def _startup():
    if not os.path.exists(LOC_CSV) or not os.path.exists(PLC_CSV):
        raise RuntimeError(f"CSV files not found. Expected at {LOC_CSV} and {PLC_CSV}")
    load_data(LOC_CSV, PLC_CSV)


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "locations_cached": len(location_dict),
        "places_cached": len(places_lookup),
    }

@app.post("/recommend")
def recommend(body: RecommendRequest):
    result = recommend_places_knn(body.query, k=body.k, candidate_source=body.candidate_source)
    if result.get("status") == "ok":
        return result
    # Map other statuses to HTTP errors where appropriate
    status = result.get("status")
    if status == "not_found":
        raise HTTPException(status_code=404, detail=result.get("message", "Not found"))
    if status == "no_candidates":
        raise HTTPException(status_code=422, detail=result.get("message", "No candidates"))
    raise HTTPException(status_code=400, detail=result)


@app.get("/")
def root():
    return {"message": "Location Recommendation API is running. Use POST /recommend to get results."}
