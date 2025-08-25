from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import re
from typing import Dict, Any, List

# Globals populated at startup
location_dict: Dict[str, Dict[str, Any]] = {}
places_lookup: Dict[str, Dict[str, float]] = {}

def load_data(location_csv_path: str, places_csv_path: str) -> None:
    """Load CSVs and build lookup dictionaries once at startup."""
    global location_dict, places_lookup
    location_df = pd.read_csv(location_csv_path)
    places_df = pd.read_csv(places_csv_path)

    # Build location lookup (name -> {Description, Latitude, Longitude})
    location_dict = (
        location_df
        .set_index(location_df["Name"].str.strip().str.lower())[["Description", "Latitude", "Longitude"]]
        .to_dict(orient="index")
    )

    # Build places lookup (place -> {Latitude, Longitude}), averaging duplicates
    places_lookup = (
        places_df
          .assign(key=places_df["Place"].str.strip().str.lower())
          .groupby("key", as_index=True)
          .agg(Latitude=( "Latitude", "mean"), Longitude=("Longitude", "mean"))
          .to_dict(orient="index")
    )

def parse_places_from_description(desc: str) -> List[str]:
    """Parse comma/semicolon-separated place names from a Description string."""
    if not isinstance(desc, str) or not desc.strip():
        return []
    parts = [p.strip() for p in re.split(r",|;", desc) if p.strip()]
    return parts

def get_location_info(query: str) -> Dict[str, Any]:
    """Resolve a user query into a main location with coordinates."""
    if not isinstance(query, str):
        return {"status": "error", "message": "Invalid input"}

    q = query.lower().strip()

    # Exact match
    if q in location_dict:
        info = location_dict[q]
        return {
            "status": "exact",
            "query": q,
            "selected_names": [q],
            "lat": float(info["Latitude"]),
            "lon": float(info["Longitude"]),
            "description": info["Description"],
        }

    # Partial word match (any word contained in the location name)
    q_words = q.split()
    matches = []
    for name, info in location_dict.items():
        if any(word in name for word in q_words):
            matches.append((name, info))

    if matches:
        avg_lat = float(np.mean([m[1]["Latitude"] for m in matches]))
        avg_lon = float(np.mean([m[1]["Longitude"] for m in matches]))
        merged_desc = "; ".join(m[1]["Description"] for m in matches)
        return {
            "status": "merged",
            "query": q,
            "selected_names": [m[0] for m in matches],  # lowercased keys already
            "lat": avg_lat,
            "lon": avg_lon,
            "description": merged_desc,
        }

    return {"status": "not_found", "query": q}

def get_candidate_places(selected_names, candidate_source="desc") -> pd.DataFrame:
    """Build candidate places from descriptions or names."""
    candidates = {}
    selected_names = [name.strip().lower() for name in selected_names]

    if candidate_source == "desc":
        for name in selected_names:
            if name in location_dict:
                desc_places = parse_places_from_description(location_dict[name]["Description"])
                for p in desc_places:
                    key = p.lower()
                    if key in places_lookup:
                        # Original-case "Place" string as seen in description
                        candidates[key] = {
                            "Place": p,
                            "Latitude": places_lookup[key]["Latitude"],
                            "Longitude": places_lookup[key]["Longitude"],
                        }
    elif candidate_source == "names":
        for name in selected_names:
            if name in places_lookup:
                candidates[name] = {
                    "Place": name,  # lowercase; could title-case if desired
                    "Latitude": places_lookup[name]["Latitude"],
                    "Longitude": places_lookup[name]["Longitude"],
                }
    else:
        raise ValueError("candidate_source must be 'desc' or 'names'")

    if not candidates:
        return pd.DataFrame(columns=["key", "Place", "Latitude", "Longitude"])

    df = pd.DataFrame.from_dict(candidates, orient="index").reset_index(names="key")
    return df[["key", "Place", "Latitude", "Longitude"]]

def _to_radians_df(df, lat_col="Latitude", lon_col="Longitude"):
    return np.radians(df[[lat_col, lon_col]].values)

def _to_radians_point(lat, lon):
    return np.radians([[lat, lon]])

def recommend_places_knn(user_query: str, k=2, candidate_source="desc") -> Dict[str, Any]:
    """Return JSON-ready dict with recommendations."""
    info = get_location_info(user_query)
    if info.get("status") == "not_found":
        return {"status": "not_found", "message": f"No location matched '{user_query}'."}
    if info.get("status") == "error":
        return info

    cand_df = get_candidate_places(info["selected_names"], candidate_source=candidate_source)
    if cand_df.empty:
        return {"status": "no_candidates", "message": "No candidate places found for the matched locations."}

    # Fit KNN over candidate places (haversine distance expects radians)
    X = _to_radians_df(cand_df, "Latitude", "Longitude")
    knn = NearestNeighbors(metric="haversine", algorithm="ball_tree")
    knn.fit(X)

    # Query point in radians
    q_rad = _to_radians_point(info["lat"], info["lon"])

    # Ask for min(k, number of candidates)
    k_eff = int(max(1, min(k, len(cand_df))))
    distances_rad, indices = knn.kneighbors(q_rad, n_neighbors=k_eff)

    # Convert to km
    distances_km = distances_rad[0] * 6371.0
    idxs = indices[0]

    # Assemble the results
    results = cand_df.iloc[idxs].copy().reset_index(drop=True)
    results["Distance_km"] = distances_km
    results.insert(0, "Main_Location_Query", info["query"])
    results.insert(1, "Resolved_Location_Type", info["status"])  # exact or merged
    results.insert(2, "Resolved_Location_Names", ", ".join(info["selected_names"])) 
    results.insert(3, "Main_Latitude", info["lat"]) 
    results.insert(4, "Main_Longitude", info["lon"]) 

    results = results[[
        "Main_Location_Query", "Resolved_Location_Type", "Resolved_Location_Names",
        "Main_Latitude", "Main_Longitude",
        "Place", "Latitude", "Longitude", "Distance_km"
    ]]

    return {"status": "ok", "data": results.to_dict(orient="records")}    
