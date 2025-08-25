# Location Recommender API (FastAPI)

This API wraps your haversine-KNN recommendation logic and returns JSON.

## 1) Project layout

```
reco_api/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app (POST /recommend)
│   └── recommender.py     # Core logic: load_data(), recommend_places_knn(), etc.
├── data/
│   ├── location_dataset.csv
│   └── places_dataset.csv
└── requirements.txt
```

> Put your `location_dataset.csv` and `places_dataset.csv` inside the `data/` folder.

## 2) Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Run

```bash
uvicorn app.main:app --reload
```

Open docs at: http://127.0.0.1:8000/docs

## 4) Request (POST)

`POST /recommend`

Body JSON:
```json
{
  "query": "Thamel",
  "k": 3,
  "candidate_source": "desc"
}
```

Example cURL:
```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{ "query": "Thamel", "k": 2, "candidate_source": "desc" }'
```

## 5) Response (200)

```json
{
  "status": "ok",
  "data": [
    {
      "Main_Location_Query": "thamel",
      "Resolved_Location_Type": "exact",
      "Resolved_Location_Names": "thamel",
      "Main_Latitude": 27.717,
      "Main_Longitude": 85.312,
      "Place": "Narayanhiti Palace",
      "Latitude": 27.7173,
      "Longitude": 85.3152,
      "Distance_km": 0.45
    }
  ]
}
```

Errors:
- 404: location not found
- 422: no candidates for matched locations
- 400: other errors
