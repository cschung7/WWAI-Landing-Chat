"""
WWAI Centralized Scores API Router
Serves pre-computed signal scores (m/t/v/o/u/d) for KRX and USA tickers.
All data loaded from JSON files at startup (no DB required).
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/scores", tags=["WWAI Scores"])

# Global data store (loaded at startup)
_data = {"krx": {}, "usa": {}}


def load_scores_data():
    """Load signal score JSON files into memory."""
    global _data
    base = Path(__file__).parent

    # KRX scores (keyed by Korean stock name)
    krx_path = base / "signal_scores.json"
    if krx_path.exists():
        with open(krx_path, "r", encoding="utf-8") as f:
            _data["krx"] = json.load(f)
        print(f"KRX scores loaded: {len(_data['krx'])} tickers")
    else:
        print(f"WARNING: {krx_path} not found. KRX scores unavailable.")

    # USA scores (keyed by ticker symbol)
    usa_path = base / "us_signal_scores.json"
    if usa_path.exists():
        with open(usa_path, "r", encoding="utf-8") as f:
            _data["usa"] = json.load(f)
        print(f"USA scores loaded: {len(_data['usa'])} tickers")
    else:
        print(f"WARNING: {usa_path} not found. USA scores unavailable.")


# --- Endpoints ---

@router.get("/meta")
async def scores_meta():
    """Get metadata about loaded score datasets."""
    krx = _data["krx"]
    usa = _data["usa"]

    # Extract latest date from first entry with a date
    krx_date = next((v.get("d") for v in krx.values() if v.get("d")), None)
    usa_date = next((v.get("d") for v in usa.values() if v.get("d")), None)

    return {
        "krx_count": len(krx),
        "usa_count": len(usa),
        "krx_date": krx_date,
        "usa_date": usa_date,
        "total": len(krx) + len(usa),
    }


@router.get("/krx/{name:path}")
async def krx_score(name: str):
    """Get KRX signal scores by Korean stock name (e.g. 삼성전자)."""
    scores = _data["krx"].get(name)
    if not scores:
        raise HTTPException(status_code=404, detail=f"'{name}' not found in {len(_data['krx'])} KRX tickers.")
    return {"name": name, "market": "krx", **scores}


@router.get("/usa/{ticker}")
async def usa_score(ticker: str):
    """Get USA signal scores by ticker symbol (e.g. AAPL)."""
    t = ticker.upper()
    scores = _data["usa"].get(t)
    if not scores:
        raise HTTPException(status_code=404, detail=f"'{t}' not found in {len(_data['usa'])} USA tickers.")
    return {"name": t, "market": "usa", **scores}


@router.get("/search")
async def search_scores(
    q: str = Query(..., min_length=1),
    market: Optional[str] = Query("all", regex="^(krx|usa|all)$"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search scores by partial name/ticker match."""
    results = []
    q_lower = q.lower()

    if market in ("krx", "all"):
        for name, scores in _data["krx"].items():
            if q_lower in name.lower():
                results.append({"name": name, "market": "krx", **scores})
                if len(results) >= limit:
                    break

    if market in ("usa", "all") and len(results) < limit:
        q_upper = q.upper()
        for ticker, scores in _data["usa"].items():
            if q_upper in ticker:
                results.append({"name": ticker, "market": "usa", **scores})
                if len(results) >= limit:
                    break

    return {"query": q, "market": market, "count": len(results), "results": results}


@router.get("/bulk")
async def bulk_scores(
    market: str = Query(..., regex="^(krx|usa)$"),
    names: str = Query(..., description="Comma-separated names/tickers (max 50)"),
):
    """Batch lookup scores for multiple tickers."""
    name_list = [n.strip() for n in names.split(",") if n.strip()]
    if len(name_list) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 tickers per request.")

    source = _data[market]
    results = {}
    for name in name_list:
        key = name if market == "krx" else name.upper()
        if key in source:
            results[key] = source[key]
        else:
            results[key] = None

    found = sum(1 for v in results.values() if v is not None)
    return {"market": market, "requested": len(name_list), "found": found, "results": results}
