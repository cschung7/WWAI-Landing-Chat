"""
ETF Intelligence API Router
Serves pre-computed ETF classification, holdings, and theme data.
All data loaded from etf_intelligence.json at startup (no DB required).
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/etf", tags=["ETF Intelligence"])

# Global data store (loaded at startup)
_data = {}


def load_etf_data():
    """Load etf_intelligence.json into memory."""
    global _data
    json_path = Path(__file__).parent / "etf_intelligence.json"
    if not json_path.exists():
        print(f"WARNING: {json_path} not found. ETF routes will return empty data.")
        _data = {
            "meta": {},
            "theme_distribution": {},
            "theme_etf_lists": {},
            "etf_lookup": {},
            "reverse_index": {},
            "active_etfs": [],
        }
        return

    with open(json_path, "r", encoding="utf-8") as f:
        _data = json.load(f)

    meta = _data.get("meta", {})
    print(f"ETF Intelligence loaded: {meta.get('etf_count', 0)} ETFs, "
          f"{meta.get('theme_count', 0)} themes, "
          f"{meta.get('stocks_in_reverse', 0)} stocks in reverse index")


# --- Response Models ---

class ETFInfo(BaseModel):
    ticker: str
    fund_name: str
    theme: str
    confidence: str
    category: str
    aum: str
    expense_ratio: str
    holdings_count: int
    similarity: float
    dna_themes: list
    top_holdings: list


class ThemeInfo(BaseModel):
    theme: str
    count: int
    etfs: list


# --- Endpoints ---

@router.get("/meta")
async def etf_meta():
    """Get metadata about the ETF Intelligence dataset."""
    return _data.get("meta", {})


@router.get("/themes")
async def etf_themes():
    """Get theme distribution (15 themes + ETF counts)."""
    dist = _data.get("theme_distribution", {})
    # Sort by count descending
    sorted_themes = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    return {
        "theme_count": len(sorted_themes),
        "themes": [{"theme": t, "count": c} for t, c in sorted_themes]
    }


@router.get("/themes/{theme}")
async def etf_theme_list(theme: str):
    """Get ETFs belonging to a specific theme."""
    lists = _data.get("theme_etf_lists", {})
    dist = _data.get("theme_distribution", {})

    # Try exact match first, then case-insensitive
    matched_theme = None
    for t in lists:
        if t == theme or t.lower() == theme.lower():
            matched_theme = t
            break

    if not matched_theme:
        # Try partial match
        theme_lower = theme.lower().replace("-", " ").replace("_", " ")
        for t in lists:
            if theme_lower in t.lower():
                matched_theme = t
                break

    if not matched_theme:
        raise HTTPException(status_code=404, detail=f"Theme '{theme}' not found. Available: {list(lists.keys())}")

    return {
        "theme": matched_theme,
        "total_count": dist.get(matched_theme, len(lists[matched_theme])),
        "etfs": lists[matched_theme]
    }


@router.get("/lookup/{ticker}")
async def etf_lookup(ticker: str):
    """Single ETF lookup - theme, holdings, metadata."""
    lookup = _data.get("etf_lookup", {})
    t = ticker.upper()
    if t not in lookup:
        raise HTTPException(status_code=404, detail=f"Ticker '{t}' not found in {len(lookup)} ETFs.")
    return lookup[t]


@router.post("/classify")
async def etf_classify(tickers: list[str]):
    """Batch classify up to 10 tickers."""
    if len(tickers) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 tickers per request.")

    lookup = _data.get("etf_lookup", {})
    results = []
    for ticker in tickers:
        t = ticker.strip().upper()
        if t in lookup:
            results.append(lookup[t])
        else:
            results.append({"ticker": t, "error": "Not found"})

    return {"count": len(results), "results": results}


@router.get("/holdings/{ticker}")
async def etf_holdings(ticker: str):
    """Get ETF holdings list."""
    lookup = _data.get("etf_lookup", {})
    t = ticker.upper()
    if t not in lookup:
        raise HTTPException(status_code=404, detail=f"Ticker '{t}' not found.")

    etf = lookup[t]
    holdings = etf.get("top_holdings", [])

    # Calculate concentration
    top5_weight = sum(h["weight"] for h in holdings[:5]) if len(holdings) >= 5 else sum(h["weight"] for h in holdings)
    top10_weight = sum(h["weight"] for h in holdings[:10]) if len(holdings) >= 10 else sum(h["weight"] for h in holdings)

    return {
        "ticker": t,
        "fund_name": etf.get("fund_name", ""),
        "holdings_count": etf.get("holdings_count", 0),
        "top_5_weight": round(top5_weight, 2),
        "top_10_weight": round(top10_weight, 2),
        "holdings": holdings
    }


@router.get("/reverse/{stock}")
async def reverse_lookup(stock: str):
    """Reverse lookup: which ETFs hold this stock."""
    reverse = _data.get("reverse_index", {})
    s = stock.upper()
    if s not in reverse:
        raise HTTPException(status_code=404, detail=f"Stock '{s}' not found in any ETF holdings.")

    etfs = reverse[s]
    return {
        "stock": s,
        "etf_count": len(etfs),
        "etfs": etfs[:30]  # Cap at 30
    }


@router.get("/active")
async def active_etfs():
    """Get active ETF 3-axis classification grid."""
    active = _data.get("active_etfs", [])
    return {
        "count": len(active),
        "etfs": active
    }


@router.get("/frontier")
async def frontier_themes():
    """Get frontier themes and novel clusters info."""
    dist = _data.get("theme_distribution", {})
    core_themes = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    frontier_themes = ["AI & Autonomous Systems", "Cybersecurity & Privacy", "Infrastructure & Industry"]

    return {
        "core_theme_count": len(core_themes),
        "core_themes": [{"theme": t, "count": c} for t, c in core_themes],
        "frontier_themes": frontier_themes,
        "description": "Frontier themes represent emerging thematic areas beyond the 15 core classifications. "
                       "These include AI & Autonomous Systems (beyond broad Tech & AI), "
                       "Cybersecurity & Privacy, and Infrastructure & Industry specializations."
    }


@router.get("/search")
async def etf_search(q: str, limit: int = 20):
    """Search ETFs by ticker or name."""
    lookup = _data.get("etf_lookup", {})
    q_upper = q.upper()
    q_lower = q.lower()
    results = []

    for ticker, data in lookup.items():
        if q_upper == ticker:
            results.insert(0, data)  # Exact ticker match first
        elif q_upper in ticker or q_lower in data.get("fund_name", "").lower():
            results.append(data)

        if len(results) >= limit:
            break

    return {"query": q, "count": len(results), "results": results}
