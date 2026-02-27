"""
Smart Kitchen - PredictHQ Events API Integration
=================================================
Fetches local events and formats them for the ML model.
"""

import urllib.request
import urllib.parse
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("TrC2MVYhElJKn0qQF2eF3NkRRNW-CZBoKdyv5W04")
BASE_URL  = "https://api.predicthq.com/v1"

EVENT_IMPACT = {
    "concerts":        1.40,
    "sports":          1.35,
    "festivals":       1.50,
    "conferences":     1.20,
    "expos":           1.15,
    "public-holidays": 0.60,
    "school-holidays": 1.10,
    "community":       1.10,
    "observances":     1.00,
    "default":         1.10,
}

DEMAND_LABEL = {
    "concerts":        "HIGH",
    "sports":          "HIGH",
    "festivals":       "HIGH",
    "conferences":     "MED",
    "expos":           "MED",
    "public-holidays": "LOW",
    "school-holidays": "MED",
    "community":       "LOW",
    "default":         "MED",
}


def fetch_events(city: str = "London", days: int = 7) -> list[dict]:
    if not API_TOKEN:
        print("⚠️  PREDICTHQ_TOKEN not found in .env — using fallback events")
        return _fallback_events(days)

    today     = datetime.today()
    date_from = today.strftime("%Y-%m-%d")
    date_to   = (today + timedelta(days=days)).strftime("%Y-%m-%d")

    params = urllib.parse.urlencode({
        "q":          city,
        "active.gte": date_from,
        "active.lte": date_to,
        "limit":      50,
        "sort":       "rank",
        "rank.gte":   30,
    })

    req = urllib.request.Request(
        f"{BASE_URL}/events/?{params}",
        headers={
            "Authorization": f"Bearer {API_TOKEN}",
            "Accept":        "application/json",
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"⚠️  PredictHQ API error: {e} — using fallback events")
        return _fallback_events(days)

    results = []
    for evt in data.get("results", []):
        category = evt.get("category", "default")
        start    = evt.get("start", "")[:10]
        end      = evt.get("end",   "")[:10] or start
        results.append({
            "id":            evt.get("id", ""),
            "title":         evt.get("title", "Unknown Event"),
            "category":      category,
            "start_date":    start,
            "end_date":      end,
            "rank":          evt.get("rank", 0),
            "demand_impact": EVENT_IMPACT.get(category, EVENT_IMPACT["default"]),
            "demand_label":  DEMAND_LABEL.get(category, "MED"),
        })

    print(f"✅ Events fetched for {city} — {len(results)} events found")
    return results


def get_events_by_date(city: str = "London", days: int = 7) -> dict:
    events   = fetch_events(city, days)
    today    = datetime.today()
    by_date  = {(today + timedelta(days=i)).strftime("%Y-%m-%d"): [] for i in range(days)}

    for evt in events:
        current = datetime.strptime(evt["start_date"], "%Y-%m-%d")
        end_dt  = datetime.strptime(evt["end_date"],   "%Y-%m-%d")
        while current <= end_dt:
            d = current.strftime("%Y-%m-%d")
            if d in by_date:
                by_date[d].append(evt)
            current += timedelta(days=1)

    return by_date


def get_demand_multiplier_for_date(city: str, date_str: str) -> dict:
    by_date = get_events_by_date(city, days=14)
    events  = by_date.get(date_str, [])
    if not events:
        return {"multiplier": 1.0, "label": "NORMAL", "events": [], "top_event": None}
    top = max(events, key=lambda e: e["demand_impact"])
    return {
        "multiplier": top["demand_impact"],
        "label":      top["demand_label"],
        "events":     events,
        "top_event":  top["title"],
    }


def _fallback_events(days: int) -> list[dict]:
    return [{
        "id":            "fallback-1",
        "title":         "Sample Local Concert",
        "category":      "concerts",
        "start_date":    (datetime.today() + timedelta(days=2)).strftime("%Y-%m-%d"),
        "end_date":      (datetime.today() + timedelta(days=2)).strftime("%Y-%m-%d"),
        "rank":          75,
        "demand_impact": 1.40,
        "demand_label":  "HIGH",
    }]


if __name__ == "__main__":
    import sys
    city = sys.argv[1] if len(sys.argv) > 1 else "London"
    print(f"\n🎪 Fetching events for: {city}\n")
    for evt in fetch_events(city, days=7)[:10]:
        icon = {"concerts":"🎵","sports":"⚽","festivals":"🎉","public-holidays":"🏖","conferences":"💼"}.get(evt["category"],"📅")
        print(f"  {icon} {evt['start_date']}  [{evt['demand_label']}]  {evt['title'][:50]}")
