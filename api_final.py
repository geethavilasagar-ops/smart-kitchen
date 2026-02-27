"""
Smart Kitchen Waste Monitor - Final API (Weather + Events + ML)
===============================================================
Run with: uvicorn api_final:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import tempfile, os

from model   import generate_dashboard_data, recommend_order_quantity
from weather import fetch_forecast, fetch_today, weather_to_model_features
from events  import fetch_events, get_events_by_date, get_demand_multiplier_for_date

app = FastAPI(
    title="Smart Kitchen Waste Monitor",
    description="ML waste prediction + Live Weather + Live Events",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_cache = {}
CITY = "HYDERABAD"  # ← Change to your restaurant's city


@app.on_event("startup")
async def startup():
    print(f"🚀 Starting KitchenZero for {CITY}...")
    _cache["data"]    = generate_dashboard_data()
    _cache["weather"] = fetch_forecast(CITY, days=7)
    _cache["today"]   = fetch_today(CITY)
    _cache["events"]  = fetch_events(CITY, days=7)
    _cache["events_by_date"] = get_events_by_date(CITY, days=7)
    print("✅ All systems ready!")


# ── Dashboard ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "version": "3.0", "city": CITY}


@app.get("/dashboard")
def get_dashboard():
    """Complete dashboard — ML data + live weather + live events."""
    return {
        **_cache.get("data", {}),
        "weather_forecast":  _cache.get("weather", []),
        "today_weather":     _cache.get("today", {}),
        "upcoming_events":   _cache.get("events", []),
        "events_by_date":    _cache.get("events_by_date", {}),
        "city":              CITY,
    }


# ── Weather ───────────────────────────────────────────────────────

@app.get("/weather")
def get_weather(days: int = Query(default=7, ge=1, le=7)):
    return {
        "city":     CITY,
        "forecast": _cache.get("weather", [])[:days],
        "today":    _cache.get("today", {}),
    }


# ── Events ────────────────────────────────────────────────────────

@app.get("/events")
def get_events(days: int = Query(default=7, ge=1, le=14)):
    """Upcoming local events that impact restaurant demand."""
    return {
        "city":           CITY,
        "events":         _cache.get("events", []),
        "events_by_date": _cache.get("events_by_date", {}),
    }


# ── Smart Predictions ─────────────────────────────────────────────

@app.get("/predict/today")
def predict_today():
    """
    Today's waste prediction combining:
    - ML model baseline
    - Live weather adjustment
    - Live events adjustment
    """
    today_weather = _cache.get("today", {})
    data          = _cache.get("data", {})
    today_str     = datetime.today().strftime("%Y-%m-%d")

    # Get factors
    weather       = weather_to_model_features(today_weather)
    event_info    = get_demand_multiplier_for_date(CITY, today_str)

    rain_factor   = 0.85 if weather["rainfall_mm"] > 5 else 1.0
    temp_factor   = 1 + (weather["temperature_c"] - 18) * 0.01
    event_factor  = event_info["multiplier"]

    predictions = []
    for item in data.get("item_stats", []):
        adjusted = round(
            item["predicted_waste_kg"] * rain_factor * temp_factor * event_factor, 2
        )
        rec = recommend_order_quantity(adjusted, item["avg_daily_sold"])
        predictions.append({
            "item":               item["item"],
            "baseline_waste_kg":  item["predicted_waste_kg"],
            "predicted_waste_kg": adjusted,
            "recommended_order":  rec["recommended_order_kg"],
            "potential_saving_kg": rec["potential_saving_kg"],
            "cost_saving":        rec["estimated_cost_saving"],
        })

    total_waste = round(sum(p["predicted_waste_kg"] for p in predictions), 1)

    return {
        "date":             today_str,
        "city":             CITY,
        "weather":          today_weather,
        "event_impact":     event_info,
        "total_waste_kg":   total_waste,
        "risk_level":       "HIGH" if total_waste > 70 else "MED" if total_waste > 50 else "LOW",
        "predictions":      predictions,
        "factors_applied":  {
            "rain_factor":   rain_factor,
            "temp_factor":   round(temp_factor, 3),
            "event_factor":  event_factor,
            "top_event":     event_info.get("top_event"),
        }
    }


@app.get("/predict/week")
def predict_week():
    """
    Full 7-day forecast combining weather + events for each day.
    The most powerful endpoint for weekly kitchen planning.
    """
    forecast      = _cache.get("weather", [])
    data          = _cache.get("data", {})
    events_by_day = _cache.get("events_by_date", {})
    results       = []

    for day_weather in forecast:
        date_str    = day_weather["date"]
        weather     = weather_to_model_features(day_weather)
        day_events  = events_by_day.get(date_str, [])

        rain_factor  = 0.85 if weather["rainfall_mm"] > 5 else 1.0
        temp_factor  = 1 + (weather["temperature_c"] - 18) * 0.01
        event_factor = max((e["demand_impact"] for e in day_events), default=1.0)
        combined     = rain_factor * temp_factor * event_factor

        total_waste = round(
            sum(i["predicted_waste_kg"] for i in data.get("item_stats", [])) * combined, 1
        )

        results.append({
            "date":               date_str,
            "condition":          day_weather.get("condition", ""),
            "temperature_c":      day_weather["temperature_c"],
            "rainfall_mm":        day_weather["rainfall_mm"],
            "events":             [e["title"] for e in day_events[:3]],
            "event_factor":       round(event_factor, 2),
            "predicted_waste_kg": total_waste,
            "risk":               "HIGH" if total_waste > 70 else "MED" if total_waste > 50 else "LOW",
        })

    return {"city": CITY, "week_forecast": results}


# ── Refresh ───────────────────────────────────────────────────────

@app.get("/refresh")
def refresh_all():
    """Refresh weather + events data from live APIs."""
    _cache["weather"]        = fetch_forecast(CITY, days=7)
    _cache["today"]          = fetch_today(CITY)
    _cache["events"]         = fetch_events(CITY, days=7)
    _cache["events_by_date"] = get_events_by_date(CITY, days=7)
    return {
        "status":       "refreshed",
        "weather_days": len(_cache["weather"]),
        "events_found": len(_cache["events"]),
    }


# ── CSV Upload ────────────────────────────────────────────────────

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files supported")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        _cache["data"] = generate_dashboard_data(use_csv=tmp_path)
        return {"status": "success", "message": "Model retrained with your data"}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
