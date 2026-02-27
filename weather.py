"""
Smart Kitchen - Weather API Integration (WeatherAPI.com)
=========================================================
Fetches today + 7-day forecast and formats it for the ML model.
"""

import urllib.request
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv("02c733f444ac4b5084470752262702")
BASE_URL = "https://api.weatherapi.com/v1"


def fetch_forecast(city: str = "London", days: int = 7) -> list[dict]:
    if not API_KEY:
        print("⚠️  WEATHER_API_KEY not found in .env — using fallback data")
        return _fallback_forecast(days)

    url = f"{BASE_URL}/forecast.json?key={API_KEY}&q={city}&days={days}&aqi=no&alerts=no"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"⚠️  Weather API error: {e} — using fallback data")
        return _fallback_forecast(days)

    results = []
    for day in data["forecast"]["forecastday"]:
        d = day["day"]
        results.append({
            "date":           day["date"],
            "temperature_c":  round(d["avgtemp_c"], 1),
            "max_temp_c":     round(d["maxtemp_c"], 1),
            "min_temp_c":     round(d["mintemp_c"], 1),
            "rainfall_mm":    round(d["totalprecip_mm"], 1),
            "humidity_pct":   round(d["avghumidity"], 1),
            "condition":      d["condition"]["text"],
            "condition_icon": d["condition"]["icon"],
            "uv_index":       d.get("uv", 0),
            "will_it_rain":   int(day["hour"][12].get("will_it_rain", 0)),
        })

    print(f"✅ Weather fetched for {city} — {len(results)} days")
    return results


def fetch_today(city: str = "London") -> dict:
    forecast = fetch_forecast(city, days=1)
    return forecast[0] if forecast else _fallback_forecast(1)[0]


def weather_to_model_features(weather: dict) -> dict:
    return {
        "temperature_c": weather["temperature_c"],
        "rainfall_mm":   weather["rainfall_mm"],
    }


def _fallback_forecast(days: int) -> list[dict]:
    today = datetime.today()
    return [
        {
            "date":           (today + timedelta(days=i)).strftime("%Y-%m-%d"),
            "temperature_c":  16.0,
            "max_temp_c":     19.0,
            "min_temp_c":     13.0,
            "rainfall_mm":    0.0,
            "humidity_pct":   65.0,
            "condition":      "Partly cloudy",
            "condition_icon": "//cdn.weatherapi.com/weather/64x64/day/116.png",
            "uv_index":       3,
            "will_it_rain":   0,
        }
        for i in range(days)
    ]


if __name__ == "__main__":
    import sys
    city = sys.argv[1] if len(sys.argv) > 1 else "London"
    print(f"\n🌤  Fetching weather for: {city}\n")
    for day in fetch_forecast(city, days=7):
        icon = "🌧" if day["rainfall_mm"] > 2 else "☀️"
        print(f"  {day['date']}  {icon}  {day['temperature_c']}°C  "
              f"Rain: {day['rainfall_mm']}mm  {day['condition']}")
