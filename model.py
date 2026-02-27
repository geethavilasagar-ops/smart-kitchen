"""
Smart Kitchen Waste Monitor - ML Prediction Engine
====================================================
Uses Random Forest regression to predict daily food waste
by correlating sales data, weather, and local events.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import json
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  DATA GENERATION (fallback when no CSV given)
# ─────────────────────────────────────────────

PERISHABLE_ITEMS = [
    "Lettuce",
    "Tomatoes",
    "Salmon",
    "Chicken Breast",
    "Fresh Herbs",
    "Strawberries",
    "Avocado",
    "Fresh Pasta",
    "Milk",
    "Greek Yogurt",
]

EVENT_TYPES = ["none", "concert", "sports_game", "festival", "holiday", "conference"]


def generate_synthetic_data(n_days: int = 365) -> pd.DataFrame:
    """
    Generate realistic synthetic restaurant data.
    Replace this with your real CSV loader.
    """
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    records = []

    for date in dates:
        # Weather features
        month = date.month
        base_temp = 15 + 10 * np.sin((month - 3) * np.pi / 6)
        temperature = base_temp + np.random.normal(0, 4)
        rainfall = max(0, np.random.exponential(2) if np.random.random() < 0.3 else 0)
        is_weekend = date.dayofweek >= 5

        # Local event
        event = np.random.choice(
            EVENT_TYPES,
            p=[0.60, 0.10, 0.10, 0.07, 0.08, 0.05],
        )
        event_boost = {
            "none": 1.0,
            "concert": 1.4,
            "sports_game": 1.35,
            "festival": 1.5,
            "holiday": 0.7,
            "conference": 1.2,
        }[event]

        for item in PERISHABLE_ITEMS:
            # Base demand varies by item category
            item_base = {
                "Lettuce": 25, "Tomatoes": 30, "Salmon": 15,
                "Chicken Breast": 40, "Fresh Herbs": 10,
                "Strawberries": 20, "Avocado": 22, "Fresh Pasta": 18,
                "Milk": 35, "Greek Yogurt": 28,
            }[item]

            # Demand modifiers
            weekend_mult = 1.3 if is_weekend else 1.0
            rain_mult = 0.85 if rainfall > 5 else 1.0
            temp_mult = 1 + (temperature - 18) * 0.01  # warmer → more salads/drinks

            demand = item_base * weekend_mult * rain_mult * temp_mult * event_boost
            demand += np.random.normal(0, item_base * 0.1)
            demand = max(5, demand)

            # Ordered quantity (slightly over-order to be safe)
            ordered = demand * np.random.uniform(1.05, 1.25)

            # Waste = ordered - sold (clipped at 0)
            sold = min(ordered, demand * np.random.uniform(0.85, 1.0))
            waste = max(0, ordered - sold)

            records.append({
                "date": date,
                "item": item,
                "temperature_c": round(temperature, 1),
                "rainfall_mm": round(rainfall, 1),
                "is_weekend": int(is_weekend),
                "local_event": event,
                "quantity_ordered": round(ordered, 1),
                "quantity_sold": round(sold, 1),
                "waste_kg": round(waste, 2),
                "day_of_week": date.dayofweek,
                "month": month,
                "week_of_year": date.isocalendar()[1],
            })

    return pd.DataFrame(records)


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load your real restaurant CSV.
    Auto-detects common column naming patterns.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # Try to map your columns to our expected names
    col_map = {}
    for col in df.columns:
        if "date" in col:
            col_map[col] = "date"
        elif "item" in col or "product" in col or "menu" in col:
            col_map[col] = "item"
        elif "waste" in col or "leftover" in col:
            col_map[col] = "waste_kg"
        elif "sold" in col or "sales" in col or "qty_sold" in col:
            col_map[col] = "quantity_sold"
        elif "order" in col or "purchased" in col:
            col_map[col] = "quantity_ordered"
        elif "temp" in col:
            col_map[col] = "temperature_c"
        elif "rain" in col or "precip" in col:
            col_map[col] = "rainfall_mm"
        elif "event" in col:
            col_map[col] = "local_event"

    df = df.rename(columns=col_map)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Fill missing columns with defaults
    if "waste_kg" not in df.columns and "quantity_ordered" in df.columns and "quantity_sold" in df.columns:
        df["waste_kg"] = (df["quantity_ordered"] - df["quantity_sold"]).clip(lower=0)

    if "temperature_c" not in df.columns:
        df["temperature_c"] = 18.0
    if "rainfall_mm" not in df.columns:
        df["rainfall_mm"] = 0.0
    if "local_event" not in df.columns:
        df["local_event"] = "none"

    return df


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encode categorical
    le = LabelEncoder()
    df["event_encoded"] = le.fit_transform(df["local_event"].astype(str))

    if "item" in df.columns:
        df["item_encoded"] = le.fit_transform(df["item"].astype(str))

    # Rolling stats per item (lag features)
    if "item" in df.columns and "date" in df.columns:
        df = df.sort_values(["item", "date"])
        df["waste_7d_avg"] = df.groupby("item")["waste_kg"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )
        df["waste_14d_avg"] = df.groupby("item")["waste_kg"].transform(
            lambda x: x.shift(1).rolling(14, min_periods=1).mean()
        )
        df["sold_7d_avg"] = df.groupby("item")["quantity_sold"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )
    else:
        df["waste_7d_avg"] = df["waste_kg"].shift(1).rolling(7, min_periods=1).mean()
        df["waste_14d_avg"] = df["waste_kg"].shift(1).rolling(14, min_periods=1).mean()
        df["sold_7d_avg"] = df.get("quantity_sold", pd.Series(0, index=df.index)).shift(1).rolling(7, min_periods=1).mean()

    df = df.fillna(df.mean(numeric_only=True))
    return df


# ─────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "temperature_c", "rainfall_mm", "is_weekend",
    "event_encoded", "day_of_week", "month", "week_of_year",
    "waste_7d_avg", "waste_14d_avg", "sold_7d_avg",
    "item_encoded",
]


def train_model(df: pd.DataFrame):
    df = engineer_features(df)
    available_features = [c for c in FEATURE_COLS if c in df.columns]

    X = df[available_features]
    y = df["waste_kg"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model trained | MAE: {mae:.3f} kg | R²: {r2:.3f}")
    return model, available_features, {"mae": round(mae, 3), "r2": round(r2, 3)}


# ─────────────────────────────────────────────
#  ORDER OPTIMIZATION
# ─────────────────────────────────────────────

def recommend_order_quantity(
    predicted_waste_kg: float,
    historical_sold_avg: float,
    safety_buffer: float = 0.10,
) -> dict:
    """
    Recommends optimal order quantity to minimize waste
    while maintaining service level.
    """
    # Expected demand = historical avg adjusted for predicted waste
    expected_demand = historical_sold_avg * (1 + safety_buffer)
    waste_reduction_target = predicted_waste_kg * 0.8  # aim to cut waste by 80%

    recommended = max(expected_demand, expected_demand - waste_reduction_target * 0.5)
    potential_saving_kg = predicted_waste_kg - max(0, recommended - expected_demand)
    cost_per_kg = 8.50  # avg perishable cost £/kg

    return {
        "recommended_order_kg": round(recommended, 1),
        "expected_demand_kg": round(expected_demand, 1),
        "predicted_waste_kg": round(predicted_waste_kg, 2),
        "potential_saving_kg": round(max(0, potential_saving_kg), 2),
        "estimated_cost_saving": round(max(0, potential_saving_kg) * cost_per_kg, 2),
    }


# ─────────────────────────────────────────────
#  GENERATE PREDICTIONS FOR DASHBOARD
# ─────────────────────────────────────────────

def generate_dashboard_data(use_csv: str = None) -> dict:
    """Main function — returns all data needed for the React dashboard."""

    # Load data
    if use_csv:
        try:
            df = load_csv_data(use_csv)
            print(f"📂 Loaded CSV: {len(df)} rows")
        except Exception as e:
            print(f"⚠️  CSV load failed ({e}), using synthetic data")
            df = generate_synthetic_data()
    else:
        print("📊 Using synthetic data (drop your CSV to use real data)")
        df = generate_synthetic_data()

    # Train model
    model, features, metrics = train_model(df)

    # Per-item stats
    item_stats = []
    items = df["item"].unique() if "item" in df.columns else ["All Items"]

    for item in items:
        sub = df[df["item"] == item] if "item" in df.columns else df
        avg_waste = sub["waste_kg"].mean()
        avg_sold = sub["quantity_sold"].mean() if "quantity_sold" in sub.columns else 50
        total_waste = sub["waste_kg"].sum()

        rec = recommend_order_quantity(avg_waste, avg_sold)
        item_stats.append({
            "item": item,
            "avg_daily_waste_kg": round(avg_waste, 2),
            "avg_daily_sold": round(avg_sold, 1),
            "total_waste_kg": round(total_waste, 1),
            **rec,
        })

    # Sort by worst waste offenders
    item_stats.sort(key=lambda x: x["total_waste_kg"], reverse=True)

    # Weekly waste trend (last 12 weeks)
    df_trend = df.copy()
    df_trend["week"] = df_trend["date"].dt.to_period("W").astype(str)
    weekly = df_trend.groupby("week")["waste_kg"].sum().reset_index()
    weekly_trend = weekly.tail(12).to_dict("records")

    # Monthly summary
    df_trend["month_label"] = df_trend["date"].dt.strftime("%b %Y")
    monthly = df_trend.groupby("month_label").agg(
        total_waste=("waste_kg", "sum"),
        total_sold=("quantity_sold", "sum") if "quantity_sold" in df.columns else ("waste_kg", "count"),
    ).reset_index()
    monthly_trend = monthly.tail(6).to_dict("records")

    # Impact summary
    total_waste_kg = df["waste_kg"].sum()
    total_potential_saving = sum(s["potential_saving_kg"] * len(df["date"].unique()) / 365 for s in item_stats)
    total_cost_saving = total_potential_saving * 8.5

    summary = {
        "total_waste_kg": round(total_waste_kg, 1),
        "avg_daily_waste_kg": round(df["waste_kg"].mean(), 2),
        "potential_annual_saving_kg": round(total_potential_saving * 365, 0),
        "potential_annual_cost_saving": round(total_cost_saving * 365, 0),
        "model_r2": metrics["r2"],
        "model_mae_kg": metrics["mae"],
        "co2_saved_kg": round(total_potential_saving * 365 * 2.5, 0),  # avg 2.5kg CO2/kg food
    }

    return {
        "summary": summary,
        "item_stats": item_stats,
        "weekly_trend": weekly_trend,
        "monthly_trend": monthly_trend,
        "model_metrics": metrics,
    }


if __name__ == "__main__":
    data = generate_dashboard_data()
    print("\n📦 SUMMARY:")
    print(json.dumps(data["summary"], indent=2))
    print("\n🥗 TOP WASTE OFFENDERS:")
    for item in data["item_stats"][:3]:
        print(f"  {item['item']}: {item['avg_daily_waste_kg']} kg/day avg → "
              f"Save {item['potential_saving_kg']} kg/day (£{item['estimated_cost_saving']}/day)")
