"""
FORESIGHT — Inventory Risk Detection
Combines the latest stock position (foresight_clean.csv) with the demand
forecast (foresight_forecasts.csv from Step 5) to calculate safety stock
and reorder point per SKU, then classifies each SKU as Stockout Risk,
Overstock, or Healthy.
"""

import numpy as np
import pandas as pd
from pathlib import Path

CLEAN_PATH = Path("data/processed/foresight_clean.csv")
FORECAST_PATH = Path("data/processed/foresight_forecasts.csv")
OUT_PATH = Path("data/processed/foresight_inventory_risk.csv")

# --- Assumptions (documented explicitly for the project report) ---
Z_SCORE = 1.65          # ~95% service level for safety stock
OVERSTOCK_MULTIPLIER = 3.0  # stock > 3x reorder point is flagged as overstock


def load_data():
    clean = pd.read_csv(CLEAN_PATH, parse_dates=["Date"])
    forecasts = pd.read_csv(FORECAST_PATH, parse_dates=["Date"])
    return clean, forecasts


def get_latest_stock_position(clean):
    # Most recent known stock level and lead time per SKU -- this represents
    # "today's" inventory position for the risk check.
    latest = clean.sort_values("Date").groupby("SKU").tail(1)
    cols = ["SKU", "Product_Name", "Category", "Date", "Current_Stock",
            "Lead_Time_Days", "Reorder_Point", "Safety_Stock"]
    latest = latest[cols].rename(columns={
        "Date": "As_Of_Date",
        "Reorder_Point": "Reorder_Point_Given",
        "Safety_Stock": "Safety_Stock_Given",
    })
    return latest


def get_demand_stats(clean, forecasts):
    # Historical daily demand variability (std dev), from actual sales history --
    # used to size safety stock against real demand volatility per SKU.
    hist_std = clean.groupby("SKU")["Units_Sold"].std().rename("Historical_Demand_Std")

    # Expected near-term daily demand, from the Step 5 model's forecast on the
    # held-out test window -- this is our best estimate of "what's coming".
    fcst_mean = forecasts.groupby("SKU")["Forecast_Demand"].mean().rename("Forecast_Avg_Daily_Demand")

    return pd.concat([hist_std, fcst_mean], axis=1).reset_index()


def calculate_risk(df):
    df["Safety_Stock_Calc"] = Z_SCORE * df["Historical_Demand_Std"] * np.sqrt(df["Lead_Time_Days"])
    df["Reorder_Point_Calc"] = (
        df["Forecast_Avg_Daily_Demand"] * df["Lead_Time_Days"] + df["Safety_Stock_Calc"]
    )

    # Days of stock remaining at current forecasted demand rate
    df["Days_Of_Stock"] = np.where(
        df["Forecast_Avg_Daily_Demand"] > 0,
        df["Current_Stock"] / df["Forecast_Avg_Daily_Demand"],
        np.inf,
    )

    def classify(row):
        if row["Current_Stock"] <= row["Reorder_Point_Calc"]:
            return "Stockout Risk"
        elif row["Current_Stock"] > row["Reorder_Point_Calc"] * OVERSTOCK_MULTIPLIER:
            return "Overstock"
        return "Healthy"

    df["Risk_Status"] = df.apply(classify, axis=1)
    return df


def main():
    clean, forecasts = load_data()
    latest_stock = get_latest_stock_position(clean)
    demand_stats = get_demand_stats(clean, forecasts)

    df = latest_stock.merge(demand_stats, on="SKU", how="left")
    df = calculate_risk(df)

    df.to_csv(OUT_PATH, index=False)

    print(f"Saved -> {OUT_PATH}")
    print(f"\nRisk breakdown ({len(df)} SKUs):")
    print(df["Risk_Status"].value_counts().to_string())

    # Sanity check: how close is our calculated reorder point to the value
    # that was already provided in inventory_snapshots.csv?
    diff = (df["Reorder_Point_Calc"] - df["Reorder_Point_Given"]).abs().mean()
    print(f"\nValidation: avg absolute difference vs given Reorder_Point = {diff:.2f} units")

    print("\n--- SKUs at Stockout Risk ---")
    cols_show = ["SKU", "Product_Name", "Current_Stock", "Reorder_Point_Calc", "Days_Of_Stock", "Lead_Time_Days"]
    print(df[df["Risk_Status"] == "Stockout Risk"][cols_show].sort_values("Days_Of_Stock").to_string(index=False))

    print("\n--- SKUs Overstocked ---")
    print(df[df["Risk_Status"] == "Overstock"][cols_show].sort_values("Days_Of_Stock", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()