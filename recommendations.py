"""
FORESIGHT — Recommendation Engine
Turns the Step 6 risk classification into concrete actions:
Reorder Immediately / Increase Stock / Reduce Procurement / Inventory Healthy,
each with a calculated recommended quantity.
"""

import pandas as pd
from pathlib import Path

RISK_PATH = Path("data/processed/foresight_inventory_risk.csv")
CLEAN_PATH = Path("data/processed/foresight_clean.csv")
OUT_PATH = Path("data/processed/foresight_recommendations.csv")

# --- Assumptions (stated explicitly for the report) ---
# Reorder Point (ROP) already covers expected lead-time demand + safety stock (Step 6).
# These multipliers define bands around ROP for finer-grained recommendations:
INCREASE_STOCK_MULTIPLIER = 1.5   # stock within 1.0x-1.5x ROP -> early warning, top up
REORDER_TARGET_MULTIPLIER = 2.0   # when reordering, order up to 2x ROP
OVERSTOCK_MULTIPLIER = 3.0        # stock above 3x ROP -> reduce procurement (matches Step 6)


def load_data():
    risk = pd.read_csv(RISK_PATH)
    clean = pd.read_csv(CLEAN_PATH, parse_dates=["Date"])
    on_order = clean.sort_values("Date").groupby("SKU").tail(1)[["SKU", "On_Order"]]
    return risk.merge(on_order, on="SKU", how="left")


def recommend(row):
    if row["Risk_Status"] == "Stockout Risk":
        return "Reorder Immediately"
    if row["Risk_Status"] == "Overstock":
        return "Reduce Procurement"
    # Within the "Healthy" band, flag SKUs trending back toward risk
    if row["Current_Stock"] <= row["Reorder_Point_Calc"] * INCREASE_STOCK_MULTIPLIER:
        return "Increase Stock"
    return "Inventory Healthy"


def recommended_quantity(row):
    available = row["Current_Stock"] + row["On_Order"]

    if row["Recommendation"] == "Reorder Immediately":
        target = row["Reorder_Point_Calc"] * REORDER_TARGET_MULTIPLIER
        return max(0, round(target - available))

    if row["Recommendation"] == "Increase Stock":
        target = row["Reorder_Point_Calc"] * INCREASE_STOCK_MULTIPLIER
        return max(0, round(target - available))

    if row["Recommendation"] == "Reduce Procurement":
        excess = row["Current_Stock"] - row["Reorder_Point_Calc"] * OVERSTOCK_MULTIPLIER
        return -round(max(0, excess))  # negative = suggested cut to next procurement

    return 0  # Inventory Healthy -- no action


def main():
    df = load_data()
    df["Recommendation"] = df.apply(recommend, axis=1)
    df["Recommended_Qty"] = df.apply(recommended_quantity, axis=1)

    df.to_csv(OUT_PATH, index=False)

    print(f"Saved -> {OUT_PATH}\n")
    print("Recommendation breakdown:")
    print(df["Recommendation"].value_counts().to_string())

    print("\n--- Reorder Immediately (sorted by urgency) ---")
    urgent = df[df["Recommendation"] == "Reorder Immediately"].sort_values("Days_Of_Stock")
    print(urgent[["SKU", "Product_Name", "Current_Stock", "On_Order", "Recommended_Qty", "Days_Of_Stock"]].to_string(index=False))

    print("\n--- Increase Stock (early warning) ---")
    warn = df[df["Recommendation"] == "Increase Stock"]
    print(warn[["SKU", "Product_Name", "Current_Stock", "Recommended_Qty", "Days_Of_Stock"]].to_string(index=False))

    print("\n--- Reduce Procurement (top 5 by excess) ---")
    excess = df[df["Recommendation"] == "Reduce Procurement"].sort_values("Recommended_Qty").head(5)
    print(excess[["SKU", "Product_Name", "Current_Stock", "Recommended_Qty", "Days_Of_Stock"]].to_string(index=False))


if __name__ == "__main__":
    main()