"""
FORESIGHT — Feature Engineering
Reads data/processed/foresight_clean.csv, adds time/lag/rolling/demand
features, and saves data/processed/foresight_features.csv (model-ready).
"""

import pandas as pd
from pathlib import Path

IN_PATH = Path("data/processed/foresight_clean.csv")
OUT_PATH = Path("data/processed/foresight_features.csv")


def load_data():
    df = pd.read_csv(IN_PATH, parse_dates=["Date"])
    df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)
    return df


def add_calendar_features(df):
    # year/month/quarter/week/day_of_week/is_weekend/season already came from
    # calendar.csv in Step 1-2. Day-of-month was missing, so add it here.
    df["Day"] = df["Date"].dt.day
    return df


def add_lag_features(df, lags=(1, 7, 14, 30)):
    for lag in lags:
        df[f"Units_Sold_lag_{lag}"] = df.groupby("SKU")["Units_Sold"].shift(lag)
    return df


def add_rolling_features(df, windows=(7, 30)):
    for w in windows:
        # shift(1) BEFORE rolling so the window only uses past days, never today's
        # own value -- otherwise the model would "see the future" during training.
        df[f"Units_Sold_roll_mean_{w}"] = df.groupby("SKU")["Units_Sold"].transform(
            lambda s: s.shift(1).rolling(w).mean()
        )
    return df


def add_previous_demand(df):
    # Explicit "previous demand" feature requested by the project brief
    # (same as lag_1, kept as its own named column for clarity in the report).
    df["Prev_Day_Demand"] = df.groupby("SKU")["Units_Sold"].shift(1)
    return df


def drop_warmup_rows(df):
    # The first 30 days of each SKU can't have a full lag/rolling history yet,
    # so those rows have NaNs in the new feature columns. Drop them rather than
    # fabricate values -- keeps the training data honest.
    feature_cols = [c for c in df.columns if "lag_" in c or "roll_mean_" in c or c == "Prev_Day_Demand"]
    before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"Dropped {before - len(df)} warm-up rows (first 30 days per SKU, no lag history yet)")
    return df


def main():
    df = load_data()
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_previous_demand(df)
    df = drop_warmup_rows(df)

    df.to_csv(OUT_PATH, index=False)

    new_cols = [c for c in df.columns if "lag_" in c or "roll_mean_" in c or c in ("Day", "Prev_Day_Demand")]
    print(f"\nSaved -> {OUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"New feature columns: {new_cols}")
    print("\nPreview:")
    preview_cols = ["Date", "SKU", "Units_Sold", "Units_Sold_lag_1", "Units_Sold_lag_7",
                     "Units_Sold_roll_mean_7", "Units_Sold_roll_mean_30", "Price", "Promotion"]
    print(df[preview_cols].head(5).to_string())


if __name__ == "__main__":
    main()