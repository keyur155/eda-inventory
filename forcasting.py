"""
FORESIGHT — Demand Forecasting
Reads data/processed/foresight_features.csv, trains a baseline model plus
Random Forest and XGBoost, evaluates with MAE/RMSE/MAPE using a proper
time-based train/test split, and saves per-SKU forecasts for Step 6.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

IN_PATH = Path("data/processed/foresight_features.csv")
OUT_FORECAST_PATH = Path("data/processed/foresight_forecasts.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

TEST_DAYS = 60  # last 60 days per SKU held out as the "future" to forecast

# Demand-driver features only. Deliberately excludes Current_Stock, Reorder_Point,
# Safety_Stock, On_Order, Inventory_Value -- those are inventory-side numbers used
# in Step 6, not causes of demand, and mixing them in here would blur the pipeline.
FEATURE_COLS = [
    "SKU_enc", "Category_enc", "Day", "month", "quarter_enc", "week",
    "day_of_week_enc", "is_weekend", "season_enc", "is_holiday",
    "Price", "Promotion",
    "Units_Sold_lag_1", "Units_Sold_lag_7", "Units_Sold_lag_14", "Units_Sold_lag_30",
    "Units_Sold_roll_mean_7", "Units_Sold_roll_mean_30",
]
TARGET_COL = "Units_Sold"


def load_and_encode():
    df = pd.read_csv(IN_PATH, parse_dates=["Date"])
    df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)

    for col in ["SKU", "Category", "day_of_week", "season", "quarter"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

    return df


def time_based_split(df, test_days=TEST_DAYS):
    # Per SKU, the last `test_days` calendar days become the test set. This
    # simulates genuinely forecasting the future rather than random shuffling,
    # which would leak future information into training.
    cutoff = df.groupby("SKU")["Date"].transform(lambda s: s.max() - pd.Timedelta(days=test_days))
    train = df[df["Date"] <= cutoff].copy()
    test = df[df["Date"] > cutoff].copy()
    return train, test


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0  # avoid divide-by-zero on no-sale days
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mp = mape(y_true, y_pred)
    print(f"{name:20s}  MAE={mae:6.2f}   RMSE={rmse:6.2f}   MAPE={mp:6.2f}%")
    return {"model": name, "MAE": mae, "RMSE": rmse, "MAPE": mp}


def main():
    df = load_and_encode()
    train, test = time_based_split(df)
    print(f"Train rows: {len(train):,}  |  Test rows: {len(test):,} (last {TEST_DAYS} days per SKU)\n")

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    results = []

    # --- Baseline: seasonal-naive (predict "same day last week") ---
    baseline_pred = X_test["Units_Sold_lag_7"]
    results.append(evaluate(y_test, baseline_pred, "Baseline (lag-7)"))

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results.append(evaluate(y_test, rf_pred, "Random Forest"))
    joblib.dump(rf, MODEL_DIR / "random_forest.pkl")

    best_name, best_pred, best_mae = "Random Forest", rf_pred, results[-1]["MAE"]

    # --- XGBoost ---
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        r = evaluate(y_test, xgb_pred, "XGBoost")
        results.append(r)
        joblib.dump(xgb, MODEL_DIR / "xgboost.pkl")
        if r["MAE"] < best_mae:
            best_name, best_pred, best_mae = "XGBoost", xgb_pred, r["MAE"]
    else:
        print("xgboost not installed -- skipping (pip install xgboost)")

    # Save forecasts -- this file feeds directly into Step 6 (inventory risk detection)
    out = test[["Date", "SKU"]].copy()
    out["Actual_Demand"] = y_test.values
    out["Forecast_Demand"] = np.clip(best_pred, a_min=0, a_max=None)  # demand can't be negative
    out.to_csv(OUT_FORECAST_PATH, index=False)

    print(f"\nBest model: {best_name}")
    print(f"Forecasts saved -> {OUT_FORECAST_PATH}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print(f"\nModel comparison:\n{results_df.to_string(index=False)}")

    # Feature importance from the best tree model (useful for the report)
    model = xgb if (HAS_XGB and best_name == "XGBoost") else rf
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\nTop 8 most important features ({best_name}):")
    print(importance.head(8).to_string())


if __name__ == "__main__":
    main()