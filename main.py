# """
# FORESIGHT — Data Merge & Cleaning Pipeline
# Combines sku_master, sales_daily, inventory_snapshots, and calendar
# into a single clean, model-ready dataset.
#
# Folder structure expected:
#     project_root/
#         data/raw/sku_master.csv
#         data/raw/sales_daily.csv
#         data/raw/inventory_snapshots.csv
#         data/raw/calendar.csv
#         data/processed/   <- output goes here
#         foresight_data_pipeline.py
# """
#
# import pandas as pd
# from pathlib import Path
#
# RAW_DIR = Path("data/raw")
# PROCESSED_DIR = Path("data/processed")
# PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
#
#
# def load_data():
#     sku_master = pd.read_csv(RAW_DIR / "sku_master.csv", parse_dates=["Launch_Date"])
#     sales = pd.read_csv(RAW_DIR / "sales_daily.csv", parse_dates=["Date"])
#     inventory = pd.read_csv(RAW_DIR / "inventory_snapshots.csv", parse_dates=["Snapshot_Date"])
#     calendar = pd.read_csv(RAW_DIR / "calendar.csv", parse_dates=["date"])
#     return sku_master, sales, inventory, calendar
#
#
# def clean_sku_master(sku_master):
#     sku_master = sku_master.drop_duplicates(subset="SKU")
#     sku_master = sku_master.dropna(subset=["SKU"])
#     return sku_master
#
#
# def clean_sales(sales):
#     sales = sales.drop_duplicates()
#     sales = sales.dropna(subset=["SKU", "Date"])
#     sales["Units_Sold"] = sales["Units_Sold"].fillna(0)
#     sales["Revenue"] = sales["Revenue"].fillna(0)
#     sales["Promotion"] = sales["Promotion"].fillna(0).astype(int)
#     sales = sales[sales["Units_Sold"] >= 0]  # drop impossible negative sales
#     return sales
#
#
# def clean_inventory(inventory):
#     inventory = inventory.drop_duplicates()
#     inventory = inventory.rename(columns={"Snapshot_Date": "Date"})
#     numeric_cols = [
#         "Current_Stock", "On_Order", "Lead_Time_Days",
#         "Safety_Stock", "Reorder_Point", "Inventory_Value",
#     ]
#     for col in numeric_cols:
#         inventory[col] = inventory[col].fillna(inventory[col].median())
#     return inventory
#
#
# def clean_calendar(calendar):
#     calendar = calendar.drop_duplicates()
#     calendar = calendar.rename(columns={"date": "Date"})
#     calendar["is_holiday"] = calendar["is_holiday"].fillna(0).astype(int)
#     calendar["promotion_event"] = calendar["promotion_event"].fillna("None")
#     return calendar
#
#
# def merge_all(sku_master, sales, inventory, calendar):
#     df = sales.merge(sku_master, on="SKU", how="left")
#     df = df.merge(inventory, on=["SKU", "Date"], how="left")
#     df = df.merge(calendar, on="Date", how="left")
#     return df
#
#
# def final_clean(df):
#     # Forward-fill any stock gaps left by imperfect date alignment between
#     # sales_daily and inventory_snapshots (e.g. snapshot taken less often than daily)
#     df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)
#     df["Current_Stock"] = df.groupby("SKU")["Current_Stock"].ffill()
#     df["Lead_Time_Days"] = df["Lead_Time_Days"].fillna(df["Lead_Time_Days"].median())
#     return df
#
#
# def main():
#     sku_master, sales, inventory, calendar = load_data()
#
#     sku_master = clean_sku_master(sku_master)
#     sales = clean_sales(sales)
#     inventory = clean_inventory(inventory)
#     calendar = clean_calendar(calendar)
#
#     merged = merge_all(sku_master, sales, inventory, calendar)
#     merged = final_clean(merged)
#
#     out_path = PROCESSED_DIR / "foresight_clean.csv"
#     merged.to_csv(out_path, index=False)
#
#     print(f"Saved cleaned dataset -> {out_path}")
#     print(f"Rows: {len(merged)}, Columns: {merged.shape[1]}")
#     print("\nMissing values remaining per column:")
#     print(merged.isnull().sum()[merged.isnull().sum() > 0])
#     print("\nPreview:")
#     print(merged.head())
#
#
# if __name__ == "__main__":
#     main()


"""
FORESIGHT — Data Merge & Cleaning Pipeline
Combines sku_master, sales_daily, inventory_snapshots, and calendar
into a single clean, model-ready dataset.

Folder structure expected:
    project_root/
        data/raw/sku_master.csv
        data/raw/sales_daily.csv
        data/raw/inventory_snapshots.csv
        data/raw/calendar.csv
        data/processed/   <- output goes here
        foresight_data_pipeline.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    sku_master = pd.read_csv(RAW_DIR / "sku_master.csv", parse_dates=["Launch_Date"])
    sales = pd.read_csv(RAW_DIR / "sales_daily.csv", parse_dates=["Date"])
    inventory = pd.read_csv(RAW_DIR / "inventory_snapshots.csv", parse_dates=["Snapshot_Date"])
    calendar = pd.read_csv(RAW_DIR / "calendar.csv", parse_dates=["date"])
    return sku_master, sales, inventory, calendar


def clean_sku_master(sku_master):
    sku_master = sku_master.drop_duplicates(subset="SKU")
    sku_master = sku_master.dropna(subset=["SKU"])
    return sku_master


def clean_sales(sales):
    sales = sales.drop_duplicates()
    sales = sales.dropna(subset=["SKU", "Date"])
    sales["Units_Sold"] = sales["Units_Sold"].fillna(0)
    sales["Revenue"] = sales["Revenue"].fillna(0)
    sales["Promotion"] = sales["Promotion"].fillna(0).astype(int)
    sales = sales[sales["Units_Sold"] >= 0]  # drop impossible negative sales
    return sales


def clean_inventory(inventory):
    inventory = inventory.drop_duplicates()
    inventory = inventory.rename(columns={"Snapshot_Date": "Date"})
    numeric_cols = [
        "Current_Stock", "On_Order", "Lead_Time_Days",
        "Safety_Stock", "Reorder_Point", "Inventory_Value",
    ]
    for col in numeric_cols:
        inventory[col] = inventory[col].fillna(inventory[col].median())
    return inventory


def clean_calendar(calendar):
    calendar = calendar.drop_duplicates()
    calendar = calendar.rename(columns={"date": "Date"})
    calendar["is_holiday"] = calendar["is_holiday"].fillna(0).astype(int)
    calendar["promotion_event"] = calendar["promotion_event"].fillna("None")
    return calendar


def merge_all(sku_master, sales, inventory, calendar):
    df = sales.merge(sku_master, on="SKU", how="left")
    df = df.merge(inventory, on=["SKU", "Date"], how="left")
    df = df.merge(calendar, on="Date", how="left")
    return df


def final_clean(df):
    # inventory_snapshots is taken monthly (1st of each month), not daily, so most
    # days have no exact match after the merge. Forward-fill each SKU's last known
    # snapshot values across the days until the next snapshot.
    df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)
    inventory_cols = [
        "Current_Stock", "On_Order", "Lead_Time_Days",
        "Safety_Stock", "Reorder_Point", "Inventory_Value",
    ]
    for col in inventory_cols:
        df[col] = df.groupby("SKU")[col].ffill().bfill()

    # holiday is a name field (only non-null on actual holidays) -- leave as label
    df["holiday"] = df["holiday"].fillna("None")
    return df


def main():
    sku_master, sales, inventory, calendar = load_data()

    sku_master = clean_sku_master(sku_master)
    sales = clean_sales(sales)
    inventory = clean_inventory(inventory)
    calendar = clean_calendar(calendar)

    merged = merge_all(sku_master, sales, inventory, calendar)
    merged = final_clean(merged)

    out_path = PROCESSED_DIR / "foresight_clean.csv"
    merged.to_csv(out_path, index=False)

    print(f"Saved cleaned dataset -> {out_path}")
    print(f"Rows: {len(merged)}, Columns: {merged.shape[1]}")
    print("\nMissing values remaining per column:")
    print(merged.isnull().sum()[merged.isnull().sum() > 0])
    print("\nPreview:")
    print(merged.head())


if __name__ == "__main__":
    main()