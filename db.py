"""
FORESIGHT — SQL Database
Builds a SQLite database (foresight.db) with Products, Sales, Inventory,
Forecasts, and Recommendations tables from the pipeline's CSV outputs,
then runs analytical SQL queries against it.
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("foresight.db")

SKU_MASTER = Path("data/raw/sku_master.csv")
SALES_CLEAN = Path("data/processed/foresight_clean.csv")
INVENTORY_RAW = Path("data/raw/inventory_snapshots.csv")
FORECASTS = Path("data/processed/foresight_forecasts.csv")
RECOMMENDATIONS = Path("data/processed/foresight_recommendations.csv")

SCHEMA = """
CREATE TABLE Products (
    SKU TEXT PRIMARY KEY,
    Product_Name TEXT,
    Category TEXT,
    Subcategory TEXT,
    Launch_Date TEXT,
    Cost_Price REAL,
    Selling_Price REAL,
    Gross_Margin_Per_Unit REAL
);

CREATE TABLE Sales (
    Date TEXT,
    SKU TEXT,
    Units_Sold INTEGER,
    Revenue REAL,
    Price REAL,
    Promotion INTEGER,
    PRIMARY KEY (Date, SKU),
    FOREIGN KEY (SKU) REFERENCES Products(SKU)
);

CREATE TABLE Inventory (
    Snapshot_Date TEXT,
    SKU TEXT,
    Current_Stock INTEGER,
    On_Order INTEGER,
    Lead_Time_Days INTEGER,
    Safety_Stock REAL,
    Reorder_Point REAL,
    Inventory_Value REAL,
    PRIMARY KEY (Snapshot_Date, SKU),
    FOREIGN KEY (SKU) REFERENCES Products(SKU)
);

CREATE TABLE Forecasts (
    Date TEXT,
    SKU TEXT,
    Actual_Demand REAL,
    Forecast_Demand REAL,
    PRIMARY KEY (Date, SKU),
    FOREIGN KEY (SKU) REFERENCES Products(SKU)
);

CREATE TABLE Recommendations (
    SKU TEXT PRIMARY KEY,
    Current_Stock REAL,
    On_Order REAL,
    Lead_Time_Days REAL,
    Reorder_Point_Calc REAL,
    Safety_Stock_Calc REAL,
    Forecast_Avg_Daily_Demand REAL,
    Days_Of_Stock REAL,
    Risk_Status TEXT,
    Recommendation TEXT,
    Recommended_Qty REAL,
    FOREIGN KEY (SKU) REFERENCES Products(SKU)
);
"""


def build_database():
    if DB_PATH.exists():
        DB_PATH.unlink()  # rebuild fresh each run

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    # Products
    products = pd.read_csv(SKU_MASTER)
    products.to_sql("Products", conn, if_exists="append", index=False)

    # Sales (from the cleaned/merged table -- one row per Date+SKU)
    sales = pd.read_csv(SALES_CLEAN)[["Date", "SKU", "Units_Sold", "Revenue", "Price", "Promotion"]]
    sales.to_sql("Sales", conn, if_exists="append", index=False)

    # Inventory -- filter to only SKUs that exist in Products, to satisfy the
    # foreign key (the raw file also tracks 150 SKUs not covered elsewhere)
    inventory = pd.read_csv(INVENTORY_RAW)
    inventory = inventory[inventory["SKU"].isin(products["SKU"])]
    inventory = inventory.rename(columns={"Snapshot_Date": "Snapshot_Date"})
    inventory.to_sql("Inventory", conn, if_exists="append", index=False)

    # Forecasts
    forecasts = pd.read_csv(FORECASTS)
    forecasts.to_sql("Forecasts", conn, if_exists="append", index=False)

    # Recommendations
    recs = pd.read_csv(RECOMMENDATIONS)[[
        "SKU", "Current_Stock", "On_Order", "Lead_Time_Days", "Reorder_Point_Calc",
        "Safety_Stock_Calc", "Forecast_Avg_Daily_Demand", "Days_Of_Stock",
        "Risk_Status", "Recommendation", "Recommended_Qty",
    ]]
    recs.to_sql("Recommendations", conn, if_exists="append", index=False)

    conn.commit()
    return conn


def run_query(conn, title, sql):
    print(f"\n--- {title} ---")
    df = pd.read_sql(sql, conn)
    print(df.to_string(index=False))
    return df


def main():
    conn = build_database()
    print(f"Database built -> {DB_PATH.resolve()}")

    for table in ["Products", "Sales", "Inventory", "Forecasts", "Recommendations"]:
        count = pd.read_sql(f"SELECT COUNT(*) AS rows FROM {table}", conn).iloc[0, 0]
        print(f"  {table}: {count:,} rows")

    run_query(conn, "Revenue & units by category", """
        SELECT p.Category,
               SUM(s.Revenue) AS Total_Revenue,
               SUM(s.Units_Sold) AS Total_Units
        FROM Sales s
        JOIN Products p ON s.SKU = p.SKU
        GROUP BY p.Category
        ORDER BY Total_Revenue DESC;
    """)

    run_query(conn, "Top 5 SKUs by revenue", """
        SELECT s.SKU, p.Product_Name, SUM(s.Revenue) AS Total_Revenue
        FROM Sales s
        JOIN Products p ON s.SKU = p.SKU
        GROUP BY s.SKU
        ORDER BY Total_Revenue DESC
        LIMIT 5;
    """)

    run_query(conn, "Monthly revenue trend (2024)", """
        SELECT strftime('%Y-%m', Date) AS Month, SUM(Revenue) AS Revenue
        FROM Sales
        WHERE Date LIKE '2024%'
        GROUP BY Month
        ORDER BY Month;
    """)

    run_query(conn, "Current recommendation distribution", """
        SELECT Recommendation, COUNT(*) AS SKU_Count
        FROM Recommendations
        GROUP BY Recommendation
        ORDER BY SKU_Count DESC;
    """)

    run_query(conn, "SKUs needing immediate reorder", """
        SELECT r.SKU, p.Product_Name, r.Current_Stock, r.Recommended_Qty, r.Days_Of_Stock
        FROM Recommendations r
        JOIN Products p ON r.SKU = p.SKU
        WHERE r.Recommendation = 'Reorder Immediately'
        ORDER BY r.Days_Of_Stock ASC;
    """)

    run_query(conn, "Forecast accuracy (MAE) per SKU -- worst 5", """
        SELECT SKU,
               ROUND(AVG(ABS(Actual_Demand - Forecast_Demand)), 2) AS MAE
        FROM Forecasts
        GROUP BY SKU
        ORDER BY MAE DESC
        LIMIT 5;
    """)

    conn.close()
    print(f"\nDone. Database file ready at {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()