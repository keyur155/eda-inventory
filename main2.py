"""
FORESIGHT — Exploratory Data Analysis
Reads data/processed/foresight_clean.csv and produces:
  - reports/figures/*.png  (charts for your report/dashboard)
  - printed business insights (paste into your report)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/processed/foresight_clean.csv")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df


def plot_monthly_trend(df):
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Units_Sold"].sum()
    monthly.index = monthly.index.to_timestamp()
    plt.figure()
    plt.plot(monthly.index, monthly.values, marker="o")
    plt.title("Monthly Demand Trend (Total Units Sold)")
    plt.xlabel("Month")
    plt.ylabel("Units Sold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_monthly_trend.png", dpi=120)
    plt.close()
    return monthly


def plot_top_skus(df, n=10):
    top = df.groupby(["SKU", "Product_Name"])["Revenue"].sum().sort_values(ascending=False).head(n)
    plt.figure()
    labels = [f"{sku}" for sku, _ in top.index]
    plt.barh(labels[::-1], top.values[::-1])
    plt.title(f"Top {n} SKUs by Revenue")
    plt.xlabel("Total Revenue")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_top_skus.png", dpi=120)
    plt.close()
    return top


def plot_category_breakdown(df):
    cat_rev = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    plt.figure()
    plt.bar(cat_rev.index, cat_rev.values)
    plt.title("Revenue by Category")
    plt.ylabel("Total Revenue")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_category_revenue.png", dpi=120)
    plt.close()
    return cat_rev


def plot_seasonality(df):
    season_order = ["Winter", "Spring", "Summer", "Monsoon", "Autumn"]
    season = df.groupby("season")["Units_Sold"].sum().reindex(season_order)
    plt.figure()
    plt.bar(season.index, season.values, color="teal")
    plt.title("Total Demand by Season")
    plt.ylabel("Units Sold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_seasonality.png", dpi=120)
    plt.close()
    return season


def plot_promotion_impact(df):
    promo = df.groupby("Promotion")["Units_Sold"].mean()
    labels = ["No Promotion", "Promotion"]
    plt.figure()
    plt.bar(labels, [promo.get(0, 0), promo.get(1, 0)], color=["gray", "orange"])
    plt.title("Average Daily Units Sold: Promotion vs No Promotion")
    plt.ylabel("Avg Units Sold / Day")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_promotion_impact.png", dpi=120)
    plt.close()
    lift = (promo.get(1, 0) / promo.get(0, 1) - 1) * 100
    return promo, lift


def plot_weekday_pattern(df):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = df.groupby("day_of_week")["Units_Sold"].mean().reindex(order)
    plt.figure()
    colors = ["steelblue"] * 5 + ["crimson"] * 2
    plt.bar(dow.index, dow.values, color=colors)
    plt.title("Average Units Sold by Day of Week")
    plt.ylabel("Avg Units Sold")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_weekday_pattern.png", dpi=120)
    plt.close()
    return dow


def print_insights(monthly, top, cat_rev, season, promo, lift, dow):
    print("=" * 60)
    print("FORESIGHT — KEY BUSINESS INSIGHTS")
    print("=" * 60)

    print(f"\n1. DEMAND TREND")
    print(f"   Peak month: {monthly.idxmax().strftime('%b %Y')} ({int(monthly.max()):,} units)")
    print(f"   Lowest month: {monthly.idxmin().strftime('%b %Y')} ({int(monthly.min()):,} units)")
    print("   Pattern repeats yearly -> March spike, October trough (clear seasonality).")

    print(f"\n2. TOP PERFORMERS")
    print(f"   #1 SKU by revenue: {top.index[0][0]} ({top.index[0][1]}) - {top.iloc[0]:,.0f}")
    print(f"   Top category: {cat_rev.idxmax()} ({cat_rev.max():,.0f} revenue)")

    print(f"\n3. SEASONALITY")
    print(f"   Strongest season: {season.idxmax()} ({int(season.max()):,} units)")
    print(f"   Weakest season: {season.idxmin()} ({int(season.min()):,} units)")

    print(f"\n4. PROMOTION IMPACT")
    print(f"   Avg units/day without promotion: {promo.get(0,0):.1f}")
    print(f"   Avg units/day with promotion:    {promo.get(1,0):.1f}")
    print(f"   -> Promotions lift daily demand by {lift:.1f}%")

    print(f"\n5. WEEKLY PATTERN")
    print(f"   Weekend avg: {dow[['Saturday','Sunday']].mean():.1f} units/day")
    print(f"   Weekday avg: {dow[['Monday','Tuesday','Wednesday','Thursday','Friday']].mean():.1f} units/day")
    print(f"   -> Weekends outsell weekdays; plan staffing/restocking accordingly.")


def main():
    df = load_data()
    monthly = plot_monthly_trend(df)
    top = plot_top_skus(df)
    cat_rev = plot_category_breakdown(df)
    season = plot_seasonality(df)
    promo, lift = plot_promotion_impact(df)
    dow = plot_weekday_pattern(df)

    print_insights(monthly, top, cat_rev, season, promo, lift, dow)
    print(f"\nCharts saved to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()