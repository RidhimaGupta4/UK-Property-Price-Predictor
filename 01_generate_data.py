"""
UK Property Price Predictor — Data Generation
==============================================
Generates a realistic UK property dataset aligned to:
  - HM Land Registry Price Paid Data
  - ONS House Price Index (regional benchmarks)
  - Ofsted school ratings
  - TfL / national rail commute time estimates
  - Zoopla / Rightmove listing feature patterns

Real data sources (use in production):
  Land Registry: https://www.gov.uk/government/collections/price-paid-data
  ONS HPI:       https://www.ons.gov.uk/economy/inflationandpriceindices
  Ofsted:        https://www.gov.uk/ofsted
  EPC Register:  https://epc.opendatacommunities.org/

Run:
    pip install pandas numpy scikit-learn
    python 01_generate_data.py
"""

import pandas as pd
import numpy as np
import json, os
from datetime import datetime, timedelta

np.random.seed(42)

# ── UK Regions with realistic price anchors ───────────────────────────────────

REGIONS = {
    "London - Inner":         {"base_price": 780_000, "price_std": 350_000, "growth": 0.048, "density": "urban"},
    "London - Outer":         {"base_price": 520_000, "price_std": 180_000, "growth": 0.042, "density": "urban"},
    "South East":             {"base_price": 420_000, "price_std": 140_000, "growth": 0.035, "density": "suburban"},
    "East of England":        {"base_price": 370_000, "price_std": 110_000, "growth": 0.033, "density": "suburban"},
    "South West":             {"base_price": 330_000, "price_std": 100_000, "growth": 0.031, "density": "suburban"},
    "East Midlands":          {"base_price": 240_000, "price_std":  80_000, "growth": 0.028, "density": "mixed"},
    "West Midlands":          {"base_price": 255_000, "price_std":  85_000, "growth": 0.027, "density": "mixed"},
    "Yorkshire & Humber":     {"base_price": 210_000, "price_std":  70_000, "growth": 0.025, "density": "mixed"},
    "North West":             {"base_price": 225_000, "price_std":  75_000, "growth": 0.026, "density": "mixed"},
    "North East":             {"base_price": 165_000, "price_std":  55_000, "growth": 0.020, "density": "mixed"},
    "Wales":                  {"base_price": 210_000, "price_std":  65_000, "growth": 0.029, "density": "mixed"},
    "Scotland":               {"base_price": 195_000, "price_std":  70_000, "growth": 0.024, "density": "mixed"},
}

PROPERTY_TYPES = ["Detached", "Semi-Detached", "Terraced", "Flat"]
TYPE_MULTIPLIERS = {"Detached": 1.55, "Semi-Detached": 1.10, "Terraced": 0.90, "Flat": 0.75}
TYPE_WEIGHTS = [0.20, 0.28, 0.28, 0.24]

TENURE = ["Freehold", "Leasehold"]
TENURE_WEIGHTS = [0.52, 0.48]
TENURE_MULTIPLIERS = {"Freehold": 1.05, "Leasehold": 0.95}

CONDITIONS = ["New Build", "Excellent", "Good", "Fair", "Poor"]
COND_WEIGHTS = [0.08, 0.22, 0.38, 0.24, 0.08]
COND_MULTIPLIERS = {"New Build": 1.12, "Excellent": 1.06, "Good": 1.00, "Fair": 0.91, "Poor": 0.80}

EPC_RATINGS = ["A", "B", "C", "D", "E", "F", "G"]
EPC_WEIGHTS = [0.03, 0.12, 0.28, 0.30, 0.17, 0.07, 0.03]
EPC_MULTIPLIERS = {"A": 1.06, "B": 1.04, "C": 1.02, "D": 1.00, "E": 0.97, "F": 0.94, "G": 0.90}

SCHOOL_RATINGS = ["Outstanding", "Good", "Requires Improvement", "Inadequate"]
SCHOOL_WEIGHTS = [0.22, 0.48, 0.22, 0.08]
SCHOOL_MULTIPLIERS = {"Outstanding": 1.08, "Good": 1.02, "Requires Improvement": 0.97, "Inadequate": 0.92}

SALE_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
YEAR_WEIGHTS = [0.14, 0.12, 0.18, 0.20, 0.19, 0.17]


def generate_postcode(region):
    """Generate a plausible UK postcode for the region."""
    postcode_prefixes = {
        "London - Inner":     ["EC", "WC", "W", "SW", "SE", "E", "N", "NW"],
        "London - Outer":     ["IG", "RM", "DA", "BR", "CR", "SM", "KT", "TW", "UB", "HA", "EN"],
        "South East":         ["GU", "RH", "TN", "CT", "ME", "SO", "PO", "BN", "SL", "HP", "AL"],
        "East of England":    ["CB", "PE", "IP", "NR", "CO", "CM", "SS", "LU", "SG"],
        "South West":         ["BS", "BA", "TA", "EX", "TQ", "PL", "TR", "GL", "SP", "BH"],
        "East Midlands":      ["NG", "DE", "LE", "CV", "NN", "MK", "LN"],
        "West Midlands":      ["B", "WV", "WS", "DY", "ST", "WR", "HR"],
        "Yorkshire & Humber": ["LS", "BD", "HX", "HD", "WF", "S", "DN", "YO", "HU"],
        "North West":         ["M", "SK", "OL", "BL", "WN", "WA", "PR", "BB", "LA", "CH", "CW"],
        "North East":         ["NE", "SR", "DH", "TS", "DL"],
        "Wales":              ["CF", "SA", "NP", "LL", "SY", "LD"],
        "Scotland":           ["EH", "G", "PA", "KA", "FK", "PH", "DD", "AB", "IV"],
    }
    prefix = np.random.choice(postcode_prefixes.get(region, ["XX"]))
    num1   = np.random.randint(1, 30)
    num2   = np.random.randint(1, 10)
    letter = np.random.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ"), 2)
    return f"{prefix}{num1} {num2}{''.join(letter)}"


def generate_property_dataset(n=8000):
    records = []

    for _ in range(n):
        region      = np.random.choice(list(REGIONS.keys()))
        reg         = REGIONS[region]
        prop_type   = np.random.choice(PROPERTY_TYPES, p=TYPE_WEIGHTS)
        tenure      = np.random.choice(TENURE, p=TENURE_WEIGHTS)
        condition   = np.random.choice(CONDITIONS, p=COND_WEIGHTS)
        epc         = np.random.choice(EPC_RATINGS, p=EPC_WEIGHTS)
        school_rate = np.random.choice(SCHOOL_RATINGS, p=SCHOOL_WEIGHTS)
        sale_year   = np.random.choice(SALE_YEARS, p=YEAR_WEIGHTS)

        # Bedrooms by type
        if prop_type == "Flat":
            bedrooms = np.random.choice([0, 1, 2, 3], p=[0.05, 0.35, 0.45, 0.15])
        elif prop_type == "Terraced":
            bedrooms = np.random.choice([1, 2, 3, 4], p=[0.10, 0.30, 0.45, 0.15])
        elif prop_type == "Semi-Detached":
            bedrooms = np.random.choice([2, 3, 4, 5], p=[0.15, 0.50, 0.28, 0.07])
        else:  # Detached
            bedrooms = np.random.choice([3, 4, 5, 6], p=[0.28, 0.42, 0.22, 0.08])

        bathrooms   = min(bedrooms, np.random.choice([1, 2, 3, 4], p=[0.38, 0.42, 0.15, 0.05]))
        floor_area  = int(np.random.normal(
            {"Flat": 65, "Terraced": 95, "Semi-Detached": 120, "Detached": 170}[prop_type],
            {"Flat": 18, "Terraced": 22, "Semi-Detached": 28, "Detached": 40}[prop_type]
        ))
        floor_area  = max(30, floor_area)

        has_garden   = 1 if prop_type in ["Detached", "Semi-Detached", "Terraced"] else int(np.random.random() < 0.25)
        has_parking  = 1 if prop_type in ["Detached", "Semi-Detached"] else int(np.random.random() < 0.35)
        has_garage   = 1 if prop_type == "Detached" and np.random.random() < 0.60 else int(np.random.random() < 0.20)

        # Commute time (minutes to nearest city centre)
        if reg["density"] == "urban":
            commute_mins = int(np.random.normal(22, 10))
        elif reg["density"] == "suburban":
            commute_mins = int(np.random.normal(42, 15))
        else:
            commute_mins = int(np.random.normal(55, 20))
        commute_mins = max(5, min(120, commute_mins))

        # Distance to station (km)
        dist_station = round(max(0.1, np.random.exponential(1.2)), 2)

        # Distance to city centre (km)
        dist_centre = round(max(0.5, np.random.exponential(
            {"urban": 4, "suburban": 12, "mixed": 18}[reg["density"]]
        )), 2)

        # Crime index (0 = safe, 100 = high crime) — urban areas higher
        crime_index = round(max(0, np.random.normal(
            {"urban": 45, "suburban": 28, "mixed": 35}[reg["density"]], 15
        )), 1)

        # Flood risk
        flood_risk = np.random.choice(["Low", "Medium", "High"], p=[0.78, 0.16, 0.06])

        # Build year
        age_band = np.random.choice(
            ["Pre-1900", "1900-1945", "1945-1975", "1975-2000", "2000-2015", "Post-2015"],
            p=[0.08, 0.15, 0.22, 0.25, 0.18, 0.12]
        )
        build_year = {
            "Pre-1900": np.random.randint(1800, 1900),
            "1900-1945": np.random.randint(1900, 1945),
            "1945-1975": np.random.randint(1945, 1975),
            "1975-2000": np.random.randint(1975, 2000),
            "2000-2015": np.random.randint(2000, 2015),
            "Post-2015": np.random.randint(2015, 2025),
        }[age_band]

        property_age = sale_year - build_year

        # ── PRICE CALCULATION ──────────────────────────────────────────────
        base = reg["base_price"]

        # Year-on-year growth
        years_from_base = sale_year - 2019
        base *= (1 + reg["growth"]) ** years_from_base

        # 2020 dip, 2021-22 surge, 2023 correction
        year_adj = {2019: 1.00, 2020: 0.97, 2021: 1.09, 2022: 1.12, 2023: 0.98, 2024: 1.01}[sale_year]
        base *= year_adj

        # Feature multipliers
        price = (base
                 * TYPE_MULTIPLIERS[prop_type]
                 * TENURE_MULTIPLIERS[tenure]
                 * COND_MULTIPLIERS[condition]
                 * EPC_MULTIPLIERS[epc]
                 * SCHOOL_MULTIPLIERS[school_rate])

        # Bedroom premium
        price *= (1 + (bedrooms - 3) * 0.08)

        # Floor area
        price *= (1 + (floor_area - 100) / 1000)

        # Commute discount
        price *= max(0.70, 1 - commute_mins * 0.003)

        # Garden / parking / garage premium
        if has_garden:  price *= 1.06
        if has_parking: price *= 1.03
        if has_garage:  price *= 1.04

        # Crime discount
        price *= (1 - crime_index * 0.001)

        # Flood risk
        flood_adj = {"Low": 1.00, "Medium": 0.97, "High": 0.92}[flood_risk]
        price *= flood_adj

        # Age discount (older = slightly cheaper, but pre-1900 = premium)
        if age_band == "Pre-1900":
            price *= 1.04
        elif property_age > 50:
            price *= 0.96

        # Random noise
        noise = np.random.normal(1.0, 0.06)
        price = max(50_000, price * noise)
        price = round(price / 1000) * 1000  # round to nearest £1k

        records.append({
            "property_id":        f"PROP{len(records)+1:06d}",
            "postcode":           generate_postcode(region),
            "region":             region,
            "property_type":      prop_type,
            "tenure":             tenure,
            "bedrooms":           bedrooms,
            "bathrooms":          bathrooms,
            "floor_area_sqm":     floor_area,
            "build_year":         build_year,
            "property_age":       property_age,
            "age_band":           age_band,
            "condition":          condition,
            "epc_rating":         epc,
            "nearest_school_ofsted": school_rate,
            "has_garden":         has_garden,
            "has_parking":        has_parking,
            "has_garage":         has_garage,
            "commute_mins":       commute_mins,
            "dist_to_station_km": dist_station,
            "dist_to_centre_km":  dist_centre,
            "crime_index":        crime_index,
            "flood_risk":         flood_risk,
            "sale_year":          sale_year,
            "density":            reg["density"],
            "price_gbp":          int(price),
        })

    return pd.DataFrame(records)


def main():
    out = "/home/claude/uk-property-predictor/data/processed"
    os.makedirs(out, exist_ok=True)

    print("Generating property dataset (8,000 properties)...")
    df = generate_property_dataset(8000)
    df.to_csv(f"{out}/properties.csv", index=False)
    print(f"  Saved: properties.csv — {len(df)} rows")

    # Summary stats
    print("\n── Price Summary by Region ──")
    summary = df.groupby("region")["price_gbp"].agg(
        count="count", median="median", mean="mean", min="min", max="max"
    ).round(0)
    print(summary.to_string())

    print("\n── Price by Property Type ──")
    print(df.groupby("property_type")["price_gbp"].median().sort_values(ascending=False).apply(lambda x: f"£{x:,.0f}").to_string())

    print("\n── Year-on-Year Median Prices ──")
    print(df.groupby("sale_year")["price_gbp"].median().apply(lambda x: f"£{x:,.0f}").to_string())

    return df


if __name__ == "__main__":
    main()
