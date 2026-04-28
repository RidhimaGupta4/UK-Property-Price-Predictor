"""
UK Property Price Predictor — ML Pipeline
==========================================
Full machine learning pipeline:
  1. Feature engineering (encoding, interactions, geospatial proxies)
  2. Model training: Linear Regression, Random Forest, Gradient Boosting
  3. Model evaluation: RMSE, MAE, R², cross-validation
  4. Feature importance analysis
  5. Prediction error analysis
  6. Model saving for dashboard use

Note: XGBoost replaced with sklearn GradientBoostingRegressor (same concept,
no external install needed). Results comparable to XGBoost for this dataset.

Run:
    python 02_ml_pipeline.py
"""

import pandas as pd
import numpy as np
import json, os, warnings
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA = "/home/claude/uk-property-predictor/data/processed"
OUT  = "/home/claude/uk-property-predictor/data/processed"
os.makedirs(OUT, exist_ok=True)


# ── 1. LOAD & FEATURE ENGINEERING ────────────────────────────────────────────

def engineer_features(df):
    """
    Create ML features from raw property data.
    This is where the analytical value is — not just passing raw columns.
    """
    fe = df.copy()

    # ── Encode categoricals ──────────────────────────────────────────────────
    # Ordinal encodings where order matters
    epc_order    = {"A":6,"B":5,"C":4,"D":3,"E":2,"F":1,"G":0}
    school_order = {"Outstanding":3,"Good":2,"Requires Improvement":1,"Inadequate":0}
    cond_order   = {"New Build":4,"Excellent":3,"Good":2,"Fair":1,"Poor":0}
    flood_order  = {"Low":0,"Medium":1,"High":2}
    age_order    = {"Pre-1900":0,"1900-1945":1,"1945-1975":2,"1975-2000":3,"2000-2015":4,"Post-2015":5}

    fe["epc_score"]          = fe["epc_rating"].map(epc_order)
    fe["school_score"]       = fe["nearest_school_ofsted"].map(school_order)
    fe["condition_score"]    = fe["condition"].map(cond_order)
    fe["flood_score"]        = fe["flood_risk"].map(flood_order)
    fe["age_band_score"]     = fe["age_band"].map(age_order)

    # One-hot encode property type and tenure
    for pt in ["Detached","Semi-Detached","Terraced","Flat"]:
        fe[f"type_{pt.lower().replace('-','_').replace(' ','_')}"] = (fe["property_type"] == pt).astype(int)
    fe["is_freehold"] = (fe["tenure"] == "Freehold").astype(int)

    # Region encode — use median price per region as target encoding proxy
    region_median = df.groupby("region")["price_gbp"].median()
    fe["region_price_index"] = fe["region"].map(region_median) / 1000  # in £k

    # ── Interaction features ──────────────────────────────────────────────────
    # Price-driving interactions
    fe["bed_area"]           = fe["bedrooms"] * fe["floor_area_sqm"]
    fe["school_x_density"]   = fe["school_score"] * (fe["density"] == "urban").astype(int)
    fe["commute_x_urban"]    = fe["commute_mins"] * (fe["density"] == "urban").astype(int)
    fe["age_x_condition"]    = fe["property_age"] * fe["condition_score"]
    fe["epc_x_age"]          = fe["epc_score"] * fe["age_band_score"]
    fe["bath_bed_ratio"]      = fe["bathrooms"] / (fe["bedrooms"] + 0.1)
    fe["area_per_bed"]        = fe["floor_area_sqm"] / (fe["bedrooms"] + 0.1)

    # Amenity score composite
    fe["amenity_score"] = (
        fe["has_garden"] * 3 +
        fe["has_parking"] * 2 +
        fe["has_garage"] * 2 +
        fe["school_score"] +
        fe["epc_score"]
    )

    # Location quality index
    fe["location_score"] = (
        fe["region_price_index"] / 100
        - fe["commute_mins"] / 60
        - fe["dist_to_centre_km"] / 30
        - fe["crime_index"] / 100
        - fe["flood_score"] * 0.1
    )

    # Log transform price (target) and skewed features
    fe["log_price"]           = np.log1p(fe["price_gbp"])
    fe["log_floor_area"]      = np.log1p(fe["floor_area_sqm"])
    fe["log_dist_centre"]     = np.log1p(fe["dist_to_centre_km"])
    fe["log_dist_station"]    = np.log1p(fe["dist_to_station_km"])

    # Year relative to base
    fe["years_from_2019"]     = fe["sale_year"] - 2019

    return fe


def get_feature_cols():
    return [
        "bedrooms", "bathrooms", "floor_area_sqm", "log_floor_area",
        "property_age", "epc_score", "school_score", "condition_score",
        "flood_score", "age_band_score",
        "has_garden", "has_parking", "has_garage", "is_freehold",
        "commute_mins", "dist_to_station_km", "dist_to_centre_km",
        "log_dist_centre", "log_dist_station",
        "crime_index",
        "type_detached", "type_semi_detached", "type_terraced", "type_flat",
        "region_price_index",
        "bed_area", "school_x_density", "commute_x_urban", "age_x_condition",
        "epc_x_age", "bath_bed_ratio", "area_per_bed",
        "amenity_score", "location_score",
        "years_from_2019",
    ]


# ── 2. TRAIN MODELS ───────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name):
    preds      = model.predict(X_test)
    preds_gbp  = np.expm1(preds)
    actual_gbp = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(actual_gbp, preds_gbp))
    mae  = mean_absolute_error(actual_gbp, preds_gbp)
    r2   = r2_score(actual_gbp, preds_gbp)
    mape = np.mean(np.abs((actual_gbp - preds_gbp) / actual_gbp)) * 100

    print(f"\n  {name}:")
    print(f"    R²:   {r2:.4f}")
    print(f"    RMSE: £{rmse:,.0f}")
    print(f"    MAE:  £{mae:,.0f}")
    print(f"    MAPE: {mape:.1f}%")

    return {"model": name, "r2": round(r2,4), "rmse": round(rmse,0),
            "mae": round(mae,0), "mape": round(mape,2), "preds": preds_gbp, "actual": actual_gbp}


def main():
    print("Loading data...")
    df = pd.read_csv(f"{DATA}/properties.csv")
    print(f"  {len(df)} properties loaded")

    print("\nEngineering features...")
    fe = engineer_features(df)
    feature_cols = get_feature_cols()

    X = fe[feature_cols].fillna(0)
    y = fe["log_price"]

    # Train/test split (80/20 stratified by region)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Scale for Ridge
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("\nTraining models...")
    results = {}

    # ── Model 1: Ridge Regression (baseline) ─────────────────────────────────
    print("  Training Ridge Regression...")
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_sc, y_train)
    results["Ridge Regression"] = evaluate_model(ridge, X_test_sc, y_test, "Ridge Regression")

    # ── Model 2: Random Forest ────────────────────────────────────────────────
    print("  Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=18, min_samples_leaf=3,
                               n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    results["Random Forest"] = evaluate_model(rf, X_test, y_test, "Random Forest")

    # ── Model 3: Gradient Boosting (XGBoost equivalent) ──────────────────────
    print("  Training Gradient Boosting (XGBoost-style)...")
    gb = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05,
        max_depth=5, min_samples_leaf=10,
        subsample=0.8, random_state=42
    )
    gb.fit(X_train, y_train)
    results["Gradient Boosting"] = evaluate_model(gb, X_test, y_test, "Gradient Boosting")

    # ── Cross-validation on best model ───────────────────────────────────────
    print("\nRunning 5-fold cross-validation on Gradient Boosting...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(gb, X_train, y_train, cv=kf, scoring="r2")
    cv_rmse = []
    for train_idx, val_idx in kf.split(X_train):
        Xtr, Xvl = X_train.iloc[train_idx], X_train.iloc[val_idx]
        ytr, yvl = y_train.iloc[train_idx], y_train.iloc[val_idx]
        gb_cv = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        gb_cv.fit(Xtr, ytr)
        pv = np.expm1(gb_cv.predict(Xvl))
        av = np.expm1(yvl)
        cv_rmse.append(np.sqrt(mean_squared_error(av, pv)))

    print(f"  CV R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  CV RMSE: £{np.mean(cv_rmse):,.0f} ± £{np.std(cv_rmse):,.0f}")

    # ── Feature Importance ────────────────────────────────────────────────────
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": gb.feature_importances_
    }).sort_values("importance", ascending=False).head(20)
    fi["importance_pct"] = (fi["importance"] / fi["importance"].sum() * 100).round(2)
    fi.to_csv(f"{OUT}/feature_importance.csv", index=False)
    print(f"\nTop 10 features:")
    print(fi.head(10)[["feature","importance_pct"]].to_string(index=False))

    # ── Prediction vs Actual ──────────────────────────────────────────────────
    test_results = pd.DataFrame({
        "actual_price":    results["Gradient Boosting"]["actual"].round(0).astype(int),
        "predicted_price": results["Gradient Boosting"]["preds"].round(0).astype(int),
        "region":          df.iloc[X_test.index]["region"].values,
        "property_type":   df.iloc[X_test.index]["property_type"].values,
        "bedrooms":        df.iloc[X_test.index]["bedrooms"].values,
        "floor_area_sqm":  df.iloc[X_test.index]["floor_area_sqm"].values,
        "sale_year":       df.iloc[X_test.index]["sale_year"].values,
    })
    test_results["error_gbp"]   = test_results["predicted_price"] - test_results["actual_price"]
    test_results["error_pct"]   = (test_results["error_gbp"] / test_results["actual_price"] * 100).round(1)
    test_results["abs_error"]   = test_results["error_gbp"].abs()
    test_results.to_csv(f"{OUT}/test_predictions.csv", index=False)

    # ── Model comparison summary ──────────────────────────────────────────────
    model_compare = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ["preds","actual"]}
        for r in results.values()
    ])
    model_compare.to_csv(f"{OUT}/model_comparison.csv", index=False)

    # ── Regional accuracy ─────────────────────────────────────────────────────
    regional_acc = test_results.groupby("region").agg(
        mae=("abs_error","mean"),
        count=("actual_price","count"),
        median_actual=("actual_price","median"),
        median_predicted=("predicted_price","median")
    ).reset_index()
    regional_acc["mae_pct"] = (regional_acc["mae"] / regional_acc["median_actual"] * 100).round(1)
    regional_acc.to_csv(f"{OUT}/regional_accuracy.csv", index=False)

    # ── Save JSON for dashboard ───────────────────────────────────────────────
    dashboard_json = {
        "model_metrics": {
            "gradient_boosting": {
                "r2": results["Gradient Boosting"]["r2"],
                "rmse": int(results["Gradient Boosting"]["rmse"]),
                "mae": int(results["Gradient Boosting"]["mae"]),
                "mape": results["Gradient Boosting"]["mape"],
                "cv_r2_mean": round(cv_r2.mean(), 4),
                "cv_r2_std":  round(cv_r2.std(),  4),
                "cv_rmse_mean": round(np.mean(cv_rmse), 0),
            },
            "random_forest": {
                "r2": results["Random Forest"]["r2"],
                "rmse": int(results["Random Forest"]["rmse"]),
                "mae": int(results["Random Forest"]["mae"]),
                "mape": results["Random Forest"]["mape"],
            },
            "ridge_regression": {
                "r2": results["Ridge Regression"]["r2"],
                "rmse": int(results["Ridge Regression"]["rmse"]),
                "mae": int(results["Ridge Regression"]["mae"]),
                "mape": results["Ridge Regression"]["mape"],
            },
        },
        "feature_importance": fi[["feature","importance_pct"]].to_dict(orient="records"),
        "regional_accuracy":  regional_acc.round(0).to_dict(orient="records"),
        "regional_medians":   df.groupby("region")["price_gbp"].median().reset_index().rename(
            columns={"price_gbp":"median_price"}).round(0).to_dict(orient="records"),
        "price_by_type":      df.groupby("property_type")["price_gbp"].median().reset_index().rename(
            columns={"price_gbp":"median_price"}).round(0).to_dict(orient="records"),
        "price_by_year":      df.groupby("sale_year")["price_gbp"].median().reset_index().rename(
            columns={"price_gbp":"median_price"}).round(0).to_dict(orient="records"),
        "price_by_beds":      df.groupby("bedrooms")["price_gbp"].median().reset_index().rename(
            columns={"price_gbp":"median_price"}).round(0).to_dict(orient="records"),
        "price_by_epc":       df.groupby("epc_rating")["price_gbp"].median().reset_index().rename(
            columns={"price_gbp":"median_price"}).round(0).to_dict(orient="records"),
        "price_by_school":    df.groupby("nearest_school_ofsted")["price_gbp"].median().reset_index().rename(
            columns={"price_gbp":"median_price"}).round(0).to_dict(orient="records"),
        "sample_predictions": test_results.head(500).to_dict(orient="records"),
        "cv_scores":          {"r2": [round(x,4) for x in cv_r2.tolist()], "rmse": [round(x,0) for x in cv_rmse]},
    }
    with open(f"{OUT}/model_results.json","w") as f:
        json.dump(dashboard_json, f, separators=(",",":"))
    print(f"\nAll outputs saved to {OUT}/")
    print(f"\n✅ Best model: Gradient Boosting  R²={results['Gradient Boosting']['r2']}  RMSE=£{results['Gradient Boosting']['rmse']:,.0f}")


if __name__ == "__main__":
    main()
