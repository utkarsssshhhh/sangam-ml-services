"""
Merge water quality tables + train a water quality model (Feature 1) - v2.

CHANGE FROM v1: v1 built its target as an average of the same columns used
as model inputs, so the model was trivially "predicting" a number derived
from itself - that's data leakage, and it's why the R^2 came back at 0.98
(misleadingly high, not a real result).

v2 reframes the task into something genuinely useful and leakage-free:
  INPUT  = fast physical sensor readings (pH, turbidity, temperature,
           conductivity, etc.) - the kind of thing a cheap real-time sensor
           can measure instantly.
  TARGET = a composite quality score built from chemical + biological lab
           parameters (BOD, COD, coliform, heavy metals, etc.) - the kind
           of thing that normally takes lab time/equipment to get.

Demo story: "Our sensor gives an instant estimate of full water quality
(including things that normally need a lab) using only fast physical
readings." That's a real, explainable value proposition, and the
evaluation numbers are now meaningful because input and target come from
genuinely different tables.

OUTPUT:
  data/cleaned/water_quality_merged.csv
  model/water_quality_model.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = Path("data")
CLEANED_DIR = DATA_DIR / "cleaned"
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MERGE_TOLERANCE_DAYS = 30

# Columns that are IDs/codes/coordinates, never real signal - excluded from
# both features and target-building no matter which table they come from.
JUNK_COLS = {
    "slno", "slno_chem", "slno_bio",
    "state_lgd_code", "state_lgd_code_chem", "state_lgd_code_bio",
    "district_lgd_code", "district_lgd_code_chem", "district_lgd_code_bio",
    "latitude", "latitude_chem", "latitude_bio",
    "longitude", "longitude_chem", "longitude_bio",
}

# Lab parameters where a HIGHER value means WORSE water quality (used to
# flip sign when building the composite target score). Edit if your actual
# column names differ from what's listed here - check the printed lab
# column list after running once.
WORSE_IF_HIGH = {
    "biochemical_oxygen_demand_mg_l",
    "chemical_oxygen_demand_mg_l",
    "total_phosphorus_mgp_l",
    "chromium_mg_l",
    "nickel_mg_l",
    "copper_mg_l",
    "zinc_mg_l",
    "nitrate_mg_l",
    "nitrite_n_nitrate_n_mgn_l",
    "total_hardness_mgcaco3_l",
}


def find_date_col(df):
    return next((c for c in df.columns if "date" in c or "acqu" in c), None)


def find_station_col(df):
    for name in ["station", "river", "local_river"]:
        if name in df.columns:
            return name
    return None


def load_category(name):
    path = CLEANED_DIR / f"{name}_cleaned.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run clean_data.py first")
    df = pd.read_csv(path)
    date_col = find_date_col(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    return df, date_col


def merge_nearest(base, base_date_col, other, other_date_col, station_col, suffix):
    merged_parts = []
    for station, base_group in base.groupby(station_col):
        other_group = other[other[station_col] == station] if station_col in other.columns else other
        if other_group.empty:
            merged_parts.append(base_group)
            continue
        other_group = other_group.sort_values(other_date_col)
        m = pd.merge_asof(
            base_group.sort_values(base_date_col),
            other_group,
            left_on=base_date_col,
            right_on=other_date_col,
            direction="nearest",
            tolerance=pd.Timedelta(days=MERGE_TOLERANCE_DAYS),
            suffixes=("", suffix),
        )
        merged_parts.append(m)
    return pd.concat(merged_parts, ignore_index=True)


def main():
    print("Loading water quality categories...")
    physical, phys_date = load_category("physical")
    chemical, chem_date = load_category("chemical")
    biological, bio_date = load_category("biological")

    station_col = find_station_col(physical)
    if station_col is None:
        raise ValueError("No station column found in physical data - can't merge per-station")

    # Remember which columns originally belonged to physical vs chemical/bio,
    # BEFORE merging, so we can split features vs target correctly after.
    physical_numeric = set(physical.select_dtypes(include="number").columns) - JUNK_COLS
    chemical_numeric = set(chemical.select_dtypes(include="number").columns) - JUNK_COLS
    biological_numeric = set(biological.select_dtypes(include="number").columns) - JUNK_COLS

    print(f"  physical: {len(physical)} rows, chemical: {len(chemical)} rows, "
          f"biological: {len(biological)} rows")

    print(f"\nMerging on nearest date within {MERGE_TOLERANCE_DAYS} days, per station...")
    merged = merge_nearest(physical, phys_date, chemical, chem_date, station_col, "_chem")
    merged = merge_nearest(merged, phys_date, biological, bio_date, station_col, "_bio")
    print(f"  Merged dataset: {len(merged)} rows")

    out_path = CLEANED_DIR / "water_quality_merged.csv"
    merged.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

    # ---- Split columns: physical = features (X), chemical+biological = target source ----
    # Column names may have picked up suffixes (_chem/_bio) during merge if there
    # were name clashes - check both original and suffixed versions.
    feature_cols = [c for c in merged.columns if c in physical_numeric]
    lab_cols = [
        c for c in merged.columns
        if (c in chemical_numeric or c in biological_numeric
            or c.replace("_chem", "") in chemical_numeric
            or c.replace("_bio", "") in biological_numeric)
        and c not in JUNK_COLS
    ]

    print(f"\nPhysical (feature) columns ({len(feature_cols)}): {feature_cols}")
    print(f"Lab (target-source) columns ({len(lab_cols)}): {lab_cols}")

    if len(feature_cols) < 2 or len(lab_cols) < 2:
        print("\n! Too few columns detected on one side - open water_quality_merged.csv "
              "and check column names manually, then adjust this script's column "
              "detection logic accordingly.")
        return

    X = merged[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    lab_df = merged[lab_cols].copy()
    lab_df = lab_df.fillna(lab_df.median(numeric_only=True))

    # Composite target from LAB columns only (no overlap with X -> no leakage)
    z = (lab_df - lab_df.mean()) / lab_df.std(ddof=0).replace(0, 1)
    for col in lab_df.columns:
        if col in WORSE_IF_HIGH:
            z[col] = -z[col]
    y = 50 + 10 * z.mean(axis=1)  # rescaled composite score, ~0-100ish range

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTraining RandomForestRegressor on {len(X_train)} rows "
          f"(testing on {len(X_test)})...")
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  Test MAE: {mae:.2f}")
    print(f"  Test R^2: {r2:.2f}")
    print("  (These numbers are now meaningful - features and target come from "
          "genuinely different tables, so this reflects real predictive signal, "
          "not the model reconstructing its own target.)")
    if len(X_test) < 10:
        print("  Note: small test set (this dataset is small overall), so treat "
              "these as rough indicators, not precise accuracy claims.")

    model_path = MODEL_DIR / "water_quality_model.pkl"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"\nSaved model -> {model_path}")
    print("Load it later with: joblib.load('model/water_quality_model.pkl')")


if __name__ == "__main__":
    main()