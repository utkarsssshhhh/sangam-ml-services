"""
Synthetic water level generator for the flood-risk feature (Feature 3) - v2.

CHANGE FROM v1: RAIN_SENSITIVITY is now auto-calibrated from the actual
rainfall data instead of a fixed constant. v1 used a fixed sensitivity that
worked for one rainfall dataset but produced "all normal, no risk events"
when re-run against a smaller/different station's rainfall (lower typical
values -> bumps too small to ever cross the thresholds). Auto-calibrating
means this keeps working no matter which station's rainfall you feed it.

HOW CALIBRATION WORKS:
We look at the 90th percentile daily rainfall in your data (a "notably
heavy rain day", not the single all-time max) and set RAIN_SENSITIVITY so
that a day like that pushes the water level bump close to the gap between
baseline and the danger level. That way: typical days stay near baseline
(normal), moderately heavy rain reaches into warning territory sometimes,
and only the heaviest rain (and rain stacked over consecutive days, thanks
to the decay/carry-over) pushes into danger. You can still nudge
CALIBRATION_TARGET_FRACTION below if the resulting mix still looks off.

OUTPUT:
  data/cleaned/water_level_synthetic.csv
  Columns: date, rainfall_mm, water_level_m, risk_label
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
CLEANED_DIR = DATA_DIR / "cleaned"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

RAINFALL_FILE = CLEANED_DIR / "rainfall_cleaned.csv"
OUT_FILE = CLEANED_DIR / "water_level_synthetic.csv"

# ---- Real Haridwar thresholds (from CWC) ----
WARNING_LEVEL = 293.0
DANGER_LEVEL = 294.0
HFL = 296.3

# ---- Simulation parameters ----
BASELINE_LEVEL = 291.5      # normal resting level, comfortably below warning
DECAY_RATE = 0.55         # how much of yesterday's rain-bump carries into today (0-1)
NOISE_STD = 0.03            # random day-to-day noise, in meters
RANDOM_SEED = 42

# A "90th percentile rain day" should push the level to roughly this
# fraction of the way from baseline to the danger level. 0.9 means a heavy
# rain day alone gets close to danger; consecutive heavy days (via decay
# carry-over) will push past it. Lower this (e.g. 0.6) for fewer danger
# events, raise it for more.
CALIBRATION_TARGET_FRACTION = 0.45


def load_rainfall() -> pd.DataFrame:
    if not RAINFALL_FILE.exists():
        raise FileNotFoundError(
            f"{RAINFALL_FILE} not found - run clean_data.py first to produce it."
        )
    df = pd.read_csv(RAINFALL_FILE)

    date_col = next((c for c in df.columns if "date" in c or "acqu" in c), None)
    if date_col is None:
        raise ValueError("Couldn't find a date column in rainfall_cleaned.csv")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    rain_col = next(
        (c for c in df.columns if "rain" in c or "mm" in c or "value" in c), None
    )
    if rain_col is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("latitude", "longitude", "slno")]
        if not numeric_cols:
            raise ValueError("Couldn't find a rainfall value column - inspect rainfall_cleaned.csv manually")
        rain_col = numeric_cols[0]

    print(f"Using date column '{date_col}' and rainfall column '{rain_col}'")

    daily = df.groupby(df[date_col].dt.date)[rain_col].mean().reset_index()
    daily.columns = ["date", "rainfall_mm"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["rainfall_mm"] = daily["rainfall_mm"].fillna(0).clip(lower=0)

    return daily


def calibrate_sensitivity(rainfall_mm: pd.Series) -> float:
    """Pick RAIN_SENSITIVITY based on the actual spread of this rainfall data."""
    p90 = rainfall_mm.quantile(0.90)
    if p90 <= 0:
        print("  Warning: 90th percentile rainfall is 0 - data may be very sparse/dry. "
              "Using a fallback sensitivity; expect mostly 'normal' results.")
        return 0.05
    target_bump = (DANGER_LEVEL - BASELINE_LEVEL) * CALIBRATION_TARGET_FRACTION
    sensitivity = target_bump / p90
    print(f"  90th percentile daily rainfall: {p90:.2f} mm -> "
          f"auto-calibrated RAIN_SENSITIVITY = {sensitivity:.5f}")
    return sensitivity


def label_risk(level: float) -> str:
    if level >= DANGER_LEVEL:
        return "danger"
    elif level >= WARNING_LEVEL:
        return "warning"
    else:
        return "normal"


def generate_water_level(daily_rainfall: pd.DataFrame, sensitivity: float) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    levels = []
    bump = 0.0
    for rain in daily_rainfall["rainfall_mm"]:
        bump = bump * DECAY_RATE + rain * sensitivity
        noise = rng.normal(0, NOISE_STD)
        level = BASELINE_LEVEL + bump + noise
        levels.append(level)

    out = daily_rainfall.copy()
    out["water_level_m"] = levels
    out["risk_label"] = out["water_level_m"].apply(label_risk)
    return out


def main():
    print("Loading real rainfall data...")
    daily_rainfall = load_rainfall()
    print(f"  {len(daily_rainfall)} days of rainfall loaded "
          f"({daily_rainfall['date'].min().date()} to {daily_rainfall['date'].max().date()})")

    print("Calibrating sensitivity to this rainfall data...")
    sensitivity = calibrate_sensitivity(daily_rainfall["rainfall_mm"])

    print("Generating synthetic water level series...")
    result = generate_water_level(daily_rainfall, sensitivity)

    print("\nRisk label distribution:")
    print(result["risk_label"].value_counts())

    result.to_csv(OUT_FILE, index=False)
    print(f"\nSaved -> {OUT_FILE}")
    print(f"Columns: {list(result.columns)}")
    print(f"\nThresholds used: warning={WARNING_LEVEL}m, danger={DANGER_LEVEL}m, HFL={HFL}m")
    print("If the mix still looks off (e.g. everything normal, or everything danger), "
          "adjust CALIBRATION_TARGET_FRACTION near the top (lower = fewer risk events, "
          "higher = more) and re-run.")


if __name__ == "__main__":
    main()