
import pandas as pd
from pathlib import Path

# ---- CONFIG ----

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "cleaned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Exact station names for water quality files (physical/chemical/biological).
# Confirmed present in your data. Add more exact names here if you want to
# widen coverage - just copy them exactly as printed in the station list.
TARGET_STATIONS = [
    "GANGA AT HARIDWAR D/S UPPER GANGA CANAL D/S BALKUMARI MANDIR AJEETPUR HARIDWAR",
    "GANGA AT RISHIKESH U/S",
]

# Rainfall stations use place names, not river names. Bahadarabad is a town
# right next to Haridwar, so it's included by default. Check the printed
# station list after your first run and add/remove names here as needed -
# look for towns close to Haridwar/Roorkee (Bahadarabad, Roorkee itself if
# it appears, Raiwala, etc.).
RAINFALL_STATIONS = [
    "Bahadarabad",
]

# Exact filenames as you have them - edit if yours differ
FILES = {
    "physical": [
        "swq_manual_physical_parameters_cpcb_uk_1961_2020.csv",
        "swq_physical_parameter_manual_cpcb_uk_2021_2025.csv",
    ],
    "chemical": [
        "swq_manual_chemical_parameters_cpcb_uk_1961_2020.csv",
        "swq_chemical_parameter_manual_cpcb_uk_2021_2025.csv",
    ],
    "biological": [
        "swq_manual_biological_parameters_cpcb_uk_1961_2020.csv",
        "swq_biological_parameter_manual_cpcb_uk_2021_2025.csv",
    ],
    "rainfall": [
        "rainfall_tel_hr_uttarakhand_uk_1991_2020.csv",
        "rainfall_tel_hr_uttarakhand_uk_2021_2025.csv",
        "rainfall_tel_hr_uttarakhand_uk_2026_2030.csv",
    ],
}

# Which categories use TARGET_STATIONS vs RAINFALL_STATIONS
STATION_LIST_FOR = {
    "physical": TARGET_STATIONS,
    "chemical": TARGET_STATIONS,
    "biological": TARGET_STATIONS,
    "rainfall": RAINFALL_STATIONS,
}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def find_date_column(df: pd.DataFrame) -> str | None:
    candidates = [c for c in df.columns if "date" in c or "acqu" in c]
    return candidates[0] if candidates else None


def find_station_column(df: pd.DataFrame) -> str | None:
    for name in ["station", "river", "local_river"]:
        if name in df.columns:
            return name
    return None


def load_and_concat(file_list: list[str]) -> pd.DataFrame:
    frames = []
    for fname in file_list:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  [skip] {fname} not found, skipping")
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print(f"  [error] couldn't read {fname}: {e}")
            continue
        df = clean_columns(df)
        if df.empty:
            print(f"  [skip] {fname} is empty (likely a future date range)")
            continue
        frames.append(df)
        print(f"  [ok] {fname}: {len(df)} rows")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def process_category(name: str, file_list: list[str]) -> pd.DataFrame:
    print(f"\n--- {name} ---")
    df = load_and_concat(file_list)
    if df.empty:
        print(f"  No usable data found for {name}.")
        return df

    date_col = find_date_column(df)
    if date_col:
        # dayfirst=True since these files use DD-MM-YYYY format
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        n_bad = df[date_col].isna().sum()
        if n_bad:
            print(f"  Note: {n_bad} rows had unparseable dates (kept as NaT)")
    else:
        print("  Warning: no obvious date column found - check manually")

    station_col = find_station_column(df)
    target_list = STATION_LIST_FOR.get(name, [])

    if station_col:
        uniques = sorted(df[station_col].dropna().unique().tolist())
        print(f"  Stations/rivers found ({len(uniques)} unique): {uniques[:25]}"
              + (" ..." if len(uniques) > 25 else ""))

        # EXACT match (case-insensitive, trimmed) instead of substring matching
        target_upper = {s.strip().upper() for s in target_list}
        mask = df[station_col].astype(str).str.strip().str.upper().isin(target_upper)
        filtered = df[mask]
        print(f"  Rows matching TARGET_STATIONS (exact): {len(filtered)} / {len(df)}")
        if filtered.empty:
            print("  ! No exact matches - copy the exact name(s) you want from the "
                  "printed station list above into TARGET_STATIONS/RAINFALL_STATIONS "
                  "at the top of this script, then re-run.")
        df = filtered
    else:
        print("  Warning: no station/river column found - saving all rows unfiltered")

    return df


def main():
    results = {}
    for category, file_list in FILES.items():
        cleaned = process_category(category, file_list)
        results[category] = cleaned
        if not cleaned.empty:
            out_path = OUT_DIR / f"{category}_cleaned.csv"
            cleaned.to_csv(out_path, index=False)
            print(f"  Saved -> {out_path}")

    print("\nDone. Check data/cleaned/ for your filtered files.")
    print("If any category showed 0 matching rows, open the printed station "
          "list above, copy the exact name(s) you want, and update "
          "TARGET_STATIONS or RAINFALL_STATIONS at the top of this script.")


if __name__ == "__main__":
    main()