"""
ml-services API (Feature 1: water quality, Feature 3: flood risk).

Serves two kinds of endpoints for each feature:
  - /predict   : takes live sensor-style input, returns a prediction
  - /replay    : returns the "next" row from your historical dataset each
                 time it's called, so your React dashboard can poll this on
                 a timer and it'll look like a live feed streaming in.

RUN:
  pip install flask flask-cors pandas joblib scikit-learn
  python app.py
  (starts on http://localhost:5001)

React side: just fetch() these endpoints every few seconds with setInterval.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)
CORS(app)  # allow your React frontend (different port) to call this API

DATA_DIR = Path("data/cleaned")
MODEL_DIR = Path("model")

# ---- Real Haridwar thresholds (from CWC) - same ones used in generate_water_level.py ----
WARNING_LEVEL = 293.0
DANGER_LEVEL = 294.0

# ---------------------------------------------------------------------------
# Load everything once at startup
# ---------------------------------------------------------------------------

print("Loading water quality model...")
wq_bundle = joblib.load(MODEL_DIR / "water_quality_model.pkl")
wq_model = wq_bundle["model"]
wq_feature_cols = wq_bundle["feature_cols"]
print(f"  Water quality model expects features: {wq_feature_cols}")

print("Loading datasets for replay...")
wq_data = pd.read_csv(DATA_DIR / "water_quality_merged.csv")
flood_data = pd.read_csv(DATA_DIR / "water_level_synthetic.csv")
print(f"  water_quality_merged: {len(wq_data)} rows")
print(f"  water_level_synthetic: {len(flood_data)} rows")

# Replay position trackers (simple in-memory counters - fine for a hackathon
# single-user demo; resets if you restart the server)
_wq_index = {"i": 0}
_flood_index = {"i": 0}


# ---------------------------------------------------------------------------
# Feature 1: Water quality
# ---------------------------------------------------------------------------

@app.route("/api/water-quality/predict", methods=["POST"])
def predict_water_quality():
    """
    Body: JSON with the physical sensor readings, e.g.
    {
      "electric_conductivity_s_cm": 210,
      "colour": 0,
      "odour": 0,
      "temperature_c": 18.5,
      "total_solids_mg_l": 150
    }
    Missing fields are filled with the training data's median.
    """
    payload = request.get_json(force=True) or {}
    row = {}
    for col in wq_feature_cols:
        row[col] = payload.get(col, wq_data[col].median() if col in wq_data.columns else 0)
    X = pd.DataFrame([row])[wq_feature_cols]
    score = float(wq_model.predict(X)[0])
    return jsonify({
        "quality_score": round(score, 2),
        "inputs_used": row,
    })


@app.route("/api/water-quality/replay", methods=["GET"])
def replay_water_quality():
    """Returns the next historical row each time it's called, cycling around."""
    i = _wq_index["i"] % len(wq_data)
    row = wq_data.iloc[i]

    # Build a prediction from this row's physical columns too, so the
    # dashboard can show "predicted score" alongside the historical reading
    filled_row = row[wq_feature_cols].fillna(wq_data[wq_feature_cols].median())
    X = pd.DataFrame([filled_row])
    score = float(wq_model.predict(X)[0])

    _wq_index["i"] += 1
    return jsonify({
        "index": i,
        "date": str(row.get("data_acquisition_time", row.get("date", ""))),
        "physical_readings": {c: round(float(filled_row[c]), 2) for c in wq_feature_cols},
        "predicted_quality_score": round(score, 2),
    })


# ---------------------------------------------------------------------------
# Feature 3: Flood risk
# ---------------------------------------------------------------------------

def label_risk(level: float) -> str:
    if level >= DANGER_LEVEL:
        return "danger"
    elif level >= WARNING_LEVEL:
        return "warning"
    return "normal"


@app.route("/api/flood-risk/predict", methods=["POST"])
def predict_flood_risk():
    """
    Body: { "water_level_m": 293.4 }
    Returns risk label based on the real CWC thresholds.
    """
    payload = request.get_json(force=True) or {}
    level = float(payload.get("water_level_m", 0))
    return jsonify({
        "water_level_m": level,
        "risk_label": label_risk(level),
        "thresholds": {"warning": WARNING_LEVEL, "danger": DANGER_LEVEL},
    })


@app.route("/api/flood-risk/replay", methods=["GET"])
def replay_flood_risk():
    """Returns the next historical (synthetic) row each time it's called."""
    i = _flood_index["i"] % len(flood_data)
    row = flood_data.iloc[i]
    _flood_index["i"] += 1
    return jsonify({
        "index": i,
        "date": str(row["date"]),
        "rainfall_mm": float(row["rainfall_mm"]),
        "water_level_m": round(float(row["water_level_m"]), 2),
        "risk_label": row["risk_label"],
        "thresholds": {"warning": WARNING_LEVEL, "danger": DANGER_LEVEL},
    })


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5001)