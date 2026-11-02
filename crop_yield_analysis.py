"""
Climate-Aware Crop Yield Prediction & Model Benchmarking
--------------------------------------------------------
This starter project uses an illustrative synthetic dataset by default.
Do not treat its metrics as evidence of real agricultural performance.
For a real portfolio result, replace sample_crop_yield_data.csv with a credible
public dataset and document the source, units, and limitations.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "sample_crop_yield_data.csv"
OUTPUT = ROOT / "outputs"

FEATURES = ["rainfall_mm", "mean_temp_c", "soil_moisture_pct", "fertilizer_kg_ha"]
TARGET = "yield_tonnes_ha"

def make_illustrative_data(path: Path, rows: int = 240) -> None:
    """Create reproducible demo data; this is synthetic, not real farm data."""
    rng = np.random.default_rng(42)
    rainfall = rng.uniform(350, 1300, rows)
    temp = rng.uniform(18, 38, rows)
    moisture = np.clip(rainfall / 16 + rng.normal(0, 9, rows), 10, 90)
    fertilizer = rng.uniform(30, 180, rows)
    noise = rng.normal(0, 0.35, rows)
    # Illustrative relationship only; not a scientific crop-yield model.
    yield_value = (
        1.0 + 0.0022 * rainfall - 0.055 * np.abs(temp - 27)
        + 0.012 * moisture + 0.006 * fertilizer + noise
    )
    df = pd.DataFrame({
        "rainfall_mm": rainfall.round(1),
        "mean_temp_c": temp.round(1),
        "soil_moisture_pct": moisture.round(1),
        "fertilizer_kg_ha": fertilizer.round(1),
        TARGET: np.maximum(yield_value, 0.1).round(3),
    })
    df.to_csv(path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="CSV path containing FEATURES and target column")
    args = parser.parse_args()
    if not args.data.exists():
        make_illustrative_data(args.data)

    df = pd.read_csv(args.data)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df[required].replace([np.inf, -np.inf], np.nan)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    if len(df) < 20:
        raise ValueError("Please provide at least 20 rows with valid target values.")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Dummy baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42
        ),
    }
    rows = []
    best_model = None
    best_rmse = float("inf")
    best_name = None
    best_predictions = None

    OUTPUT.mkdir(exist_ok=True)
    for name, estimator in models.items():
        model = make_pipeline(SimpleImputer(strategy="median"), estimator)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": rmse,
            "R2": r2_score(y_test, pred),
        })
        if rmse < best_rmse:
            best_rmse, best_model, best_name, best_predictions = rmse, model, name, pred

    results = pd.DataFrame(rows).sort_values("RMSE")
    results.to_csv(OUTPUT / "model_comparison.csv", index=False)
    pd.DataFrame({
        "actual_yield_tonnes_ha": y_test.to_numpy(),
        "predicted_yield_tonnes_ha": best_predictions,
    }).to_csv(OUTPUT / "best_model_predictions.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, best_predictions, alpha=0.75)
    low = min(float(y_test.min()), float(np.min(best_predictions)))
    high = max(float(y_test.max()), float(np.max(best_predictions)))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual yield (tonnes/ha)")
    plt.ylabel("Predicted yield (tonnes/ha)")
    plt.title(f"Actual vs Predicted — {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT / "actual_vs_predicted.png", dpi=160)
    plt.close()

    print("\nDATASET NOTE: Synthetic illustrative data" if args.data == DEFAULT_DATA else f"\nDataset: {args.data}")
    print("\nModel comparison (lower MAE/RMSE is better; higher R² is generally better):")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nLowest test RMSE in this run: {best_name}")
    print(f"Outputs saved to: {OUTPUT.resolve()}")
    print("Do not report synthetic-data metrics as real-world agricultural results.")

if __name__ == "__main__":
    main()
