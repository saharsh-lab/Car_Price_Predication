"""Train and save the high-accuracy car price prediction ensemble model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

from preprocess import (
    ALL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_MAPS,
    compute_derived_features,
    encode_input,
    prepare_car_data,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Cardetails.csv"
MODEL_JOBLIB_PATH = BASE_DIR / "model.joblib"
METADATA_PATH = BASE_DIR / "public" / "metadata.json"


def build_metadata(car_data: pd.DataFrame, metrics: dict) -> dict:
    """Construct metadata with brand statistics, options, and model performance metrics."""
    brands = sorted(car_data["name"].unique().tolist())
    metadata = {
        "metrics": metrics,
        "brands": brands,
        "options": {
            "fuel": sorted(car_data["fuel"].unique().tolist()),
            "seller_type": sorted(car_data["seller_type"].unique().tolist()),
            "transmission": sorted(car_data["transmission"].unique().tolist()),
            "owner": sorted(car_data["owner"].unique().tolist()),
        },
        "ranges": {
            "year": [int(car_data["year"].min()), int(car_data["year"].max())],
            "km_driven": [int(car_data["km_driven"].min()), int(car_data["km_driven"].max())],
            "mileage": [float(car_data["mileage"].min()), float(car_data["mileage"].max())],
            "engine": [int(car_data["engine"].min()), int(car_data["engine"].max())],
            "max_power": [float(car_data["max_power"].min()), float(car_data["max_power"].max())],
            "seats": [int(car_data["seats"].min()), int(car_data["seats"].max())],
        },
        "brand_defaults": {},
        "stats": {
            "records": int(len(car_data)),
            "brands": len(brands),
            "years": [int(car_data["year"].min()), int(car_data["year"].max())],
        },
        "market": build_market_insights(car_data),
    }

    for brand in brands:
        subset = car_data[car_data["name"] == brand]
        metadata["brand_defaults"][brand] = {
            "year": int(subset["year"].median()),
            "km_driven": int(subset["km_driven"].median()),
            "fuel": subset["fuel"].mode().iloc[0],
            "seller_type": subset["seller_type"].mode().iloc[0],
            "transmission": subset["transmission"].mode().iloc[0],
            "owner": subset["owner"].mode().iloc[0],
            "mileage": round(float(subset["mileage"].median()), 1),
            "engine": int(subset["engine"].median()),
            "max_power": round(float(subset["max_power"].median()), 1),
            "seats": int(subset["seats"].median()),
        }

    return metadata


def build_market_insights(car_data: pd.DataFrame) -> dict:
    """Summarise observed listing prices per brand for the web app's market charts."""
    brand_year = {}
    for (brand, year), group in car_data.groupby(["name", "year"]):
        brand_year.setdefault(brand, []).append(
            [int(year), int(group["selling_price"].median()), int(len(group))]
        )

    brand_median = {
        brand: [int(group["selling_price"].median()), int(len(group))]
        for brand, group in car_data.groupby("name")
    }

    return {"brand_year": brand_year, "brand_median": brand_median}


def export_metadata_only() -> None:
    """Regenerate public/metadata.json from the dataset without retraining the model."""
    metrics = json.loads(METADATA_PATH.read_text()).get("metrics", {}) if METADATA_PATH.exists() else {}

    metadata = build_metadata(prepare_car_data(DATA_PATH), metrics)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to {METADATA_PATH}")


def build_ensemble_model(cat_mask: list) -> TransformedTargetRegressor:
    """Build a balanced voting ensemble with log-transformed target for non-negative, high-accuracy predictions."""
    hgb = HistGradientBoostingRegressor(
        random_state=42,
        max_iter=300,
        learning_rate=0.07,
        l2_regularization=1.0,
        categorical_features=cat_mask,
        min_samples_leaf=15,
    )

    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.07,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
    )

    rf = RandomForestRegressor(
        n_estimators=60,
        max_depth=16,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )

    voting = VotingRegressor(
        estimators=[("hgb", hgb), ("gbr", gbr), ("rf", rf)],
        weights=[0.45, 0.35, 0.20],
    )

    return TransformedTargetRegressor(
        regressor=voting,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def train_and_save() -> None:
    print("=" * 60)
    print("Training Car Price Prediction High-Accuracy Model")
    print("=" * 60)

    raw_data = prepare_car_data(DATA_PATH)
    print(f"Loaded and cleaned {len(raw_data):,} car records.")

    enriched_data = compute_derived_features(raw_data)

    car_data = enriched_data.copy()
    for column, mapping in LABEL_MAPS.items():
        car_data[column] = car_data[column].map(mapping)
    car_data = car_data.dropna()

    x = car_data[ALL_FEATURE_COLUMNS]
    y = car_data["selling_price"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    cat_mask = [col in LABEL_MAPS for col in ALL_FEATURE_COLUMNS]

    model = build_ensemble_model(cat_mask)
    print("Fitting ensemble regressor (HGB + GBR + RF with log1p target transform)...")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    predictions = np.maximum(predictions, 0.0)

    r2 = float(r2_score(y_test, predictions))
    mae = float(mean_absolute_error(y_test, predictions))
    mape = float(mean_absolute_percentage_error(y_test, predictions))

    metrics = {
        "r2": round(r2, 4),
        "mae": round(mae),
        "mape": round(mape * 100, 2),
        "accuracy_pct": round(r2 * 100, 1),
    }

    print("\n--- Model Evaluation Results ---")
    print(f"R² Score:            {metrics['r2']:.4f} ({metrics['accuracy_pct']}%)")
    print(f"Mean Absolute Error: ₹{metrics['mae']:,}")
    print(f"Mean Abs Pct Error:  {metrics['mape']:.2f}%")

    # Evaluate by price tiers
    eval_df = pd.DataFrame({"actual": y_test, "pred": predictions})
    budget = eval_df[eval_df["actual"] < 500000]
    mid = eval_df[(eval_df["actual"] >= 500000) & (eval_df["actual"] < 1500000)]
    luxury = eval_df[eval_df["actual"] >= 1500000]

    print("\nPrice Tier Performance:")
    print(f"  • Budget (< ₹5L, n={len(budget)}):       MAE = ₹{budget['actual'].sub(budget['pred']).abs().mean():,.0f}, MAPE = {budget['actual'].sub(budget['pred']).abs().div(budget['actual']).mean():.1%}")
    print(f"  • Mid-range (₹5L-₹15L, n={len(mid)}):   MAE = ₹{mid['actual'].sub(mid['pred']).abs().mean():,.0f}, MAPE = {mid['actual'].sub(mid['pred']).abs().div(mid['actual']).mean():.1%}")
    print(f"  • Luxury (> ₹15L, n={len(luxury)}):      MAE = ₹{luxury['actual'].sub(luxury['pred']).abs().mean():,.0f}, MAPE = {luxury['actual'].sub(luxury['pred']).abs().div(luxury['actual']).mean():.1%}")

    # Build and save metadata
    metadata = build_metadata(raw_data, metrics)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved metadata to {METADATA_PATH}")

    # Save compressed model for api/predict.py
    joblib.dump(model, MODEL_JOBLIB_PATH, compress=3)
    print(f"Saved compressed model to {MODEL_JOBLIB_PATH} ({MODEL_JOBLIB_PATH.stat().st_size / 1e6:.2f} MB)")

    # Sample predictions across tiers
    test_samples = [
        ("Maruti Alto (Budget)", ["Maruti", 2012, 60000, "Petrol", "Individual", "Manual", "First Owner", 22.7, 796.0, 47.0, 5.0]),
        ("Hyundai Creta (Mid-range)", ["Hyundai", 2018, 45000, "Diesel", "Dealer", "Manual", "First Owner", 19.6, 1582.0, 126.0, 5.0]),
        ("BMW 5 Series (Luxury)", ["BMW", 2017, 35000, "Diesel", "Dealer", "Automatic", "First Owner", 18.1, 1995.0, 190.0, 5.0]),
    ]
    print("\nSample Predictions:")
    for label, row in test_samples:
        encoded_sample = encode_input(pd.DataFrame([row], columns=FEATURE_COLUMNS))
        pred_val = float(model.predict(encoded_sample)[0])
        print(f"  • {label:28s} -> ₹{pred_val:,.0f}")

    print("=" * 60)
    print("Training and artifact export completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if "--metadata-only" in sys.argv:
        export_metadata_only()
    else:
        train_and_save()
