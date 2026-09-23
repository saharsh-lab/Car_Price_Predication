"""Shared preprocessing helpers for training and inference."""

from pathlib import Path
from typing import Dict, List, Set
import numpy as np
import pandas as pd

OWNER_MAP = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
    "Test Drive Car": 5,
}
FUEL_MAP = {"Diesel": 1, "Petrol": 2, "LPG": 3, "CNG": 4}
SELLER_MAP = {"Individual": 1, "Dealer": 2, "Trustmark Dealer": 3}
TRANSMISSION_MAP = {"Manual": 1, "Automatic": 2}
NAME_MAP = {
    "Maruti": 1,
    "Skoda": 2,
    "Honda": 3,
    "Hyundai": 4,
    "Toyota": 5,
    "Ford": 6,
    "Renault": 7,
    "Mahindra": 8,
    "Tata": 9,
    "Chevrolet": 10,
    "Datsun": 11,
    "Jeep": 12,
    "Mercedes-Benz": 13,
    "Mitsubishi": 14,
    "Audi": 15,
    "Volkswagen": 16,
    "BMW": 17,
    "Nissan": 18,
    "Lexus": 19,
    "Jaguar": 20,
    "Land": 21,
    "MG": 22,
    "Volvo": 23,
    "Daewoo": 24,
    "Kia": 25,
    "Fiat": 26,
    "Force": 27,
    "Ambassador": 28,
    "Ashok": 29,
    "Isuzu": 30,
    "Opel": 31,
}

LABEL_MAPS = {
    "name": NAME_MAP,
    "fuel": FUEL_MAP,
    "seller_type": SELLER_MAP,
    "transmission": TRANSMISSION_MAP,
    "owner": OWNER_MAP,
}

FEATURE_COLUMNS = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "seats",
]

DERIVED_FEATURE_COLUMNS = [
    "car_age",
    "km_per_year",
    "power_to_engine",
    "is_luxury",
]

ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS

LUXURY_BRANDS: Set[str] = {
    "Audi",
    "BMW",
    "Jaguar",
    "Land",
    "Lexus",
    "Mercedes-Benz",
    "Volvo",
}

REFERENCE_YEAR = 2024


def get_brand_name(car_name: str) -> str:
    """Extract brand name from full car name string."""
    return str(car_name).split(" ")[0].strip()


def clean_numeric(value) -> float:
    """Clean numeric values formatted with units (e.g. '23.4 kmpl' -> 23.4)."""
    if pd.isna(value):
        return np.nan
    val_str = str(value).split(" ")[0].strip()
    if val_str in ("", "nan", "null"):
        return np.nan
    try:
        return float(val_str)
    except ValueError:
        return np.nan


def prepare_car_data(csv_path) -> pd.DataFrame:
    """Load, clean, and impute car data from raw CSV."""
    car_data = pd.read_csv(csv_path)
    if "torque" in car_data.columns:
        car_data.drop(columns=["torque"], inplace=True)
    car_data.dropna(inplace=True)
    car_data.drop_duplicates(inplace=True)

    car_data["name"] = car_data["name"].apply(get_brand_name)
    for column in ("mileage", "max_power", "engine"):
        car_data[column] = car_data[column].apply(clean_numeric)

    car_data = car_data.dropna()

    # Impute missing/zero mileage with brand median or global median
    zero_mileage = car_data["mileage"] <= 0
    if zero_mileage.any():
        car_data.loc[zero_mileage, "mileage"] = np.nan
        brand_medians = car_data.groupby("name")["mileage"].transform("median")
        car_data["mileage"] = car_data["mileage"].fillna(brand_medians)
        car_data["mileage"] = car_data["mileage"].fillna(car_data["mileage"].median())

    return car_data.dropna()


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate engineered domain features."""
    enriched = df.copy()

    year_val = pd.to_numeric(enriched["year"], errors="coerce").fillna(REFERENCE_YEAR - 5)
    km_val = pd.to_numeric(enriched["km_driven"], errors="coerce").fillna(50000)
    engine_val = pd.to_numeric(enriched["engine"], errors="coerce").fillna(1200)
    power_val = pd.to_numeric(enriched["max_power"], errors="coerce").fillna(80)

    car_age = np.maximum(REFERENCE_YEAR - year_val, 0.0)
    km_per_year = km_val / (car_age + 0.5)
    power_to_engine = power_val / np.maximum(engine_val, 100.0)

    # If brand name is still string, check luxury set; if already integer, map from inverse NAME_MAP
    if enriched["name"].dtype == object or isinstance(enriched["name"].iloc[0], str):
        is_luxury = enriched["name"].astype(str).isin(LUXURY_BRANDS).astype(float)
    else:
        inv_map = {v: k for k, v in NAME_MAP.items()}
        brands = enriched["name"].map(inv_map)
        is_luxury = brands.isin(LUXURY_BRANDS).astype(float)

    enriched["car_age"] = car_age
    enriched["km_per_year"] = km_per_year
    enriched["power_to_engine"] = power_to_engine
    enriched["is_luxury"] = is_luxury

    return enriched


def encode_input(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess and encode raw input DataFrame for model inference."""
    encoded = df.copy()

    # Clean numeric fields if any were passed as strings
    for col in ("mileage", "max_power", "engine", "year", "km_driven", "seats"):
        if col in encoded.columns:
            encoded[col] = pd.to_numeric(
                encoded[col].apply(lambda v: str(v).split(" ")[0].strip()),
                errors="coerce",
            )

    # Impute mileage <= 0 with plausible default (18.0)
    if "mileage" in encoded.columns:
        encoded.loc[encoded["mileage"] <= 0, "mileage"] = 18.0

    # Add derived features before mapping categorical names to integers
    encoded = compute_derived_features(encoded)

    # Map categorical features using LABEL_MAPS
    for column, mapping in LABEL_MAPS.items():
        if column in encoded.columns:
            if encoded[column].dtype == object or isinstance(encoded[column].iloc[0], str):
                encoded[column] = encoded[column].map(mapping).fillna(1)

    return encoded[ALL_FEATURE_COLUMNS].astype(float)
