import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import joblib
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent

LABEL_MAPS = {
    "name": {
        "Maruti": 1, "Skoda": 2, "Honda": 3, "Hyundai": 4, "Toyota": 5,
        "Ford": 6, "Renault": 7, "Mahindra": 8, "Tata": 9, "Chevrolet": 10,
        "Datsun": 11, "Jeep": 12, "Mercedes-Benz": 13, "Mitsubishi": 14,
        "Audi": 15, "Volkswagen": 16, "BMW": 17, "Nissan": 18, "Lexus": 19,
        "Jaguar": 20, "Land": 21, "MG": 22, "Volvo": 23, "Daewoo": 24,
        "Kia": 25, "Fiat": 26, "Force": 27, "Ambassador": 28, "Ashok": 29,
        "Isuzu": 30, "Opel": 31,
    },
    "fuel": {"Diesel": 1, "Petrol": 2, "LPG": 3, "CNG": 4},
    "seller_type": {"Individual": 1, "Dealer": 2, "Trustmark Dealer": 3},
    "transmission": {"Manual": 1, "Automatic": 2},
    "owner": {
        "First Owner": 1, "Second Owner": 2, "Third Owner": 3,
        "Fourth & Above Owner": 4, "Test Drive Car": 5,
    },
}

LUXURY_BRANDS = {"Audi", "BMW", "Jaguar", "Land", "Lexus", "Mercedes-Benz", "Volvo"}

# Column positions in the encoded feature row (see encode_request_payload)
YEAR_IDX, KM_IDX = 1, 2
AGE_IDX, KM_PER_YEAR_IDX = 11, 12
REFERENCE_YEAR = 2024.0
DATASET_YEAR_RANGE = (1994, 2020)


def format_inr(amount: float) -> str:
    """Format rupees with Indian digit grouping, e.g. 1155622 -> '₹ 11,55,622'."""
    digits = str(int(round(amount)))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])
    return f"₹ {digits}"


def format_inr_compact(amount: float) -> str:
    """Short INR label using lakh / crore units, e.g. 1155622 -> '₹ 11.56 Lakh'."""
    if amount >= 1e7:
        return f"₹ {amount / 1e7:.2f} Crore"
    if amount >= 1e5:
        return f"₹ {amount / 1e5:.2f} Lakh"
    return format_inr(amount)


def encode_request_payload(data: dict) -> np.ndarray:
    name_str = str(data.get("name", "Maruti")).split(" ")[0].strip()
    year = float(data.get("year", 2015))
    km = float(data.get("km_driven", 50000))
    fuel_str = str(data.get("fuel", "Petrol")).strip()
    seller_str = str(data.get("seller_type", "Individual")).strip()
    trans_str = str(data.get("transmission", "Manual")).strip()
    owner_str = str(data.get("owner", "First Owner")).strip()

    mileage = float(str(data.get("mileage", 18.0)).split(" ")[0])
    if mileage <= 0:
        mileage = 18.0
    engine = float(str(data.get("engine", 1200.0)).split(" ")[0])
    power = float(str(data.get("max_power", 80.0)).split(" ")[0])
    seats = float(str(data.get("seats", 5.0)).split(" ")[0])

    car_age = max(REFERENCE_YEAR - year, 0.0)
    km_per_year = km / (car_age + 0.5)
    power_to_engine = power / max(engine, 100.0)
    is_luxury = 1.0 if name_str in LUXURY_BRANDS else 0.0

    row = [
        float(LABEL_MAPS["name"].get(name_str, 1)),
        year,
        km,
        float(LABEL_MAPS["fuel"].get(fuel_str, 1)),
        float(LABEL_MAPS["seller_type"].get(seller_str, 1)),
        float(LABEL_MAPS["transmission"].get(trans_str, 1)),
        float(LABEL_MAPS["owner"].get(owner_str, 1)),
        mileage,
        engine,
        power,
        seats,
        car_age,
        km_per_year,
        power_to_engine,
        is_luxury,
    ]
    return np.array([row], dtype=np.float64)


def load_model():
    search_paths = [CURRENT_DIR / "model.joblib", ROOT / "model.joblib"]
    for p in search_paths:
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError(f"Model artifact not found. Searched: {[str(p) for p in search_paths]}")


MODEL = load_model()


def with_year_and_km(row: np.ndarray, year: float, km: float) -> np.ndarray:
    """Copy an encoded row, overriding year/km and recomputing the derived features."""
    updated = row.copy()
    car_age = max(REFERENCE_YEAR - year, 0.0)
    updated[YEAR_IDX] = year
    updated[KM_IDX] = km
    updated[AGE_IDX] = car_age
    updated[KM_PER_YEAR_IDX] = km / (car_age + 0.5)
    return updated


def valuation_curves(encoded: np.ndarray) -> dict:
    """Predict the same car across model years and odometer readings for the UI charts."""
    base = encoded[0]
    year, km = float(base[YEAR_IDX]), float(base[KM_IDX])

    year_hi = max(int(year), DATASET_YEAR_RANGE[1])
    year_lo = max(DATASET_YEAR_RANGE[0], min(int(year), DATASET_YEAR_RANGE[1]) - 12)
    years = list(range(year_lo, year_hi + 1))

    km_hi = max(200000.0, km * 1.6)
    km_step = max(5000.0, round(km_hi / 16 / 5000) * 5000)
    kms = sorted({1000.0, km, *np.arange(km_step, km_hi + 1, km_step).tolist()})

    grid = [with_year_and_km(base, float(y), km) for y in years]
    grid += [with_year_and_km(base, year, k) for k in kms]
    prices = np.maximum(MODEL.predict(np.array(grid)), 0.0)

    return {
        "year": [[y, round(float(p))] for y, p in zip(years, prices[: len(years)])],
        "km_driven": [[int(k), round(float(p))] for k, p in zip(kms, prices[len(years):])],
    }


class handler(BaseHTTPRequestHandler):
    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length)) if length > 0 else {}
            encoded = encode_request_payload(payload)
            price = max(float(MODEL.predict(encoded)[0]), 0.0)
            result = {
                "price": round(price),
                "price_formatted": format_inr(price),
                "price_compact": format_inr_compact(price),
            }
            if payload.get("include_curves"):
                result["curves"] = valuation_curves(encoded)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as error:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(error)}).encode())
