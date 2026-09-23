import json
import pickle as pk
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

    car_age = max(2024.0 - year, 0.0)
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
    search_paths = [
        CURRENT_DIR / "model.joblib",
        ROOT / "model.joblib",
        CURRENT_DIR / "model.pkl",
        ROOT / "model.pkl",
    ]
    for p in search_paths:
        if p.exists():
            if p.suffix == ".joblib":
                return joblib.load(p)
            with open(p, "rb") as f:
                obj = pk.load(f)
            return obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    raise FileNotFoundError(f"Model artifact not found. Searched: {[str(p) for p in search_paths]}")


MODEL = load_model()


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

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(
                json.dumps({
                    "price": round(price),
                    "price_formatted": f"₹ {price:,.0f}",
                }).encode()
            )
        except Exception as error:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(error)}).encode())
