import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocess import FEATURE_COLUMNS, encode_input  # noqa: E402


def load_model():
    """Load model with fallback between model.joblib and model.pkl."""
    joblib_path = ROOT / "model.joblib"
    pkl_path = ROOT / "model.pkl"
    if joblib_path.exists():
        return joblib.load(joblib_path)
    if pkl_path.exists():
        import pickle as pk
        with open(pkl_path, "rb") as f:
            obj = pk.load(f)
        return obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    raise FileNotFoundError("No trained model file found (expected model.joblib or model.pkl)")


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
            row = [payload.get(column) for column in FEATURE_COLUMNS]
            input_data = pd.DataFrame([row], columns=FEATURE_COLUMNS)
            encoded = encode_input(input_data)
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
