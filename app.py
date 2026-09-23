import pickle as pk

import pandas as pd
import streamlit as st
from pathlib import Path

from preprocess import FEATURE_COLUMNS, encode_input, prepare_car_data

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "model.pkl", "rb") as model_file:
    loaded = pk.load(model_file)

model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
metrics = loaded.get("metrics", {}) if isinstance(loaded, dict) else {}

car_data = prepare_car_data(BASE_DIR / "Cardetails.csv")
brands = sorted(car_data["name"].unique())


def brand_defaults(brand: str) -> dict:
    subset = car_data[car_data["name"] == brand]
    return {
        "year": int(subset["year"].median()),
        "km_driven": int(subset["km_driven"].median()),
        "fuel": subset["fuel"].mode().iloc[0],
        "seller_type": subset["seller_type"].mode().iloc[0],
        "transmission": subset["transmission"].mode().iloc[0],
        "owner": subset["owner"].mode().iloc[0],
        "mileage": float(subset["mileage"].median()),
        "engine": int(subset["engine"].median()),
        "max_power": float(subset["max_power"].median()),
        "seats": int(subset["seats"].median()),
    }


st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚘",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <div style="background-color: #2a9df4; padding: 10px; border-radius: 10px;">
        <h1 style="color: white; text-align: center;">Car Price Prediction App</h1>
        <p style="color: white; text-align: center;">Enter car details to get an estimated resale price.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if metrics:
    r2_pct = metrics.get("accuracy_pct", round(metrics.get("r2", 0) * 100, 1))
    mape_val = metrics.get("mape", 0)
    st.caption(
        f"Model accuracy: {r2_pct}% (R² {metrics.get('r2', 0):.3f}) · typical error ±₹{metrics.get('mae', 0):,.0f} ({mape_val:.1f}%)"
    )

st.sidebar.header("Enter Car Details")
name = st.sidebar.selectbox("Select Car Brand", brands)
defaults = brand_defaults(name)

year = st.sidebar.slider(
    "Car Manufactured Year",
    int(car_data["year"].min()),
    int(car_data["year"].max()),
    defaults["year"],
)
km_driven = st.sidebar.slider(
    "No. of Kilometers Driven",
    int(car_data["km_driven"].min()),
    int(car_data["km_driven"].max()),
    defaults["km_driven"],
)
fuel = st.sidebar.selectbox(
    "Fuel Type",
    sorted(car_data["fuel"].unique()),
    index=list(sorted(car_data["fuel"].unique())).index(defaults["fuel"]),
)
seller_type = st.sidebar.selectbox(
    "Seller Type",
    sorted(car_data["seller_type"].unique()),
    index=list(sorted(car_data["seller_type"].unique())).index(defaults["seller_type"]),
)
transmission = st.sidebar.selectbox(
    "Transmission Type",
    sorted(car_data["transmission"].unique()),
    index=list(sorted(car_data["transmission"].unique())).index(defaults["transmission"]),
)
owner = st.sidebar.selectbox(
    "Owner",
    sorted(car_data["owner"].unique()),
    index=list(sorted(car_data["owner"].unique())).index(defaults["owner"]),
)
mileage = st.sidebar.slider(
    "Car Mileage (kmpl)",
    float(car_data["mileage"].min()),
    float(car_data["mileage"].max()),
    defaults["mileage"],
)
engine = st.sidebar.slider(
    "Engine Capacity (CC)",
    int(car_data["engine"].min()),
    int(car_data["engine"].max()),
    defaults["engine"],
)
max_power = st.sidebar.slider(
    "Max Power (BHP)",
    float(car_data["max_power"].min()),
    float(car_data["max_power"].max()),
    defaults["max_power"],
)
seats = st.sidebar.slider(
    "Number of Seats",
    int(car_data["seats"].min()),
    int(car_data["seats"].max()),
    defaults["seats"],
)

if st.sidebar.button("Predict Price"):
    input_data = pd.DataFrame(
        [[
            name, year, km_driven, fuel, seller_type, transmission, owner,
            mileage, engine, max_power, seats,
        ]],
        columns=FEATURE_COLUMNS,
    )
    encoded = encode_input(input_data)
    predicted_price = max(float(model.predict(encoded)[0]), 0)

    st.markdown(
        f"""
        <div style="background-color: #4CAF50; padding: 20px; border-radius: 10px;">
            <h2 style="color: white; text-align: center;">The Predicted Car Price is:</h2>
            <h1 style="color: white; text-align: center; font-weight: bold;">₹ {predicted_price:,.0f}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.balloons()

st.markdown(
    "<hr><p style='text-align: center; color: gray;'>Machine Learning Project</p>",
    unsafe_allow_html=True,
)
