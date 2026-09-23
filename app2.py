import pickle as pk

import pandas as pd
import streamlit as st
from pathlib import Path

from preprocess import FEATURE_COLUMNS, encode_input, prepare_car_data

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "model.pkl", "rb") as model_file:
    loaded = pk.load(model_file)
model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded

car_data = prepare_car_data(BASE_DIR / "Cardetails.csv")

st.header("Car Price Prediction ML Project")

name = st.selectbox("Select Car Brand", sorted(car_data["name"].unique()))
subset = car_data[car_data["name"] == name]

year = st.slider("Car Manufactured Year", int(subset["year"].min()), int(subset["year"].max()), int(subset["year"].median()))
km_driven = st.slider("NO. of KiloMeters Driven", int(subset["km_driven"].min()), int(subset["km_driven"].max()), int(subset["km_driven"].median()))
fuel = st.selectbox("Fuel Type", sorted(subset["fuel"].unique()))
seller_type = st.selectbox("Seller Type", sorted(subset["seller_type"].unique()))
transmission = st.selectbox("Transmission Type", sorted(subset["transmission"].unique()))
owner = st.selectbox("Owner", sorted(subset["owner"].unique()))
mileage = st.slider("Car Mileage", float(subset["mileage"].min()), float(subset["mileage"].max()), float(subset["mileage"].median()))
engine = st.slider("Engine CC", int(subset["engine"].min()), int(subset["engine"].max()), int(subset["engine"].median()))
max_power = st.slider("Max Power", float(subset["max_power"].min()), float(subset["max_power"].max()), float(subset["max_power"].median()))
seats = st.slider("NO. of Seats", int(subset["seats"].min()), int(subset["seats"].max()), int(subset["seats"].median()))

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=FEATURE_COLUMNS,
    )
    predicted_price = max(float(model.predict(encode_input(input_data))[0]), 0)
    st.markdown(
        f"<h2 style='text-align: center; color: #2a9df4;'>The predicted car price is: ₹ {predicted_price:,.0f}</h2>",
        unsafe_allow_html=True,
    )
