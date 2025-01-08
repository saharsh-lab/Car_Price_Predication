import pandas as pd
import pickle as pk
import streamlit as st

# Load the trained model
model = pk.load(open('/Users/Sai/car price/model.pkl', 'rb'))

# Load car details data
car_data = pd.read_csv("/Users/Sai/Downloads/Cardetails.csv")

# Function to extract car brand from car name
def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

# Apply the function to the car data
car_data['name'] = car_data['name'].apply(get_brand_name)

# Set a custom page title and layout
st.set_page_config(
    page_title="🚗 Car Price Prediction",
    page_icon="🚘",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add a fancy header
st.markdown(
    """
    <div style="background-color: #2a9df4; padding: 10px; border-radius: 10px;">
        <h1 style="color: white; text-align: center;">🚗 Car Price Prediction App 🚘</h1>
        <p style="color: white; text-align: center;">Enter the car details below to predict its price!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# User inputs for car details
st.sidebar.header("📝 Enter Car Details")
name = st.sidebar.selectbox('Select Car Brand', car_data['name'].unique())
year = st.sidebar.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.sidebar.slider('No. of Kilometers Driven', 1, 200000)
fuel = st.sidebar.selectbox('Fuel Type', car_data['fuel'].unique())
seller_type = st.sidebar.selectbox('Seller Type', car_data['seller_type'].unique())
transmission = st.sidebar.selectbox('Transmission Type', car_data['transmission'].unique())
owner = st.sidebar.selectbox('Owner', car_data['owner'].unique())
mileage = st.sidebar.slider('Car Mileage (kmpl)', 5, 40)
engine = st.sidebar.slider('Engine Capacity (CC)', 700, 5000)
max_power = st.sidebar.slider('Max Power (BHP)', 0, 200)
seats = st.sidebar.slider('Number of Seats', 2, 10)

# Predict button
if st.sidebar.button("🚀 Predict Price"):
    # Prepare input data for the model
    input_data = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )
    
    # Replace categorical values with numeric codes for the model
    input_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)
    input_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data['name'].replace(
        ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata', 'Chevrolet', 
         'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus', 
         'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
        range(1, 32), inplace=True
    )
    
    # Predict the price
    predicted_price = model.predict(input_data)[0]

    # Handle negative values
    if predicted_price < 0:
        predicted_price = 0  # Set to 0 or handle with any default value of your choice

    # Format the predicted price
    formatted_price = f"₹ {predicted_price:,.2f}"

    # Display the price prediction
    st.markdown(
        f"""
        <div style="background-color: #4CAF50; padding: 20px; border-radius: 10px;">
            <h2 style="color: white; text-align: center;">🎉 The Predicted Car Price is:</h2>
            <h1 style="color: white; text-align: center; font-weight: bold;">{formatted_price}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.balloons()  # Add celebratory balloons!

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align: center; color: gray;">Machine Learning Project</p>
    """,
    unsafe_allow_html=True
)

