import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# Google Drive model link
google_drive_url = "https://drive.google.com/file/d/1Jt4cKgBTX7G7FN8xFUzyRZsNBaNmTGns/view?usp=sharing"

model_path = "best_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        gdown.download(google_drive_url, model_path, quiet=False, fuzzy=True)
    model = joblib.load(model_path)
    return model

model = load_model()

st.title("Housing Price Predictor")

st.write("Enter the details of the property below:")

longitude = st.number_input("Longitude", value=-122.0)
latitude = st.number_input("Latitude", value=37.0)
housing_median_age = st.slider("Housing Median Age", 1, 52, 30)
total_rooms = st.number_input("Total Rooms", value=2000.0)
total_bedrooms = st.number_input("Total Bedrooms", value=400.0)
population = st.number_input("Population", value=1000.0)
households = st.number_input("Households", value=350.0)
median_income = st.number_input("Median Income (in 10k USD)", value=3.5)

input_df = pd.DataFrame([[
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income
]], columns=[
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income"
])

if st.button("Predict Median House Value"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Median House Value: ${prediction:,.2f}")
