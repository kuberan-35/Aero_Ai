# ---------------------------------------------------------
# STREAMLIT APP – Drone ETA Prediction
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# Load Model + Dataset
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("D:\data science\Aaro ai\eta_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("D:\data science\Aaro ai\yolov5\cleaned_drone_eta.csv")

model = load_model()
df = load_data()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Drone ETA Prediction", page_icon="🚁", layout="wide")

st.title("🚁 Drone Delivery ETA Prediction")
st.write("Predict Estimated Delivery Time based on drone metrics.")

st.markdown("---")

# Sidebar
st.sidebar.header("🔧 Input Parameters")

# Prepare options
drone_types = df["drone_type"].unique()
weather_types = df["weather_condition"].unique()

# ------------------------------
# Input Form
# ------------------------------
distance = st.sidebar.number_input("Distance (km)", min_value=1.0, max_value=50.0, step=0.5)
payload = st.sidebar.number_input("Payload Weight (kg)", min_value=0.1, max_value=5.0, step=0.1)
speed = st.sidebar.number_input("Drone Speed (km/h)", min_value=10.0, max_value=60.0, step=1.0)
battery = st.sidebar.number_input("Battery Efficiency (%)", min_value=50.0, max_value=100.0, step=1.0)

drone_type = st.sidebar.selectbox("Drone Type (Encoded)", sorted(drone_types))
weather = st.sidebar.selectbox("Weather Condition (Encoded)", sorted(weather_types))

predict_btn = st.sidebar.button("Predict ETA ⏱️")

# ------------------------------
# Prediction Logic
# ------------------------------
if predict_btn:
    input_data = np.array([[distance, payload, speed, battery, drone_type, weather]])
    eta = model.predict(input_data)[0]

    st.success(f"### Predicted ETA: **{eta:.2f} minutes**")
    st.balloons()

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
