import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

MODEL_FILE = "california_housing_model.pkl"


@st.cache_resource
def load_trained_model():
  """Load the persisted model artifact."""
  if not os.path.exists(MODEL_FILE):
    return None
  return joblib.load(MODEL_FILE)


model = load_trained_model()

st.title("🏠 California Housing Price Predictor")
st.markdown(
    "Predict median house prices across California districts using the trained"
    " **HistGradientBoosting** model."
)

if model is None:
  st.error(
      f"Model artifact '{MODEL_FILE}' not found! Run"
      " `california_housing_pipeline.py` first to train and save the model."
  )
  st.stop()

st.divider()

# Organize inputs into two columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
  st.subheader("📊 Socioeconomic & Property Features")

  med_inc = st.slider(
      "Median Income (in $10,000s)",
      min_value=0.5,
      max_value=15.0,
      value=5.0,
      step=0.1,
      help="Median income in block group (e.g., 5.0 = $50,000).",
  )

  house_age = st.slider(
      "House Age (years)",
      min_value=1.0,
      max_value=52.0,
      value=25.0,
      step=1.0,
      help="Median age of a house within a block.",
  )

  ave_rooms = st.number_input(
      "Average Rooms per Household",
      min_value=1.0,
      max_value=15.0,
      value=5.5,
      step=0.1,
  )

  ave_bedrms = st.number_input(
      "Average Bedrooms per Household",
      min_value=0.5,
      max_value=5.0,
      value=1.0,
      step=0.05,
  )

  population = st.number_input(
      "Block Population",
      min_value=10.0,
      max_value=10000.0,
      value=1200.0,
      step=50.0,
  )

  ave_occup = st.number_input(
      "Average Household Occupancy",
      min_value=1.0,
      max_value=10.0,
      value=3.0,
      step=0.1,
  )

with col2:
  st.subheader("📍 Geographical Location")

  latitude = st.slider(
      "Latitude",
      min_value=32.5,
      max_value=42.0,
      value=34.05,
      step=0.01,
      help="District latitude coordinate.",
  )

  longitude = st.slider(
      "Longitude",
      min_value=-124.5,
      max_value=-114.0,
      value=-118.25,
      step=0.01,
      help="District longitude coordinate.",
  )

  # Display interactive district map marker
  st.markdown("**Selected Coordinates on California Map:**")
  map_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
  st.map(map_df, zoom=5)

st.divider()

# Prediction trigger
if st.button("Calculate Estimated Price", type="primary", use_container_width=True):
  # Construct input DataFrame matching training column names exactly
  input_data = pd.DataFrame({
      "MedInc": [med_inc],
      "HouseAge": [house_age],
      "AveRooms": [ave_rooms],
      "AveBedrms": [ave_bedrms],
      "Population": [population],
      "AveOccup": [ave_occup],
      "Latitude": [latitude],
      "Longitude": [longitude],
  })

  # Generate prediction (model target is in $100k units)
  raw_pred = model.predict(input_data)[0]
  price_usd = max(0.0, raw_pred * 100000)

  res_col1, res_col2 = st.columns([1, 1])
  with res_col1:
    st.metric(
        label="Estimated Median House Value",
        value=f"${price_usd:,.0f}",
        delta=f"Raw Model Output: {raw_pred:.3f} ($100k)",
    )
  with res_col2:
    st.dataframe(input_data, hide_index=True)