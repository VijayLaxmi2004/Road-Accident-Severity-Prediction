import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# -----------------------------
# Load Model and Encoders
# -----------------------------
model = joblib.load("/content/drive/MyDrive/Road_Accident_Severity_Prediction/model/random_forest_model.pkl")
encoders = joblib.load("/content/drive/MyDrive/Road_Accident_Severity_Prediction/model/label_encoders.pkl")

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🚗 Road Accident Severity Prediction")

st.markdown("""
Predict the severity of a road accident using a Machine Learning model.

Fill in the details below and click **Predict Severity**.
""")

st.divider()
# ============================
# User Input Section
# ============================

st.header("📝 Enter Accident Details")

# -------- Date & Time --------
col1, col2 = st.columns(2)

with col1:
    accident_date = st.date_input("Accident Date")

with col2:
    hour = st.slider("Hour of Accident", 0, 23, 12)

# -------- Location --------
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox(
        "City",
        encoders["city"].classes_
    )

with col2:
    state = st.selectbox(
        "State",
        encoders["state"].classes_
    )

latitude = st.number_input(
    "Latitude",
    value=20.0,
    format="%.6f"
)

longitude = st.number_input(
    "Longitude",
    value=78.0,
    format="%.6f"
)

# -------- Road Information --------

road_type = st.selectbox(
    "Road Type",
    encoders["road_type"].classes_
)

lanes = st.slider(
    "Number of Lanes",
    1,
    6,
    2
)

traffic_signal = st.selectbox(
    "Traffic Signal",
    [0, 1]
)

# -------- Weather --------

weather = st.selectbox(
    "Weather",
    encoders["weather"].classes_
)

visibility = st.selectbox(
    "Visibility",
    encoders["visibility"].classes_
)

temperature = st.slider(
    "Temperature (°C)",
    15,
    40,
    28
)

traffic_density = st.selectbox(
    "Traffic Density",
    encoders["traffic_density"].classes_
)

# -------- Accident Details --------

cause = st.selectbox(
    "Cause",
    encoders["cause"].classes_
)

vehicles_involved = st.slider(
    "Vehicles Involved",
    1,
    5,
    2
)

casualties = st.slider(
    "Casualties",
    0,
    5,
    1
)

risk_score = st.slider(
    "Risk Score",
    0.10,
    1.00,
    0.50
)
# ============================
# Predict Button
# ============================

if st.button("🚀 Predict Severity"):

    # ---------------------------------
    # Extract Date Information
    # ---------------------------------
    year = accident_date.year
    month = accident_date.month
    day = accident_date.day

    day_name = accident_date.strftime("%A")

    is_weekend = 1 if day_name in ["Saturday", "Sunday"] else 0

    is_peak_hour = 1 if hour in [8, 9, 10, 17, 18, 19] else 0

    # ---------------------------------
    # Encode Categorical Features
    # ---------------------------------

    city_encoded = encoders["city"].transform([city])[0]
    state_encoded = encoders["state"].transform([state])[0]
    day_encoded = encoders["day_of_week"].transform([day_name])[0]
    road_encoded = encoders["road_type"].transform([road_type])[0]
    weather_encoded = encoders["weather"].transform([weather])[0]
    visibility_encoded = encoders["visibility"].transform([visibility])[0]
    traffic_density_encoded = encoders["traffic_density"].transform([traffic_density])[0]
    cause_encoded = encoders["cause"].transform([cause])[0]

    # ---------------------------------
    # Create Input DataFrame
    # IMPORTANT:
    # This order exactly matches X.columns
    # ---------------------------------

    input_data = pd.DataFrame([[
        city_encoded,
        state_encoded,
        latitude,
        longitude,
        hour,
        day_encoded,
        is_weekend,
        road_encoded,
        lanes,
        traffic_signal,
        weather_encoded,
        visibility_encoded,
        temperature,
        traffic_density_encoded,
        cause_encoded,
        vehicles_involved,
        casualties,
        is_peak_hour,
        risk_score,
        year,
        month,
        day
    ]],
    columns=[
        'city',
        'state',
        'latitude',
        'longitude',
        'hour',
        'day_of_week',
        'is_weekend',
        'road_type',
        'lanes',
        'traffic_signal',
        'weather',
        'visibility',
        'temperature',
        'traffic_density',
        'cause',
        'vehicles_involved',
        'casualties',
        'is_peak_hour',
        'risk_score',
        'year',
        'month',
        'day'
    ])

    # ---------------------------------
    # Make Prediction
    # ---------------------------------

    prediction = model.predict(input_data)

    severity = encoders["accident_severity"].inverse_transform(prediction)[0]
    severity = encoders["accident_severity"].inverse_transform(prediction)[0]
        # ---------------------------------
    # Display Prediction
    # ---------------------------------

    st.divider()

    st.subheader("Prediction Result")

    if severity == "minor":
        st.success("✅ Predicted Accident Severity: MINOR")

    elif severity == "major":
        st.warning("⚠️ Predicted Accident Severity: MAJOR")

    else:
        st.error("🚨 Predicted Accident Severity: FATAL")

    # ---------------------------------
    # Prediction Confidence
    # ---------------------------------

    probabilities = model.predict_proba(input_data)[0]

    confidence = max(probabilities) * 100

    st.info(f"Prediction Confidence: {confidence:.2f}%")

    # ---------------------------------
    # Show Prediction Probabilities
    # ---------------------------------

    st.subheader("Prediction Probabilities")

    probability_df = pd.DataFrame({
        "Severity": encoders["accident_severity"].classes_,
        "Probability": probabilities
    })

    st.dataframe(probability_df)

    st.bar_chart(
        probability_df.set_index("Severity")
    )

# ============================
# Footer
# ============================

st.divider()

st.markdown(
    """
    ### About
    This application predicts the severity of road accidents using a **Random Forest Machine Learning model** trained on historical accident data.

    **Possible Predictions:**
    - Minor
    - Major
    - Fatal
    """
)
