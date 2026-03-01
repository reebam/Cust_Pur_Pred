
import streamlit as st
import pandas as pd
import joblib
import os

# 1. Load the model and encoder from the local directory
# (These were created by your train.py and copied to root)
MODEL_PATH = "tourism_model.pkl"
ENCODER_PATH = "encoder.joblib"

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
else:
    st.error("Model or Encoder not found! Please ensure training was successful.")
    st.stop()

# 2. Streamlit UI Design
st.set_page_config(page_title="Tourism Purchase Predictor", layout="centered")

st.title("🌴 Holiday Package Purchase Prediction")
st.write("""
This app predicts whether a customer will purchase a newly introduced **Wellness Tourism Package** based on their profile and previous interactions.
""")

st.divider()

# 3. User Inputs (Categorical)
col1, col2 = st.columns(2)

with col1:
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Unmarried", "Divorced"])

with col2:
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])

# 4. User Inputs (Numerical)
st.subheader("Customer Details")
age = st.slider("Age", 18, 70, 30)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 0, 120, 15)
number_of_trips = st.number_input("Number of Trips per Year", 1, 20, 2)
monthly_income = st.number_input("Monthly Income", 0, 100000, 25000)

# 5. Assemble input into DataFrame
input_dict = {
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfTrips': number_of_trips,
    'MaritalStatus': marital_status,
    'MonthlyIncome': monthly_income,
    'OwnCar': 1 if own_car == "Yes" else 0,
    'Designation': designation
}

input_df = pd.DataFrame([input_dict])

# 6. Prediction Logic
if st.button("Predict Purchase Intent"):
    try:
        # The model pipeline handles scaling/encoding automatically
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        st.divider()
        if prediction[0] == 1:
            st.success(f"🎯 **High Potential!** The customer is likely to purchase the package. (Probability: {probability:.2%})")
        else:
            st.warning(f"⏳ **Low Potential.** The customer is unlikely to purchase at this time. (Probability: {probability:.2%})")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")
