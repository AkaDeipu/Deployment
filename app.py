import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🚢 Titanic Survival Prediction")

# Collect user inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["female", "male"])
age = st.slider("Age", 0, 80, 30)
fare = st.slider("Fare Paid", 0.0, 500.0, 50.0)
embarked = st.selectbox("Embarked Location", ["S", "C", "Q"])
had_cabin = st.selectbox("Had Cabin Info", ["No", "Yes"])
family_size = st.slider("Family Size", 1, 10, 1)

# Encode inputs
sex = 1 if sex == "male" else 0
embarked = {"S": 0, "C": 1, "Q": 2}[embarked]
had_cabin = 1 if had_cabin == "Yes" else 0

# Prepare input for prediction
input_data = np.array([[pclass, sex, age, fare, embarked, had_cabin, family_size]])
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# Display result
if prediction == 1:
    st.success(f"🎉 Survived! Probability: {probability:.2f}")
else:
    st.error(f"❌ Did not survive. Probability: {probability:.2f}")
