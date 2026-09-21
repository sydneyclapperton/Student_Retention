import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("retention_model.pkl")

st.title("Student Retention Prediction App")
st.write("Enter student information to predict probability of retention next semester.")

# GPA bins
gpa_bins = [0.0, 0.49, 0.9, 1.49, 1.99, 2.49, 2.99, 3.49, 3.99, 4.0]
gpa_labels = [
    "0-0.49", "0.5-0.9", "1-1.49", "1.5-1.99", "2-2.49",
    "2.5-2.99", "3-3.49", "3.5-3.99", "4.0"
]

# Age bins
age_bins = [18, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74]
age_labels = [
    "18-19", "20-24", "25-29", "30-34", "35-39", "40-44",
    "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"
]

# Sidebar inputs
st.sidebar.header("Student Inputs")

gender = st.sidebar.selectbox("Gender", ["MALE", "FEMALE", "OTHER"])
age = st.sidebar.number_input("Age", min_value=17, max_value=90, value=20)
cumul_gpa = st.sidebar.number_input("Cumulative GPA", min_value=0.0, max_value=4.0, value=3.0)
term_gpa = st.sidebar.number_input("Term GPA", min_value=0.0, max_value=4.0, value=3.0)
credit_load = st.sidebar.number_input("Credit Load", min_value=0, max_value=30, value=12)
advising = st.sidebar.number_input("Advising Appointments", min_value=0, max_value=20, value=1)
alerts = st.sidebar.number_input("Alerts", min_value=0, max_value=20, value=0)

# Calculate full-time internally (not shown to user)
full_time = "1" if credit_load >= 12 else "0"

# Convert to bins
cumul_gpa_bin = pd.cut([cumul_gpa], bins=gpa_bins, labels=gpa_labels, include_lowest=True)[0]
term_gpa_bin = pd.cut([term_gpa], bins=gpa_bins, labels=gpa_labels, include_lowest=True)[0]
age_bin = pd.cut([age], bins=age_bins, labels=age_labels, include_lowest=True)[0]

# Handle out-of-range values
if pd.isna(cumul_gpa_bin):
    cumul_gpa_bin = "Unknown"
if pd.isna(term_gpa_bin):
    term_gpa_bin = "Unknown"
if pd.isna(age_bin):
    age_bin = "Unknown"

# Build input dataframe
input_data = pd.DataFrame({
    "Cumul_GPA": [cumul_gpa],
    "Term_GPA": [term_gpa],
    "Gender": [gender],
    "Age_bin": [age_bin],
    "Cumul_GPA_bin": [cumul_gpa_bin],
    "Advising_Appointments": [advising],
    "Alerts": [alerts],
    "Credit_Load": [credit_load],
    "Full_Time": [full_time], #internal
    "Term_GPA_bin": [term_gpa_bin]
})

# Predict
if st.button("Predict Retention"):
    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.write(f"**Retention Probability:** {prob:.2f}")

    if pred == 1:
        st.success("This student is likely to be retained.")
    else:
        st.error("This student is at risk of not being retained.")

# RISK / POSITIVE FACTORS

risk_factors = []
positive_factors = []

# GPA thresholds
if term_gpa < 2.0:
    risk_factors.append(("Low Term GPA", "Term GPA below 2.0 increases risk"))
else:
    positive_factors.append(("Strong Term GPA", "Term GPA above 2.0 supports retention"))

if cumul_gpa < 2.0:
    risk_factors.append(("Low Cumulative GPA", "Cumulative GPA below 2.0 increases risk"))
else:
    positive_factors.append(("Strong Cumulative GPA", "Cumulative GPA above 2.0 supports retention"))

# Advising
if advising == 0:
    risk_factors.append(("No Advising Appointments", "Students with no advising are more at risk"))
elif advising >= 2:
    positive_factors.append(("Advising Engagement", "Multiple advising appointments support retention"))

# Alerts
if alerts >= 3:
    risk_factors.append(("High Alerts", "Multiple alerts indicate academic risk"))
elif alerts == 0:
    positive_factors.append(("No Alerts", "No alerts is a strong positive indicator"))

# Credit Load
if credit_load < 9:
    risk_factors.append(("Low Credit Load", "Taking fewer than 9 credits increases risk"))
else:
    positive_factors.append(("Healthy Credit Load", "Higher credit load supports retention"))

# Age bin risk patterns (based on your model coefficients)
if age_bin in ["55-59", "60-64"]:
    risk_factors.append(("Age Group Risk", f"Age group {age_bin} has lower retention historically"))
elif age_bin in ["20-24", "25-29", "30-34"]:
    positive_factors.append(("Age Group Strength", f"Age group {age_bin} shows stronger retention"))

# Display tables
st.subheader("Risk Factors")
if len(risk_factors) > 0:
    risk_df = pd.DataFrame(risk_factors, columns=["Factor", "Reason"])
    st.table(risk_df)
else:
    st.write("No major risk factors detected.")

st.subheader("Positive Factors")
if len(positive_factors) > 0:
    pos_df = pd.DataFrame(positive_factors, columns=["Factor", "Reason"])
    st.table(pos_df)
else:
    st.write("No major positive factors detected.")
