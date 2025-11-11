import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("loan_default_model.pkl")
scaler = joblib.load("loan_scaler.pkl")

st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="centered")

st.title("💰 Loan Default Prediction App")
st.write("Enter customer details below to predict whether they are likely to **default on a loan**.")

# Sidebar inputs
st.sidebar.header("Input Customer Details")

# Numeric Inputs
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("Annual Income ($)", min_value=0, max_value=500000, value=50000)
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=0, max_value=1000000, value=10000)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, value=700)
months_employed = st.sidebar.number_input("Months Employed", min_value=0, max_value=480, value=24)
num_credit_lines = st.sidebar.number_input("Number of Credit Lines", min_value=0, max_value=30, value=5)
interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=10.5)
loan_term = st.sidebar.number_input("Loan Term (months)", min_value=6, max_value=360, value=60)
dti_ratio = st.sidebar.number_input("DTI Ratio", min_value=0.0, max_value=100.0, value=20.0)

# Categorical Inputs
education = st.sidebar.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
employment_type = st.sidebar.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
has_mortgage = st.sidebar.selectbox("Has Mortgage", ["Yes", "No"])
has_dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Home", "Car", "Education", "Business", "Personal"])
has_cosigner = st.sidebar.selectbox("Has Co-Signer", ["Yes", "No"])

# Convert categorical to numeric (must match training encoding)
education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
employment_map = {"Salaried": 0, "Self-Employed": 1, "Unemployed": 2}
marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
binary_map = {"Yes": 1, "No": 0}
loan_purpose_map = {"Home": 0, "Car": 1, "Education": 2, "Business": 3, "Personal": 4}

# Create DataFrame for prediction
input_data = pd.DataFrame({
    'LoanID': [0],  # Dummy numeric ID (scaler expects this column but it won't affect prediction)
    'Age': [age],
    'Income': [income],
    'LoanAmount': [loan_amount],
    'CreditScore': [credit_score],
    'MonthsEmployed': [months_employed],
    'NumCreditLines': [num_credit_lines],
    'InterestRate': [interest_rate],
    'LoanTerm': [loan_term],
    'DTIRatio': [dti_ratio],
    'Education': [education_map[education]],
    'EmploymentType': [employment_map[employment_type]],
    'MaritalStatus': [marital_map[marital_status]],
    'HasMortgage': [binary_map[has_mortgage]],
    'HasDependents': [binary_map[has_dependents]],
    'LoanPurpose': [loan_purpose_map[loan_purpose]],
    'HasCoSigner': [binary_map[has_cosigner]]
})

# Scale numeric data
scaled_input = scaler.transform(input_data)

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The customer is **likely to default** on the loan. (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ The customer is **not likely to default**. (Confidence: {(1-probability)*100:.2f}%)")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")
