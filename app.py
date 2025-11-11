import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler with graceful error handling (shows useful debug info in the app)
try:
    model = joblib.load("loan_default_model.pkl")
    scaler = joblib.load("loan_scaler.pkl")
except Exception as e:
    # If running in a normal Python session this will raise; in Streamlit show a helpful message
    try:
        # Use Streamlit UI to display debugging hints
        st.set_page_config(page_title="Loan Default Predictor - Error", page_icon="💥")
        st.title("Model load error")
        st.error("Failed to load model or scaler. See details below and check your environment.")
        st.write(f"Error: {e}")
        import sys, platform
        st.write(f"Python executable: {sys.executable}")
        st.write(f"Python version: {sys.version}")
        try:
            import sklearn
            st.write(f"scikit-learn: {sklearn.__version__}")
        except Exception:
            st.write("scikit-learn: not installed or failed to import")
        try:
            import joblib as _joblib
            st.write(f"joblib: {_joblib.__version__}")
        except Exception:
            st.write("joblib: not available")
        st.stop()
    except Exception:
        # If Streamlit isn't available (e.g., running from plain python), re-raise original exception
        raise

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
education = st.sidebar.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
employment_type = st.sidebar.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
has_mortgage = st.sidebar.selectbox("Has Mortgage", ["Yes", "No"])
has_dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
has_cosigner = st.sidebar.selectbox("Has Co-Signer", ["Yes", "No"])

# Convert categorical to numeric (must match training encoding - alphabetical order)
education_map = {"Bachelor's": 0, "High School": 1, "Master's": 2, "PhD": 3}
employment_map = {"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3}
marital_map = {"Divorced": 0, "Married": 1, "Single": 2}
binary_map = {"Yes": 1, "No": 0}
loan_purpose_map = {"Auto": 0, "Business": 1, "Education": 2, "Home": 3, "Other": 4}

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
