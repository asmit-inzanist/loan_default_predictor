# 💰 Loan Default Prediction App

A Streamlit web application that predicts whether a customer is likely to default on a loan based on their financial and personal information.

## Features

- Interactive web interface built with Streamlit
- Real-time loan default prediction
- Probability scoring for risk assessment
- User-friendly input fields for customer details

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/loan_predictor.git
cd loan_predictor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
python -m streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Input Features

The model accepts the following inputs:

- **Age**: Customer's age (18-100)
- **Annual Income**: Customer's yearly income in dollars
- **Loan Amount**: Requested loan amount in dollars
- **Credit Score**: Customer's credit score (300-900)
- **Months Employed**: Number of months at current employment
- **Number of Credit Lines**: Total number of credit lines
- **Interest Rate**: Loan interest rate percentage
- **Loan Term**: Loan duration in months
- **DTI Ratio**: Debt-to-Income ratio
- **Education**: Education level (High School, Bachelor, Master, PhD)
- **Employment Type**: Salaried, Self-Employed, or Unemployed
- **Marital Status**: Single, Married, or Divorced
- **Has Mortgage**: Yes or No
- **Has Dependents**: Yes or No
- **Loan Purpose**: Home, Car, Education, Business, or Personal
- **Has Co-Signer**: Yes or No

## Model Files

The app requires two pre-trained model files:
- `loan_default_model.pkl`: Trained machine learning model
- `loan_scaler.pkl`: Feature scaler for data normalization

## Technologies Used

- **Python 3.11+**
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Joblib**: Model serialization

## License

MIT License

## Author

Built with ❤️ using Streamlit and Scikit-learn
