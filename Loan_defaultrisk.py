import streamlit as st
import joblib
import pandas as pd
from dotenv import load_dotenv
import os
from openai import OpenAI

# --- Page Title ---
st.title("Loan Default Risk Prediction")
st.markdown("This tool predicts whether a loan applicant is likely to **default** or **repay** the loan.")

# --- Load Model and Tools ---
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("encoder.pkl")  # If you used LabelEncoder, otherwise remove this line

# --- Input Fields ---
st.header("Applicant Details")

age = st.slider("Age", 18, 100, 30)
income = st.number_input("Monthly Income ($)", min_value=100, max_value=1000000, value=3000)
loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=1000000, value=50000)
loan_term = st.number_input("Loan Term (months)", min_value=6, max_value=360, value=36)

credit_score = st.slider("Credit Score", 300, 850, 700)
months_employed = st.slider("Months Employed", 0, 600, 24)
num_credit_lines = st.slider("Number of Credit Lines", 0, 20, 5)
interest_rate = st.slider("Interest Rate (%)", 1.0, 30.0, 8.5)
dti_ratio = st.slider("Debt-to-Income Ratio (DTI)", 0.0, 1.0, 0.2)

education = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])
employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
has_mortgage = st.selectbox("Has Mortgage", [0, 1])
has_dependents = st.selectbox("Has Dependents", [0, 1])
loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Home", "Auto", "Education"])
has_cosigner = st.selectbox("Has CoSigner", [0, 1])

# --- Prediction ---
if st.button("Predict Loan Default Risk"):
    # Prepare input dataframe
    input_data = pd.DataFrame([[
        age, income, loan_amount, credit_score, months_employed, num_credit_lines,
        interest_rate, loan_term, dti_ratio, education, employment_type,
        marital_status, has_mortgage, has_dependents, loan_purpose, has_cosigner
    ]], columns=[
        'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines',
        'InterestRate', 'LoanTerm', 'DTIRatio', 'Education', 'EmploymentType',
        'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
    ])

    # Encode categorical variables
    input_data.replace({
        'Education': {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3},
        'EmploymentType': {'Salaried': 0, 'Self-Employed': 1},
        'MaritalStatus': {'Single': 0, 'Married': 1},
        'LoanPurpose': {'Personal': 0, 'Home': 1, 'Auto': 2, 'Education': 3}
    }, inplace=True)

    # Scale numeric features (same as training)
    numeric_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Predict
    proba = model.predict_proba(input_data)[0][1]  # class 1 = Default
    threshold = 0.4
    prediction = 1 if proba > threshold else 0
    prediction_label = "Likely to **Default**" if prediction == 1 else "Likely to **Repay**"

    # Display result
    st.subheader("Prediction Result")
    st.markdown(f"The model predicts the applicant is **{prediction_label}** the loan.")
    st.markdown(f"**Probability of Default:** {proba:.2f}")

    # Explanation Prompt
    explanation = f"""
    Based on the applicant's financial profile, the model considered factors like age, income, loan amount, and credit score.
    The predicted probability of default is **{proba:.2f}**.
    """

    st.subheader("LLM Explanation")
    st.markdown(explanation)

    # --- GPT-3.5 LLM Explanation ---
    load_dotenv("API.env")
    api_key = os.getenv("OPENAI_API_KEY")

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that explains loan risk decisions clearly."},
                {"role": "user", "content": explanation}
            ]
        )
        st.subheader("Detailed LLM Interpretation")
        st.write(response.choices[0].message.content)

    except Exception as e:
        st.warning(f"LLM explanation failed: {e}")
