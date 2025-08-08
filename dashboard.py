import streamlit as st
import requests
import pandas as pd
import joblib
import os
from fpdf import FPDF

# --- Configuration & Setup ---
API_URL = "http://127.0.0.1:5000" 
COLUMNS_PATH = "training_columns.joblib"

# --- Caching & Data Loading ---
@st.cache_resource(show_spinner="Loading model assets...")
def load_training_columns(path: str) -> list:
    if not os.path.exists(path):
        st.error(f"Error: The required file '{path}' was not found.")
        st.stop()
    return joblib.load(path)

training_columns = load_training_columns(COLUMNS_PATH)

# --- Helper Functions ---

def get_rejection_analysis_and_suggestions(inputs: dict) -> tuple[list, list]:
    """Analyzes user inputs to provide reasons for ineligibility and actionable suggestions."""
    reasons = []
    suggestions = []
    
    if inputs['cibil'] < 650:
        reasons.append("Low CIBIL Score")
        suggestions.append("Improve your CIBIL score by paying all existing bills and EMIs on time without any delays.")
        
    annual_emi = inputs['annuity'] * 12
    dti_ratio = (annual_emi / inputs['income']) if inputs['income'] > 0 else 1
    
    if dti_ratio > 0.5:
        reasons.append("High Debt-to-Income Ratio")
        suggestions.append("The proposed monthly payment is high for your current income. Consider reducing the loan amount or extending the loan tenure to lower the EMI.")
        
    if inputs['emp'] < 1:
        reasons.append("Short Employment History")
        suggestions.append("Lenders prefer applicants with a stable employment history of at least 1-2 years. Building a longer track record at your current job will help.")
        
    if inputs['loan_amount'] > inputs['income'] * 5:
        reasons.append("High Loan Amount Relative to Income")
        suggestions.append("The requested loan amount is very high compared to your annual income.")

    if inputs.get('has_prev_loans'):
        if inputs.get('prev_loan_count', 0) > 3:
            reasons.append("High Number of Existing Loans")
            suggestions.append("Having multiple active loans can indicate high financial leverage. It's advisable to close some existing loans before applying for new ones.")
        if inputs.get('prev_outstanding_amt', 0) > inputs['income'] * 2:
            reasons.append("High Existing Debt Burden")
            suggestions.append("Your existing outstanding loan amount is high compared to your income. Reducing this debt will significantly improve your eligibility.")

    if not reasons:
        reasons.append("Overall Profile Risk")
        suggestions.append("While individual factors may be acceptable, your overall profile combination is assessed as high-risk by the model. Improving your CIBIL score is the most effective way to boost your eligibility.")
        
    return reasons, suggestions

def get_approval_suggestions() -> list:
    """Provides suggestions for maintaining a good credit score."""
    return [
        "Continue paying all your bills and EMIs on time without fail.",
        "Keep your credit utilization ratio low (ideally below 30%).",
        "Avoid applying for multiple new loans or credit cards in a short period.",
        "Regularly review your full credit report for any errors."
    ]

def map_to_full_payload(inputs: dict) -> dict:
    """Creates the full data payload required by the model from a dictionary of user inputs."""
    base = {col: 0 for col in training_columns}
    base['CIBIL_SCORE'] = (inputs['cibil'] - 300) / 600
    base["DAYS_BIRTH"] = -inputs['age'] * 365
    base["DAYS_EMPLOYED"] = -inputs['emp'] * 365
    base['AMT_INCOME_TOTAL'] = inputs['income']
    base['AMT_CREDIT'] = inputs['loan_amount']
    base['AMT_ANNUITY'] = inputs['annuity']
    
    if inputs.get('has_prev_loans'):
        base['PREV_LOAN_COUNT'] = inputs.get('prev_loan_count', 0)
        base['PREV_AMT_OUTSTANDING'] = inputs.get('prev_outstanding_amt', 0)
        base['PREV_REMAINING_EMI'] = inputs.get('prev_remaining_emi', 0)
        
    return base

def generate_pdf_report(score: int, prob: float, eligibility: str, reasons: list = [], suggestions: list = []) -> bytes:
    """Generates a comprehensive PDF report including analysis and suggestions."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # --- Report Header ---
    pdf.set_fill_color(240, 240, 240) # Light grey background
    pdf.cell(0, 12, "Credit Scoring & Eligibility Report", ln=True, align="C", fill=True)
    pdf.ln(10)
    
    # --- Assessment Result Section ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Assessment Result", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, f"Predicted Credit Score: {score}")
    pdf.multi_cell(0, 6, f"Eligibility Status: {eligibility}")
    pdf.ln(5)
    
    # --- Analysis & Suggestions Sections ---
    if reasons or suggestions:
        pdf.line(10, pdf.get_y(), 200, pdf.get_y()) # Draw a line separator
        pdf.ln(5)

    if reasons:
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(220, 50, 50) # Red color for reasons
        pdf.cell(0, 8, "Key Factors for Ineligibility", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.set_text_color(0, 0, 0) # Reset to black
        for i, reason in enumerate(reasons, 1):
            pdf.multi_cell(0, 6, f"{i}. {reason}")
        pdf.ln(5)

    if suggestions:
        suggestion_title = "Suggestions for Improvement" if reasons else "Recommendations to Maintain Your Good Score"
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(0, 100, 0) # Green color for suggestions
        pdf.cell(0, 8, suggestion_title, ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.set_text_color(0, 0, 0) # Reset to black
        for i, suggestion in enumerate(suggestions, 1):
            pdf.multi_cell(0, 6, f"{i}. {suggestion}")
        pdf.ln(5)
        
    # --- Final Output ---
    # This is a more robust way to get the byte output for download
    return pdf.output(dest="S").encode("latin-1")

# --- Streamlit UI ---
st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide", page_icon="💳")
st.title("📊 Advanced Credit Scoring Dashboard")
st.markdown("Enter applicant details to receive a comprehensive, real-time credit risk assessment.")

# This radio button is outside the form to allow the UI to update instantly
has_prev_loans_choice = st.radio(
    "Do you have any existing loans?",
    ("No", "Yes"),
    horizontal=True,
    label_visibility="visible"
)

with st.form("prediction_form"):
    
    st.subheader("Personal Information")
    col1, col2, col3 = st.columns(3)
    cibil = col1.slider("CIBIL Score", 300, 900, 650)
    age = col2.slider("Age", 18, 100, 30)
    emp = col3.slider("Years Employed", 0, 50, 5)

    st.subheader("Financial Information")
    col4, col5, col6 = st.columns(3)
    income = col4.number_input("Total Annual Income", min_value=0, value=500000, step=10000)
    loan_amount = col5.number_input("Loan Amount Requested", min_value=0, value=1000000, step=25000)
    annuity = col6.number_input("Proposed Monthly Payment (EMI/Annuity)", min_value=0, value=25000, step=1000)
    
    # Conditionally display the extra inputs inside the form
    if has_prev_loans_choice == "Yes":
        st.subheader("Existing Loan Details")
        col7, col8, col9 = st.columns(3)
        prev_loan_count = col7.number_input("Number of existing loans", min_value=0, value=1)
        prev_outstanding_amt = col8.number_input("Total outstanding loan amount", min_value=0, value=200000, step=10000)
        prev_remaining_emi = col9.number_input("Total remaining EMIs on all loans", min_value=0, value=12)
    else:
        prev_loan_count, prev_outstanding_amt, prev_remaining_emi = 0, 0, 0

    predict_button = st.form_submit_button("🔎 Calculate Eligibility & Score")

if predict_button:
    user_inputs = {
        'cibil': cibil, 'age': age, 'emp': emp,
        'income': income, 'loan_amount': loan_amount, 'annuity': annuity,
        'has_prev_loans': (has_prev_loans_choice == "Yes"), 
        'prev_loan_count': prev_loan_count,
        'prev_outstanding_amt': prev_outstanding_amt, 
        'prev_remaining_emi': prev_remaining_emi
    }
    st.session_state.user_inputs = user_inputs

    with st.spinner("Analyzing profile..."):
        try:
            full_payload = map_to_full_payload(user_inputs)
            predict_url = f"{API_URL}/predict"
            response = requests.post(predict_url, json=full_payload, timeout=10)
            response.raise_for_status()
            st.session_state.prediction_result = response.json()
        except requests.RequestException as e:
            st.error(f"API Call Failed: {e}")
            st.session_state.prediction_result = None

# --- Display Results ---
if "prediction_result" in st.session_state and st.session_state.prediction_result:
    result = st.session_state.prediction_result
    score = result['credit_score']
    prob = result['probability_of_default']
    
    st.success("Analysis Complete!")
    st.metric("Predicted Credit Score", score)
    
    if score >= 670:
        eligibility = "Eligible for Loan"
        st.success(f"✅ {eligibility}")
        reasons, suggestions = [], get_approval_suggestions()
        st.subheader("Congratulations & Recommendations")
        st.info("Your financial profile is strong. Here are some tips to maintain your excellent credit score:")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
    else:
        eligibility = "Not Eligible for Loan"
        st.error(f"❌ {eligibility}")
        reasons, suggestions = get_rejection_analysis_and_suggestions(st.session_state.user_inputs)
        st.subheader("Detailed Analysis")
        st.warning("Our analysis indicates the following key factors impacted your score:")
        for reason in reasons:
            st.markdown(f"- **{reason}**")
        st.subheader("Actionable Suggestions")
        st.info("To improve your eligibility for future applications, we recommend the following:")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

    pdf_bytes = generate_pdf_report(score, prob, eligibility, reasons, suggestions)
    st.download_button(
        label="📄 Download Full Report as PDF",
        data=pdf_bytes,
        file_name=f"credit_report_score_{score}.pdf",
        mime="application/pdf"
    )
