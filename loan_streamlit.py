import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os

# --- Configuration and Data Loading ---
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

st.title("Loan Approval Prediction App")
st.write("This application predicts loan approval based on various financial and personal factors.")

# Define a function to load data and cache it
@st.cache_data
def load_data():
    try:
        loan_df = pd.read_csv("loan_approval_dataset.csv")
    except FileNotFoundError:
        st.error("Error: 'loan_approval_dataset.csv' not found. Please ensure the CSV file is in the same directory as the app.")
        st.stop()
    
    loan_df.dropna(inplace=True)
    loan_df.columns = loan_df.columns.str.strip()

    label_encoder = LabelEncoder()
    # It's better to fit_transform on the original columns, then reassign
    loan_df['education_encoded'] = label_encoder.fit_transform(loan_df['education'])
    # Make sure 'self_employed' is treated correctly as a string/object before encoding
    loan_df['self_employed_encoded'] = label_encoder.fit_transform(loan_df['self_employed'])
    
    # Overwrite original columns for consistency with feature list
    loan_df['self_employed'] = loan_df['self_employed_encoded']
    loan_df['education'] = loan_df['education_encoded']
    
    # Drop the temporary numeric columns if not needed explicitly for display
    loan_df.drop('education_encoded', axis=1, inplace=True)
    loan_df.drop('self_employed_encoded', axis=1, inplace=True)

    return loan_df

loan_df = load_data()

# --- Model Training and Saving (Run only once or when model needs retraining) ---
model_path = 'random_forest_model.joblib'
# Define features here, so they are available even if the model is loaded
features = [
    'no_of_dependents', 'education', 'self_employed',
    'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value'
]
target = 'loan_status' # Define target as well

if not os.path.exists(model_path):
    st.info("Training model... This may take a moment.")
    
    X = loan_df[features]
    y = loan_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    st.success("Successfully Predicted")
else:
    st.info("Loading......")
    model = joblib.load(model_path)


# --- Exploratory Data Analysis (EDA) Section ---
st.header("Loan Applicants")

if st.checkbox("Show Raw Data"):
    st.subheader("Raw Loan Data")
    st.dataframe(loan_df.head())

# Loan Status Distribution
if st.checkbox("Show Loan Status Distribution", value=True, key="loan_status_dist_checkbox"):
    st.subheader("Loan Status Distribution")
    loan_status_counts = loan_df['loan_status'].value_counts()
    # Drastically reduced figure size: (2, 1.5) inches
    fig1, ax1 = plt.subplots(figsize=(1, 0.5))
    ax1.bar(loan_status_counts.index, loan_status_counts.values, color=['green', 'red'])
    ax1.set_title('Loan Status', fontsize=7) #  title
    ax1.set_xlabel('Status', fontsize=6) #  label
    ax1.set_ylabel('Count', fontsize=6)
    ax1.tick_params(axis='both', which='major', labelsize=5) # tick labels
    st.pyplot(fig1)
    plt.close(fig1)

# Education Distribution
if st.checkbox("Show Education Distribution", value=False, key="education_dist_checkbox"):
    st.subheader("Education Distribution")
    education_counts = loan_df['education'].value_counts()
  
    fig2, ax2 = plt.subplots(figsize=(2, 2))
    ax2.pie(education_counts, labels=["Grad", "Not Grad"], autopct='%1.0f%%', colors=['blue', 'yellow'], textprops={'fontsize': 6}) # Shorter labels, no decimals
    ax2.set_title('Education', fontsize=7) # Smaller title
    st.pyplot(fig2)
    plt.close(fig2)

# Loan Status based on Education
if st.checkbox("Show Loan Status by Education Level", value=False, key="education_loan_status_checkbox"):
    st.subheader("Loan Status by Education Level")
    education_loan_status_counts = loan_df.groupby(['education', 'loan_status']).size().unstack()
    
    fig3, ax3 = plt.subplots(figsize=(2.5, 1.8))
    education_loan_status_counts.plot(kind='bar', stacked=True, ax=ax3)
    ax3.set_title('Loan by Education', fontsize=7)
    ax3.set_xlabel('Education (0:Grad, 1:Not Grad)', fontsize=6)
    ax3.set_ylabel('Count', fontsize=6)
    ax3.legend(title='Status', loc='upper right', labels=['Approved', 'Rejected'], fontsize=5, title_fontsize=6)
    ax3.tick_params(axis='both', which='major', labelsize=5)
    plt.xticks(rotation=0)
    st.pyplot(fig3)
    plt.close(fig3)

# CIBIL Score Distribution
if st.checkbox("Show CIBIL Score Distribution", value=False, key="cibil_dist_checkbox"):
    st.subheader("CIBIL Score Distribution by Loan Status")
    
    fig4, ax4 = plt.subplots(figsize=(2.5, 1.8))
    sns.histplot(data=loan_df, x='cibil_score', bins=4, hue='loan_status', ax=ax4)
    ax4.set_title('CIBIL Score', fontsize=7)
    ax4.set_xlabel('Score', fontsize=6)
    ax4.set_ylabel('Count', fontsize=6)
    ax4.tick_params(axis='both', which='major', labelsize=5)
    st.pyplot(fig4)
    plt.close(fig4)

# Effects of Dependents
if st.checkbox("Show Loan Status by Number of Dependents", value=False, key="dependents_checkbox"):
    st.subheader("Loan Status by Number of Dependents")
    
    fig5, ax5 = plt.subplots(figsize=(2.5, 1.8))
    sns.countplot(x='no_of_dependents', data=loan_df, hue='loan_status', ax=ax5)
    ax5.set_title('Loan by Dependents', fontsize=7)
    ax5.set_xlabel('Dependents', fontsize=6)
    ax5.set_ylabel('Count', fontsize=6)
    ax5.tick_params(axis='both', which='major', labelsize=5)
    st.pyplot(fig5)
    plt.close(fig5)

# Effects of Employment and Education
if st.checkbox("Show Self Employed vs. Education", value=False, key="employment_education_checkbox"):
    st.subheader("Self Employed vs. Education")
   
    fig6, ax6 = plt.subplots(figsize=(2.5, 1.8))
    sns.countplot(x='self_employed', data=loan_df, hue='education', ax=ax6)
    ax6.set_title('Self Employed vs. Education', fontsize=7)
    ax6.set_xlabel('Self Employed (0:No, 1:Yes)', fontsize=6)
    ax6.set_ylabel('Count', fontsize=6)
    ax6.tick_params(axis='both', which='major', labelsize=5)
    st.pyplot(fig6)
    plt.close(fig6)

# --- Loan Approval Prediction Section ---
st.header("2. Predict Loan Approval")
st.write("Enter the details below to predict loan approval.")

# --- Custom Input / User Input for Prediction ---
with st.form("loan_prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        no_of_dependents = st.slider("Number of Dependents", 0, 5, 0)
        education_input = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed_input = st.selectbox("Self-Employed", ["No", "Yes"])
        income_annum = st.number_input("Annual Income", min_value=0.0, value=500000.0, step=10000.0, format="%.2f")
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=200000.0, step=10000.0, format="%.2f")
        loan_term = st.slider("Loan Term (in months)", 12, 360, 120)

    with col2:
        cibil_score = st.slider("CIBIL Score", 300, 900, 700)
        residential_assets_value = st.number_input("Residential Assets Value", min_value=0.0, value=100000.0, step=10000.0, format="%.2f")
        commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0.0, value=50000.0, step=10000.0, format="%.2f")
        luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0.0, value=75000.0, step=10000.0, format="%.2f")
        bank_asset_value = st.number_input("Bank Asset Value", min_value=0.0, value=30000.0, step=10000.0, format="%.2f")

    education_numeric = 0 if education_input == "Graduate" else 1
    self_employed_numeric = 1 if self_employed_input == "Yes" else 0

    submitted = st.form_submit_button("Predict Loan Approval")

    if submitted:
        user_data = pd.DataFrame(
            [[no_of_dependents, education_numeric, self_employed_numeric, income_annum, loan_amount, loan_term, cibil_score,
              residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]],
            columns=['no_of_dependents', 'education', 'self_employed',
                     'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                     'residential_assets_value', 'commercial_assets_value',
                     'luxury_assets_value', 'bank_asset_value']
        )

        prediction = model.predict(user_data)[0]

        st.subheader("Prediction Result:")
        if prediction == ' Approved':
            st.success("🎉 **Loan Application: APPROVED!** Congratulations! Your loan is likely to be approved based on the provided details.")
            st.balloons()
        else:
            
            st.error("😞 **Sorry! Loan Application: REJECTED.** Based on the provided details, the loan is likely to be rejected. Please review the input factors or try adjusting them.")
            st.snow() 