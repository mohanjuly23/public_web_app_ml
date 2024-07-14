import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App
st.set_page_config(page_title="E-commerce Customer Retention Analysis", page_icon="🛒")

st.title("🛒 E-commerce Customer Retention Analysis")
st.markdown("Predict whether a customer will churn based on their profile and usage data. Please enter the customer details below:")

# Sidebar inputs
st.sidebar.title("Customer Data Input")
tenure = st.sidebar.number_input("Tenure (in months)", min_value=0, value=12, step=1)
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3], index=0)
cc_contacted_ly = st.sidebar.number_input("CC Contacted Last Year", min_value=0, value=0, step=1)
payment = st.sidebar.selectbox("Payment Method", [0, 1], index=0)  # 0: Other, 1: Credit Card
gender = st.sidebar.selectbox("Gender", [0, 1], index=0)  # 0: Female, 1: Male
service_score = st.sidebar.slider("Service Score", min_value=0, max_value=10, value=5)
account_user_count = st.sidebar.number_input("Account User Count", min_value=1, value=1, step=1)
account_segment = st.sidebar.selectbox("Account Segment", [1, 2, 3], index=0)
cc_agent_score = st.sidebar.slider("CC Agent Score", min_value=0, max_value=10, value=5)
marital_status = st.sidebar.selectbox("Marital Status", [0, 1], index=0)  # 0: Single, 1: Married
rev_per_month = st.sidebar.number_input("Revenue per Month", min_value=0, value=100, step=10)
complain_ly = st.sidebar.number_input("Complain Last Year", min_value=0, value=0, step=1)
rev_growth_yoy = st.sidebar.number_input("Revenue Growth YOY", min_value=-100, value=0, step=1)
coupon_used_for_payment = st.sidebar.number_input("Coupons Used for Payment", min_value=0, value=0, step=1)
day_since_cc_connect = st.sidebar.number_input("Days Since CC Connect", min_value=0, value=30, step=1)
cashback = st.sidebar.number_input("Cashback Received", min_value=0, value=0, step=1)
login_device = st.sidebar.selectbox("Login Device", [1, 2, 3], index=0)  # 1: Mobile, 2: Desktop, 3: Tablet

# Input data dictionary
input_data = {
    'Tenure': tenure,
    'City_Tier': city_tier,
    'CC_Contacted_LY': cc_contacted_ly,
    'Payment': payment,
    'Gender': gender,
    'Service_Score': service_score,
    'Account_user_count': account_user_count,
    'account_segment': account_segment,
    'CC_Agent_Score': cc_agent_score,
    'Marital_Status': marital_status,
    'rev_per_month': rev_per_month,
    'Complain_ly': complain_ly,
    'rev_growth_yoy': rev_growth_yoy,
    'coupon_used_for_payment': coupon_used_for_payment,
    'Day_Since_CC_connect': day_since_cc_connect,
    'cashback': cashback,
    'Login_device': login_device
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

if st.sidebar.button("Predict Churn"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[0]

    if prediction[0] == 1:
        st.error("Prediction: Churn")
    else:
        st.success("Prediction: No Churn")
    
    st.write("Prediction Probability:")
    st.write(f"No Churn: {prediction_proba[0]:.2f}")
    st.write(f"Churn: {prediction_proba[1]:.2f}")

# Additional Information and Tips
st.sidebar.title("Tips")
st.sidebar.info("""
- Ensure all input fields are filled correctly.
- Click on "Predict Churn" to see the results.
- The model uses logistic regression to predict the likelihood of churn.
""")

st.sidebar.title("About")
st.sidebar.info("""
This application uses a machine learning model to predict customer churn in an e-commerce setting. The model was trained on historical customer data and uses various customer features to make predictions.
""")
