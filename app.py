# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("salary_model.pkl")  # Update filename if needed

# Title
st.title("💼 Employee Salary Class Prediction")
st.markdown("Predict whether an employee earns >50K or <=50K based on input details.")

# Sidebar for individual input
st.sidebar.header("Enter Employee Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=70, value=30)
education = st.sidebar.selectbox("Education Level", ['Bachelors', 'HS-grad', 'Masters', 'Some-college', 'Assoc-acdm'])
occupation = st.sidebar.selectbox("Occupation", ['Exec-managerial', 'Craft-repair', 'Adm-clerical', 'Sales', 'Other-service'])
hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)
experience = st.sidebar.number_input("Years of Experience", 0, 50, 5)

# Prepare single input
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.subheader("🔎 Input Data Preview")
st.write(input_df)

# Predict button
if st.button("🔮 Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch Prediction
st.markdown("---")
st.subheader("📂 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview", batch_data.head())
    
    try:
        batch_preds = model.predict(batch_data)
        batch_data['PredictedClass'] = batch_preds
        st.success("✅ Batch Prediction Completed")
        st.write(batch_data.head())

        # Download button
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV with Predictions", csv, "salary_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"⚠️ Error during batch prediction: {e}")
