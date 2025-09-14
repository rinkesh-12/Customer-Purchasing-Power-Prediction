import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('E:\\Data_Analysis_Datasets\\ML\\Social_Network_Ads.csv')

le = LabelEncoder()
le.fit(df['Gender'])

# --- Load Data and Pipeline ---
# model = pk.load(open(r"C:\Users\91787\Desktop\PY YT\ML\Project\Customer Purchasing Power ML Model\pickle.pkl", "rb"))

# pickle_path = r"C:\Users\91787\Desktop\PY YT\ML\Project\Customer Purchasing Power ML Model\pickle.pkl"
pickle_path = r"C:\Users\91787\Desktop\PY YT\ML\Project\Customer Purchasing Power ML Model\pickle_multiple_model.pkl"

if os.path.exists(pickle_path):
    model = pk.load(open(pickle_path, 'rb'))
else:
    st.error("❌ Model file not found! Check the path or filename.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Customer Purchasing Power ML Model", layout="wide")
st.title("Customer Purchasing Power ML Model")
st.write("This application predicts the Customer Purchasing Power based on various features.")

# Load dataset


gender = st.selectbox('Select Gender Brand', ['Male', 'Female'])
age	= st.text_input("Enter your Age:")
estimatedSalary = st.text_input("Enter your Estimated Salary:")

if st.button("Predict"):
    if age.isdigit() and estimatedSalary.isdigit():
        # Encode Gender with the loaded encoder
        gender_encoded = le.transform([gender])[0]
        input_data = pd.DataFrame([[gender_encoded, age, estimatedSalary]], columns=['Gender', 'Age', 'EstimatedSalary'])
        prediction = model.predict(input_data)
        
        # prob = model.predict_proba(input_data)[0]
        if prediction[0] == 0:
            st.error("🛑 Sorry! Based on your profile, you are **not eligible to purchase** at this time.")
        else:
            st.success("✅ Congratulations! Based on your profile, you are **eligible to purchase**.")

        # --- Visualization ---
        # st.bar_chart(pd.DataFrame({"Probability": prob}, index=["Not Purchased (0)", "Purchased (1)"]))
        # if hasattr(model, "predict_proba"):
        #     prob = model.predict_proba(input_data)[0]
        #     st.bar_chart(pd.DataFrame({"Probability": prob}, index=["Not Purchased (0)", "Purchased (1)"]))
        # else:
        #     st.warning("⚠️ This model does not support probability estimates.")

    else:
        st.error("Please enter valid numeric values for Age and Estimated Salary.")

# st.sidebar.markdown("### 🏆 Best Model Info")
# st.sidebar.write(f"**Model:** {best_model_name}")
# st.sidebar.write(f"**Test Accuracy:** {best_model.score(X_test, y_test):.2f}")
