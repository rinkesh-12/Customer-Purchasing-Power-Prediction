# Customer Purchasing Power Prediction

This project predicts whether a customer is likely to **purchase a vehicle** based on demographic and financial attributes.  
The solution involves **data analysis, machine learning model development, hyperparameter tuning, and deployment with Streamlit**.

---

## Features

- **Exploratory Data Analysis (EDA):**
  - Age and salary distribution
  - Gender-wise purchasing trends
  - Outlier detection with boxplots
  - Correlation heatmap  

- **Machine Learning Models:**
  - Logistic Regression  
  - Random Forest Classifier  
  - Support Vector Classifier (SVC)  
  - K-Nearest Neighbors (KNN)  
  - Gradient Boosting Classifier  

- **Model Selection & Tuning:**
  - Applied **GridSearchCV** with cross-validation  
  - Best model: **Random Forest (max_depth=5, n_estimators=50)**  
  - Achieved **92.5% accuracy** on test data  

- **Deployment:**
  - Built an interactive **Streamlit app**  
  - Users can input **Gender, Age, and Estimated Salary**  
  - Predicts whether a person will **purchase (1)** or **not purchase (0)**  

---

## Tech Stack

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Model Deployment:** Streamlit  
- **Serialization:** Pickle  

---

## Project Structure

├── app.py # Streamlit web app

├── Customer Purchasing Power.ipynb # Jupyter Notebook (EDA & ML training)

├── pickle_multiple_model.pkl # Trained model file

├── Social_Network_Ads.csv # Dataset

├── requirements.txt # Dependencies

└── README.md # Documentation
