import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Configuration ---
MODEL_PATH = 'salary_model.pkl'
ENCODER_GENDER_PATH = 'le_gender.pkl'
ENCODER_EDUCATION_PATH = 'le_education.pkl'
ENCODER_ROLE_PATH = 'le_role.pkl'
METRICS_PATH = 'model_performance_metrics.json'

# --- Load Model and Encoders ---
@st.cache_resource
@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        le_gender = joblib.load(ENCODER_GENDER_PATH)
        le_education = joblib.load(ENCODER_EDUCATION_PATH)
        le_role = joblib.load(ENCODER_ROLE_PATH)
        return model, le_gender, le_education, le_role
    except FileNotFoundError as e:
        st.error(f"Error loading model assets: {e}. Run 'train_model.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading assets: {e}")
        st.stop()

def load_metrics():
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return metrics
    except:
        return None

metrics = load_metrics()

# --- Page Functions ---
def home_page():
    st.title("Welcome to the Employee Salary Prediction App! 💰")
    
    st.markdown("""
        This application allows you to predict an employee's potential income based on their
        age, gender, experience, education level, and job role.

        ---
        
        How it works:
        This app uses a Machine Learning model (Random Forest Regressor) trained
        on a dataset of employee information and their corresponding incomes.
    """)
    st.markdown("---")
    st.markdown("Developed with ❤ using Streamlit and Scikit-learn.")

def salary_predictor_page():
    st.title("Predict Employee Salary 📈")
    st.markdown("Enter employee details to get an estimated income.")

    GENDER_OPTIONS = ['Male', 'Female', 'Other']
    EDUCATION_OPTIONS = ['High School', 'Bachelors', 'Masters', 'PhD']
    ROLE_OPTIONS = [
        'Software Engineer', 'Data Scientist', 'Project Manager',
        'HR Assistant', 'Marketing Specialist', 'Financial Analyst',
        'UX Designer', 'Intern', 'Senior Developer', 'Director'
    ]

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=18, max_value=60, value=30, step=1)
        gender = st.selectbox("Gender", options=GENDER_OPTIONS)

        if age == 18:
            experience = 0
            st.write("Experience (Years): 0 (Minimum possible due to age)")
        else:
            max_experience = age - 18
            experience = st.slider("Experience (Years)", min_value=0, max_value=max_experience, value=0)

    with col2:
        education = st.selectbox("Education Level", options=EDUCATION_OPTIONS)
        role = st.selectbox("Job Role", options=ROLE_OPTIONS)

    if st.button("Predict Salary"):
        st.subheader("Prediction Result")
        try:
            gender_encoded = le_gender.transform([gender])[0]
            education_encoded = le_education.transform([education])[0]
            role_encoded = le_role.transform([role])[0]

            input_df = pd.DataFrame([[age, gender_encoded, experience, education_encoded, role_encoded]],
                                    columns=['Age', 'Gender', 'Experience', 'Education', 'Role'])

            predicted_income = model.predict(input_df)[0]

            st.success(f"Predicted Income: ₹{predicted_income:,.2f}")
        except ValueError as ve:
            st.error(f"Input Error: {ve}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    st.markdown("---")

def performance_page():
    st.title("Model Performance Overview 📊")
    st.markdown("Here are the key metrics indicating how well the model performs on unseen data:")

    if metrics:
        col_r2, col_mae, col_rmse = st.columns(3)

        with col_r2:
            st.metric("R² Score", f"{metrics['r2_score']:.2f}")
        with col_mae:
            st.metric("MAE (Avg Error)", f"₹{metrics['mae']:,.0f}")
        with col_rmse:
            st.metric("RMSE", f"₹{metrics['rmse']:,.0f}")

        st.markdown("---")
        st.subheader("Performance Metrics Graph")
        chart_data = pd.DataFrame({
            'Metric': ['R² Score', 'MAE', 'RMSE'],
            'Value': [metrics['r2_score'], metrics['mae'], metrics['rmse']]
        })
        st.bar_chart(chart_data.set_index('Metric'))
        st.info("R² is a measure of fit (closer to 1 is better), while MAE and RMSE are error magnitudes (lower is better).")

    else:
        st.info("Model performance metrics could not be loaded.")
    st.markdown("---")

def overview_page():
    st.title("Salary Overview 📊")

    try:
        df = pd.read_csv("employee_data.csv")

        st.subheader("📌 Key Statistics")
        st.write(f"Average Salary: ₹{df['Income'].mean():,.2f}")
        st.write(f"Minimum Salary: ₹{df['Income'].min():,.2f}")
        st.write(f"Maximum Salary: ₹{df['Income'].max():,.2f}")

        st.markdown("---")
        st.subheader("📘 Average Salary by Education Level")
        edu_salary = df.groupby("Education")["Income"].mean().sort_values()
        st.bar_chart(edu_salary)

        st.markdown("---")
        st.subheader("📗 Average Salary by Job Role")
        role_salary = df.groupby("Role")["Income"].mean()

        fig, ax = plt.subplots()
        role_salary.sort_values().plot(kind='barh', color='skyblue', ax=ax)
        ax.set_xlabel("Average Salary (₹)")
        ax.set_title("Average Salary by Job Role")
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("📉 Actual vs Predicted Salary")

        X = df[['Age', 'Gender', 'Experience', 'Education', 'Role']]
        X['Gender'] = le_gender.transform(X['Gender'])
        X['Education'] = le_education.transform(X['Education'])
        X['Role'] = le_role.transform(X['Role'])

        y_actual = df['Income']
        y_predicted = model.predict(X)

        comparison_df = pd.DataFrame({'Actual': y_actual, 'Predicted': y_predicted})

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(comparison_df['Actual'].values, label='Actual', marker='o')
        ax2.plot(comparison_df['Predicted'].values, label='Predicted', linestyle='--', marker='x')
        ax2.set_title("Actual vs Predicted Salary")
        ax2.set_ylabel("Salary (₹)")
        ax2.set_xlabel("Employee Index")
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("---")
    except Exception as e:
        st.error(f"Error loading overview data: {e}")

# --- Main App ---
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💰",
    layout="centered"
)

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Home", "Salary Predictor", "Performance", "Overview"])

# Load model and encoders once
model, le_gender, le_education, le_role = load_assets()

if page == "Home":
    home_page()
elif page == "Salary Predictor":
    salary_predictor_page()
elif page == "Performance":
    performance_page()
elif page == "Overview":
    overview_page()
