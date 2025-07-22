import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# --- Load the trained model and label encoder ---
# Ensure this path is correct if the model file is not in the same directory
try:
    model = joblib.load('salary_prediction_model_lgbm.pkl')
    le = LabelEncoder()
    le.fit(['<=50K', '>50K']) # Fit with the exact classes the model was trained on
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'salary_prediction_model_lgbm.pkl' not found. Please train and save the model first.")
    st.stop()


# --- Streamlit App Layout ---
st.set_page_config(page_title="Employee Salary Prediction", layout="wide")

st.title("💰 Employee Salary Prediction (Income Bracket)")
st.markdown("""
    This application predicts whether an individual's income is **<=50K** or **>50K**
    based on various demographic and work-related attributes.
""")

st.sidebar.header("User Input Features")

# --- Function to get user input ---
def get_user_input():
    # Define the unique values for each categorical feature based on your data analysis
    # These lists should match the categories seen during model training
    workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay']
    education_options = ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc', '11th', 'Assoc-acdm',
                         '10th', '7th-8th', 'Prof-school', '9th', '12th', 'Doctorate', '5th-6th', '1st-4th', 'Preschool']
    marital_status_options = ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed',
                              'Married-spouse-absent', 'Married-AF-spouse']
    occupation_options = ['Craft-repair', 'Prof-specialty', 'Exec-managerial', 'Adm-clerical', 'Sales',
                          'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners',
                          'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
    relationship_options = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
    race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
    gender_options = ['Male', 'Female']
    native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada',
                              'El-Salvador', 'India', 'Cuba', 'England', 'China', 'Jamaica', 'South', 'Italy',
                              'Dominican-Republic', 'Japan', 'Guatemala', 'Vietnam', 'Columbia', 'Poland',
                              'Haiti', 'Portugal', 'Iran', 'Taiwan', 'Greece', 'Nicaragua', 'Peru', 'Ecuador',
                              'France', 'Ireland', 'Thailand', 'Hong', 'Cambodia', 'Trinadad&Tobago', 'Yugoslavia',
                              'Outlying-US(Guam-USVI-etc)', 'Laos', 'Scotland', 'Honduras', 'Hungary',
                              'Holand-Netherlands']

    age = st.sidebar.slider("Age", 17, 90, 35)
    workclass = st.sidebar.selectbox("Workclass", workclass_options)
    fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=13492, max_value=1490400, value=189734)
    education = st.sidebar.selectbox("Education", education_options)
    educational_num = st.sidebar.slider("Educational Num (Years of Education)", 1, 16, 10)
    marital_status = st.sidebar.selectbox("Marital Status", marital_status_options)
    occupation = st.sidebar.selectbox("Occupation", occupation_options)
    relationship = st.sidebar.selectbox("Relationship", relationship_options)
    race = st.sidebar.selectbox("Race", race_options)
    gender = st.sidebar.selectbox("Gender", gender_options)
    capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
    capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=4356, value=0)
    hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
    native_country = st.sidebar.selectbox("Native Country", native_country_options)

    # Create a DataFrame from user inputs
    data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = get_user_input()

st.subheader("User Input Features")
st.write(input_df)

# --- Prediction ---
if st.button("Predict Income"):
    prediction_encoded = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    prediction_label = le.inverse_transform(prediction_encoded)

    st.subheader("Prediction Result")
    if prediction_label[0] == '>50K':
        st.success(f"Based on the provided features, the predicted income is: **{prediction_label[0]}** 🎉")
    else:
        st.info(f"Based on the provided features, the predicted income is: **{prediction_label[0]}**")

    st.subheader("Prediction Probability")
    proba_df = pd.DataFrame(prediction_proba, columns=le.classes_)
    st.write(proba_df)

    st.markdown("""
    **Interpretation of Probability:**
    - The first column (`<=50K`) shows the probability of income being less than or equal to 50K.
    - The second column (`>50K`) shows the probability of income being greater than 50K.
    """)

st.markdown("""
---
**About the Model:**
This application uses a LightGBM Classifier trained on the Adult Income dataset.
""")
