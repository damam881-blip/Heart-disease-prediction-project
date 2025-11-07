import streamlit as st
import pandas as pd
import joblib
import os
from pyngrok import ngrok

script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_dir, '..', 'models', 'final_model.pkl')
scaler_path = os.path.join(script_dir, '..', 'models', 'scaler.pkl')
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.write("--- Model and Scaler Loaded Successfully (for debugging) ---")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

port = 8501

try:
    public_url = ngrok.connect(port)
    print(f"--- Ngrok tunnel established at: {public_url} ---")
    st.success(f"Public URL: {public_url}")
except Exception as e:
    print(f"Error connecting to Ngrok: {e}")
    pass

original_features = [
    'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak',
    'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
    'restecg_normal', 'restecg_st-t abnormality'
]

st.title('Heart Disease Prediction AI 🩺')
st.write("This app predicts the likelihood of heart disease based on user input.")

st.sidebar.header('Patient Health Data')
st.sidebar.write("Please fill in the details below:")

age = st.sidebar.slider('Age', 20, 80, 50)
sex = st.sidebar.selectbox('Sex (0=Female, 1=Male)', [0, 1])
trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 80, 200, 120)
chol = st.sidebar.slider('Cholesterol (chol)', 100, 600, 200)
fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
thalach = st.sidebar.slider('Max Heart Rate (thalach)', 60, 220, 150)
exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', [0, 1])
oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.2, 1.0, step=0.1)

st.sidebar.subheader("Chest Pain Type (cp)")

cp_type = st.sidebar.radio("Select one type:", 
                           ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic (No Pain)'])

cp_atypical_angina = 1 if cp_type == 'Atypical Angina' else 0
cp_non_anginal = 1 if cp_type == 'Non-Anginal Pain' else 0
cp_typical_angina = 1 if cp_type == 'Typical Angina' else 0

st.sidebar.subheader("Resting ECG (restecg)")
restecg_type = st.sidebar.radio("Select one type:",
                                ['Normal', 'ST-T Abnormality', 'LV Hypertrophy'])

restecg_normal = 1 if restecg_type == 'Normal' else 0
restecg_st_t_abnormality = 1 if restecg_type == 'ST-T Abnormality' else 0

if st.sidebar.button('Predict Heart Disease Risk'):

    user_input = {
        'age': age,
        'sex': sex,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'cp_atypical angina': cp_atypical_angina,
        'cp_non-anginal': cp_non_anginal,
        'cp_typical angina': cp_typical_angina,
        'restecg_normal': restecg_normal,
        'restecg_st-t abnormality': restecg_st_t_abnormality
    }
    
    input_df = pd.DataFrame([user_input], columns=original_features)
    
    try:
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        st.subheader('Prediction Result:')
        if prediction[0] == 0:
            st.success('**Result: Low Risk**')
            st.write(f"The model predicts you have a **low risk** of heart disease.")
            st.write(f"Confidence (Probability of No Disease): {prediction_proba[0][0]*100:.2f}%")
        else:
            st.error('**Result: High Risk**')
            st.write(f"The model predicts you have a **high risk** of heart disease.")
            st.write(f"Confidence (Probability of Disease): {prediction_proba[0][1]*100:.2f}%")

    except Exception as e:
        st.error(f"Error during prediction: {e}")