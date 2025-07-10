import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------ Load Your Model ------------------

@st.cache_resource
def load_model():
    with open('student_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

model = load_model()

# ------------------ Streamlit App UI ------------------

st.title("🎓 Student Exam Score Predictor")
st.write("Fill in the student's details below to predict their exam score:")

# ------------------ User Input ------------------

attendance = st.slider('Attendance (%)', 0, 100, 75)
hours_studied = st.slider('Hours Studied', 0, 100, 30)
previous_scores = st.slider('Previous Scores', 0, 100, 50)
tutoring_sessions = st.slider('Tutoring Sessions per week', 0, 10, 2)
sleep_hours = st.slider('Average Sleep Hours', 0, 12, 7)
physical_activity = st.slider('Physical Activity Hours per week', 0, 20, 3)

# Example categorical fields as selects
parental_involvement = st.selectbox('Parental Involvement', ['High', 'Medium', 'Low'])
access_to_resources = st.selectbox('Access to Resources', ['High', 'Medium', 'Low'])
motivation_level = st.selectbox('Motivation Level', ['High', 'Medium', 'Low'])
peer_influence = st.selectbox('Peer Influence', ['Negative', 'Neutral', 'Positive'])
teacher_quality = st.selectbox('Teacher Quality', ['High', 'Medium', 'Low'])
learning_disabilities_yes = st.checkbox('Learning Disabilities')

# ------------------ Prepare Input Data ------------------

# Initialize input dict with numeric features
input_dict = {
    'Attendance': attendance,
    'Hours_Studied': hours_studied,
    'Previous_Scores': previous_scores,
    'Tutoring_Sessions': tutoring_sessions,
    'Sleep_Hours': sleep_hours,
    'Physical_Activity': physical_activity,
    'Learning_Disabilities_Yes': int(learning_disabilities_yes),
}

# One-hot encode categorical inputs manually
# Following columns were generated in your training with drop_first=True, so exclude first categories

# Parental_Involvement_Low, Parental_Involvement_Medium (High is baseline, so no column)
input_dict['Parental_Involvement_Low'] = 1 if parental_involvement == 'Low' else 0
input_dict['Parental_Involvement_Medium'] = 1 if parental_involvement == 'Medium' else 0

# Access_to_Resources_Low, Access_to_Resources_Medium
input_dict['Access_to_Resources_Low'] = 1 if access_to_resources == 'Low' else 0
input_dict['Access_to_Resources_Medium'] = 1 if access_to_resources == 'Medium' else 0

# Motivation_Level_Low, Motivation_Level_Medium
input_dict['Motivation_Level_Low'] = 1 if motivation_level == 'Low' else 0
input_dict['Motivation_Level_Medium'] = 1 if motivation_level == 'Medium' else 0

# Peer_Influence_Positive, Peer_Influence_Neutral (Negative is baseline)
input_dict['Peer_Influence_Positive'] = 1 if peer_influence == 'Positive' else 0
input_dict['Peer_Influence_Neutral'] = 1 if peer_influence == 'Neutral' else 0

# Teacher_Quality_Low, Teacher_Quality_Medium (High is baseline)
input_dict['Teacher_Quality_Low'] = 1 if teacher_quality == 'Low' else 0
input_dict['Teacher_Quality_Medium'] = 1 if teacher_quality == 'Medium' else 0

input_data = pd.DataFrame([input_dict])

# ------------------ Make Prediction ------------------

if st.button('Predict Exam Score'):
    # Make sure input_data columns match model's expected features exactly
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_features]

    prediction = model.predict(input_data)
    st.success(f"Predicted Exam Score: {round(prediction[0], 2)}")
