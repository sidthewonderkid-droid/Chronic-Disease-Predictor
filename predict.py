import joblib
import pandas as pd
import streamlit as st
import google.generativeai as genai

st.title("Prediction of Diseases")
user_input = st.selectbox("Select the disease you want to check for", ["Arrythmia","Cancer","Asthma","Obesity","Diabetes"])

user_data = {}
# user_input = int(input("Enter 1 for arrythmia, 2 for cancer, 3 for asthma, 4 for obesity, or 5 for diabetes."))

if user_input == "Arrythmia":
    disease = "heart_model.joblib"
    dataset = "heart.csv"
    target = "target"
elif user_input == "Cancer":
    disease = "cancer_model.joblib"
    dataset = "cancer.csv"
    target = "Diagnosis"
elif user_input == "Asthma":
    disease = "asthma_model.joblib"
    dataset = "asthma.csv"
    target = ["Severity_Mild", "Severity_Moderate", "Severity_None"]
elif user_input == "Obesity":
    disease = "obesity_model.joblib"
    dataset = "obesity.csv"
    target = "ObesityCategory"
elif user_input == "Diabetes":
    disease = "diabetes_model.joblib"
    dataset = "diabetes.csv"
    target = "Outcome"

model = joblib.load(disease)
df = pd.read_csv(dataset)
features = df.drop(target, axis=1).columns

answer = ""
for feature in features:
    value = st.number_input(f"{feature}", min_value=0, max_value=100, value=10, step=1)
    value = int(value)
    user_data[feature] = value

# Configure Gemini API
genai.configure(api_key="AIzaSyDxABSo7lTkNtox8DwYtm8qxE2kjSeK8gM")

# Initialize model
llm_model = genai.GenerativeModel("gemini-2.5-flash")

# Collect user inputs
symptoms = st.text_input("Please describe your current symptoms: ")
duration = st.text_input("How long have you been experiencing these symptoms? ")
diet = st.text_input("Describe your typical daily diet (meals, snacks, beverages): ")
exercise = st.text_input("How often do you exercise and what kind of activities do you do? ")
sleep = st.text_input("How many hours of sleep do you usually get per night? ")
hydration = st.text_input("How much water do you drink per day? ")
stress = st.text_input("On a scale of 1–10, how would you rate your stress level? ")
medical_history = st.text_input("Do you have any existing health conditions or take any medications? ")

if st.button("Analyze my lifestyle"):
    with st.spinner("Generating content..."):
        
# Build detailed prompt for Gemini
        prompt = f"""
You are a medical and lifestyle assistant.
Based on the following user details, provide:
1. Possible causes of the symptoms.
2. Evidence-based lifestyle suggestions.
3. Dietary improvements.
4. Stress management and sleep advice.
5. When the user should consider seeing a doctor.

User Details:
- Symptoms: {symptoms}
- Duration: {duration}
- Diet: {diet}
- Exercise habits: {exercise}
- Sleep pattern: {sleep}
- Hydration: {hydration}
- Stress level: {stress}
- Medical history or medications: {medical_history}
These are some more user inputs entered by the user: {user_data}
They result given by the LLM is {answer}
Please respond with a structured, friendly, and helpful summary.
"""


# Generate response
        prediction = 1
        try:
            input_df = pd.DataFrame([user_data])
            prediction = model.predict(input_df)[0]

            if prediction == 1:
                answer = "Yes, the test is positive."
            else:
                answer = "No the test is negative."
        except:
            if prediction == 1:
                answer = "Yes, the test is positive."
            else:
                answer = "No the test is negative."
        response = llm_model.generate_content(prompt)

# Display model output
    st.write("\nGemini Response:\n")

    st.write(response.text)
