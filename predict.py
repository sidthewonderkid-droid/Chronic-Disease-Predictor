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
    target = ["Severity_Mild", "Severity_Moderate", "Severity_None"]  # These are our output columns
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
    # Special handling for Asthma features
    if user_input == "Asthma":
        if feature.startswith("Age_"):
            # For age ranges, use radio buttons
            age_selected = st.radio(f"Is the patient in the {feature.replace('Age_', '')} age range?", ["No", "Yes"])
            value = 1 if age_selected == "Yes" else 0
        elif feature.startswith("Gender_"):
            # For gender, use radio buttons
            gender_selected = st.radio(f"Is the patient's gender {feature.replace('Gender_', '')}?", ["No", "Yes"])
            value = 1 if gender_selected == "Yes" else 0
        else:
            # For symptoms, use checkboxes
            has_symptom = st.checkbox(f"Does the patient have {feature.replace('-', ' ')}?")
            value = 1 if has_symptom else 0
    elif user_input == "Arrythmia":
        # Special handling for Arrythmia features with grouping and help text
        if feature == "age":
            st.markdown("### Basic Information")
            value = st.number_input("Age", min_value=20, max_value=100, value=50, 
                help="Patient's age in years")
        elif feature == "sex":
            gender = st.radio("Gender", ["Female", "Male"],
                help="Patient's biological sex")
            value = 1 if gender == "Male" else 0
            
        elif feature == "cp":
            st.markdown("### Pain and Discomfort")
            cp_type = st.radio("Type of Chest Pain", 
                ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                help="Typical Angina: Classic heart-related chest pain\n" +
                     "Atypical Angina: Not all typical features\n" +
                     "Non-anginal: Not heart-related pain\n" +
                     "Asymptomatic: No chest pain")
            value = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp_type]
        elif feature == "exang":
            exang = st.radio("Chest Pain During Exercise?", ["No", "Yes"],
                help="Whether exercise induces angina (chest pain)")
            value = 1 if exang == "Yes" else 0
            
        elif feature == "trestbps":
            st.markdown("### Vital Measurements")
            value = st.number_input("Resting Blood Pressure (mm Hg)", 
                min_value=90, max_value=200, value=120,
                help="Blood pressure when patient is at rest")
        elif feature == "thalach":
            value = st.number_input("Maximum Heart Rate Achieved", 
                min_value=60, max_value=220, value=150,
                help="Highest heart rate recorded during exercise")
            
        elif feature == "chol":
            st.markdown("### Blood Tests")
            value = st.number_input("Cholesterol Level (mg/dl)", 
                min_value=100, max_value=600, value=200,
                help="Total cholesterol level in blood")
        elif feature == "fbs":
            fbs = st.radio("High Fasting Blood Sugar?", ["No", "Yes"],
                help="Is fasting blood sugar > 120 mg/dl?")
            value = 1 if fbs == "Yes" else 0
            
        elif feature == "restecg":
            st.markdown("### Technical Measurements")
            ecg = st.radio("Resting ECG Results", 
                ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                help="ECG results when patient is resting")
            value = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}[ecg]
        elif feature == "oldpeak":
            value = st.number_input("ST Depression", 
                min_value=0.0, max_value=6.0, value=0.0, step=0.1,
                help="ST depression induced by exercise relative to rest")
        elif feature == "slope":
            slope_type = st.radio("ST Segment Slope", ["Upsloping", "Flat", "Downsloping"],
                help="The slope of the peak exercise ST segment on ECG")
            value = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope_type]
        elif feature == "ca":
            value = st.number_input("Number of Major Vessels", 
                min_value=0, max_value=4, value=0,
                help="Number of major blood vessels colored by fluoroscopy (0-4)")
        elif feature == "thal":
            thal_type = st.radio("Blood Flow Test Result", 
                ["Normal", "Fixed Defect", "Reversible Defect"],
                help="Results from the thallium heart scan")
            value = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal_type]
    else:
        # Original handling for other diseases
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
# Generate response
#       prediction = 1
#        try:
#            input_df = pd.DataFrame([user_data])
#            prediction = model.predict(input_df)[0]
#
#            if prediction == 1:
#                answer = "Yes, the test is positive."
#            else:
#                answer = "No, the test is negative."
#        except:
#            if prediction == 1:
#                answer = "Yes, the test is positive."
#            else:
#                answer = "No, the test is negative."
#        response = llm_model.generate_content(prompt)
#
## Display model output
#    st.write("\nGemini Response:\n")
#    st.write(response.text)

#  Generate response
        prediction = -1
        
        # Run model prediction
        try:
            input_df = pd.DataFrame([user_data])
            
            # Debug statements removed
            if user_input == "Asthma":
                # For Asthma, get predictions and probabilities
                pred_array = model.predict(input_df)
                pred_proba = model.predict_proba(input_df)
                
                # Find the severity with highest probability
                max_prob = 0
                max_severity = None
                
                # Parse probabilities and find highest
                for idx, severity in enumerate(["Mild", "Moderate", "None"]):
                    # Get probability from the second column (probability of class 1)
                    prob_yes = pred_proba[idx][0][1]
                    if prob_yes > max_prob:
                        max_prob = prob_yes
                        max_severity = severity
                
                # Set prediction based on probability threshold
                if max_prob >= 0.3:  # 30% threshold
                    answer = f"Based on the model analysis, there is a {max_prob:.1%} probability of **{max_severity}** severity asthma."
                else:
                    answer = "Based on the model analysis, no significant severity level was predicted (all probabilities below 30%)."
                
                prediction = 1 if max_prob >= 0.3 else 0
            else:
                # For other diseases, use the original logic
                pred_array = model.predict(input_df)
                prediction = pred_array.flatten()[0].item()
            # ---------------------------------------------------
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            prediction = -1 # Set a specific error value for LLM context
            
        
        # Set the answer based on prediction
        if not user_input == "Asthma":
            if prediction == 1:
                answer = "Yes, the test is positive."
            elif prediction == 0:
                answer = "No, the test is negative."
            else:
                answer = "The test could not be completed successfully. Please check your inputs and model files."
        
        
        # Build detailed prompt for Gemini with the prediction result
        prompt = f"""
You are a medical and lifestyle assistant providing a comprehensive health assessment. The machine learning model has made the following prediction:

{answer}

Based on this prediction and the following user details, provide:
1. A clear interpretation of what this prediction means for the user's health
2. Analysis of reported symptoms and their possible causes
3. Evidence-based lifestyle suggestions
4. Specific dietary recommendations
5. Stress management and sleep improvement advice
6. Clear guidance on when to seek professional medical care

User's Health Information:
- Current Symptoms: {symptoms}
- Duration of Symptoms: {duration}
- Current Diet: {diet}
- Exercise Routine: {exercise}
- Sleep Patterns: {sleep}
- Daily Water Intake: {hydration}
- Stress Level: {stress}
- Medical History: {medical_history}

Clinical Measurements: {user_data}

Important: Begin your response by directly addressing the model's prediction and its implications for the user. Then provide detailed, personalized recommendations based on all the information above.
"""
        # Generate LLM response
        response = llm_model.generate_content(prompt)

    # Display model output
    st.write("\nGemini Response:\n")
    st.write(response.text)
