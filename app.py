import streamlit as st
import pandas as pd
import joblib 

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Heart Risk Predictor", 
    page_icon="🫀", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS for a Pro UI Look
st.markdown("""
    <style>
    /* Style the main Predict button */
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        border: none;
        font-size: 20px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #ff1a1a;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    /* Style the header */
    .main-header {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        color: #ff4b4b;
    }
    .sub-header {
        text-align: center;
        color: #888888;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Cache the heavy lifting (loading models) for better performance
@st.cache_resource
def load_assets():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_assets()

# 4. Beautiful Headers
st.markdown("<h1 class='main-header'>🫀 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='sub-header'>Engineered by Saurabh ✨</h4>", unsafe_allow_html=True)
st.write("Fill in the patient's clinical metrics below to assess heart risk using our K-Nearest Neighbors algorithm.")
st.divider()

# 5. Grouping inputs logically using Columns
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.subheader("👤 Personal Details")
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"], help="Select Biological Sex")
    chest_pain = st.selectbox(
        "Chest Pain Type", 
        ["ATA", "NAP", "TA", "ASY"], 
        help="ATA: Atypical Angina | NAP: Non-Anginal Pain | TA: Typical Angina | ASY: Asymptomatic"
    )

with col2:
    st.subheader("🩺 Clinical Vitals")
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, step=1)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200, step=1)
    fasting_bs = st.radio(
        "Fasting Blood Sugar > 120 mg/dL", 
        [0, 1], 
        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
        horizontal=True
    )

with col3:
    st.subheader("📈 ECG & Exercise")
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.radio("Exercise-Induced Angina", ["Y", "N"], horizontal=True)
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.divider()

# 6. Center the Predict Button using dummy columns
_, btn_col, _ = st.columns([1, 1, 1])

with btn_col:
    predict_btn = st.button("🔍 Predict Risk Status")

# 7. Prediction Logic with feedback indicators
if predict_btn:
    # Adding a spinner gives the user visual feedback that the app is "thinking"
    with st.spinner("Analyzing patient data..."):
        
        # Create a raw input dictionary
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        # Create input dataframe
        input_df = pd.DataFrame([raw_input])

        # Fill in missing columns with 0s
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_columns]

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

    # Show enhanced result cards
    if prediction == 1:
        st.error("### ⚠️ High Risk of Heart Disease Detected")
        st.write("The model indicates a high probability of heart disease based on the provided metrics. Please consult a healthcare professional for a formal diagnosis.")
    else:
        st.success("### ✅ Low Risk of Heart Disease")
        st.write("The model indicates a low probability of heart disease. Keep up the healthy lifestyle!")
        st.balloons() # Pro touch: triggers a balloon animation on a good result