import streamlit as st
import pandas as pd
import joblib

# --- Page Config ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

# --- Custom Background, Font, and Color Styles ---
page_style = """
<style>
/* Background for the main app area */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.pexels.com/photos/7001550/pexels-photo-7001550.jpeg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Add a subtle overlay for better text contrast */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 50, 0.4); /* dark blue transparent overlay */
    z-index: 0;
}

/* Sidebar background and text color */
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.85);
    color: #001f3f;
}

/* General text color for the main container */
body, [data-testid="stMarkdownContainer"], .stMarkdown, .stTextInput, .stSelectbox, .stNumberInput {
    color: #f0f0f0 !important;  /* Light font color */
    font-family: 'Segoe UI', sans-serif;
    z-index: 1;
}

/* Titles and headers */
h1, h2, h3, h4 {
    color: #ffcc00 !important;  /* Gold/yellow headings */
    text-align: center;
}

/* Input labels and radio buttons */
label, .stRadio, .stSelectbox, .stNumberInput label {
    color: #ffffff !important;
}

/* Buttons */
div.stButton > button {
    background-color: #003366;
    color: #ffffff;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #0055aa;
    color: #ffcc00;
}

/* Result boxes */
.stSuccess {
    background-color: rgba(0, 128, 0, 0.7);
    color: white;
}
.stError {
    background-color: rgba(178, 34, 34, 0.7);
    color: white;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# --- Load Model ---
loaded_model = joblib.load(r"D:\Codes\MachineLearning\models\Titanic.pkl")

# --- Title & Description ---
st.title("Titanic Survival Prediction App")
st.write("Enter passenger details below to predict whether they would survive or not.")

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.radio("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)

with col2:
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=32.0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# --- Predict Button ---
if st.button("Predict Survival"):
    input_df = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked
    }])

    # --- Prediction ---
    prediction = loaded_model.predict(input_df)[0]
    prob = loaded_model.predict_proba(input_df)[0][1]

    # --- Display Result ---
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"Survived! (Probability: {prob:.2f})")
    else:
        st.error(f"Did Not Survive (Probability: {prob:.2f})")
