# --- Import necessary libraries ---
import streamlit as st            # For building the web app (frontend + backend)
import pandas as pd               # For handling tabular data (user inputs and dataset)
import joblib                     # For loading the saved machine learning model
from sklearn.preprocessing import LabelEncoder  # For encoding/decoding obesity labels
import matplotlib.pyplot as plt   # For creating static plots (Bar charts, Line charts)
import seaborn as sns             # For advanced visualizations (Risk factor plot)
import plotly.graph_objects as go # For interactive visualizations (BMI Gauge, Radar plot)
import numpy as np                # For numerical calculations
import re                         # For regular expressions (removing surrogate characters)
import os                         # For interacting with files in the project folder


# --- CONFIGURATION ---
st.set_page_config(page_title="Obesity Level Predictor", layout="wide") # Set Streamlit app title and make the layout wide (full screen)
st.title("🍔 Obesity Level Predictor") # Show main heading on the page
st.markdown("Fill in your details below to predict your **Obesity Level** and view insights and visualizations.") # Display a message below the title

# --- REMOVE SURROGATE CHARACTERS ---
def remove_surrogates(text):
    return re.sub(r'[\ud800-\udfff]', '', text) # Function to remove unwanted characters that could cause display errors

# --- LOAD MODELS AND DATA ---
@st.cache_resource
def load_all_models():
    # Look for file '1_knn_model.pkl', etc.
    model_files = [f for f in os.listdir() if f.endswith('_models.pkl')] # Find all files ending with '_models.pkl' (your trained models)
    models = {}
    for file in sorted(model_files):  # sort to maintain consistent order
        try:
            model = joblib.load(file)
            models[file] = model  # keep full filename for rank extraction
        except Exception as e:
            st.warning(f"⚠️ Failed to load {file}: {e}") # Show warning if loading fails
    return models


@st.cache_resource
def get_label_encoder():
    try:
        df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")  # Load dataset
        le = LabelEncoder()
        le.fit(df["NObeyesdad"])  # Train label encoder on target column
        return le
    except Exception as e:
        st.error(f"Error loading label encoder: {str(e)}")
        return None

# --- Load models and label encoder ---
models = load_all_models()
le = get_label_encoder()

if not models or le is None:
    st.error("❌ Could not load models or label encoder.")
    st.stop()
# If model loading fails, stop the app

# Use first model for visualization
pipe = models[list(models.keys())[0]]

# --- DYNAMIC CALORIE ESTIMATION (Simple Example) ---
def estimate_calories(gender, age, height, weight, activity):
    bmr = 0
    if gender == "Male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    activity_factors = [1.2, 1.375, 1.55, 1.725] # Sedentary to Very Active
    return bmr * activity_factors[activity]
# Estimate daily calorie needs based on gender, age, height, weight, activity

# --- USER INPUT FORM ---
with st.form("user_input_form"):
    col1, col2 = st.columns(2) # Create 2 columns to arrange input fields neatly

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 10, 100, 25)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
        fam_hist = st.selectbox("Family History with Overweight", ["yes", "no"])
        high_cal_food = st.selectbox("Frequent consumption of high caloric food", ["yes", "no"])
        veggies = st.slider("Frequency of consumption of vegetables", 1, 3, 2,
                            help="1: Never, 2: Sometimes, 3: Always")
        meals = st.slider("Number of main meals", 1, 4, 3)

    with col2:
        snacks = st.selectbox("Consumption of food between meals",
                            ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Do you smoke?", ["yes", "no"])
        water = st.slider("Consumption of water daily", 1, 3, 2,
                           help="1: Less than 1L, 2: 1-2L, 3: More than 2L")
        cal_monitor = st.selectbox("Calories consumption monitoring", ["yes", "no"])
        activity = st.slider("Physical activity frequency (0=none to 3=high)", 0, 3, 1)
        tech_time = st.slider("Time using technology devices (0=low to 2=high)", 0, 2, 1)
        alcohol = st.selectbox("Consumption of alcohol",
                             ["no", "Sometimes", "Frequently", "Always"])
        transport = st.selectbox("Transportation used",
                                 ["Automobile", "Walking", "Bike", "Motorbike", "Public Transportation"])

    submitted = st.form_submit_button("Predict")  # Button to submit all inputs

if submitted:
    # Create input dictionary
    sample_input = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Family History with Overweight': fam_hist,
        'Frequent consumption of high caloric food': high_cal_food,
        'Frequency of consumption of vegetables': veggies,
        'Number of main meals': meals,
        'Consumption of food between meals': snacks,
        'Smoke': smoke,
        'Consumption of water daily': water,
        'Calories consumption monitoring': cal_monitor,
        'Physical activity frequency': activity,
        'Time using technology devices': tech_time,
        'Consumption of alcohol': alcohol,
        'Transportation used': transport
    }
    # Prepare input dictionary

    df = pd.DataFrame([sample_input]) # Convert dictionary to pandas DataFrame (needed by model)
    
    # Store predictions from all models
    model_predictions = {}
    for model_name, model_pipe in models.items():
        try:
            encoded_pred = model_pipe.predict(df)[0] # Predict the encoded label (numeric)
            class_label = le.inverse_transform([encoded_pred])[0] # Decode numeric prediction to human-readable class
            model_predictions[model_name] = class_label
        except Exception as e:
            model_predictions[model_name] = f"❌ Error: {str(e)}"

     # --- Display Predictions ---
    st.subheader("📍 Prediction from KNeighbors Classifier Model :")
    for name, pred in model_predictions.items():
        # Extract model rank from filename like '0_svm_model.pkl'
        try:
            rank = int(name.split('_')[0])
        except:
            rank = 99  # fallback in case of unexpected filename

        clean_name = "_".join(name.split('_')[1:]).replace('_model.pkl', '').upper()

    # Color code the predictions based on rank
        if rank == 0:
            st.success(f"✅ **{clean_name}** predicted: `{pred}`")
        elif rank == 1:
            st.warning(f"⚠️ **{clean_name}** predicted: `{pred}`")
        elif rank == 2:
            st.error(f"❌ **{clean_name}** predicted: `{pred}`")
        else:
            st.info(f"ℹ️ **{clean_name}** predicted: `{pred}`")


    # Use first model's prediction for visualizations
    class_label = list(model_predictions.values())[0]

    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    bmi_category = ""
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif 18.5 <= bmi < 25:
        bmi_category = "Normal weight"
    elif 25 <= bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    st.info(f"**BMI:** {bmi:.1f} ({bmi_category})")

    # Calculate values for visualizations
    user_estimated = 2500 if gender == "Male" else 2000
    categories = ["Veggies", "Water", "Activity", "Tech Use", "High-Cal Food"]
    values = [veggies/3*5, water/3*5, activity/3*5,
              (2-tech_time)/2*5, 1 if high_cal_food == "no" else 4]
    features = ["Low Activity", "No Calorie Monitor", "High Tech Time",
                "Frequent Snacks", "Alcohol Use"]
    scores = [
        (3 - activity) / 3 * 100,
        80 if cal_monitor == "no" else 20,
        (tech_time / 2) * 100,
        25 if snacks == "no" else 50 if snacks == "Sometimes" else 75 if snacks == "Frequently" else 100,
        25 if alcohol == "no" else 50 if alcohol == "Sometimes" else 75 if alcohol == "Frequently" else 100
    ]
    days = list(range(0, 365, 30))
    current_trend = [weight - (i * 0.3 if activity > 1 else i * 0.2) for i in range(len(days))]
    improved_trend = [weight - (i * 0.5) for i in range(len(days))]

    # Create visualizations
    st.subheader("Health Insights")

    # Tab 1: BMI Gauge
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "BMI Analysis", "Calorie Comparison", "Lifestyle Radar",
        "Risk Factors", "Weight Projection"
    ])

    with tab1:
        fig1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=bmi,
            title={'text': "BMI (Body Mass Index)"},
            gauge={
                'axis': {'range': [0, 50]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 18.5], 'color': "lightblue"},
                    {'range': [18.5, 24.9], 'color': "lightgreen"},
                    {'range': [25, 29.9], 'color': "yellow"},
                    {'range': [30, 34.9], 'color': "orange"},
                    {'range': [35, 50], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': bmi
                }
            }
        ))
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2, ax = plt.subplots()
        bars = ax.bar(["Recommended", "Your Estimated Need"], [user_estimated, user_estimated],
                      color=["skyblue", "salmon"])
        ax.set_title("Daily Calorie Intake Comparison")
        ax.set_ylabel("Calories (kcal)")
        st.pyplot(fig2)
        plt.close(fig2)

    with tab3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name="Your Lifestyle"
        ))
        fig3.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5])
            ),
            showlegend=False,
            title="Lifestyle Radar Chart"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        fig4, ax = plt.subplots()
        sns.barplot(x=scores, y=features, hue=features, palette="Reds_r", ax=ax, legend=False)
        ax.set_title("Obesity Risk Contributors")
        ax.set_xlabel("Risk Score (%)")
        ax.set_xlim(0, 100)
        st.pyplot(fig4)
        plt.close(fig4)

    with tab5:
        fig5, ax = plt.subplots()
        ax.plot(days, current_trend, marker="o", label="Current Trend")
        ax.plot(days, improved_trend, marker="o", label="With Improved Activity")
        ax.set_title("1-Year Weight Projection")
        ax.set_xlabel("Days")
        ax.set_ylabel("Weight (kg)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig5)
        plt.close(fig5)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
**Note:** This tool provides estimates based on statistical models and should not replace
professional medical advice. Always consult with a healthcare provider for personalized guidance.
""")
#  streamlit run main.py