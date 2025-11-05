import os
import io
import joblib
import random
from typing import List, Optional
import pandas as pd
import numpy as np
import streamlit as st
from fpdf import FPDF
from datetime import datetime

# -----------------------------
# Feature Validation Data for Diabetes (WHO-based) - UPDATED with ASCII characters
# -----------------------------
VALIDATION_FEATURES_DIABETES = {
    "age": {
        "description": "Risk for Type 2 Diabetes increases with age, particularly after 45.",
        "role": "Demographic",
        "link": "https://www.who.int/news-room/fact-sheets/detail/diabetes"
    },
    "pulse_rate": {
        "description": "Pulse rate itself isn't a direct diabetes indicator but can be affected by related cardiovascular conditions.",
        "role": "Cardiovascular",
        "link": ""
    },
    "systolic_bp": {
        "description": "Systolic blood pressure. WHO defines hypertension as >=140 mmHg systolic OR >=90 mmHg diastolic.", # Replaced ≥
        "role": "Cardiovascular",
        "link": "https://www.who.int/news-room/fact-sheets/detail/hypertension"
    },
    "diastolic_bp": {
        "description": "Diastolic blood pressure. WHO defines hypertension as >=140 mmHg systolic OR >=90 mmHg diastolic.", # Replaced ≥
        "role": "Cardiovascular",
        "link": "https://www.who.int/news-room/fact-sheets/detail/hypertension"
    },
    "glucose": {
        "description": "Fasting plasma glucose level. WHO criteria: >=126 mg/dL (7.0 mmol/L) suggests diabetes, 100-125 mg/dL (5.6-6.9 mmol/L) suggests Impaired Fasting Glucose (prediabetes).", # Replaced ≥
        "role": "Metabolic Marker",
        "link": "https://www.who.int/data/gho/indicator-metadata-registry/imr-details/2380"
    },
    "height": {
        "description": "Height is used in conjunction with weight to calculate BMI.",
        "role": "Biometric",
        "link": ""
    },
    "weight": {
        "description": "Weight is used in conjunction with height to calculate BMI. Excess weight increases diabetes risk.",
        "role": "Biometric",
        "link": ""
    },
    "bmi": {
        "description": "Body Mass Index (BMI). WHO criteria: >=30 kg/m² indicates Obesity, 25-29.9 kg/m² indicates Overweight. Both are major risk factors for Type 2 Diabetes.", # Replaced ≥
        "role": "Biometric",
        "link": "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight"
    },
    "hypertensive": { # Description for the condition based on input
        "description": "A prior diagnosis or current state of hypertension (BP >=140/90 mmHg) significantly increases the risk for Type 2 Diabetes.", # Replaced ≥
        "role": "Medical History",
        "link": "https://www.cdc.gov/diabetes/basics/risk-factors.html"
    },
    "diagnostic_label": { # Input, but clinical meaning is relevant
        "description": "Previous diagnostic status (Normal, Prediabetes, Diabetes) provides crucial context for current risk.",
        "role": "Medical History / Context",
        "link": ""
    }
}
# Add entry for the specific input key used in explain_risk_factors
VALIDATION_FEATURES_DIABETES['hypertension_input'] = VALIDATION_FEATURES_DIABETES['hypertensive']
# Add specific entry for BP reporting in explain_risk_factors
VALIDATION_FEATURES_DIABETES['blood_pressure'] = {
    "description": "Blood pressure reading. Hypertension (>=140/90 mmHg) is a major risk factor for diabetes and cardiovascular disease.", # Replaced ≥
    "role": "Cardiovascular",
    "link": "https://www.who.int/news-room/fact-sheets/detail/hypertension"
}


# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    # Try loading the model, handle potential file not found or other errors
    try:
        model = joblib.load(model_path)
        # Attempt to access feature_names_in_ to verify it's likely the correct model type
        _ = model.feature_names_in_
        return model
    except FileNotFoundError:
        st.error(f"**Fatal Error:** Model file not found at '{model_path}'. Ensure it's in the correct directory.")
        st.stop()
    except AttributeError:
         st.error(f"**Fatal Error:** The loaded object from '{model_path}' doesn't seem to be a trained scikit-learn model with feature names. Please check the file.")
         st.stop()
    except Exception as e:
        st.error(f"**Fatal Error:** Could not load model from '{model_path}'. Details: {e}")
        st.stop()


def risk_label_from_proba(p_high: float) -> str:
    # Using thresholds from OvaPredict example, adjust if needed for diabetes
    if p_high < 0.40: return "Low Risk"
    elif p_high < 0.70: return "Moderate Risk"
    else: return "High Risk"

# --- Data Preprocessing (Placeholder - Assuming direct input for diabetes model) ---
def preprocess_for_diabetes(df_raw, model):
    # Check if the model object has feature names defined
    if hasattr(model, 'feature_names_in_'):
         model_feature_order = model.feature_names_in_
         missing_cols = set(model_feature_order) - set(df_raw.columns)
         if missing_cols:
              st.error(f"Missing required columns in input data: {missing_cols}")
              return None
         # Ensure columns are float where needed before returning
         df_processed = df_raw[model_feature_order].copy()
         for col in df_processed.columns:
             # Attempt conversion, ignore errors for columns already numeric
             df_processed[col] = pd.to_numeric(df_processed[col], errors='ignore')
         return df_processed
    else:
         # This case should ideally not happen if model loading checks feature_names_in_
         st.error("Model object does not contain expected feature names ('feature_names_in_'). Cannot proceed.")
         return None


# -----------------------------
# Rule-based Explanation + Clinical Facts for Diabetes - UPDATED THRESHOLDS & CHARACTERS
# -----------------------------
def explain_risk_factors_diabetes(patient_data):
    risk_indicators_keys = [] # Store keys that trigger rules
    if patient_data.get('age', 0) > 45: risk_indicators_keys.append("age")
    if patient_data.get('hypertensive', 0) == 1: risk_indicators_keys.append("hypertension_input")
    # WHO Hypertension definition (using >=)
    if patient_data.get('systolic_bp', 0) >= 140 or patient_data.get('diastolic_bp', 0) >= 90:
         risk_indicators_keys.append("blood_pressure") # Use combined key
    # WHO Glucose thresholds (using >=)
    if patient_data.get('glucose', 0) >= 100: # Includes prediabetes and diabetes
        risk_indicators_keys.append("glucose")
    # WHO BMI thresholds (using >=)
    if patient_data.get('bmi', 0) >= 25: # Includes overweight and obesity
        risk_indicators_keys.append("bmi")

    if not risk_indicators_keys: return None, None

    table_data = []
    for feat_key in risk_indicators_keys:
        # Use generic lookup key, handle special cases inside
        lookup_key = feat_key.replace('_input', '')
        info = VALIDATION_FEATURES_DIABETES.get(lookup_key, {})
        feat_display = lookup_key.replace('_', ' ').title()

        if feat_key == 'blood_pressure':
            val_display = f"{patient_data.get('systolic_bp', 'N/A')}/{patient_data.get('diastolic_bp', 'N/A')} mmHg"
            # Description based on confirmed hypertension
            sys_val = patient_data.get('systolic_bp', 0)
            dia_val = patient_data.get('diastolic_bp', 0)
            if sys_val >= 140 or dia_val >= 90:
                desc = "Indicates Hypertension (Stage 2), a major risk factor."
                role = "Cardiovascular (High Risk)"
            elif 130 <= sys_val <= 139 or 80 <= dia_val <= 89: # Add check for stage 1 based on original code logic
                desc = "Indicates Hypertension (Stage 1). Monitoring and lifestyle changes important."
                role = "Cardiovascular (Elevated)"
            else: # Elevated but below Stage 1 - This might not be hit if caught by >=140/90
                 desc = "Blood pressure reading above normal but below Hypertension Stage 1."
                 role = "Cardiovascular"
            info = {"role": role, "description": desc} # Overwrite info for dynamic BP desc
            feat_display = "Blood Pressure" # Ensure display name is consistent
        elif feat_key == 'glucose':
            val = patient_data.get('glucose', 0)
            val_display = f"{val:.2f} mg/dL" if isinstance(val, float) else str(val)
            if val >= 126:
                desc = "Fasting glucose >=126 mg/dL suggests diabetes.".replace("≥", ">=") # Replace char
                role = "Metabolic (Diabetes Range)"
            else: # Must be 100-125
                desc = "Fasting glucose 100-125 mg/dL suggests prediabetes."
                role = "Metabolic (Prediabetes Range)"
        elif feat_key == 'bmi':
            val = patient_data.get('bmi', 0)
            val_display = f"{val:.2f} kg/m²" if isinstance(val, float) else str(val)
            if val >= 30:
                desc = "BMI >=30 kg/m² indicates Obesity, a major risk factor.".replace("≥", ">=") # Replace char
                role = "Biometric (Obese)"
            else: # Must be 25-29.9
                desc = "BMI 25-29.9 kg/m² indicates Overweight, increasing risk."
                role = "Biometric (Overweight)"
        elif feat_key == 'hypertension_input':
            val_display = "Yes"
            desc = info.get("description", "Patient reported hypertension.").replace("≥", ">=") # Replace char
            role = info.get("role", "Medical History")
            feat_display = "Hypertension Status" # More descriptive feature name
        else: # Default for age or others
            val = patient_data.get(lookup_key, 'N/A')
            val_display = str(val)
            desc = info.get("description", "").replace("≥", ">=") # Replace char
            role = info.get("role", "")


        table_data.append({
            "Feature": feat_display, "Value": val_display, "Risk": "High",
            "Role": role, "Interpretation": desc
        })

    df_table = pd.DataFrame(table_data)
    df_table.drop_duplicates(subset=['Feature'], keep='first', inplace=True)
    table_html = df_table.to_html(index=False, escape=False)
    styled_html = f"""
    <div style='overflow-x:auto;'>
        <style>
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px;}}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left !important; vertical-align: top; white-space: normal; color: #000000 !important; }} /* Ensure table text is black */
            th {{ background-color: #f2f2f2; font-weight: bold; color: #000000 !important;}} /* Ensure header text is black */
        </style>
        {table_html}
    </div>
    """
    return df_table, styled_html

# --- Map short names to full display names for PDF ---
def get_feature_display_name_diabetes(short_name):
    mapping = {
        'age': 'Age (years)',
        'pulse_rate': 'Pulse Rate (per min)',
        'systolic_bp': 'Systolic BP (mmHg)',
        'diastolic_bp': 'Diastolic BP (mmHg)',
        'glucose': 'Glucose Level (mg/dL)',
        'height': 'Height (cm)',
        'weight': 'Weight (kg)',
        'bmi': 'Body Mass Index (kg/m²)',
        'hypertensive': 'Patient is Hypertensive?',
        'diagnostic_label': 'Diagnostic Label'
    }
    return mapping.get(short_name, short_name.replace('_', ' ').title())

# -----------------------------
# PDF Generation Function (Adapted for Diabetes)
# -----------------------------
def generate_pdf_report(user_vals_dict_pdf, risk_label, percent, df_table=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 0) # Black title
    pdf.cell(0, 10, "DiaPredict AI - Diabetes Risk Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "1. Patient Input Values", ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", "", 9)
    col_widths = [80, 100]
    for feature_key, value in user_vals_dict_pdf.items():
        # Skip internal text keys used only for PDF display formatting
        if feature_key.endswith('_text'):
             continue
        display_name = get_feature_display_name_diabetes(feature_key)
        # Use companion text keys for Yes/No & labels if available
        if feature_key == 'hypertensive': display_value = user_vals_dict_pdf.get('hypertensive_text', ('Yes' if value == 1 else 'No'))
        elif feature_key == 'diagnostic_label':
             label_map_rev = {0: 'Normal', 1: 'Prediabetes', 2: 'Diabetes'}
             display_value = user_vals_dict_pdf.get('diagnostic_label_text', label_map_rev.get(value, 'Unknown'))
        elif isinstance(value, float): display_value = f"{value:.2f}"
        else: display_value = str(value)

        pdf.cell(col_widths[0], 6, display_name, border=1)
        pdf.cell(col_widths[1], 6, display_value, border=1, ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "2. Prediction Result", ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", "B", 14)
    # Set text color based on risk, but keep text black for High/Moderate on white background later
    risk_color_tuple = (0, 0, 0) # Default black
    if risk_label == "High Risk": risk_color_tuple = (220, 53, 69) # Red
    elif risk_label == "Moderate Risk": risk_color_tuple = (255, 193, 7) # Amber/Yellow
    elif risk_label == "Low Risk": risk_color_tuple = (76, 175, 80) # Green
    pdf.set_text_color(*risk_color_tuple)
    pdf.cell(0, 8, f"Risk Assessment: {risk_label} ({percent:.2f}%)", ln=True)
    pdf.set_text_color(0, 0, 0) # Reset to black for subsequent text
    pdf.ln(5)

    if df_table is not None and not df_table.empty:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "3. High-Risk Indicators & Clinical Interpretation", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", "B", 8)
        # Adjusted widths for diabetes features
        col_widths_risk = [35, 25, 15, 35, 70]
        headers = ["Feature", "Value", "Risk", "Role", "Interpretation"]
        for i, header in enumerate(headers):
            pdf.cell(col_widths_risk[i], 6, header, border=1)
        pdf.ln()

        pdf.set_font("Arial", "", 7)
        for _, row in df_table.iterrows():
            start_y = pdf.get_y()
            # Ensure text being sent to PDF doesn't contain the bad character
            feature_text = str(row["Feature"])
            value_text = str(row["Value"])
            risk_text = str(row["Risk"])
            role_text = str(row["Role"])
            interpretation_text = str(row["Interpretation"]).replace("≥", ">=") # Replace here too as fallback

            pdf.multi_cell(col_widths_risk[0], 5, feature_text, border=1, align='L')
            y1 = pdf.get_y()
            pdf.set_xy(pdf.l_margin + col_widths_risk[0], start_y)

            pdf.multi_cell(col_widths_risk[1], 5, value_text, border=1, align='L')
            y2 = pdf.get_y()
            pdf.set_xy(pdf.l_margin + sum(col_widths_risk[:2]), start_y)

            pdf.multi_cell(col_widths_risk[2], 5, risk_text, border=1, align='L')
            y3 = pdf.get_y()
            pdf.set_xy(pdf.l_margin + sum(col_widths_risk[:3]), start_y)

            pdf.multi_cell(col_widths_risk[3], 5, role_text, border=1, align='L')
            y4 = pdf.get_y()
            pdf.set_xy(pdf.l_margin + sum(col_widths_risk[:4]), start_y)

            pdf.multi_cell(col_widths_risk[4], 5, interpretation_text, border=1, align='L')
            y5 = pdf.get_y()
            pdf.set_y(max(y1, y2, y3, y4, y5))

    pdf.ln(10)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 8, "Generated by DiaPredict AI - For clinical decision support only", ln=True, align="R")

    return bytes(pdf.output())

# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config(page_title="DiaPredict AI", layout="wide")

# --- Custom CSS Injection ---
st.markdown(
    """
    <style>
    /* Main App Styling - White Background */
    .main, .stApp {
        background-color: #ffffff !important; /* White background */
        font-family: 'Arial', sans-serif;
    }
    .block-container {
        padding: 2rem 3rem; /* Adjust padding as needed */
    }
    /* Text visibility - Black Text */
    body, p, label, div, span, .st-emotion-cache-16idsys p, .st-emotion-cache-1y4p8pa,
    .stTextInput label, .stNumberInput label, .stSelectbox label, /* Target specific widget labels */
    h1, h2, h3, h4, h5, h6, /* Ensure headers are black */
    .stMarkdown, .stAlert, .stMetricLabel, .stMetricValue, /* Common Streamlit elements */
    th, td /* Table text */
     {
        color: #000000 !important; /* Black text */
    }

     /* Target text within selectbox options specifically */
    .stSelectbox [data-baseweb="list-item"] div, /* Standard selectbox option text */
    .stSelectbox [role="option"] /* Alternative for some renderings */
    {
        color: #000000 !important;
    }

    /* Also ensure the selected value displayed in the selectbox widget itself is black */
    .stSelectbox div[data-baseweb="select"] > div {
         color: #000000 !important;
    }


    /* Keep Title blue for hierarchy, or make black if truly all black */
    h1 {
       color: #1c4e80 !important; /* Original blue */
       /* color: #000000 !important; */ /* Uncomment for black title */
       text-align: center;
    }
     h3 {
       color: #1c4e80 !important; /* Original blue */
       /* color: #000000 !important; */ /* Uncomment for black subheaders */
       text-align: left; /* Adjust alignment if needed */
     }
     h4 { /* Subtitle */
       color: #555555 !important; /* Keep subtitle grey or make black */
       /* color: #000000 !important; */
        text-align: center;
     }


    /* Risk Indicator Table Headers */
     th {
        background-color: #f2f2f2 !important;
        font-weight: bold;
        color: #000000 !important;
     }

    /* Result Card - Adjust text color based on background */
    .risk-card-low { background-color: #4CAF50; color: white !important; }
    .risk-card-moderate { background-color: #FFEB3B; color: black !important; } /* Black text on yellow */
    .risk-card-high { background-color: #F44336; color: white !important; }

    /* Ensure links in reference tab are visible */
    .stMarkdown a {
        color: #0056b3 !important; /* Standard link blue */
    }

     /* Reference Card Headers */
    div[style*="background:#E8F4FF"] h4, /* Blue cards */
    div[style*="background:#FFF1F2"] h4, /* Pink cards */
    div[style*="background:#E8FFF3"] h4, /* Green cards */
    div[style*="background:#FFF4E6"] h4  /* Peach cards */
    {
       color: #102a43 !important; /* Dark blue for card headers */
    }
     /* Reference Card Text */
    div[style*="background:"] p {
        color: #243b53 !important; /* Dark grey for card text */
     }


    /* Button Styling (Optional: Adjust if needed) */
    .stButton>button {
        width: 100%; border-radius: 10px; border: none;
        background-color: #1c4e80; color: white; transition: all 0.2s ease-in-out;
        padding: 12px 0; font-size: 16px; font-weight: bold;
    }
    .stButton>button:hover { background-color: #2769b3; }

    </style>
    """,
    unsafe_allow_html=True
)


st.title("🩺 DiaPredict AI: Diabetes Risk Prediction")



# --- Model Loading (Check if successful) ---
model = None
try:
    # Ensure this is the correct path to your diabetes model
    model = load_model("best_global_stacking_model.pkl")
except Exception as e:
    # Error is displayed by load_model, st.stop() halts execution
    pass # Keep linters happy

# Stop execution if model loading failed
if model is None:
    st.stop()


# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["Clinical Interpretation", "Clinical Reference"])

with tabs[0]:
    st.subheader("Patient Prediction")
    st.markdown("Enter the patient's feature values:")
    user_vals_input = {} # Store user selections with text (e.g., "Yes") for PDF/display
    user_vals_model = {} # Store numerically mapped values for the model & risk explain

    # Get feature names expected by the loaded model
    try:
        model_feature_names = list(model.feature_names_in_)
    except AttributeError:
        st.error("Fatal Error: Loaded model object does not have 'feature_names_in_'. Cannot determine required features.")
        st.stop()
    except Exception as e:
        st.error(f"Fatal Error: Could not retrieve feature names from model. Details: {e}")
        st.stop()


    with st.form("single_prediction_form"):
        # Use 2 columns as in the original diabetes app layout
        col1, col2 = st.columns(2)
        with col1:
            # Store numerical inputs directly
            age_val = st.number_input("Age (years)", 1, 120, 55, key='s_age')
            pulse_rate_val = st.number_input("Pulse Rate (per min)", 30, 200, 75, key='s_pulse')
            systolic_bp_val = st.number_input("Systolic BP (mmHg)", 80, 250, 145, key='s_sys_bp')
            diastolic_bp_val = st.number_input("Diastolic BP (mmHg)", 50, 150, 92, key='s_dia_bp')
            glucose_val = st.number_input("Glucose Level (mg/dL)", 50, 500, 115, key='s_glucose')
        with col2:
            height_val = st.number_input("Height (cm)", 100, 250, 165, key='s_height')
            weight_val = st.number_input("Weight (kg)", 30, 200, 90, key='s_weight')
            bmi_val = st.number_input("Body Mass Index (kg/m²)", 10.0, 60.0, 33.1, format="%.2f", key='s_bmi')
            # Store text input separately
            hypertensive_input = st.selectbox("Patient is Hypertensive?", ["No", "Yes"], key='s_hyper', index=1)
            diagnostic_label_input = st.selectbox("Diagnostic Label", ["Normal", "Prediabetes", "Diabetes"], key='s_diag', index=1)

        predict_button = st.form_submit_button("Predict", type="primary")

    if predict_button:
        try:
             # Populate the dictionaries after submission
             user_vals_model = {
                'age': age_val, 'pulse_rate': pulse_rate_val, 'systolic_bp': systolic_bp_val,
                'diastolic_bp': diastolic_bp_val, 'glucose': glucose_val, 'height': height_val,
                'weight': weight_val, 'bmi': bmi_val
             }
             user_vals_input = user_vals_model.copy() # Start with numerical values for PDF dict

             # Map categorical inputs for the model dict
             user_vals_model['hypertensive'] = 1 if hypertensive_input == "Yes" else 0
             diagnostic_label_map = {"Normal": 0, "Prediabetes": 1, "Diabetes": 2}
             user_vals_model['diagnostic_label'] = diagnostic_label_map[diagnostic_label_input]

             # Add text versions to the input dict for PDF/Display
             user_vals_input['hypertensive'] = hypertensive_input
             user_vals_input['diagnostic_label'] = diagnostic_label_input


             # Prepare dataframe for model prediction using numerically mapped values
             input_df = pd.DataFrame([user_vals_model])
             # Ensure only the columns the model expects are passed, in the right order
             missing_model_keys = set(model_feature_names) - set(user_vals_model.keys())
             if missing_model_keys:
                  st.error(f"Internal Error: Missing keys needed for the model: {missing_model_keys}")
                  st.stop()

             input_df_ordered = input_df[model_feature_names]

             # Assuming preprocess_for_diabetes just orders/cleans columns if needed
             processed_df = preprocess_for_diabetes(input_df_ordered, model)

             if processed_df is None: # Handle preprocessing error
                 st.stop()

             proba = model.predict_proba(processed_df)[0]
             p_high = float(proba[1]) # Assuming class 1 is diabetes
             risk_label = risk_label_from_proba(p_high)
             percent = p_high * 100

             # Define risk card class based on label
             risk_card_class = "risk-card-low"
             if risk_label == "Moderate Risk": risk_card_class = "risk-card-moderate"
             elif risk_label == "High Risk": risk_card_class = "risk-card-high"

             st.markdown(f'<div class="{risk_card_class}" style="padding:20px; border-radius:10px; font-size:28px; font-weight:bold; text-align:center; margin-bottom:20px;">{risk_label} ({percent:.2f}%)</div>', unsafe_allow_html=True)


             st.markdown("### Risk Indicators")
             # Pass the DICTIONARY WITH NUMERICAL VALUES (user_vals_model) to explanation function
             df_table, table_html = explain_risk_factors_diabetes(user_vals_model)
             if table_html:
                 st.markdown(table_html, unsafe_allow_html=True)
             else:
                 st.markdown("✅ All features appear to be within a normal range based on simple rules.")

             st.markdown("### Download Complete Report")
             # Pass the DICTIONARY containing user-friendly text for PDF display
             pdf = generate_pdf_report(user_vals_input, risk_label, percent, df_table)
             st.download_button(label="Download Full Report (PDF)", data=pdf, file_name="DiaPredictAI_Report.pdf", mime="application/pdf", type="primary")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e) # Show full traceback for debugging

with tabs[1]:
    st.header("Clinical Reference for Diabetes Risk Factors")

    categories_diabetes = {
        "Metabolic Markers": {"features": ["glucose", "bmi"], "color": "#E8F4FF"},
        "Cardiovascular Health": {"features": ["systolic_bp", "diastolic_bp", "hypertensive", "pulse_rate"], "color": "#FFF1F2"},
        "Demographics & Biometrics": {"features": ["age", "height", "weight"], "color": "#E8FFF3"},
        "Context": {"features": ["diagnostic_label"], "color": "#FFF4E6"}
    }

    # UPDATED expanded texts to use '>=' instead of '≥'
    expanded_texts_diabetes = {
         "glucose": (
            "Fasting blood glucose is a key test for diabetes. WHO criteria state a level of >=126 mg/dL (7.0 mmol/L) or higher on two separate tests indicates diabetes.",
            "Levels between 100 and 125 mg/dL (5.6-6.9 mmol/L) suggest Impaired Fasting Glucose (prediabetes). Lifestyle changes are crucial in this range.",
            "Normal fasting glucose is generally below 100 mg/dL (5.6 mmol/L). Maintaining this level is important for preventing diabetes and its complications."
         ),
        "bmi": (
            "Body Mass Index (BMI) estimates body fat based on height and weight.",
            "A BMI of >=30 kg/m² is classified as obesity by the WHO. Obesity significantly increases insulin resistance and the risk of developing Type 2 Diabetes.",
             "A BMI between 25 and 29.9 kg/m² is considered overweight according to WHO. Even being overweight increases diabetes risk, especially if combined with other factors."
        ),
         "systolic_bp": (
             "Systolic blood pressure (the top number) measures the pressure in arteries when the heart beats.",
             "WHO defines hypertension primarily by readings >=140 mmHg systolic or >=90 mmHg diastolic. High blood pressure damages blood vessels and is strongly linked with insulin resistance and diabetes.",
             "Consistently elevated blood pressure above normal levels (even if below 140/90) increases cardiovascular risk and warrants monitoring and lifestyle adjustments."
         ),
         "diastolic_bp": (
             "Diastolic blood pressure (the bottom number) measures the pressure in arteries when the heart rests.",
              "WHO defines hypertension primarily by readings >=140 mmHg systolic or >=90 mmHg diastolic.",
             "Elevated diastolic pressure contributes to overall cardiovascular risk."
         ),
         "hypertensive": (
             "A prior diagnosis or current state of hypertension (typically BP >=140/90 mmHg) means the patient has established high blood pressure.",
            "Hypertension is a major component of metabolic syndrome and significantly increases the likelihood of developing Type 2 Diabetes and cardiovascular complications."
         ),
         "age": (
             "The risk of developing Type 2 Diabetes increases significantly as people get older, especially after age 45.",
             "Screening is generally recommended starting at age 45, or earlier for individuals with other risk factors. Age-related changes can affect insulin sensitivity and production."
         ),
         "pulse_rate": (
             "Pulse rate indicates how many times the heart beats per minute.",
             "While not a direct WHO criterion for diabetes, abnormal pulse rates can indicate underlying cardiovascular issues which often coexist with diabetes."
         ),
        "height": ("Height is a fixed biometric measurement used primarily to calculate BMI."),
        "weight": ("Body weight, particularly excess weight (overweight/obesity based on BMI), is a major modifiable risk factor for Type 2 Diabetes according to WHO."),
        "diagnostic_label": (
             "Knowing a patient's previous diabetes status (Normal, Prediabetes, or diagnosed Diabetes) is essential context.",
            "This helps interpret current results and understand the progression or stability of their metabolic health based on WHO diagnostic categories."
        )
    }

    for cat_name, cat_info in categories_diabetes.items():
        st.subheader(cat_name)
        for feat_key in cat_info["features"]:
            info = VALIDATION_FEATURES_DIABETES.get(feat_key, {})
            link_html = f'<a href="{info.get("link", "")}" target="_blank" style="text-decoration:none; color:#0056b3; font-weight:600;">Learn More</a>' if info.get("link") else ""

            texts = expanded_texts_diabetes.get(feat_key, info.get("description", "No detailed summary available."))
            if isinstance(texts, tuple): paragraphs = [p for p in texts if p and p.strip()]
            else: paragraphs = [texts] if texts else ["No detailed summary available."]

            # Ensure text displayed in UI is also corrected
            cleaned_paragraphs = [p.replace("≥", ">=") for p in paragraphs]

            # Use st.markdown for card structure to ensure Streamlit handles text rendering correctly
            st.markdown(f"""
            <div style="background:{cat_info['color']}; border:1px solid rgba(0,0,0,0.06); border-radius:12px; padding:16px; margin-bottom:16px;">
                <h4 style="margin:0 0 8px 0;">{get_feature_display_name_diabetes(feat_key)}</h4>
            """, unsafe_allow_html=True) # Header needs black color override removed or set explicitly

            # Render paragraphs ensuring black text
            for p in cleaned_paragraphs:
                 st.markdown(f'<p style="margin:6px 0; font-size:15px; line-height:1.5;">{p}</p>', unsafe_allow_html=True) # Rely on body style for black text

            # Render link separately if it exists
            if link_html:
                 st.markdown(f'<div style="margin-top:10px; text-align:right;">{link_html}</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True) # Close card div

