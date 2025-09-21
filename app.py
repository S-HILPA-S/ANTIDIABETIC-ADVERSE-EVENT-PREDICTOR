import streamlit as st
import pickle
import numpy as np
import pycountry_convert as pc
import pycountry
import joblib


# ---------------- Load Pickle Files ----------------
scaler = joblib.load(open("model/scaler.pkl", "rb"))
ohe_encoder = joblib.load(open("model/ohe_encoder.pkl", "rb"))
label_encoder = joblib.load(open("model/label_encoder (2).pkl", "rb"))
model = joblib.load(open("model/final_faers_model.pkl", "rb"))
pca = joblib.load(open("model/faers_pca.pkl", "rb"))

# ---------------- Outcome Mapping ----------------
outcome_mapping = {
    "DE": "Death",
    "HO": "Hospitalization",
    "DS": "Disability",
    "LT": "Life-threatening",
    "OT": "Required Intervention"
}


# ---------------- Country to Continent ----------------
def country_to_continent(country_name):
    try:
        country_code = pc.country_name_to_country_alpha2(country_name, cn_name_format="default")
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        continent_map = {
            "AF": "Africa",
            "AS": "Asia",
            "EU": "Europe",
            "NA": "North America",
            "SA": "South America",
            "OC": "Oceania"
        }
        return continent_map.get(continent_code, "UNK")
    except:
        return "UNK"


# ---------------- Streamlit UI ----------------
st.title("💊Antidiabetic Adverse Event Predictor")
st.markdown("""
This application predicts the **seriousness of adverse events** reported for selected **antidiabetic drugs** 
using the **FDA FAERS database**.  

Provide patient and case details below, and the model will predict the most likely clinical outcome.
""")

# ---------------- Input Form ----------------
with st.form("prediction_form"):
    # drugname
    drug = st.selectbox("Select Antidiabetic Drug",
                        ['DAPAGLIFLOZIN', 'EMPAGLIFLOZIN', 'GLIMEPIRIDE',
                         'GLIPIZIDE', 'LINAGLIPTIN', 'METFORMIN',
                         'SAXAGLIPTIN', 'SITAGLIPTIN'])

    # age_in_years
    # ---------------- Age Input ----------------
    age_input = st.text_input("Age in Years", placeholder="Enter a number between 1 and 120")
    # Validate age input
    if age_input:
        if not age_input.isdigit():
            st.warning("⚠️ Please enter a valid number.")
        else:
            age = int(age_input)
            if age < 1 or age > 120:
                st.warning("⚠️ Age must be between 1 and 120.")
            else:
                st.success(" Age recorded.")

    # sex
    sex = st.selectbox("Sex", ["Male", "Female", "Prefer Not To Say"])
    sex = {"Male": "M", "Female": "F", "Prefer Not To Say": "UNK"}[sex]

    # occr_continent (derived from country)
    country_list = [c.name for c in pycountry.countries]
    country = st.selectbox("Country", country_list, index=country_list.index("United States"))
    continent = country_to_continent(country)

    # indi_category
    indi_cat = st.selectbox("What medical condition was the drug prescribed for?",
                            ['Blood disorders', 'Cardiovascular', 'Diabetes', 'Gastrointestinal',
                             'Infections', 'Mental/Nervous disorders', 'Metabolic (non-diabetes)',
                             'Neoplasms', 'Other', 'Procedural/Product issues',
                             'Renal/Urinary', 'Respiratory'])
    # dechal
    dechal = st.selectbox("Did the patient’s condition improve after stopping the drug ?",
                          ["Improved", "Not Improved", "Uncertain"])
    dechal = {"Improved": "Y", "Not Improved": "N", "Uncertain": "U", "Uncertain": "D"}[dechal]

    # rechal
    rechal = st.selectbox("Did the adverse event return after restarting the drug?",
                          ["Yes", "No","Not Applicable"])
    rechal = {"Yes": "Y", "No": "N","Not Applicable":"U","Not Applicable":"D"}[rechal]


    # pt_soc
    pt_cat = st.selectbox("Select the type of disorder affected by the drug:",
                          ['Cardiac disorders', 'Diabetes-related', 'Eye disorders',
                           'Gastrointestinal disorders', 'Infections', 'Metabolic/Blood disorders',
                           'Musculoskeletal disorders', 'Neoplasms', 'Nervous/Autoimmune disorders',
                           'Other', 'Product/Procedural issues', 'Renal/Hepatic disorders',
                           'Respiratory disorders', 'Skin/General disorders', 'Vascular disorders'])

    submit = st.form_submit_button("Predict")

# ---------------- Prediction ----------------
import pandas as pd

# ---------------- Prediction ----------------
if submit:
    # Build input row in exact feature order
    input_row = pd.DataFrame([{
        "drugname": drug,
        "age_in_years": age,
        "sex": sex,
        "occr_continent": continent,
        "indi_category": indi_cat,
        "dechal": dechal,
        "rechal": rechal,
        "pt_soc": pt_cat
    }])

    # Separate categorical + numeric features
    categorical_cols = ["drugname", "dechal", "rechal", "sex","occr_continent", "indi_category", "pt_soc"]
    numeric_cols = ["age_in_years"]

    # Encode only categorical
    X_cat = ohe_encoder.transform(input_row[categorical_cols])

    # Scale numeric (reshape because scaler expects 2D)
    X_num = scaler.transform(input_row[numeric_cols])

    # Concatenate encoded categorical + scaled numeric
    X_processed = np.hstack([X_cat, X_num])

    # Apply PCA
    X_pca = pca.transform(X_processed)

    # Predict
    y_pred = model.predict(X_pca)
    y_label = label_encoder.inverse_transform(y_pred)[0]

    st.success(f"Predicted Outcome: **{outcome_mapping[y_label]}**")

