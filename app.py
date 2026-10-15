import pickle
from pathlib import Path

import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "svc_trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_artifacts()

st.title("❤️ Heart Disease Prediction")
st.write("Educational demonstration for AIC354 Assignment 2.")

st.warning(
    "Educational use only. This model is not a medical diagnostic tool and must not "
    "be used to make clinical decisions."
)

def options(column):
    return list(encoders[column].classes_)

cp = st.selectbox("Chest Pain Type (cp)", options("cp"))
slope = st.selectbox("Slope (slope)", options("slope"))
ca = st.selectbox("Major Vessels (ca)", options("ca"))
thal = st.selectbox("Thal (thal)", options("thal"))

if st.button("Predict Heart Disease", type="primary"):
    values = {"cp": cp, "slope": slope, "ca": ca, "thal": thal}

    encoded = [
        encoders[column].transform([str(values[column])])[0]
        for column in ["cp", "slope", "ca", "thal"]
    ]

    prediction = int(model.predict(np.array(encoded).reshape(1, -1))[0])

    if prediction == 1:
        st.error("Prediction: HEART DISEASE PRESENT")
    else:
        st.success("Prediction: NO HEART DISEASE")

    st.write("Encoded input:", encoded)
