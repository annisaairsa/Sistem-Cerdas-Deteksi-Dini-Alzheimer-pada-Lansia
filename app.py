import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="Deteksi Risiko Alzheimer Lansia",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Sistem Deteksi Risiko Alzheimer pada Lansia")
st.write("Berbasis Machine Learning (Logistic Regression)")

# ==============================
# LOAD & TRAIN MODEL
# ==============================
@st.cache_resource
def load_model():
    # DATA CONTOH (WAJIB ADA >1 BARIS)
    data = {
        "usia": [60, 65, 70, 75, 80],
        "tekanan_darah": [120, 130, 140, 150, 160],
        "kolesterol": [180, 200, 220, 240, 260],
        "memori": [8, 7, 5, 4, 3],
        "label": ["Normal", "Normal", "Alzheimer", "Alzheimer", "Alzheimer"]
    }

    df = pd.DataFrame(data)

    X = df.drop("label", axis=1)
    y = df["label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    return model, scaler, label_encoder


model, scaler, label_encoder = load_model()

# ==============================
# INPUT USER
# ==============================
st.subheader("📝 Input Data Lansia")

usia = st.number_input("Usia", 50, 100, 65)
tekanan = st.number_input("Tekanan Darah", 90, 200, 120)
kolesterol = st.number_input("Kolesterol", 100, 350, 200)
memori = st.slider("Skor Daya Ingat", 1, 10, 7)

# ==============================
# PREDIKSI
# ==============================
if st.button("🔍 Deteksi Risiko"):
    input_data = np.array([[usia, tekanan, kolesterol, memori]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    result = label_encoder.inverse_transform(prediction)[0]

    if result == "Normal":
        st.success("✅ Kondisi Normal")
    else:
        st.error("⚠️ Berisiko Alzheimer")
