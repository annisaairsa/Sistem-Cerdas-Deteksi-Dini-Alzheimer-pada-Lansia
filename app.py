import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Deteksi Alzheimer Lansia",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Sistem Deteksi Risiko Alzheimer pada Lansia")
st.caption("Berbasis Machine Learning (Logistic Regression)")
st.divider()

# =========================
# LOAD & TRAIN MODEL
# =========================
@st.cache_resource
def load_model():
    # Load dataset
    df = pd.read_csv("alzheimers_disease_data.csv")

    fitur = [
        "Age",
        "MMSE",
        "FunctionalAssessment",
        "MemoryComplaints",
        "BehavioralProblems",
        "ADL"
    ]
    target = "Diagnosis"

    # Ambil kolom penting
    df = df[fitur + [target]]

    # Fokus lansia
    df = df[df["Age"] >= 60]

    # Hanya 2 kelas
    df = df[df[target].isin(["Normal", "Alzheimer"])]

    # Cek jumlah data
    if df[target].value_counts().min() < 2:
        st.error("❌ Data Alzheimer / Normal tidak cukup untuk training model.")
        st.stop()

    # Encode target
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])

    X = df.drop(target, axis=1)
    y = df[target]

    # Split TANPA stratify (anti error)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Model Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler, le


# Load model
model, scaler, label_encoder = load_model()

# =========================
# INPUT USER
# =========================
st.subheader("Input Data Pasien")

age = st.number_input("Usia (≥ 60 tahun)", min_value=60, max_value=120, value=65)
mmse = st.slider("Skor MMSE (0–30)", 0, 30, 24)
functional = st.slider("Skor Fungsi Kognitif", 0, 10, 7)
adl = st.slider("Skor Aktivitas Harian (ADL)", 0, 10, 7)
memory = st.radio("Keluhan Memori", ["Tidak", "Ya"])
behavior = st.radio("Masalah Perilaku", ["Tidak", "Ya"])

st.divider()

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Deteksi Risiko"):
    with st.spinner("Menganalisis data..."):
        time.sleep(1)

        input_data = np.array([[
            age,
            mmse,
            functional,
            1 if memory == "Ya" else 0,
            1 if behavior == "Ya" else 0,
            adl
        ]])

        input_scaled = scaler.transform(input_data)

        prob_alzheimer = model.predict_proba(input_scaled)[0][1]

        threshold = 0.5
        hasil = "Alzheimer" if prob_alzheimer >= threshold else "Normal"

    st.subheader("Hasil Deteksi")

    if hasil == "Alzheimer":
        st.error(f"⚠️ Risiko Alzheimer Terdeteksi ({prob_alzheimer:.1%})")
        st.write(
            "Hasil menunjukkan indikasi risiko Alzheimer. "
            "Disarankan untuk melakukan pemeriksaan medis lanjutan."
        )
    else:
        st.success(f"✅ Kondisi Kognitif Normal ({(1 - prob_alzheimer):.1%})")
        st.write(
            "Tidak ditemukan indikasi gangguan kognitif yang signifikan."
        )

st.divider()
st.caption("⚠️ Sistem ini hanya digunakan sebagai alat bantu skrining awal, bukan diagnosis medis.")
