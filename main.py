import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="📱 Prediksi Smartphone 5G",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
# 📱 Prediksi Dukungan 5G Smartphone  
Aplikasi Machine Learning yang memprediksi apakah sebuah smartphone **mendukung 5G atau tidak**, menggunakan 3 model:

- 🌲 Random Forest  
- 🔵 Logistic Regression  
- 🟣 Ensemble Voting (RF + LR)

Akurasi model sudah mencapai **>90%**.
""")

@st.cache_resource
def load_models():
    model_rf = joblib.load("model_random_forest.pkl")
    model_lr = joblib.load("model_logistic_regression.pkl")
    model_ens = joblib.load("model_ensemble_voting.pkl")
    return model_rf, model_lr, model_ens

model_rf, model_lr, model_ens = load_models()

models = {
    "Random Forest": model_rf,
    "Logistic Regression": model_lr,
    "Ensemble Voting": model_ens
}

df = pd.read_csv("smartphones.csv")

# drop kolom target
feature_cols = df.drop(columns=["5G_or_not"]).columns.tolist()

st.subheader("📥 Masukkan Data Smartphone")

input_data = {}

col1, col2 = st.columns(2)

for i, col in enumerate(feature_cols):
    if df[col].dtype == "object":
        unique_vals = sorted(df[col].dropna().unique().tolist())
        chosen_val = col1.selectbox(f"{col}", unique_vals) if i % 2 == 0 else col2.selectbox(f"{col}", unique_vals)
        input_data[col] = chosen_val
    else:
        default_val = float(df[col].median())
        val = col1.number_input(f"{col}", value=default_val) if i % 2 == 0 else col2.number_input(f"{col}", value=default_val)
        input_data[col] = val

# Convert ke DataFrame
input_df = pd.DataFrame([input_data])

st.subheader("🤖 Pilih Model Prediksi")

chosen_model = st.radio(
    "Silakan pilih model:",
    ["Random Forest", "Logistic Regression", "Ensemble Voting"]
)

model = models[chosen_model]

predict_btn = st.button("🚀 Prediksi Sekarang", type="primary")

if predict_btn:
    with st.spinner("Sedang memproses..."):
        try:
            prediction = model.predict(input_df)[0]

            # Probabilitas (sof voting only)
            if chosen_model == "Ensemble Voting":
                proba = model.predict_proba(input_df)[0]
                confidence = max(proba) * 100
            else:
                # Model RF & LR juga punya predict_proba
                proba = model.predict_proba(input_df)[0]
                confidence = max(proba) * 100

            st.success("### 🎯 Hasil Prediksi")
            if prediction == 1:
                st.markdown("## ✅ **Smartphone ini MENDUKUNG 5G**")
            else:
                st.markdown("## ❌ **Smartphone ini TIDAK MENDUKUNG 5G**")

            st.info(f"**Tingkat Keyakinan Model:** {confidence:.2f}%")

            colA, colB = st.columns(2)
            with colA:
                st.metric("Probabilitas 0 (Tidak 5G)", f"{proba[0]*100:.2f}%")
            with colB:
                st.metric("Probabilitas 1 (5G)", f"{proba[1]*100:.2f}%")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

st.markdown("""
---
👨‍💻 *Sistem Prediksi 5G — Dibuat dengan Machine Learning (RF, LR, Ensemble Voting)*  
""")
