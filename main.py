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
Sistem Machine Learning untuk memprediksi apakah smartphone **mendukung 5G atau tidak**, menggunakan:

- 🌲 Random Forest  
- 🔵 Logistic Regression  
- 🟣 Ensemble Voting (RF + LR)
""")

@st.cache_resource
def load_models():
    try:
        model_rf = joblib.load("model_random_forest.pkl")
        model_lr = joblib.load("model_logistic_regression.pkl")
        model_ensemble = joblib.load("model_ensemble_voting.pkl")
        return model_rf, model_lr, model_ensemble
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

model_rf, model_lr, model_ensemble = load_models()

models = {
    "Random Forest": model_rf,
    "Logistic Regression": model_lr,
    "Ensemble Voting": model_ensemble
}

df = pd.read_csv("smartphones.csv")
feature_cols = df.drop(columns=["5G_or_not"]).columns.tolist()

st.subheader("📥 Masukkan Data Smartphone")

input_data = {}
col1, col2 = st.columns(2)

for i, col in enumerate(feature_cols):
    if df[col].dtype == "object":
        vals = sorted(df[col].dropna().unique())
        selected = (col1 if i % 2 == 0 else col2).selectbox(col, vals)
        input_data[col] = selected
    else:
        default = float(df[col].median())
        selected = (col1 if i % 2 == 0 else col2).number_input(col, value=default)
        input_data[col] = selected

input_df = pd.DataFrame([input_data])

st.subheader("Pilih Model Prediksi")
chosen_model = st.radio("Pilih model:", list(models.keys()))

model = models[chosen_model]

if st.button("🚀 Prediksi Sekarang"):
    try:
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        confidence = max(prob) * 100

        st.success("### 🎯 Hasil Prediksi")
        st.metric("Probabilitas Tidak 5G", f"{prob[0]*100:.2f}%")
        st.metric("Probabilitas 5G", f"{prob[1]*100:.2f}%")

        if prediction == 1:
            st.markdown("## ✅ **Smartphone ini MENDUKUNG 5G**")
        else:
            st.markdown("## ❌ **Smartphone ini TIDAK MENDUKUNG 5G**")

        st.info(f"**Tingkat Keyakinan Model:** {confidence:.2f}%")

    except Exception as e:
        st.error(f"Terjadi kesalahan prediksi: {e}")

st.markdown("---")
st.caption("👨‍💻 Sistem Prediksi 5G — Machine Learning")
