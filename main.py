import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Prediksi 5G Smartphone", page_icon="📱", layout="wide")

st.title("📱 Prediksi Dukungan 5G Smartphone")
st.write("Aplikasi ini menggunakan Random Forest, Logistic Regression, dan Ensemble Voting.")


@st.cache_resource
def load_all():
    preprocessor = joblib.load("preprocessor.pkl")
    model_rf = joblib.load("model_rf.pkl")
    model_lr = joblib.load("model_lr.pkl")
    ensemble = joblib.load("model_ensemble.pkl")
    return preprocessor, model_rf, model_lr, ensemble

preprocessor, model_rf, model_lr, model_ensemble = load_all()

models = {
    "Random Forest": model_rf,
    "Logistic Regression": model_lr,
    "Ensemble Voting": model_ensemble
}

df = pd.read_csv("smartphones.csv")
feature_cols = df.drop(columns=["5G_or_not"]).columns.tolist()

st.header("📥 Input Data Smartphone Baru")

input_data = {}
col1, col2 = st.columns(2)

for i, col in enumerate(feature_cols):
    if df[col].dtype == "object":
        val = (col1 if i % 2 == 0 else col2).selectbox(col, df[col].unique())
    else:
        median_val = float(df[col].median())
        val = (col1 if i % 2 == 0 else col2).number_input(col, value=median_val)
    input_data[col] = val

input_df = pd.DataFrame([input_data])


st.header("🧠 Pilih Model")

selected = st.radio("Model:", ["Random Forest", "Logistic Regression", "Ensemble Voting"])
model = models[selected]

if st.button("🚀 Prediksi"):
    try:
        # preprocessing dulu
        transformed = preprocessor.transform(input_df)

        # prediksi
        pred = model.predict(transformed)[0]
        proba = model.predict_proba(transformed)[0]

        st.success("Hasil Prediksi:")
        if pred == 1:
            st.write("### ✅ Smartphone MENDUKUNG 5G")
        else:
            st.write("### ❌ Smartphone TIDAK MENDUKUNG 5G")

        st.metric("Probabilitas 0 (Tidak 5G)", f"{proba[0]*100:.2f}%")
        st.metric("Probabilitas 1 (5G)", f"{proba[1]*100:.2f}%")

    except Exception as e:
        st.error(f"ERROR: {e}")
