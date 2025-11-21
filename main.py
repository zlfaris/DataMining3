import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Prediksi 5G Smartphone",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Prediksi Smartphone Mendukung 5G atau Tidak")
st.write("Masukkan spesifikasi smartphone untuk memprediksi apakah mendukung 5G.")

@st.cache_resource
def load_model():
    return joblib.load("model_ensemble.pkl")

model = load_model()

st.subheader("Masukkan Spesifikasi Smartphone")

brand = st.text_input("Brand")
model_name = st.text_input("Model")
price = st.number_input("Harga (Rp)", min_value=0)
avg_rating = st.number_input("Rating Rata-rata", min_value=0.0, max_value=10.0, step=0.1)
processor_brand = st.text_input("Processor Brand")
num_cores = st.number_input("Jumlah Core CPU", min_value=1)
processor_speed = st.number_input("Kecepatan CPU (GHz)", min_value=0.0)
battery_capacity = st.number_input("Kapasitas Baterai (mAh)", min_value=0)
ram_capacity = st.number_input("RAM (GB)", min_value=1)
internal_memory = st.number_input("Memori Internal (GB)", min_value=1)
screen_size = st.number_input("Ukuran Layar (inci)", min_value=0.0)
refresh_rate = st.number_input("Refresh Rate (Hz)", min_value=1)
num_rear_cameras = st.number_input("Jumlah Kamera Belakang", min_value=1)

os = st.text_input("Sistem Operasi (Android/iOS)")
primary_camera_rear = st.number_input("Kamera Belakang Utama (MP)", min_value=0)
primary_camera_front = st.number_input("Kamera Depan Utama (MP)", min_value=0)
resolution_height = st.number_input("Resolusi Tinggi (px)", min_value=0)
resolution_width = st.number_input("Resolusi Lebar (px)", min_value=0)

if st.button("Prediksi"):
    input_data = pd.DataFrame([{
        "brand_name": brand,
        "model": model_name,
        "price": price,
        "avg_rating": avg_rating,
        "processor_brand": processor_brand,
        "num_cores": num_cores,
        "processor_speed": processor_speed,
        "battery_capacity": battery_capacity,
        "ram_capacity": ram_capacity,
        "internal_memory": internal_memory,
        "screen_size": screen_size,
        "refresh_rate": refresh_rate,
        "num_rear_cameras": num_rear_cameras,
        "os": os,
        "primary_camera_rear": primary_camera_rear,
        "primary_camera_front": primary_camera_front,
        "resolution_height": resolution_height,
        "resolution_width": resolution_width
    }])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    st.subheader("Hasil Prediksi")
    if prediction == 1:
        st.success(f"✔ Smartphone ini MENDUKUNG 5G — Probabilitas {prob[1]*100:.2f}%")
    else:
        st.error(f"✖ Smartphone ini TIDAK mendukung 5G — Probabilitas {prob[0]*100:.2f}%")
