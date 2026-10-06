import streamlit as st
import numpy as np
import tensorflow as tf
import os
import gdown
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input


# Load Model
@st.cache_resource
def load_model():
    # URL Google Drive file
    drive_url = "https://drive.google.com/file/d/1XA6DRBC27UdDlX4-nEcno_KKeCgVA6Ju/view?usp=sharing"
    local_path = "best_model_vgg16.keras"

    # Download hanya kalau file belum ada
    if not os.path.exists(local_path):
        gdown.download(drive_url, local_path, quiet=False)

    model = tf.keras.models.load_model(local_path)
    return model


model = load_model()

# Daftar Kelas Masakan Padang
class_names = [
    "ayam_goreng",
    "ayam_pop",
    "daging_rendang",
    "dendeng_batokok",
    "gulai_ikan",
    "gulai_tambusu",
    "gulai_tunjang",
    "telur_balado",
    "telur_dadar",
]


# Preprocess Image
def preprocess_image(image):

    # Pastikan RGB
    image = image.convert("RGB")

    # Resize sesuai input VGG16
    image = image.resize((224, 224))

    # Ubah menjadi numpy array
    image_array = np.array(image)

    # Tambahkan dimensi batch
    image_array = np.expand_dims(image_array, axis=0)

    # Preprocessing khusus VGG16
    image_array = preprocess_input(image_array)

    return image_array


def run():

    st.title("Prediksi Masakan Padang")

    st.write("Upload gambar Masakan Padang untuk mengetahui jenis makanan")

    uploaded_file = st.file_uploader(
        "Masukkan Gambar Masakan Padang yang ingin diprediksi",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:

        # Baca gambar
        image = Image.open(uploaded_file)

        # Tampilkan gambar
        st.image(image, caption="Gambar yang diupload")

        if st.button("Predict"):

            # Preprocessing
            image_processed = preprocess_image(image)

            # Prediction
            prediction = model.predict(image_processed)

            # Ambil index probabilitas terbesar
            predicted_index = np.argmax(prediction[0])

            # Ubah index menjadi nama kelas
            predicted_label = class_names[predicted_index]

            # Confidence
            confidence = np.max(prediction[0]) * 100

            st.write("### Hasil Prediksi")
            st.success(f"Masakan: {predicted_label}")
            st.write(f"Confidence: **{confidence:.2f}%**")


if __name__ == "__main__":
    run()
