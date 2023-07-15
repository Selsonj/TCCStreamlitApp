import io
import numpy as np
import streamlit as st
import tensorflow as tf
from google.cloud import storage
from PIL import Image

model_path = "gs://tcc_covid19/model_architecture1.h5"

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def load_model_from_gcs(model_path):
    bucket_name = "seu-bucket"
    blob_name = "model_architecture1.h5"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.download_to_filename("model_architecture1.h5")
    model = tf.keras.models.load_model("model_architecture1.h5")

    return model

def main():
    st.title("Classificação de COVID-19 em uma imagem de Raio-X")
    st.write("Carregue uma imagem de Raio-X para analisar a probabilidade de ser COVID-19.")

    uploaded_file = st.file_uploader("Escolha uma imagem de Raio-X", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem carregada", use_column_width=True)

        preprocessed_image = preprocess_image(image)

        model = load_model_from_gcs(model_path)
        prediction = model.predict(preprocessed_image)
        class_index = np.argmax(prediction)
        class_label = "COVID-19" if class_index == 0 else "Non-COVID-19"
        confidence = prediction[0][class_index] * 100

        st.success(f"A imagem é classificada como {class_label} com {confidence:.2f}% de confiança.")

if __name__ == "__main__":
    main()
