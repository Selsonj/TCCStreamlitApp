!pip install streamlit -q
!pip install google-cloud-storage

import subprocess
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

model_path = "gs://tcc_covid19/model_architecture1.h5"
model = load_model(model_path)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def main():
    st.title("Classificação de COVID-19 em uma imagem de Raio-X")
    st.write("Carregue uma imagem de Raio-X para analisar a probabilidade de ser COVID-19.")

    uploaded_file = st.file_uploader("Escolha uma imagem de Raio-X", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, caption="Imagem carregada", use_column_width=True)

        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        class_index = np.argmax(prediction)
        class_label = "COVID-19" if class_index == 0 else "Non-COVID-19"
        confidence = prediction[0][class_index] * 100

        st.success(f"A imagem é classificada como {class_label} com {confidence:.2f}% de confiança.")

if __name__ == "__main__":
    main()
