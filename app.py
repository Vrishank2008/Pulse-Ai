
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="PULSE AI", page_icon="🫁")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "pulse_ai_xray_model.keras",
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False
    )

st.title("🫁 PULSE AI")
st.subheader("AI-Assisted Tuberculosis Detection")
st.write("Upload a chest X-ray image for experimental AI classification.")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Chest X-ray", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", width=350)

    if st.button("Analyze X-ray"):
        img = img.resize((224, 224))
        arr = np.array(img).astype("float32")
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        score = float(model.predict(arr, verbose=0)[0][0])

        st.subheader("Prediction")
        st.metric("Tuberculosis score", f"{score * 100:.2f}%")
        st.metric("Normal score", f"{(1-score) * 100:.2f}%")

        st.info(
            "Experimental AI classification only. "
            "Not a medical diagnosis. Consult a qualified doctor."
        )
