import streamlit as st
from inference import load_model, predict_image
from PIL import Image
import tempfile
import os


st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Pneumonia Detection from Chest X-ray")
st.markdown(
    """
    Upload a **Chest X-ray image** and this application will predict
    whether the patient is **NORMAL** or has **PNEUMONIA**.
    """
)

st.divider()

@st.cache_resource
def get_model():
    return load_model("best_model.pth")

model = get_model()

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📷 Uploaded X-ray Image")
    st.image(image, caption="Chest X-ray", use_container_width=True)

    st.divider()

    if st.button("🔍 Predict Pneumonia"):
        with st.spinner("Analyzing X-ray..."):
    
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                image.save(tmp.name)
                temp_path = tmp.name
    
            result = predict_image(temp_path, model)
            os.remove(temp_path)
    
        st.subheader("🧠 Prediction Result")
    
        if result["prediction"] == "PNEUMONIA":
            st.error("🦠 **PNEUMONIA detected**")
        else:
            st.success("✅ **NORMAL chest X-ray**")
    
        st.info(f"Confidence: {result['confidence']}%")
