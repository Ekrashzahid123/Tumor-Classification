import streamlit as st
import io
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered",
)

st.title("Brain Tumor MRI Classifier 🧠")
st.markdown("Upload a brain MRI scan and get an AI-powered classification (MobileNetV2) into **Glioma**, **Meningioma**, **No Tumor**, or **Pituitary Tumor**.")
st.markdown("**developed by Ekrash Zahid**")

# Use a cached function to load the model to speed up multiple requests
@st.cache_resource
def get_model_and_predict_module():
    from app.predict import load_model, predict
    load_model() # prime the model in background
    return predict

predict_func = None
try:
    predict_func = get_model_and_predict_module()
except Exception as e:
    st.error(f"Error loading model: {e}. Make sure you've trained the model first.")

uploaded_file = st.file_uploader("Choose an MRI image file", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded_file is not None and predict_func is not None:
    # Read image
    bytes_data = uploaded_file.getvalue()
    
    # Display the uploaded image
    st.image(bytes_data, caption="Uploaded MRI Scan", use_container_width=True)
    
    if st.button("Analyze MRI"):
        with st.spinner("Analyzing scan using AI..."):
            try:
                result = predict_func(bytes_data)
                
                # Colors map
                color_map = {
                    "None": "green",
                    "Medium": "orange",
                    "High": "red"
                }
                
                # Show results in metric-like format
                st.subheader(f"Prediction: {result['label']}")
                st.metric(label="Confidence", value=f"{result['confidence']}%")
                
                severity_color = color_map.get(result.get('severity', "None"), "blue")
                st.markdown(f"**Severity:** :{severity_color}[{result.get('severity', 'Unknown')}]")
                st.info(result['description'])
                
                # Show probability bars
                st.subheader("Confidence Scores")
                for cls_name, score in sorted(result['scores'].items(), key=lambda x: x[1], reverse=True):
                    st.progress(score / 100, text=f"{cls_name.capitalize()}: {score}%")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

st.divider()
st.caption("⚠️ This tool is for educational and research purposes only. It is not a substitute for professional medical advice or clinical diagnosis.")
