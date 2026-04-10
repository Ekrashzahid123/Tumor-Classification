import streamlit as st
import io
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Header Styling */
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px !important;
        padding-bottom: 0px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #71717a;
        margin-top: -10px;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .author-badge {
        display: inline-block;
        padding: 4px 12px;
        background-color: #f3f4f6;
        color: #52525b;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 5px;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(236, 72, 153, 0.39);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.5);
    }
    
    /* Subheader Styling */
    h3 {
        font-weight: 700 !important;
        color: #3f3f46;
    }
    
    /* Alert Info Panel */
    .info-panel {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Brain Tumor MRI Classifier 🧠</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a brain MRI scan and get an AI-powered classification using MobileNetV2.</p>', unsafe_allow_html=True)
st.markdown('<div class="author-badge">Developed by Ekrash Zahid</div><br><br>', unsafe_allow_html=True)

# Use a cached function to load the model securely
@st.cache_resource(show_spinner=False)
def get_model_and_predict_module():
    from app.predict import load_model, predict
    load_model() # prime the model in background
    return predict

predict_func = None
with st.spinner("Loading AI Engine..."):
    try:
        predict_func = get_model_and_predict_module()
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure you've trained the model first.")

uploaded_file = st.file_uploader("📂 Choose a brain MRI image file", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded_file is not None and predict_func is not None:
    # Read image
    bytes_data = uploaded_file.getvalue()
    
    # We use a 2-column layout for a modern, side-by-side dashboard feel
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📷 Uploaded Scan")
        # Display the uploaded image with rounded corners via CSS wrap
        st.markdown('<div style="border-radius: 15px; overflow: hidden; box-shadow: 0 10px 20px rgba(0,0,0,0.08);">', unsafe_allow_html=True)
        st.image(bytes_data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🧬 Analyze MRI Scan", use_container_width=True)
        
    with col2:
        if analyze_btn:
            with st.spinner("Analyzing scan using AI patterns..."):
                try:
                    result = predict_func(bytes_data)
                    st.markdown("### 📊 AI Analysis Results")
                    
                    # Highlight Metrics
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("Predicted Diagnosis", result['label'])
                    m_col2.metric("AI Confidence", f"{result['confidence']}%")
                    
                    # Custom Severity Card
                    color = result.get('color', '#3b82f6')
                    severity = result.get('severity', 'Unknown')
                    
                    # Generate a beautiful info card
                    st.markdown(f'''
                    <div style="background-color: {color}15; border-left: 6px solid {color}; padding: 16px; border-radius: 8px; margin-top: 10px; margin-bottom: 20px;">
                        <h4 style="margin-top: 0; color: {color}; font-size: 1.1rem; font-weight: 700;">Severity: {severity}</h4>
                        <p style="margin-bottom: 0; font-size: 0.95rem; color: #4b5563; line-height: 1.5;">{result['description']}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Probabilities Breakdown
                    st.markdown("<h4 style='font-size: 1.1rem; color: #52525b; margin-bottom:15px'>Confidence Breakdown</h4>", unsafe_allow_html=True)
                    for cls_name, score in sorted(result['scores'].items(), key=lambda x: x[1], reverse=True):
                        st.write(f"**{cls_name.capitalize()}**")
                        st.progress(score / 100, text=f"{score}%")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
        else:
            # Placeholder instruction before clicking analyze
            st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; mt: 50px; opacity: 0.6">
                <h1 style="font-size: 4rem; margin: 0">👈</h1>
                <h3 style="text-align: center; color: #71717a">Click analyze to get AI predictions</h3>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("""
<div style="text-align: center; color: #a1a1aa; font-size: 0.9rem;">
    ⚠️ <b>Disclaimer:</b> This tool is for educational and research purposes only. It is not a substitute for professional medical advice or clinical diagnosis.
</div>
""", unsafe_allow_html=True)
