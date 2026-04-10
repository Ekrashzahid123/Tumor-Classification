import streamlit as st
import io
from PIL import Image
import numpy as np
import time

# Set page configuration
st.set_page_config(
    page_title="AI Brain Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Advanced CSS for Glassmorphism & Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Animated Gradient Background for the App */
    .stApp {
        background: radial-gradient(circle at 15% 50%, rgba(20, 30, 48, 1), rgba(36, 59, 85, 0) 50%),
                    radial-gradient(circle at 85% 30%, rgba(13, 14, 21, 1), rgba(13, 14, 21, 0) 50%);
        background-color: #0a0e17;
        background-size: cover;
    }

    /* Glowing Title */
    .main-title {
        font-size: 4rem !important;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(to right, #00d4ff, #7303c0, #f80759);
        background-size: 200% auto;
        color: #fff;
        background-clip: text;
        text-fill-color: transparent;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 5s linear infinite;
        margin-bottom: 0px !important;
        padding-bottom: 0px;
    }
    
    @keyframes shine {
        to {
            background-position: 200% center;
        }
    }
    
    /* Center text block */
    .subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 3rem;
        font-weight: 300;
    }

    /* Glassmorphism Upload Container */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        background: rgba(0, 212, 255, 0.05);
        border-color: rgba(0, 212, 255, 0.5);
    }
    
    /* Cyberpunk Button */
    .stButton>button {
        background: linear-gradient(90deg, #020024 0%, #090979 35%, #00d4ff 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 1.2rem;
        letter-spacing: 1px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    .stButton>button:hover {
        transform: scale(1.02) translateY(-3px);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.8);
    }

    /* Glassmorphism Results Card */
    .glass-card {
        background: rgba(21, 27, 43, 0.7);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 24px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-top: 10px;
    }
    
    /* Metrics Override */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        color: #fff;
    }
    
    /* Image styling border */
    .scan-image-wrapper img {
        border-radius: 16px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 0 40px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">NEURO-VISION AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Next-Generation MobileNetV2 Brain MRI Classification<br><span style="font-size:0.9rem;opacity:0.5">Developed by Ekrash Zahid</span></p>', unsafe_allow_html=True)

# Use a cached function to load the model securely
@st.cache_resource(show_spinner=False)
def get_model_and_predict_module():
    from app.predict import load_model, predict
    load_model() # prime the model in background
    return predict

predict_func = None
try:
    predict_func = get_model_and_predict_module()
except Exception as e:
    st.error(f"Error loading AI Engine: {e}. Has the model been trained?")

uploaded_file = st.file_uploader("Drop a high-res MRI scan here", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded_file is not None and predict_func is not None:
    bytes_data = uploaded_file.getvalue()
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, spacing, col2 = st.columns([1.2, 0.1, 1.2])
    
    with col1:
        st.markdown("<h3 style='text-align: center; color: #94a3b8;'>Scan Preview</h3>", unsafe_allow_html=True)
        st.markdown('<div class="scan-image-wrapper">', unsafe_allow_html=True)
        st.image(bytes_data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("📡 INITIATE AI SCAN", use_container_width=True)
        
    with col2:
        if analyze_btn:
            with st.spinner("Processing deep latent features..."):
                time.sleep(0.5) # Slight delay for dramatic effect
                try:
                    result = predict_func(bytes_data)
                    
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("<h3 style='margin-top:0; color: #00d4ff;'>Diagnostic Results</h3>", unsafe_allow_html=True)
                    
                    # Large Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("Diagnosis Focus", result['label'])
                    m2.metric("Neural Confidence", f"{result['confidence']}%")
                    
                    # Futuristic Severity Card
                    color = result.get('color', '#3b82f6')
                    severity = result.get('severity', 'Unknown')
                    
                    st.markdown(f'''
                    <div style="background: linear-gradient(145deg, rgba(20,20,30,1) 0%, rgba(10,10,15,1) 100%); 
                                border: 1px solid {color}40; 
                                border-left: 5px solid {color}; 
                                padding: 20px; 
                                border-radius: 12px; 
                                margin-top: 20px; 
                                box-shadow: 0 10px 30px {color}20;">
                        <div style="display:flex; align-items:center;">
                            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {color}; box-shadow: 0 0 10px {color}; margin-right: 15px;"></div>
                            <h4 style="margin: 0; color: white; font-size: 1.2rem; font-weight: 800;">Risk Level: <span style="color:{color}">{severity}</span></h4>
                        </div>
                        <p style="margin-top: 15px; margin-bottom: 0; font-size: 1rem; color: #a1a1aa; line-height: 1.6;">{result['description']}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<h4 style='color: #94a3b8; font-size: 1.1rem;'>Distribution Graph</h4>", unsafe_allow_html=True)
                    
                    # Styled Progress Bars
                    for cls_name, score in sorted(result['scores'].items(), key=lambda x: x[1], reverse=True):
                        st.markdown(f'''
                        <div style="margin-bottom: 8px; display:flex; justify-content:space-between; color:#cbd5e1; font-weight:500;">
                            <span>{cls_name.capitalize()}</span>
                            <span>{score}%</span>
                        </div>
                        ''', unsafe_allow_html=True)
                        st.progress(score / 100)

                    st.markdown('</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"System Error: {str(e)}")
        else:
            # Standby screen
            st.markdown('<div class="glass-card" style="text-align:center; padding: 60px 20px;">', unsafe_allow_html=True)
            st.markdown('<h1 style="font-size: 4rem; margin:0; opacity: 0.8;">🔮</h1>', unsafe_allow_html=True)
            st.markdown('<h2 style="color: #64748b; margin-top:20px;">Awaiting Initialization</h2>', unsafe_allow_html=True)
            st.markdown('<p style="color: #475569;">Click the scan button to begin pattern recognition.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #475569; font-size: 0.85rem; border-top: 1px solid #1e293b; padding-top: 20px;">
    <b>CLINICAL DISCLAIMER:</b> This system uses experimental AI protocols. 
    It is not certified for medical diagnosis, treatment, or clinical use. Always consult a qualified neurologist.
</div>
""", unsafe_allow_html=True)
