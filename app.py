import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime
import plotly.express as px

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="LUNATWIN",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CLEAN LIGHT THEME CSS ======================
st.markdown("""
    <style>
    .main { background-color: #f8fafc; color: #1e2937; }
    h1 {
        background: linear-gradient(90deg, #14b8a6, #0f766e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .stButton>button {
        background: linear-gradient(90deg, #14b8a6, #2dd4bf);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        height: 3.5em;
        border: none;
        box-shadow: 0 4px 20px rgba(45, 212, 191, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(45, 212, 191, 0.4);
    }
    .risk-high { color: #ef4444; background: #fee2e2; padding: 8px 20px; border-radius: 50px; font-weight: 700; }
    .risk-medium { color: #f59e0b; background: #fef3c7; padding: 8px 20px; border-radius: 50px; font-weight: 700; }
    .risk-low { color: #10b981; background: #d1fae5; padding: 8px 20px; border-radius: 50px; font-weight: 700; }
    .report-container {
        background: white;
        padding: 35px;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.title("🫁 LUNATWIN")
st.markdown("**LUNA16 3D Vision Transformer • Digital Twin • Clinical Intelligence**")
st.caption("Advanced Pulmonary Nodule Analysis System for Early Lung Cancer Detection")

# ====================== SIDEBAR ======================
st.sidebar.header("🩺 Patient Profile")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    patient_id = st.text_input("Patient ID", value="P001")
    age = st.number_input("Age", 18, 100, 62)
with col_b:
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking History", ["Never", "Former", "Current"])

st.sidebar.markdown("---")
st.sidebar.info("🔬 Currently running in Mock Mode (Training not completed yet)")

# ====================== SESSION STATE ======================
if "digital_twin" not in st.session_state:
    st.session_state.digital_twin = {
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "smoking": smoking,
        "scan_history": []
    }

st.session_state.digital_twin.update({
    "patient_id": patient_id,
    "age": age,
    "gender": gender,
    "smoking": smoking
})

# ====================== MOCK INFERENCE (No Real Model Loading) ======================
def run_inference(uploaded_file=None):
    # Always use mock for now
    if uploaded_file:
        return [
            {"coord": [-42, -28, 118, 13.8], "prob": 0.978},
            {"coord": [22, 38, 162, 9.2], "prob": 0.865},
            {"coord": [8, -52, 98, 15.6], "prob": 0.941}
        ]
    return [
        {"coord": [-45, -30, 120, 12.4], "prob": 0.962},
        {"coord": [25, 40, 165, 8.7], "prob": 0.881}
    ]

# ====================== HELPERS ======================
def extract_features(detections):
    count = len(detections)
    sizes = [d["coord"][3] for d in detections]
    probs = [d["prob"] for d in detections]
    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "nodule_count": count,
        "avg_size_mm": round(sum(sizes) / count, 2) if count > 0 else 0.0,
        "max_prob": round(max(probs), 3) if probs else 0.0,
        "total_volume_est": round(sum(sizes) * 4.1888, 1)
    }

def get_risk_level(state):
    if state["nodule_count"] >= 3 or state["avg_size_mm"] > 13:
        return "HIGH", "risk-high"
    elif state["nodule_count"] >= 2 or state["avg_size_mm"] > 8:
        return "MEDIUM", "risk-medium"
    return "LOW", "risk-low"

# ====================== TABS ======================
tab1, tab2, tab3 = st.tabs(["📸 New Scan Analysis", "📊 Digital Twin Dashboard", "🧠 Clinical Report"])

with tab1:
    st.subheader("Upload CT Scan Image")
    uploaded_file = st.file_uploader(
        "Upload CT Slice (PNG / JPG / DICOM)", 
        type=["png", "jpg", "jpeg", "dcm"]
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded CT Slice", use_container_width=True)

    if st.button("🚀 Run 3D Vision Transformer Analysis", type="primary", use_container_width=True):
        if not uploaded_file:
            st.error("Please upload a CT image first!")
        else:
            with st.spinner("Processing CT Scan..."):
                detections = run_inference(uploaded_file)
                features = extract_features(detections)
                
                history = st.session_state.digital_twin["scan_history"]
                if history:
                    prev = history[-1]
                    delta = features["avg_size_mm"] - prev["avg_size_mm"]
                    features["progression"] = "🔴 Growing" if delta > 0.5 else "🟢 Shrinking" if delta < -0.5 else "⚪ Stable"
                    features["delta_mm"] = round(delta, 2)
                else:
                    features["progression"] = "📍 Baseline"
                    features["delta_mm"] = 0.0
                
                st.session_state.digital_twin["scan_history"].append(features)
                st.success("✅ Analysis Complete!")
                st.balloons()

with tab2:
    st.subheader("Digital Twin Overview")
    history = st.session_state.digital_twin["scan_history"]
    
    if history:
        latest = history[-1]
        risk_level, risk_class = get_risk_level(latest)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Nodule Count", latest["nodule_count"])
        with c2: st.metric("Avg Size", f"{latest['avg_size_mm']} mm")
        with c3: st.metric("Max Confidence", f"{latest['max_prob']:.1%}")
        with c4:
            st.markdown("**Risk Level**")
            st.markdown(f"<span class='{risk_class}'>{risk_level} RISK</span>", unsafe_allow_html=True)
        
        view_option = st.radio("View Mode", ["Cards", "Detailed Table", "Trend Chart"], horizontal=True)
        
        if view_option == "Detailed Table":
            df = pd.DataFrame(history)
            st.dataframe(
                df.style.format({
                    "avg_size_mm": "{:.2f} mm",
                    "max_prob": "{:.1%}",
                    "total_volume_est": "{:.1f} mm³"
                }),
                use_container_width=True,
                hide_index=True
            )
        elif view_option == "Trend Chart" and len(history) > 1:
            df = pd.DataFrame(history)
            fig = px.line(df, x="date", y="avg_size_mm", 
                         markers=True, title="Nodule Size Progression",
                         color_discrete_sequence=["#14b8a6"])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No scans yet. Go to New Scan Analysis tab.")

with tab3:
    st.subheader("🧠 Clinical Reasoning Report")
    if st.session_state.digital_twin["scan_history"]:
        latest = st.session_state.digital_twin["scan_history"][-1]
        risk_level, risk_class = get_risk_level(latest)
        
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.markdown(f"""
        **Patient:** {patient_id} | **Age:** {age} | **Gender:** {gender}  
        **Smoking History:** {smoking}  
        **Scan Date:** {latest['date']}
        
        ### Scan Summary
        - **Nodules Detected:** `{latest['nodule_count']}`
        - **Average Diameter:** `{latest['avg_size_mm']}` mm
        - **Estimated Total Volume:** `{latest.get('total_volume_est', 0)}` mm³
        - **Highest Confidence:** `{latest['max_prob']:.1%}`
        - **Progression:** `{latest.get('progression', 'Baseline')}` ({latest.get('delta_mm', 0):+.2f} mm)
        
        ### Risk Assessment
        <span class="{risk_class}">{risk_level} RISK</span>
        
        ### Recommendations
        • Immediate pulmonologist consultation recommended  
        • Follow-up LDCT in **3 months**  
        • Consider **biopsy/PET-CT** if any nodule >12mm or growing  
        • Monitor symptoms: cough, hemoptysis, weight loss
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Run a scan to generate clinical report.")

st.divider()
st.caption("LUNATWIN • Powered by 3D Vision Transformer + Digital Twin Technology • Mock Mode Active")