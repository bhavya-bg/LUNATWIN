# 🫁 LUNATWIN

**LUNA16 3D Vision Transformer + Digital Twin for Pulmonary Nodule Analysis**

An advanced system for automated lung nodule detection, progression tracking, and clinical decision support using 3D Vision Transformer and Digital Twin technology.

### 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-LIVE-LINK-HERE.streamlit.app)

---

### Features

- 3D Vision Transformer based nodule detection from CT scans
- Patient-specific Digital Twin with longitudinal progression tracking
- Automatic risk stratification (Low / Medium / High)
- AI-powered Clinical Reasoning Report with recommendations
- Interactive Streamlit web interface

### Dataset

Trained on the **LUNA16** dataset consisting of 888 thoracic CT scans.

### Model Architecture

- Hybrid 3D CNN + Vision Transformer (VitDet3D)
- End-to-end training for nodule detection and bounding box regression
- Achieves high detection performance on LUNA16 benchmark

### How to Run Locally

```bash
# Clone the repository
git clone https://github.com/bhavya-bg/LUNATWIN.git
cd LUNATWIN

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
