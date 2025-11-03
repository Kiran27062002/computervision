import streamlit as st

# --- Safe import for OpenCV ---
try:
    import cv2
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "opencv-python-headless"])
    import cv2

from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

# -------------------------
# Load YOLOv11 Model
# -------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolo11n.pt")  # your trained model
    return model

model = load_model()

# -------------------------
# Streamlit UI Setup
# -------------------------
st.set_page_config(page_title="ASL Live Classifier", page_icon="🖐")
st.title("🖐 American Sign Language Classifier (YOLOv11)")
st.write("Use your webcam for **real-time hand sign detection**!")

# Initialize session state
if "run" not in st.session_state:
    st.session_state.run = False

# -------------------------
# Start/Stop Buttons
# -------------------------
col1, col2 = st.columns(2)
with col1:
    start = st.button("▶️ Start Live Detection")
with col2:
    stop = st.button("⏹️ Stop Detection")

# Control logic
if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

# -------------------------
# Live Stream
# -------------------------
frame_window = st.image([])

# OpenCV webcam capture
cap = cv2.VideoCapture(0)

while st.session_state.run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame from webcam.")
        break

    # Inference
    results = model.predict(frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # Convert BGR → RGB for display
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame_rgb)

cap.release()
st.write("Webcam stopped.")
