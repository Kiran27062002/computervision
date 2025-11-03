import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time

# --- Load model once ---

@st.cache_resource
def load_model():
return YOLO("yolo11n.pt")

model = load_model()

st.title("🖐 Real-Time Hand Sign Detection (YOLOv11)")
st.markdown("Detects hand signs live from your webcam and predicts the letter.")

# --- Start/Stop control ---

run = st.checkbox("Start webcam")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

if not camera.isOpened():
st.error("❌ Cannot open webcam. Please check your camera or permissions.")

while run:
ret, frame = camera.read()
if not ret:
st.warning("⚠️ Failed to grab frame from webcam.")
break

```
# Run YOLO detection
results = model(frame, verbose=False)

# Annotate frame with YOLO predictions
annotated_frame = results[0].plot()

# Convert from BGR (OpenCV) to RGB (Streamlit)
annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

# Display live video
FRAME_WINDOW.image(annotated_frame, channels="RGB")

time.sleep(0.03)  # ~30 FPS limit
```

else:
if camera.isOpened():
camera.release()
st.info("🛑 Webcam stopped.")
