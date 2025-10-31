import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🖐 American Sign Language Classifier (YOLOv11)")
st.write("Use your webcam for real-time hand sign detection!")

# Load your trained YOLOv11 model
model = YOLO("best.pt")  # path to your fine-tuned weights

# Use Streamlit's camera input (browser-based)
img_file = st.camera_input("Capture a hand sign")

if img_file is not None:
    # Convert the image to an array
    image = Image.open(img_file)
    st.image(image, caption="Captured Image")

    # Convert to numpy array
    img_array = np.array(image)

    # Perform YOLOv11 inference
    results = model.predict(img_array, imgsz=640, conf=0.5)

    # Display results
    for result in results:
        st.image(result.plot(), caption="Predicted Output")
        st.write(result.names)
