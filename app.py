import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🦷 Dental X-Ray Detector")
model = YOLO("best.pt")

uploaded = st.file_uploader("Upload X-Ray", type=['jpg','png'])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Original")
    if st.button("Detect"):
        result = model.predict(uploaded, conf=0.25)
        st.image(result[0].plot())
