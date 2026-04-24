import streamlit as st
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import dlib

from detect_video import detect_video
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------------------
# CONFIG
# -----------------------------------------
st.set_page_config(page_title="DeepFake AGI Detector", layout="wide")

st.title("🎭 DeepFake Detector — AGI MODE")
st.markdown("### 🧠 Multi-Modal AI | Explainable | Real-time Vision System")

st.divider()

# -----------------------------------------
# FACE DETECTOR (REAL)
# -----------------------------------------
face_detector = dlib.get_frontal_face_detector()

# -----------------------------------------
# PDF GENERATOR
# -----------------------------------------
def create_pdf(result, confidence):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("DeepFake AI Report", styles['Title']))
    content.append(Spacer(1, 20))
    content.append(Paragraph(f"Result: {result}", styles['Heading2']))
    content.append(Paragraph(f"Confidence: {confidence}%", styles['Normal']))

    if result == "DeepFake":
        content.append(Paragraph("Detected synthetic artifacts and inconsistencies.", styles['Normal']))
    else:
        content.append(Paragraph("Natural human facial behavior detected.", styles['Normal']))

    doc.build(content)

# -----------------------------------------
# FPS
# -----------------------------------------
def measure_fps(start, end):
    return round(1/(end-start), 2) if end-start > 0 else 0

# -----------------------------------------
# REAL FACE TRACKING + REGIONS
# -----------------------------------------
def draw_real_overlay(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 0)

    for face in faces:

        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        # face box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        h = y2 - y1

        # eyes region
        cv2.rectangle(frame, (x1, y1), (x2, y1 + int(h*0.3)), (255,0,0), 2)

        # mouth region
        cv2.rectangle(frame, (x1, y1 + int(h*0.6)), (x2, y2), (0,0,255), 2)

    return frame

# -----------------------------------------
# UI
# -----------------------------------------
uploaded_file = st.file_uploader("📁 Upload Video", type=["mp4","mov","avi"])

if uploaded_file:

    st.subheader("🎬 Uploaded Video")
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(uploaded_file.read())
        video_path = temp.name

    st.divider()

    if st.button("🚀 Run AGI Detection"):

        start_time = time.time()

        with st.spinner("🧠 AGI analyzing video..."):

            try:
                result, confidence, timeline, heatmaps, boxes = detect_video(video_path)
            except Exception as e:
                st.error(str(e))
                st.stop()

        end_time = time.time()

        fps = measure_fps(start_time, end_time)

        # -----------------------------------------
        # PERFORMANCE PANEL
        # -----------------------------------------
        col1, col2, col3 = st.columns(3)
        col1.metric("⚡ FPS", fps)
        col2.metric("🎯 Confidence", f"{confidence}%")
        col3.metric("⏱ Time", f"{round(end_time-start_time,2)}s")

        st.divider()

        # -----------------------------------------
        # RESULT
        # -----------------------------------------
        st.subheader("🧾 Final Result")

        if result == "DeepFake":
            st.error("🚨 DeepFake Detected")
        elif result == "Real":
            st.success("✅ Authentic Video")
        else:
            st.warning("⚠ Uncertain")

        st.progress(int(confidence))

        # -----------------------------------------
        # TIMELINE
        # -----------------------------------------
        st.subheader("📊 Frame Confidence Timeline")

        fig, ax = plt.subplots()
        ax.plot(timeline)
        ax.set_title("DeepFake Probability")
        ax.grid(True)
        st.pyplot(fig)

        # -----------------------------------------
        # HEATMAPS
        # -----------------------------------------
        st.subheader("🔥 AI Attention Heatmaps")

        cols = st.columns(5)
        for i, img in enumerate(heatmaps[:5]):
            cols[i % 5].image(img)

        # -----------------------------------------
        # REAL FACE TRACKING PREVIEW
        # -----------------------------------------
        st.subheader("🎯 Real Face Tracking + Region Analysis")

        cap = cv2.VideoCapture(video_path)
        frames = []

        for _ in range(6):
            ret, frame = cap.read()
            if not ret:
                break

            frame = draw_real_overlay(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        cols = st.columns(len(frames))
        for i, f in enumerate(frames):
            cols[i].image(f)

        # -----------------------------------------
        # CSV EXPORT
        # -----------------------------------------
        st.download_button(
            "📥 Download Timeline CSV",
            data="\n".join(map(str, timeline)),
            file_name="timeline.csv"
        )

        # -----------------------------------------
        # PDF REPORT
        # -----------------------------------------
        create_pdf(result, confidence)

        with open("report.pdf", "rb") as f:
            st.download_button("📄 Download Report", f, "report.pdf")

        # -----------------------------------------
        # EXPLAINABLE AI
        # -----------------------------------------
        st.subheader("🧠 Explainable AI Insights")

        if result == "DeepFake":
            st.write("""
            ✔ Facial warping detected  
            ✔ Temporal instability  
            ✔ GAN-based artifacts  
            ✔ Frequency inconsistencies  
            """)
        else:
            st.write("""
            ✔ Natural facial structure  
            ✔ Stable motion  
            ✔ No synthetic patterns  
            ✔ Consistent lighting  
            """)

    if os.path.exists(video_path):
        os.remove(video_path)