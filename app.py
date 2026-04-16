import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Helmet Detection",
    page_icon="⛑️",
    layout="wide"
)



@st.cache_resource
def load_model():
    try:
        return YOLO("Helmet_model.pt")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

# ── Helper: run detection on a frame ─────────────────────────
def detect(frame_bgr, conf_threshold):
    results = model(frame_bgr, conf=conf_threshold, verbose=False)[0]
    annotated = results.plot()

    helmet_count    = 0
    no_helmet_count = 0

    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        if cls_name.lower() in ["helmet", "with_helmet", "with helmet"]:
            helmet_count += 1
        else:
            no_helmet_count += 1

    return annotated, helmet_count, no_helmet_count

# ── Helper: draw metric cards ─────────────────────────────────
def show_metrics(helmet, no_helmet):
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ With Helmet",    helmet)
    c2.metric("🚨 Without Helmet", no_helmet)
    c3.metric("👷 Total People",   helmet + no_helmet)

    if no_helmet > 0:
        st.error(f"⚠️ Alert: {no_helmet} person(s) detected WITHOUT a helmet!")
    else:
        st.success("✅ All persons are wearing helmets.")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hard-hat.png", width=80)
    st.title("⛑️ Helmet Detection")
    st.markdown("---")

    mode = st.radio(
        "Detection Mode",
        ["📷 Image Upload", "🎥 Video File", "📸 Webcam Snapshot"],
        index=0
    )

    st.markdown("---")
    conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.45, 0.05,
                     help="Lower = detect more (may include false positives)")
    st.markdown("---")
    st.caption("Model: YOLOv8 · Custom trained")
    st.caption("Classes: Helmet / No Helmet")

# ═══════════════════════════════════════════════════════════════
# MODE 1 — Image Upload
# ═══════════════════════════════════════════════════════════════
if mode == "📷 Image Upload":
    st.header("📷 Image Detection")
    st.write("Upload one or more images to detect helmets.")

    uploaded_files = st.file_uploader(
        "Choose image(s)", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded in uploaded_files:
            image = Image.open(uploaded).convert("RGB")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            with st.spinner(f"Detecting in {uploaded.name}..."):
                annotated, helmets, no_helmets = detect(frame, conf)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_column_width=True)
            with col2:
                st.image(annotated_rgb, caption="Detected", use_column_width=True)

            show_metrics(helmets, no_helmets)
            st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# MODE 2 — Video File
# ═══════════════════════════════════════════════════════════════
elif mode == "🎥 Video File":
    st.header("🎥 Video Detection")
    st.write("Upload a video file. Frames are sampled and analyzed.")

    video_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])

    if video_file:
        # Save to temp file (OpenCV needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        duration     = total_frames / fps if fps > 0 else 0

        st.info(f"📹 {total_frames} frames · {fps:.1f} FPS · {duration:.1f}s duration")

        # Sample every Nth frame (process ~60 frames max for speed)
        sample_every = max(1, total_frames // 60)

        process_btn = st.button("🚀 Run Detection", type="primary")

        if process_btn:
            progress    = st.progress(0, text="Processing video...")
            frame_disp  = st.empty()
            metrics_disp = st.empty()

            frame_idx       = 0
            processed       = 0
            total_helmet    = 0
            total_no_helmet = 0
            annotated_frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_every == 0:
                    annotated, h, nh = detect(frame, conf)
                    total_helmet    += h
                    total_no_helmet += nh
                    annotated_frames.append(annotated)
                    processed += 1

                    # Live preview
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_disp.image(rgb, caption=f"Frame {frame_idx}", use_column_width=True)

                    with metrics_disp.container():
                        show_metrics(total_helmet, total_no_helmet)

                    progress.progress(
                        min(frame_idx / total_frames, 1.0),
                        text=f"Processing frame {frame_idx}/{total_frames}"
                    )

                frame_idx += 1

            progress.progress(1.0, text="✅ Done!")
            cap.release()
            os.unlink(tmp_path)

            st.success(f"Processed {processed} sampled frames out of {total_frames} total.")

            # Save annotated video
            if annotated_frames:
                h_px, w_px = annotated_frames[0].shape[:2]
                out_path = tempfile.mktemp(suffix=".mp4")
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps / sample_every, (w_px, h_px))
                for f in annotated_frames:
                    out.write(f)
                out.release()

                with open(out_path, "rb") as vf:
                    st.download_button(
                        "⬇️ Download Annotated Video",
                        vf, file_name="helmet_detection_output.mp4",
                        mime="video/mp4"
                    )
                os.unlink(out_path)

# ═══════════════════════════════════════════════════════════════
# MODE 3 — Webcam Snapshot
# ═══════════════════════════════════════════════════════════════
elif mode == "📸 Webcam Snapshot":
    st.header("📸 Webcam Snapshot")
    st.write("Take a photo using your device camera and detect helmets instantly.")

    img_data = st.camera_input("📷 Click to take a photo")

    if img_data:
        image = Image.open(img_data).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting..."):
            annotated, helmets, no_helmets = detect(frame, conf)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Your photo", use_column_width=True)
        with col2:
            st.image(annotated_rgb, caption="Detected", use_column_width=True)

        show_metrics(helmets, no_helmets)

        # Download button
        result_pil = Image.fromarray(annotated_rgb)
        import io
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Result",
            buf.getvalue(),
            file_name="helmet_detection.png",
            mime="image/png"
        )