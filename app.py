import json
from pathlib import Path
import base64
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import torch
import torchvision.transforms as T

# -------------------- Paths --------------------
ART = Path("Data_Directory/artifacts")
TS_PATH   = ART / "model.torchscript.pt"
CFG_PATH  = ART / "config_inference.json"
LAB_PATH  = ART / "labels.json"
TEMP_PATH = ART / "temperature.json"

BANNER   = "header_banner.jpg"
APP_LOGO = "logo 2.jpg"

# -------------------- Page + Styling --------------------
st.set_page_config(
    page_title="AI-Powered Apple Leaf Specialist",
    page_icon=APP_LOGO if Path(APP_LOGO).exists() else "🍎",
    layout="wide",
)

st.markdown("""
<style>
/* Reset Streamlit Column Padding */
div[data-testid="column"] > div:first-child {
  margin-top: 0 !important;
  padding-top: 0 !important;
}

/* Header Text Styles */
.block-head {
  display: flex;
  flex-direction: column;
  margin-bottom: 8px;
}
.title {
  font-size: 1rem;
  font-weight: 600;
  color: #1f2937;
  margin: 0;
}
.sub {
  font-size: 0.875rem;
  font-weight: 400;
  color: #6b7280;
  margin: 0;
}

/* Alignment Spacer */
.right-spacer {
  height: 45px;
  width: 100%;
}

/* Uploader Styling */
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
  border: 1.5px solid #E6E9EF;
  background: #F6F8FB;
  border-radius: 12px;
  padding: 12px;
}

/* Camera Card Styling */
.camera-card-container {
  position: relative;
  display: flex;
  align-items: flex-start;
  border: 1.5px solid #E6E9EF;
  background: #F6F8FB;
  border-radius: 12px;
  padding: 16px 12px;
  min-height: 64px;
}

.camera-hint {
  font-size: 0.875rem;
  color: #6b7280;
  margin: 0;
  padding-right: 150px;
}

/* Target the Streamlit Native Button to look like your design */
.stButton > button {
  position: absolute !important;
  right: 16px !important;
  top: 8px !important;
  background: #ffffff !important;
  color: #111827 !important;
  font-size: 0.875rem !important;
  font-weight: 400 !important;
  border: 1px solid #D1D5DB !important;
  border-radius: 8px !important;
  padding: 0.45rem 0.8rem !important;
  height: auto !important;
  width: auto !important;
  line-height: 1.2 !important;
  transition: all 0.2s;
}

.stButton > button:hover {
  border-color: #9CA3AF !important;
  background: #f9fafb !important;
}

@media (max-width: 680px) {
  .camera-card-container { flex-direction: column; }
  .camera-hint { padding-right: 0; margin-bottom: 10px; }
  .stButton > button { position: static !important; width: 100% !important; }
  .right-spacer { height: 16px; }
}
</style>
""", unsafe_allow_html=True)

if Path(BANNER).exists():
    st.image(BANNER, use_container_width=True)

# -------------------- Model Logic --------------------
def _load_json(p: Path, default):
    try:
        with open(p, "r") as f: return json.load(f)
    except: return default

@st.cache_resource(show_spinner=False)
def load_model_only_ts():
    model = torch.jit.load(str(TS_PATH), map_location="cpu").eval()
    cfg = _load_json(CFG_PATH, {"img_size": 256, "mean":[0.485,0.456,0.406], "std":[0.229,0.224,0.225]})
    labels = _load_json(LAB_PATH, ["healthy","scab","rust","black_rot"])
    temp = float(_load_json(TEMP_PATH, {"temperature": 1.0}).get("temperature", 1.0))
    img_size = int(cfg["img_size"])
    transform = T.Compose([
        T.Resize(img_size + 32),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])
    return model, labels, img_size, temp, transform

model, labels, IMG_SIZE, TEMPERATURE, transform = load_model_only_ts()

def load_pil(obj) -> Image.Image:
    im = obj if isinstance(obj, Image.Image) else Image.open(obj).convert("RGB")
    return ImageOps.exif_transpose(im)

def predict_probs(pil_img: Image.Image) -> np.ndarray:
    x = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        if TEMPERATURE > 0: logits = logits / TEMPERATURE
        return torch.softmax(logits, dim=1).cpu().numpy()[0]

def compute_brightness(pil_img: Image.Image) -> float:
    arr = np.asarray(pil_img.resize((256, 256))).astype(np.float32) / 255.0
    return float((0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2]).mean())

# -------------------- State Logic --------------------
if "show_camera" not in st.session_state: st.session_state.show_camera = False
if "source" not in st.session_state: st.session_state.source = None
if "captured" not in st.session_state: st.session_state.captured = None
if "upload" not in st.session_state: st.session_state.upload = None

def on_upload_change():
    st.session_state.upload = st.session_state.get("uploader")
    st.session_state.source = "upload"
    st.session_state.show_camera = False

# -------------------- Sidebar --------------------
with st.sidebar:
    st.subheader("Settings")
    THRESHOLD = st.slider("Decision threshold", 0.0, 0.99, 0.85, 0.01)
    dark_thr = st.slider("Dark threshold", 0.05, 0.50, 0.25, 0.01)
    bright_thr = st.slider("Bright threshold", 0.50, 0.99, 0.90, 0.01)
    st.write("---")
    st.write("### Contacts")
    st.markdown("[GitHub](https://github.com/akthammomani)")

# -------------------- Main UI --------------------
st.subheader("Add a leaf photo")
left, right = st.columns([1,1], gap="large")

with left:
    st.markdown('<div class="block-head"><div class="title">Upload Photo</div><div class="sub">Drop a JPG/PNG here, or browse</div></div>', unsafe_allow_html=True)
    st.file_uploader(label="", type=["jpg", "jpeg", "png"], key="uploader", on_change=on_upload_change, label_visibility="collapsed")

with right:
    st.markdown('<div class="block-head"><div class="title">Record Photo</div><div class="sub">Use your device camera</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="right-spacer"></div>', unsafe_allow_html=True)

    if not st.session_state.show_camera:
        st.markdown('<div class="camera-card-container"><p class="camera-hint">Tap “Open camera” to take a photo.</p></div>', unsafe_allow_html=True)
        # This button is positioned inside the container above via CSS
        if st.button("Open camera", key="open_cam_btn"):
            st.session_state.show_camera = True
            st.session_state.source = "camera"
            st.rerun()
    else:
        cap = st.camera_input("Take a photo", label_visibility="collapsed")
        if cap:
            st.session_state.captured = cap
            st.session_state.source = "camera"
            st.session_state.show_camera = False
            st.rerun()
        if st.button("Cancel"):
            st.session_state.show_camera = False
            st.rerun()

# -------------------- Results --------------------
file = st.session_state.captured if st.session_state.source == "camera" else st.session_state.upload

if file:
    pil = load_pil(file)
    b = compute_brightness(pil)
    
    if b < dark_thr or b > bright_thr:
        st.warning(f"Lighting issue detected (Brightness: {b:.2f}). Please retake.")
    else:
        probs = predict_probs(pil)
        idx = np.argmax(probs)
        conf = probs[idx]
        label = labels[idx] if conf >= THRESHOLD else "Unknown"

        c1, c2 = st.columns(2)
        with c1:
            st.image(pil, caption="Your Image", use_container_width=True)
        with c2:
            st.markdown(f"### Result: **{label.title()}**")
            st.markdown(f"**Confidence:** {conf*100:.1f}%")
            for i, l in enumerate(labels):
                st.write(f"{l.title()}: {probs[i]*100:.1f}%")
                st.progress(float(probs[i]))

else:
    st.info("Upload a photo or open the camera to begin.")
