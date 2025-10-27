# app.py
import json
from pathlib import Path
import base64

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import torch
import torchvision.transforms as T
import streamlit.components.v1 as components
from streamlit_javascript import st_javascript

# -------------------- Paths --------------------
ART = Path("Data_Directory/artifacts")
TS_PATH   = ART / "model.torchscript.pt"
CFG_PATH  = ART / "config_inference.json"
LAB_PATH  = ART / "labels.json"
TEMP_PATH = ART / "temperature.json"

BANNER   = "header_banner.jpg"
APP_LOGO = "logo 2.jpg"

# -------------------- Page + light styling --------------------
st.set_page_config(
    page_title="AI-Powered Apple Leaf Specialist",
    page_icon=APP_LOGO if Path(APP_LOGO).exists() else "🍎",
    layout="wide",
)

# Card‑like look for uploader and camera
st.markdown("""
<style>
.section { margin-bottom:.05rem; }
.section .title { font-size:1.4rem; font-weight:700; margin:0 0 .15rem 0; color:#2c313f; }
.section .sub   { color:#6b7280; margin:0 0 .25rem 0; }

/* Uploader card */
div[data-testid="stFileUploader"]{ margin-top:.25rem; }
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]{
  border:1.5px solid #E6E9EF; background:#F6F8FB; border-radius:12px; padding:12px;
}

/* Camera card */
.camera-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px;
  border: 1.5px solid #E6E9EF;
  background: #F6F8FB;
  border-radius: 12px;
  min-height: 64px;
  box-sizing: border-box;
}
/* keep room for the button */
.camera-hint {
  font-size: 14px;
  color: #6b7280;
  line-height: 1.3;

/* Button inside camera card */
.custom-cam-btn {
  background: #ffffff;
  color: #111827;
  border: 1px solid #D1D5DB;
  border-radius: 8px;
  padding: .4rem .8rem;
  cursor: pointer;
  font-size: 14px;
}
.custom-cam-btn:hover {
  border-color:#9CA3AF;
}

/* responsive adjustments */
@media (max-width:680px){
  .camera-hint{ padding-right:0; }
  .custom-cam-btn { position: static !important; margin-top:.5rem !important; }
}
</style>
""", unsafe_allow_html=True)

if Path(BANNER).exists():
    st.image(BANNER, use_container_width=True)

# -------------------- Helpers --------------------
def _load_json(p: Path, default):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return default

@st.cache_resource(show_spinner=False)
def load_model_only_ts():
    if not TS_PATH.exists():
        raise FileNotFoundError("artifacts/model.torchscript.pt not found")
    model = torch.jit.load(str(TS_PATH), map_location="cpu").eval()

    cfg  = _load_json(CFG_PATH, {"img_size": 256, "mean":[0.485,0.456,0.406], "std":[0.229,0.224,0.225]})
    labels = _load_json(LAB_PATH, ["healthy","scab","rust","black_rot"])
    temperature = float(_load_json(TEMP_PATH, {"temperature": 1.0}).get("temperature", 1.0))

    img_size = int(cfg["img_size"])
    mean, std = cfg["mean"], cfg["std"]

    pad = 32 if img_size >= 224 else int(img_size * 0.125)
    transform = T.Compose([
        T.Resize(img_size + pad),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return model, labels, img_size, temperature, transform

model, labels, IMG_SIZE, TEMPERATURE, transform = load_model_only_ts()

def load_pil(obj) -> Image.Image:
    im = obj if isinstance(obj, Image.Image) else Image.open(obj).convert("RGB")
    return ImageOps.exif_transpose(im)

def predict_probs(pil_img: Image.Image) -> np.ndarray:
    x = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        if TEMPERATURE and TEMPERATURE > 0:
            logits = logits / TEMPERATURE
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

def compute_brightness(pil_img: Image.Image) -> float:
    arr = np.asarray(pil_img.resize((256, 256))).astype(np.float32) / 255.0
    y = 0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2]
    return float(y.mean())

def green_coverage_soft(pil_img: Image.Image) -> float:
    hsv = np.array(pil_img.convert("HSV"))
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    mask_hsv = (H >= 11) & (H <= 85) & (S >= 20) & (V >= 20)
    rgb = np.asarray(pil_img.convert("RGB"))
    R, G, B = rgb[...,0].astype(np.int16), rgb[...,1].astype(np.int16), rgb[...,2].astype(np.int16)
    mask_gdom = (G >= R + 8) & (G >= B + 8)
    return float((mask_hsv | mask_gdom).mean())

def sobel_texture_np(pil_img: Image.Image) -> float:
    gray = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    pad = np.pad(gray, 1, mode="reflect")
    gx = (pad[:-2, :-2]*kx[0,0] + pad[:-2,1:-1]*kx[0,1] + pad[:-2,2:]*kx[0,2] +
          pad[1:-1, :-2]*kx[1,0] + pad[1:-1,1:-1]*kx[1,1] + pad[1:-1,2:]*kx[1,2] +
          pad[2:, :-2]*kx[2,0] + pad[2:,1:-1]*kx[2,1] + pad[2:,2:]*kx[2,2])
    gy = (pad[:-2, :-2]*ky[0,0] + pad[:-2,1:-1]*ky[0,1] + pad[:-2,2:]*ky[0,2] +
          pad[1:-1, :-2]*ky[1,0] + pad[1:-1,1:-1]*ky[1,1] + pad[1:-1,2:]*ky[1,2] +
          pad[2:, :-2]*ky[2,0] + pad[2:,1:-1]*ky[2,1] + pad[2:,2:]*ky[2,2])
    mag = np.hypot(gx, gy)
    return float(mag.var() * 1000.0)

def is_leaf_like(pil_img: Image.Image, cov_min=0.04, cov_max=0.98, tex_min=25.0):
    cov = green_coverage_soft(pil_img)
    tex = sobel_texture_np(pil_img)
    ok = (cov_min <= cov <= cov_max) and (tex >= tex_min)
    return ok, cov, tex

def decide(probs: np.ndarray, labels, threshold: float):
    k = int(np.argmax(probs)); p = float(probs[k])
    return (labels[k], p, k) if p >= threshold else ("unknown", p, k)

# ---- UI helpers ----
def _pretty(lab: str) -> str:
    return lab.replace("_", " ").title()

def vspace(rows: int = 2, row_px: int = 12):
    st.markdown(f"<div style='height:{rows*row_px}px'></div>", unsafe_allow_html=True)

def preview_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    return ImageOps.contain(img, (max_w, max_h))  # preserve aspect ratio

def render_prob_bars_native(prob_map: dict):
    st.markdown("**Apple Disease Probability**")
    order = ["black_rot", "healthy", "scab", "rust"]
    for lab in order:
        p = float(prob_map.get(lab, 0.0))
        c1, c2, c3 = st.columns([1.6, 6, 1.2])
        with c1: st.write(_pretty(lab))
        with c2:
            try: st.progress(p)
            except Exception: st.progress(int(p*100))
        with c3: st.write(f"{p*100:.1f}%")

# -------------------- Posters --------------------
CARE_POSTERS = {
    "black_rot": "black_rot_care_v1.jpg",
    "healthy":   "healthy_care_v1.jpg",
    "scab":      "scab_care_v1.jpg",
    "rust":      "rust_care_v1.jpg",
}

# -------------------- Session state --------------------
if "show_camera" not in st.session_state:   st.session_state.show_camera = False
if "source" not in st.session_state:        st.session_state.source = None
if "captured" not in st.session_state:      st.session_state.captured = None
if "upload" not in st.session_state:        st.session_state.upload = None
if "keep_camera_on" not in st.session_state: st.session_state.keep_camera_on = False

def open_camera():
    st.session_state.show_camera = True
    st.session_state.source = "camera"
    st.session_state.upload = None

def close_camera():
    st.session_state.show_camera = False

def on_upload_change():
    st.session_state.upload = st.session_state.get("uploader")
    st.session_state.source = "upload"
    st.session_state.show_camera = False

# -------------------- Sidebar --------------------
def sidebar_logo(title:str, path:str):
    if Path(path).exists():
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        ext = Path(path).suffix.lstrip(".").lower() or "png"
        img_html = f'<img src="data:image/{ext};base64,{b64}" alt="logo" />'
    else:
        img_html = '<div style="font-size:48px">🍎</div>'
    st.markdown(
        f'<div style="display:flex;flex-direction:column;align-items:center;gap:.4rem;margin-bottom:1rem;text-align:center">'
        f'{img_html}<div style="font-weight:700;font-size:1.0rem;line-height:1.2;color:#2c313f">{title}</div></div>',
        unsafe_allow_html=True
    )

with st.sidebar:
    sidebar_logo("AI‑Powered Apple Leaf Specialist", APP_LOGO)
    st.subheader("Settings")
    THRESHOLD = st.slider("Decision threshold (τ)", 0.0, 0.99, 0.85, 0.01)
    dark_thr   = st.slider("Too dark threshold", 0.05, 0.50, 0.25, 0.01)
    bright_thr = st.slider("Too bright threshold", 0.50, 0.99, 0.90, 0.01)
    cov_min = st.slider("Min green coverage (camera gate)", 0.00, 0.50, 0.04, 0.01)
    tex_min = st.slider("Min texture score (camera gate)", 0.0, 300.0, 25.0, 1.0)
    PREVIEW_MAX_W = st.slider("Image preview max width (px)", 280, 1000, 520, 10)
    PREVIEW_MAX_H = st.slider("Image preview max height (px)", 200, 900, 520, 10)
    st.checkbox("Keep camera open after capture", value=st.session_state.keep_camera_on, key="keep_camera_on")
    st.caption(f"Engine: TorchScript · Temperature: {TEMPERATURE:.2f} · Image size: {IMG_SIZE}")
    st.markdown("**Classes**: " + " · ".join(labels))

# -------------------- Inputs --------------------
st.subheader("Add a leaf photo")
left, right = st.columns([1,1], gap="large")

with left:
    st.markdown('<div class="section"><div class="title">Upload Photo</div>'
                '<div class="sub">Drop a JPG/PNG here, or browse</div>', unsafe_allow_html=True)
    st.file_uploader(label="", type=["jpg","jpeg","png"], key="uploader", on_change=on_upload_change)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section"><div class="title">Record Photo</div>'
                '<div class="sub">Use your device camera</div>', unsafe_allow_html=True)

    if not st.session_state.show_camera:
        st.markdown("""
            <div class="camera-card">
              <div class="camera-hint">Tap “Open camera” to take a photo.</div>
              <button id="open_cam_real" class="custom-cam-btn">Open camera</button>
            </div>
        """, unsafe_allow_html=True)

        js_result = st_javascript("""
        (function(){
          const btn = window.parent.document.getElementById("open_cam_real");
          if(btn){
            btn.onclick = function(){
              window.parent.postMessage({type:"streamlit:setComponentValue", value:true}, "*");
            };
          }
        })();
        """, key="open_cam_js")

        if js_result:
            st.session_state.show_camera = True
            st.session_state.source = "camera"
            st.session_state.upload = None

        cap = None
    else:
        cap = st.camera_input("", key="camera_input")
        if cap is not None:
            st.session_state.captured = cap
            st.session_state.source = "camera"
            if not st.session_state.keep_camera_on:
                close_camera()
        st.button("Close camera", on_click=close_camera, key="close_cam_btn")

    st.markdown("</div>", unsafe_allow_html=True)

# Active source
file = st.session_state.captured if st.session_state.source == "camera" else (
    st.session_state.upload if st.session_state.source == "upload" else None
)

# -------------------- Main inference path --------------------
if file:
    pil = load_pil(file)

    # Quality gates
    b = compute_brightness(pil)
    if b < dark_thr:
        st.warning(f"Image appears too dark (brightness {b:.2f}). Retake under brighter, even lighting.")
        st.stop()
    if b > bright_thr:
        st.warning(f"Image appears too bright/washed‑out (brightness {b:.2f}). Retake avoiding direct glare.")
        st.stop()
    if st.session_state.source == "camera":
        bypass_gate = st.checkbox("Bypass leaf check for this camera image", value=False)
        ok_leaf, cov, tex = is_leaf_like(pil, cov_min=cov_min, cov_max=0.98, tex_min=tex_min)
        if not (ok_leaf or bypass_gate):
            st.warning(
                f"This photo might not be a single leaf (green_coverage≈{cov:.2f}, texture≈{tex:.0f}). "
                "Retake: fill the frame with one leaf in even lighting, sharp focus — or tick the bypass to proceed."
            )
            st.stop()

    # Inference
    probs = predict_probs(pil)
    pred_label, pred_conf, _ = decide(probs, labels, THRESHOLD)
    prob_map = {lab: float(probs[i]) for i, lab in enumerate(labels)}

    # -------- Row 1: image + prediction --------
    r1_left, r1_right = st.columns([1,1], gap="large")
    with r1_left:
        st.markdown("### Your Image:")
        st.image(ImageOps.contain(pil, (PREVIEW_MAX_W, PREVIEW_MAX_H)), use_container_width=False)

    with r1_right:
        st.markdown("### Predicted Apple Disease Label is:")
        st.markdown(f"**{_pretty(pred_label)}** with **{pred_conf*100:.0f}%** Confidence")
        render_prob_bars_native(prob_map)
        st.caption("Model: Calibrated ResNet‑18 (TorchScript). Low‑confidence predictions route to ‘unknown’.")
    vspace(3)

    # -------- Row 2: Title --------
    st.markdown(f"### Apple – {_pretty(pred_label)} Care Recommendations:")
    vspace(2)

    # -------- Row 3: Poster image --------
    poster_path = CARE_POSTERS.get(pred_label, CARE_POSTERS["healthy"])
    if not Path(poster_path).exists():
        st.info("Care poster not found. Please add the JPGs next to app.py.")
    else:
        st.image(poster_path, use_container_width=True)

else:
    st.info("Upload a photo or open the camera to begin.")
