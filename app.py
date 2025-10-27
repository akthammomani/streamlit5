import json
from pathlib import Path
import base64

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import torch
import torchvision.transforms as T
from streamlit_javascript import st_javascript

# -------------------- Paths --------------------
ART = Path("Data_Directory/artifacts")
TS_PATH   = ART / "model.torchscript.pt"
CFG_PATH  = ART / "config_inference.json"
LAB_PATH  = ART / "labels.json"
TEMP_PATH = ART / "temperature.json"

BANNER   = "header_banner.jpg"
APP_LOGO = "logo 2.jpg"

# -------------------- Page --------------------
st.set_page_config(
    page_title="AI-Powered Apple Leaf Specialist",
    page_icon=APP_LOGO if Path(APP_LOGO).exists() else "🍎",
    layout="wide",
)

# -------------------- CSS --------------------
st.markdown("""
<style>

/* ===========================
   GLOBAL COLUMN CLEANUP
   =========================== */

/* Streamlit gives each st.columns cell its own inner div with padding.
   Kill that so both columns start at the same vertical origin. */
div[data-testid="column"] > div:first-child {
  margin-top: 0 !important;
  padding-top: 0 !important;
}

/* We'll wrap each column in .leaf-left / .leaf-right in the Python code */
.leaf-left { /* left column wrapper */ }
.leaf-right { /* right column wrapper */ }

/* Inner container for each column’s content. */
.leaf-block {
  display: block;
  margin: 0;
  padding: 0;
}


/* ===========================
   HEADER (TITLE + SUBTITLE)
   =========================== */

/* Wrapper around "Upload Photo" / "Record Photo" + subtitle text */
.block-head {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  margin: 0;
  padding: 0;
  line-height: 1.4;
}

/* Header title ("Upload Photo", "Record Photo") */
.block-head .title {
  font-size: 1rem;
  font-weight: 600;
  color: #1f2937;      /* slate-800-ish */
  margin: 0;
  line-height: 1.4;
}

/* Header subtitle ("Drop a JPG/PNG...", "Use your device camera") */
.block-head .sub {
  font-size: 0.875rem;
  font-weight: 400;
  color: #6b7280;      /* gray-500/600 */
  margin: 0;
  line-height: 1.4;
}

/* A consistent little gap below the header before its card (both columns) */
.block-head {
  margin-bottom: 8px;
}


/* ===========================
   ALIGNMENT SPACER (RIGHT ONLY)
   =========================== */

/* This spacer will ONLY exist in the right column,
   and it will push the camera card down so that the top
   of the camera card lines up with the top of the upload card.
   Adjust height until visually perfect. */
.right-spacer {
  height: 45px;   /* try 45, bump to 50 if camera card still too high */
  width: 100%;
}


/* ===========================
   CARD ROW WRAPPER
   =========================== */

.block-card {
  margin: 0 !important;
  padding: 0 !important;
}

/* On the left side, Streamlit gives the uploader widget
   a default top margin. Kill it so the upload card hugs
   its header closely. */
.leaf-left div[data-testid="stFileUploader"] {
  margin-top: 0 !important;
}

/* Normalize spacing inside uploader layers */
.upload-wrapper,
.upload-wrapper > div[data-testid="stFileUploader"],
.upload-wrapper section[data-testid="stFileUploaderDropzone"] {
  margin: 0 !important;
  padding: 0 !important;
}


/* ===========================
   UPLOADER CARD (LEFT COLUMN)
   =========================== */

div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
  border: 1.5px solid #E6E9EF;
  background: #F6F8FB;
  border-radius: 12px;
  padding: 12px;
}


/* ===========================
   CAMERA CARD (RIGHT COLUMN)
   =========================== */

.camera-card {
  position: relative;
  display: flex;
  align-items: flex-start;

  border: 1.5px solid #E6E9EF;
  background: #F6F8FB;
  border-radius: 12px;

  padding: 16px 12px;
  min-height: 78px;
  color: #6b7280;
  box-sizing: border-box;

  margin: 0 !important; /* prevent Streamlit surprises */
}

/* Text inside camera card */
.camera-hint {
  font-size: 0.875rem;
  line-height: 1.4;
  color: #6b7280;
  margin: 0;

  /* Make room for the "Open camera" button on desktop */
  padding-right: 150px;
}

/* "Open camera" button */
.custom-cam-btn {
  position: absolute;
  right: 16px;
  top: 8px;

  background: #ffffff;
  color: #111827;
  font-size: 0.875rem;
  line-height: 1.2;

  border: 1px solid #D1D5DB;
  border-radius: 8px;
  padding: .45rem .8rem;

  cursor: pointer;
  white-space: nowrap;
}

.custom-cam-btn:hover {
  border-color: #9CA3AF;
}


/* ===========================
   RESPONSIVE BEHAVIOR
   =========================== */

@media (max-width: 680px) {

  /* Stack vertical inside the camera card on small screens */
  .camera-card {
    flex-direction: column;
  }

  .camera-hint {
    padding-right: 0;
  }

  .custom-cam-btn {
    position: static !important;
    margin-top: .5rem !important;
  }

  /* On narrow screens, we don't want a huge offset,
     so shrink the spacer. */
  .right-spacer {
    height: 16px;
  }
}


</style>
""", unsafe_allow_html=True)


# -------------------- Banner --------------------
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
    R, G, B = rgb[...,0].astype(np.int16), rgb[...,1].astype*np.int16(), rgb[...,2].astype*np.int16()
    # ^^ I'll fix that small typo below in final output to keep it valid Python

def green_coverage_soft(pil_img: Image.Image) -> float:
    hsv = np.array(pil_img.convert("HSV"))
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    mask_hsv = (H >= 11) & (H <= 85) & (S >= 20) & (V >= 20)
    rgb = np.asarray(pil_img.convert("RGB"))
    R = rgb[...,0].astype(np.int16)
    G = rgb[...,1].astype(np.int16)
    B = rgb[...,2].astype(np.int16)
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

def _pretty(lab: str) -> str:
    return lab.replace("_", " ").title()

def vspace(rows: int = 2, row_px: int = 12):
    st.markdown(f"<div style='height:{rows*row_px}px'></div>", unsafe_allow_html=True)

def render_prob_bars_native(prob_map: dict):
    st.markdown("**Apple Disease Probability**")
    order = ["black_rot", "healthy", "scab", "rust"]
    for lab in order:
        p = float(prob_map.get(lab, 0.0))
        c1, c2, c3 = st.columns([1.6, 6, 1.2])
        with c1: st.write(_pretty(lab))
        with c2:
            try:
                st.progress(p)
            except Exception:
                st.progress(int(p*100))
        with c3: st.write(f"{p*100:.1f}%")

# Posters
CARE_POSTERS = {
    "black_rot": "black_rot_care_v1.jpg",
    "healthy":   "healthy_care_v1.jpg",
    "scab":      "scab_care_v1.jpg",
    "rust":      "rust_care_v1.jpg",
}

# Session state
if "show_camera" not in st.session_state:   st.session_state.show_camera = False
if "source" not in st.session_state:        st.session_state.source = None
if "captured" not in st.session_state:      st.session_state.captured = None
if "upload" not in st.session_state:        st.session_state.upload = None
if "keep_camera_on" not in st.session_state: st.session_state.keep_camera_on = False

def close_camera():
    st.session_state.show_camera = False

def on_upload_change():
    st.session_state.upload = st.session_state.get("uploader")
    st.session_state.source = "upload"
    st.session_state.show_camera = False

# Sidebar
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
    sidebar_logo("AI-Powered Apple Leaf Specialist", APP_LOGO)
    st.subheader("Settings")

    THRESHOLD = st.slider("Decision threshold (τ)", 0.0, 0.99, 0.85, 0.01)
    dark_thr   = st.slider("Too dark threshold", 0.05, 0.50, 0.25, 0.01)
    bright_thr = st.slider("Too bright threshold", 0.50, 0.99, 0.90, 0.01)
    cov_min    = st.slider("Min green coverage (camera gate)", 0.00, 0.50, 0.04, 0.01)
    tex_min    = st.slider("Min texture score (camera gate)", 0.0, 300.0, 25.0, 1.0)
    PREVIEW_MAX_W = st.slider("Image preview max width (px)", 280, 1000, 520, 10)
    PREVIEW_MAX_H = st.slider("Image preview max height (px)", 200, 900, 520, 10)

    st.checkbox("Keep camera open after capture", value=st.session_state.keep_camera_on, key="keep_camera_on")

    st.caption(f"Engine: TorchScript · Temperature: {TEMPERATURE:.2f} · Image size: {IMG_SIZE}")
    st.markdown("**Classes**: " + " · ".join(labels))

# Inputs row
st.subheader("Add a leaf photo")
left, right = st.columns([1,1], gap="large")

with left:
    # wrap LEFT column with .leaf-left so we can target its uploader
    st.markdown('<div class="leaf-left"><div class="leaf-block">', unsafe_allow_html=True)

    st.markdown(
        '<div class="block-head">'
        '<div class="title">Upload Photo</div>'
        '<div class="sub">Drop a JPG/PNG here, or browse</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="block-card upload-wrapper">', unsafe_allow_html=True)
    st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png"],
        key="uploader",
        on_change=on_upload_change
    )
    st.markdown('</div>', unsafe_allow_html=True)  # close block-card

    st.markdown('</div></div>', unsafe_allow_html=True)  # close leaf-block + leaf-left

with right:
    st.markdown('<div class="leaf-right"><div class="leaf-block">', unsafe_allow_html=True)

    # Header (stays where it is)
    st.markdown(
        '<div class="block-head">'
        '<div class="title">Record Photo</div>'
        '<div class="sub">Use your device camera</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # NEW: spacer to push the card down
    st.markdown('<div class="right-spacer"></div>', unsafe_allow_html=True)

    if not st.session_state.show_camera:
        # Camera closed view
        st.markdown(
            '<div class="block-card">'
            '  <div class="camera-card">'
            '    <p class="camera-hint">Tap “Open camera” to take a photo.</p>'
            '    <button id="open_cam_real" class="custom-cam-btn">Open camera</button>'
            '  </div>'
            '</div>',
            unsafe_allow_html=True
        )

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
        # Camera open view
        st.markdown('<div class="block-card">', unsafe_allow_html=True)

        cap = st.camera_input("", key="camera_input")
        if cap is not None:
            st.session_state.captured = cap
            st.session_state.source = "camera"
            if not st.session_state.keep_camera_on:
                st.session_state.show_camera = False

        st.button("Close camera", on_click=lambda: setattr(st.session_state, "show_camera", False), key="close_cam_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)  # close leaf-block + leaf-right


# Pick active file
file = st.session_state.captured if st.session_state.source == "camera" else (
    st.session_state.upload if st.session_state.source == "upload" else None
)

# Inference / output
if file:
    pil = load_pil(file)

    b = compute_brightness(pil)
    if b < dark_thr:
        st.warning(f"Image appears too dark (brightness {b:.2f}). Retake under brighter, even lighting.")
        st.stop()
    if b > bright_thr:
        st.warning(f"Image appears too bright/washed-out (brightness {b:.2f}). Retake avoiding direct glare.")
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

    probs = predict_probs(pil)
    pred_label, pred_conf, _ = decide(probs, labels, THRESHOLD)
    prob_map = {lab: float(probs[i]) for i, lab in enumerate(labels)}

    r1_left, r1_right = st.columns([1,1], gap="large")
    with r1_left:
        st.markdown("### Your Image:")
        st.image(ImageOps.contain(pil, (PREVIEW_MAX_W, PREVIEW_MAX_H)), use_container_width=False)

    with r1_right:
        st.markdown("### Predicted Apple Disease Label is:")
        st.markdown(f"**{_pretty(pred_label)}** with **{pred_conf*100:.0f}%** Confidence")
        render_prob_bars_native(prob_map)
        st.caption("Model: Calibrated ResNet-18 (TorchScript). Low-confidence predictions route to ‘unknown’.")
    vspace(3)

    st.markdown(f"### Apple – {_pretty(pred_label)} Care Recommendations:")
    vspace(2)

    poster_path = CARE_POSTERS.get(pred_label, CARE_POSTERS["healthy"])
    if not Path(poster_path).exists():
        st.info("Care poster not found. Please add the JPGs next to app.py.")
    else:
        st.image(poster_path, use_container_width=True)

else:
    st.info("Upload a photo or open the camera to begin.")
