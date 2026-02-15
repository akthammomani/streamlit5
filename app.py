# app.py
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

# -------------------- Defaults (Settings are now fixed) --------------------
THRESHOLD     = 0.85
dark_thr      = 0.25
bright_thr    = 0.90
PREVIEW_MAX_W = 420
PREVIEW_MAX_H = 420

# Camera leaf-gate defaults (used later)
cov_min = 0.04
tex_min = 25.0

# -------------------- Page config (force sidebar open) --------------------
st.set_page_config(
    page_title="AI-Powered Apple Leaf Specialist",
    page_icon=APP_LOGO if Path(APP_LOGO).exists() else "🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- CSS --------------------
# Notes:
# - Streamlit does NOT provide a true "uncollapsible sidebar" API.
#   We do best-effort: keep it expanded + hide the collapse control.
# - The “attached shapes” are usually Streamlit empty widgets (often st.text_input/search)
#   rendered as rounded boxes. We hide common ones inside the sidebar.
st.markdown(
    """
<style>
/* =========================
   SIDEBAR: wider + gray bg
   ========================= */
:root{
  --sb-w: 420px;             /* tweak: 400 / 420 / 460 */
  --sb-bg: #F3F4F6;          /* solid gray */
  --card-bg: #F3F4F6;        /* keep cards same gray (solid look) */
  --border: #E5E7EB;
  --text: #111827;
  --muted: #6B7280;
}
section[data-testid="stSidebar"]{
  width: var(--sb-w) !important;
  min-width: var(--sb-w) !important;
  background: var(--sb-bg) !important;
}
section[data-testid="stSidebar"] > div{
  width: var(--sb-w) !important;
  background: var(--sb-bg) !important;
}

/* Best-effort: prevent collapsing by removing the UI control */
button[data-testid="collapsedControl"]{ display:none !important; }
section[data-testid="stSidebar"] div[data-testid="stSidebarNav"]{ display:none; }

/* Keep sidebar background solid (avoid random white blocks) */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div,
section[data-testid="stSidebar"] .block-container{
  background: var(--sb-bg) !important;
}

/* =========================
   REMOVE "ATTACHED SHAPES"
   ========================= */
/* These are commonly: empty st.text_input / search input rendered as rounded boxes.
   Hide the entire widget container when it's a text input inside the sidebar. */
section[data-testid="stSidebar"] div[data-testid="stTextInput"]{
  display:none !important;
}
/* Also hide any stray empty input-like boxes (defensive) */
section[data-testid="stSidebar"] input[type="text"],
section[data-testid="stSidebar"] input[type="search"]{
  display:none !important;
}

/* =========================
   TYPOGRAPHY / HIERARCHY
   ========================= */
.sb-app-title{
  font-size: 1.35rem;
  font-weight: 850;
  color: var(--text);
  line-height: 1.2;
  text-align: center;
  margin: .35rem 0 .55rem 0;
}
.sb-divider{
  height: 3px;
  background: #D1D5DB;
  border-radius: 999px;
  margin: .4rem 0 1rem 0;
}
.sb-h1{
  font-size: 1.05rem;
  font-weight: 800;
  color: var(--text);
  margin: 0 0 .35rem 0;
}
.sb-sub{
  font-size: .86rem;
  color: var(--muted);
  margin: 0 0 .85rem 0;
}

/* =========================
   SOLID SECTION LOOK
   ========================= */
.sb-card{
  border: 1.5px solid var(--border);
  background: var(--card-bg) !important;
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 10px;
}
.sb-sec-title{
  font-size: .95rem;
  font-weight: 750;
  color: var(--text);
  margin: 0 0 2px 0;
}
.sb-sec-sub{
  font-size: .82rem;
  color: var(--muted);
  margin: 0 0 10px 0;
}
.sb-mini-sep{
  height: 1px;
  background: #E5E7EB;
  margin: 10px 2px;
}

/* File uploader dropzone should match the same gray */
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"]{
  background: var(--card-bg) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 12px !important;
}

/* Tighten sidebar spacing */
section[data-testid="stSidebar"] .block-container{ padding-top: .6rem; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap: .65rem; }

/* Buttons */
section[data-testid="stSidebar"] .stButton button{
  border-radius: 8px !important;
}

/* Main area column cleanup */
div[data-testid="column"] > div:first-child{
  margin-top: 0 !important;
  padding-top: 0 !important;
}
</style>
""",
    unsafe_allow_html=True
)

st.markdown(
"""
<style>
/* ✅ Remove ONLY the isolated rounded empty “bars” in the sidebar */
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:empty{
  display: none !important;
}

/* ✅ Some Streamlit builds don’t leave it truly empty; they contain only whitespace/BR.
   Hide containers that have no visible children (common in these ghost bars). */
section[data-testid="stSidebar"] div[data-testid="stElementContainer"] > div:empty{
  display:none !important;
}
section[data-testid="stSidebar"] div[data-testid="stElementContainer"] > div > div:empty{
  display:none !important;
}

/* ✅ Defensive: if the ghost bar is a bordered container with no widgets, kill its border/background */
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]{
  box-shadow: none !important;
}
</style>
""",
unsafe_allow_html=True
)


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

def _pretty(lab: str) -> str:
    return lab.replace("_", " ").title()

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
                st.progress(int(p * 100))
        with c3: st.write(f"{p*100:.1f}%")

# -------------------- Posters --------------------
CARE_POSTERS = {
    "black_rot": "black_rot_care_v2.jpg",
    "healthy":   "healthy_care_v2.jpg",
    "scab":      "scab_care_v2.jpg",
    "rust":      "rust_care_v2.jpg",
}

# -------------------- Session state --------------------
if "show_camera" not in st.session_state:     st.session_state.show_camera = False
if "source" not in st.session_state:         st.session_state.source = None
if "captured" not in st.session_state:       st.session_state.captured = None
if "upload" not in st.session_state:         st.session_state.upload = None
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

# -------------------- Sidebar UI --------------------
with st.sidebar:
    # Logo
    if Path(APP_LOGO).exists():
        b64 = base64.b64encode(Path(APP_LOGO).read_bytes()).decode()
        ext = Path(APP_LOGO).suffix.lstrip(".").lower() or "png"
        st.markdown(
            f"""
            <div style="display:flex;justify-content:center;margin-top:.2rem;">
              <img src="data:image/{ext};base64,{b64}" style="max-width:120px;height:auto;" />
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div style="text-align:center;font-size:52px">🍎</div>', unsafe_allow_html=True)

    # Title + thick separator
    st.markdown('<div class="sb-app-title">AI-Powered Apple Leaf Specialist</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Section header
    st.markdown('<div class="sb-h1">Add a leaf photo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Upload a file or take a photo using your camera.</div>', unsafe_allow_html=True)

    # Upload card
    st.markdown('<div class="sb-card">', unsafe_allow_html=True)
    st.markdown('<div class="sb-sec-title">Upload Photo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sec-sub">Drop a JPG/PNG here, or browse</div>', unsafe_allow_html=True)
    st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png"],
        key="uploader",
        on_change=on_upload_change,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Small separator inside section
    st.markdown('<div class="sb-mini-sep"></div>', unsafe_allow_html=True)

    # Record card
    st.markdown('<div class="sb-card">', unsafe_allow_html=True)
    st.markdown('<div class="sb-sec-title">Record Photo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sec-sub">Use your device camera</div>', unsafe_allow_html=True)

    if not st.session_state.show_camera:
        c_hint, c_btn = st.columns([6, 4], vertical_alignment="center")
        with c_hint:
            st.caption('Tap "Open camera" to take a photo.')
        with c_btn:
            if st.button("Open camera", key="open_cam_btn"):
                open_camera()
    else:
        cap = st.camera_input("", key="camera_input")
        if cap is not None:
            st.session_state.captured = cap
            st.session_state.source = "camera"
            if not st.session_state.keep_camera_on:
                st.session_state.show_camera = False
        st.button("Close camera", on_click=close_camera, key="close_cam_btn")

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("---")
    st.write("""
    ### Contacts
    [![](https://img.shields.io/badge/GitHub-Follow-informational)](https://github.com/akthammomani)
    [![](https://img.shields.io/badge/Linkedin-Connect-informational)](https://www.linkedin.com/in/akthammomani/)
    [![](https://img.shields.io/badge/Open-Issue-informational)](https://github.com/akthammomani/ai_powered_apple_leaf_specialist/issues)
    [![MAIL Badge](https://img.shields.io/badge/-aktham.momani81@gmail.com-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:aktham.momani81@gmail.com)](mailto:aktham.momani81@gmail.com)
    ###### © Aktham Momani, 2025. All rights reserved.
    """)

# -------------------- Active source --------------------
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

    # Inference
    probs = predict_probs(pil)
    pred_label, pred_conf, _ = decide(probs, labels, THRESHOLD)
    prob_map = {lab: float(probs[i]) for i, lab in enumerate(labels)}

    # -------- Row 1: image + prediction --------
    r1_left, med, r1_right = st.columns([0.5, 0.5, 1], gap="large")
    with r1_left:
        st.markdown("### Your Image:")
        st.image(ImageOps.contain(pil, (PREVIEW_MAX_W, PREVIEW_MAX_H)), use_container_width=False)

    with med:
        st.markdown("### Learn More")
        st.markdown("[![](https://img.shields.io/badge/GitHub%20-AI--Powered%20Apple%20Leaf%20Specialist-informational)](https://github.com/akthammomani/ai_powered_apple_leaf_specialist)")

    with r1_right:
        st.markdown("### Predicted Apple Disease Label is:")
        st.markdown(f"**{_pretty(pred_label)}** with **{pred_conf*100:.0f}%** Confidence")
        render_prob_bars_native(prob_map)
        st.write("#### Learn More")
        st.markdown("[![](https://img.shields.io/badge/GitHub%20-Calibrated%20ResNet-18%20Model-informational)](https://github.com/akthammomani/ai_powered_apple_leaf_specialist/blob/main/Notebooks/Modeling_AI_Powered_Apple_Leaf_Specialist.ipynb)")

    # -------- Care poster --------
    st.markdown(f"### Apple – {_pretty(pred_label)} Care Recommendations:")
    poster_path = CARE_POSTERS.get(pred_label, CARE_POSTERS["healthy"])
    if not Path(poster_path).exists():
        st.info("Care poster not found. Please add the JPGs next to app.py.")
    else:
        st.image(poster_path, use_container_width=True)
        st.write("""
        ###### ***Disclaimer***
        *This app is not a substitute for professional agricultural advice, diagnosis, or treatment. Field conditions, pests, and diseases can vary widely. Always consult a qualified agronomist, crop advisor, or local extension service before making decisions that could affect tree health, spray plans, or harvest.*
        """)
else:
    st.info("Use the sidebar to upload a photo or open the camera to begin.")
