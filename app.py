import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os

st.set_page_config(page_title="Fluorescence Segmentation", layout="wide")

# ----------------- Session State -----------------
if "files" not in st.session_state:          # list of dicts: {name, bytes}
    st.session_state.files = []
if "results" not in st.session_state:        # list of dicts with counts/params
    st.session_state.results = []
if "method" not in st.session_state:
    st.session_state.method = "HSV (GFP-like)"

# ----------------- Utils -----------------
def to_uint8(img):
    if img is None: return None
    if img.dtype == np.uint8: return img
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        scale = 255.0 / float(info.max if info.max > 0 else 1)
        return cv2.convertScaleAbs(img, alpha=scale)
    img = np.nan_to_num(img)
    m, M = float(img.min()), float(img.max())
    if M > m:
        img = (img - m) / (M - m)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def ensure_bgr_u8(arr):
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return to_uint8(arr)

def morph(mask, ksize=5):
    if ksize and ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def apply_hsv(bgr_u8, lower, upper, ksize=5):
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    return morph(m, ksize)

def apply_lab(bgr_u8, L, a, b, ksize=5):
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
    Lc, ac, bc = cv2.split(lab)
    cond = ((Lc >= L[0]) & (Lc <= L[1]) &
            (ac >= a[0]) & (ac <= a[1]) &
            (bc >= b[0]) & (bc <= b[1]))
    m = np.where(cond, 255, 0).astype(np.uint8)
    return morph(m, ksize)

def apply_rgb(bgr_u8, g_min=90, g_margin=25, ksize=5):
    b, g, r = cv2.split(bgr_u8)
    g16, r16, b16 = g.astype(np.int16), r.astype(np.int16), b.astype(np.int16)
    cond = (g16 >= g_min) & (g16 >= r16 + g_margin) & (g16 >= b16 + g_margin)
    m = np.where(cond, 255, 0).astype(np.uint8)
    return morph(m, ksize)

def overlay_mask(bgr_u8, mask, alpha=0.6, color=(0,255,0)):
    color_layer = np.zeros_like(bgr_u8)
    color_layer[mask > 0] = color
    return cv2.addWeighted(color_layer, alpha, bgr_u8, 1 - alpha, 0)

# ----------------- UI: Settings -----------------
st.title("Fluorescence Mask & Quantification")

st.sidebar.header("Settings")
method = st.sidebar.selectbox(
    "Segmentation method",
    ["HSV (GFP-like)", "Lab (greenish a*<128)", "RGB heuristic (G dominant)"],
    index=["HSV (GFP-like)", "Lab (greenish a*<128)", "RGB heuristic (G dominant)"].index(st.session_state.method),
)
st.session_state.method = method

ksize = st.sidebar.slider("Morph kernel size (ellipse)", 0, 21, 5, step=2)
alpha = st.sidebar.slider("Overlay opacity", 0.05, 1.0, 0.6, step=0.05)

st.sidebar.markdown("---")
if method.startswith("HSV"):
    st.sidebar.subheader("HSV thresholds (OpenCV scale)")
    h_low  = st.sidebar.slider("H low",  0, 179, 35)
    h_high = st.sidebar.slider("H high", 0, 179, 85)
    s_low  = st.sidebar.slider("S low",  0, 255, 40)
    v_low  = st.sidebar.slider("V low",  0, 255, 40)
    s_high, v_high = 255, 255
elif method.startswith("Lab"):
    st.sidebar.subheader("Lab thresholds (OpenCV 8-bit scaled)")
    Lmin = st.sidebar.slider("L min", 0, 255, 0)
    Lmax = st.sidebar.slider("L max", 0, 255, 255)
    amin = st.sidebar.slider("a* min", 0, 255, 0)
    amax = st.sidebar.slider("a* max (greenish <128)", 0, 255, 120)
    bmin = st.sidebar.slider("b* min", 0, 255, 0)
    bmax = st.sidebar.slider("b* max", 0, 255, 255)
else:
    st.sidebar.subheader("RGB heuristic")
    g_min   = st.sidebar.slider("G minimum", 0, 255, 90)
    g_margin= st.sidebar.slider("G margin over R and B", 0, 100, 25)

st.sidebar.markdown("---")
# ------- NEW: robust output folder handling (Downloads by default) -------
default_downloads = Path.home() / "Downloads" / "fluorescence_outputs"
out_dir_input = st.sidebar.text_input("Output folder", str(default_downloads))

# Expand ~, resolve absolute path, and ensure directories exist
out_dir = Path(out_dir_input).expanduser().resolve()
masks_dir = out_dir / "masks"
out_dir.mkdir(parents=True, exist_ok=True)
masks_dir.mkdir(parents=True, exist_ok=True)

# Optional: warn if not writeable (rare on local machines)
if not os.access(out_dir, os.W_OK):
    st.warning(f"Output folder may not be writeable: {out_dir}")

# ----------------- Upload (multi-file) -----------------
uploaded_files = st.file_uploader(
    "Upload one or more images (tif/png/jpg)",
    type=["tif","tiff","png","jpg","jpeg","bmp"],
    accept_multiple_files=True
)

# Cache uploaded file bytes in session_state so we can reuse them
if uploaded_files:
    st.session_state.files = [{"name": uf.name, "bytes": uf.getvalue()} for uf in uploaded_files]

colL, colR = st.columns(2)
with colL:
    process_btn = st.button("Process & Preview")
with colR:
    save_btn = st.button("Save CSV + Masks")

# ----------------- Process -----------------
if process_btn and st.session_state.files:
    st.session_state.results = []  # reset results for a fresh run
    cols = st.columns(2)

    for i, f in enumerate(st.session_state.files):
        arr = cv2.imdecode(np.frombuffer(f["bytes"], np.uint8), cv2.IMREAD_UNCHANGED)
        bgr = ensure_bgr_u8(arr)
        H, W = bgr.shape[:2]
        total = int(H * W)

        if method.startswith("HSV"):
            mask = apply_hsv(bgr, (h_low, s_low, v_low), (h_high, s_high, v_high), ksize)
            method_key = "hsv"
            param_dict = {"h_low": h_low, "h_high": h_high, "s_low": s_low, "s_high": s_high, "v_low": v_low, "v_high": v_high}
        elif method.startswith("Lab"):
            mask = apply_lab(bgr, (Lmin, Lmax), (amin, amax), (bmin, bmax), ksize)
            method_key = "lab"
            param_dict = {"Lmin": Lmin, "Lmax": Lmax, "amin": amin, "amax": amax, "bmin": bmin, "bmax": bmax}
        else:
            mask = apply_rgb(bgr, g_min, g_margin, ksize)
            method_key = "rgb"
            param_dict = {"g_min": g_min, "g_margin": g_margin}

        count = int((mask > 0).sum())
        ratio = count / total if total > 0 else 0.0
        ov = overlay_mask(bgr, mask, alpha=alpha, color=(0,255,0))

        with cols[i % 2]:
            st.markdown(f"**{f['name']}** — {H}×{W} px")
            st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), caption=f"{method_key.upper()} overlay", use_column_width=True)
            st.image(mask, caption=f"{method_key.upper()} mask", clamp=True, use_column_width=True)
            st.write(f"Fluorescent pixels: **{count} / {total}**  (ratio = **{ratio:.6f}**)")

        row = {
            "image": f["name"], "height": H, "width": W, "total_pixels": total,
            f"fluorescent_pixels_{method_key}": count, f"ratio_{method_key}": ratio,
            "method": method_key, "ksize": ksize, "alpha": alpha
        }
        row.update(param_dict)
        st.session_state.results.append(row)

    if st.session_state.results:
        st.subheader("Summary table")
        st.dataframe(pd.DataFrame(st.session_state.results), use_container_width=True)

# ----------------- Save -----------------
if save_btn:
    if not st.session_state.results:
        st.warning("Click 'Process & Preview' first.")
    else:
        try:
            df = pd.DataFrame(st.session_state.results)
            csv_path = out_dir / "results.csv"

            # Append or create CSV
            if csv_path.exists():
                old = pd.read_csv(csv_path)
                df = pd.concat([old, df], ignore_index=True)
            df.to_csv(csv_path, index=False)

            # Save masks using the same parameters used during processing
            for r, f in zip(st.session_state.results, st.session_state.files):
                arr = cv2.imdecode(np.frombuffer(f["bytes"], np.uint8), cv2.IMREAD_UNCHANGED)
                bgr = ensure_bgr_u8(arr)
                if r["method"] == "hsv":
                    mask = apply_hsv(
                        bgr,
                        (int(r["h_low"]), int(r["s_low"]), int(r["v_low"])),
                        (int(r["h_high"]), int(r["s_high"]), int(r["v_high"])),
                        int(r["ksize"])
                    )
                elif r["method"] == "lab":
                    mask = apply_lab(
                        bgr,
                        (int(r["Lmin"]), int(r["Lmax"])),
                        (int(r["amin"]), int(r["amax"])),
                        (int(r["bmin"]), int(r["bmax"])),
                        int(r["ksize"])
                    )
                else:
                    mask = apply_rgb(bgr, int(r["g_min"]), int(r["g_margin"]), int(r["ksize"]))

                base = Path(r["image"]).stem
                cv2.imwrite(str(masks_dir / f"{base}_mask_{r['method']}.png"), mask)

            st.success(f"Saved CSV to {csv_path} and masks to {masks_dir}")

        except Exception as e:
            st.error("Failed to save outputs.")
            st.exception(e)
