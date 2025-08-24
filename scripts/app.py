# scripts/app.py
import os, glob, time, shlex, subprocess, tempfile
from datetime import datetime
import numpy as np
import cv2
import streamlit as st
from ultralytics import YOLO

# -------------------- Utils --------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def list_frames(dirpath): return sorted(glob.glob(os.path.join(dirpath, "frame_*.jpg")))

@st.cache_resource
def load_model(name):  # cache across reruns
    return YOLO(name)

def ffmpeg_extract(video_path, frames_dir, fps=30, tone_map=False):
    """Extract frames with ffmpeg and stream logs live to the UI."""
    ensure_dir(frames_dir)
    existing = list_frames(frames_dir)
    if existing:
        st.info(f"Frames already present in **{frames_dir}** ({len(existing)}).")
        return len(existing)

    vf = f"fps={fps}"
    if tone_map:
        # HDR/HLG → SDR tonemap
        vf = f"fps={fps},zscale=t=linear:npl=100,tonemap=hable,zscale=t=bt709:m=bt709:r=tv"

    out_pattern = os.path.join(frames_dir, "frame_%05d.jpg")
    cmd = f'ffmpeg -hide_banner -y -i "{video_path}" -map 0:v:0 -vf "{vf}" "{out_pattern}"'

    st.write(f"**Running:** `{cmd}`")
    log_box = st.empty()
    status = st.status("Extracting frames…", expanded=True) if hasattr(st, "status") else None

    t0 = time.time()
    proc = subprocess.Popen(shlex.split(cmd),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    lines = []
    for line in proc.stderr:  # progress via stderr
        lines.append(line.rstrip())
        log_box.code("\n".join(lines[-40:]), language="bash")

    code = proc.wait()
    dt = time.time() - t0
    saved = len(list_frames(frames_dir))

    if code == 0 and saved > 0:
        if status: status.update(label=f"Done in {dt:.1f}s — frames: {saved}", state="complete")
        st.success(f"✅ Extracted **{saved}** frames → **{frames_dir}**")
    else:
        if status: status.update(label="Extraction failed", state="error")
        st.error("❌ ffmpeg failed. Last log lines:")
        st.code("\n".join(lines[-80:]), language="bash")
    return saved

def motion_kernel(length_px, angle_deg):
    length_px = max(3, int(length_px) | 1)
    k = np.zeros((length_px, length_px), np.float32)
    k[length_px//2,:] = 1.0 / length_px
    M = cv2.getRotationMatrix2D((length_px/2, length_px/2), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length_px, length_px))
    s = k.sum()
    if s != 0: k /= s
    return k

def estimate_flow_angle(prev_bgr, curr_bgr):
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    flow = dis.calc(prev_gray, curr_gray, None)
    mag = np.linalg.norm(flow, axis=2)
    thr = np.percentile(mag, 75)
    mask = mag > thr
    if mask.sum() == 0: return 0.0
    dx = flow[...,0][mask].mean()
    dy = flow[...,1][mask].mean()
    return np.degrees(np.arctan2(dy, dx))

# -------------------- Segmentation (fixed) --------------------
def segment_people(model, img, imgsz=1280):
    """
    Stable person masks for Action Pan:
    - YOLOv11-seg with retina masks, person class only
    - Threshold -> clip to box -> polygon fallback if weird
    """
    H, W = img.shape[:2]

    r = model(
        img,
        imgsz=imgsz,
        retina_masks=True,
        conf=0.25,
        classes=[0],          # 0 = person
        verbose=False
    )[0]

    if r.masks is None:
        return [], []

    # boxes (Nx4 int) for clipping
    boxes = r.boxes.xyxy.int().cpu().numpy() if r.boxes is not None else None

    # 1) start from retina masks
    m = r.masks.data.float().cpu().numpy()     # (N, Hm, Wm), 0..1
    m = (m > 0.5).astype(np.uint8)             # hard threshold

    Hm, Wm = m.shape[1], m.shape[2]
    if (Hm, Wm) != (H, W):
        m = np.stack([cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST) for mi in m], axis=0)

    masks = []
    for i, mi in enumerate(m):
        # 2) clip to its own box (defensive)
        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = boxes[i]
            x1 = int(np.clip(x1, 0, W-1)); x2 = int(np.clip(x2, 0, W-1))
            y1 = int(np.clip(y1, 0, H-1)); y2 = int(np.clip(y2, 0, H-1))
            box_mask = np.zeros((H, W), dtype=np.uint8)
            box_mask[y1:y2+1, x1:x2+1] = 1
            mi = (mi & box_mask).astype(np.uint8)

        # 3) sanity check: if the mask looks like a big rectangle (artifact), rebuild from polygons
        area = mi.sum()
        rectish = False
        if boxes is not None and i < len(boxes):
            # how much of the box is filled? if ~100%, treat as suspicious
            x1, y1, x2, y2 = boxes[i]
            box_area = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
            fill_ratio = area / float(box_area)
            rectish = fill_ratio > 0.95  # almost full box

        if rectish and hasattr(r.masks, "xy") and len(r.masks.xy) > i and len(r.masks.xy[i]) > 0:
            # polygon rasterization fallback
            mi_poly = np.zeros((H, W), dtype=np.uint8)
            pts = [p.astype(np.int32) for p in r.masks.xy[i]]  # list of (Kx2) arrays
            cv2.fillPoly(mi_poly, pts, 1)
            if boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = boxes[i]
                mi_poly[y1:y2+1, x1:x2+1] = mi_poly[y1:y2+1, x1:x2+1]  # already inside
            mi = mi_poly

        # small cleanup
        mi = cv2.morphologyEx(mi, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        masks.append(mi)

    return masks, boxes if boxes is not None else []

def paint_overlay(img, masks, selected):
    """Nice visual: colored fill + crisp black contours; selected subjects in orange."""
    overlay = img.copy()
    base_color = (80, 255, 120)   # green
    sel_color  = (255, 140, 80)   # orange
    for i, m in enumerate(masks):
        color = sel_color if i in selected else base_color
        idx = m.astype(bool)
        # semi-transparent fill
        alpha = 0.45
        overlay[idx] = (overlay[idx] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        # edge contour
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0,0,0), 2)
    return overlay

# -------------------- Smooth transition --------------------
def apply_curve(alpha_lin, curve="Linear", gamma=2.2):
    a = np.clip(alpha_lin, 0.0, 1.0)
    if curve == "Linear":
        return a
    if curve == "Smoothstep":
        return a*a*(3.0 - 2.0*a)            # 3x^2 - 2x^3
    if curve == "Cosine":
        return 0.5 - 0.5*np.cos(np.pi*a)
    if curve == "Gamma":
        return np.power(a, max(0.1, float(gamma)))
    return a

def smooth_transition_mask(mask_u8, blend_radius=30, curve="Linear", gamma=2.2):
    mask = (mask_u8 > 0).astype(np.uint8)
    dist_to_bg = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    alpha_lin = np.clip(dist_to_bg / float(max(1, blend_radius)), 0.0, 1.0)
    return apply_curve(alpha_lin, curve=curve, gamma=gamma)

def composite_transition(img, masks, chosen, blur_len, angle, blend_radius=30, curve="Linear", gamma=2.2):
    kernel = motion_kernel(blur_len, angle)
    blurred = cv2.filter2D(img, -1, kernel)

    keep = np.zeros(img.shape[:2], np.uint8)
    for i in chosen:
        keep = np.logical_or(keep, masks[i]).astype(np.uint8)

    # gentle cleanup so the ramp starts from a clean edge
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    alpha = smooth_transition_mask(keep, blend_radius=blend_radius, curve=curve, gamma=gamma)
    alpha3 = np.repeat(alpha[:, :, None], 3, axis=2)
    out = (alpha3 * img + (1.0 - alpha3) * blurred).astype(np.uint8)
    return out

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Neuromotion – Action Pan", layout="wide")
st.title("Neuromotion – Action Pan (Flow-aware, Smooth Transition)")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Segmentation model",
        ["yolo11n-seg.pt","yolo11s-seg.pt","yolo11m-seg.pt","yolo11l-seg.pt"],
        index=1
    )
    imgsz = st.slider("Inference size (imgsz)", 640, 1920, 1280, 64)
    blur_len = st.slider("Blur length (px)", 3, 151, 45, 2)

    st.markdown("**Edge transition**")
    blend_radius = st.slider("Edge softness (px)", 5, 120, 30, 5)
    curve = st.selectbox("Transition curve", ["Linear","Smoothstep","Cosine","Gamma"], index=1)
    gamma_val = 2.2
    if curve == "Gamma":
        gamma_val = st.slider("Gamma", 0.5, 4.0, 2.2, 0.1)

    batch_mode = st.selectbox("Mode", ["interactive","largest","first"], index=0)
    tone_map = st.checkbox("Tone-map HDR/HLG → SDR (ffmpeg)", value=False)

    st.markdown("---")
    st.caption("Video or frames:")
    uploaded = st.file_uploader("Upload video (MP4/MOV)", type=["mp4","mov","MP4","MOV"])
    frames_dir = st.text_input("…or existing frames dir", "data/extracted_frames/C0031")
    go_extract = st.button("Extract frames (or refresh)")

# Timestamped output run dir: outputs/action_pan/<clip>/<YYYYmmdd_HHMMSS>/
clip_name = os.path.basename(os.path.normpath(frames_dir)) or "session"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_out_dir = os.path.join("outputs", "action_pan", clip_name)
run_dir = os.path.join(base_out_dir, timestamp)
if "run_dir" not in st.session_state:
    st.session_state.run_dir = run_dir
ensure_dir(st.session_state.run_dir)
st.write(f"**Output run:** `{st.session_state.run_dir}`")

# Upload/extraction
tmp_video = None
if uploaded is not None:
    tmp_video = os.path.join(tempfile.gettempdir(), uploaded.name)
    with open(tmp_video, "wb") as f:
        f.write(uploaded.read())

if go_extract:
    if tmp_video:
        ffmpeg_extract(tmp_video, frames_dir, fps=30, tone_map=tone_map)
    elif os.path.exists(frames_dir):
        st.info(f"Frames dir exists, found {len(list_frames(frames_dir))} frames.")
    else:
        st.error("Provide a video or a valid frames directory.")

frames = list_frames(frames_dir)
st.write(f"**Frames found:** {len(frames)} in `{frames_dir}`")
if len(frames) < 2:
    st.info("Provide a video and click **Extract frames**, or point to a frames directory.")
    st.stop()

# Controls
colA, colB = st.columns([1,3], gap="large")
with colA:
    idx = st.slider("Frame index", 0, len(frames)-1, 0, 1)
    save_current = st.button("Save current")
    run_batch = st.button("Run batch")

# Load frame(s)
curr = cv2.imread(frames[idx])
prev = cv2.imread(frames[idx-1]) if idx > 0 else curr

# Inference
model = load_model(model_name)
masks, boxes = segment_people(model, curr, imgsz=imgsz)
angle = estimate_flow_angle(prev, curr)

# Selection state
if "selected" not in st.session_state:
    st.session_state.selected = set()
selected = st.session_state.selected

with colA:
    if masks:
        areas = [int(m.sum()) for m in masks]
        options = [f"subject #{i} (area={a})" for i,a in enumerate(areas)]
        default = []
        if batch_mode != "interactive" and not selected and len(areas) > 0:
            default = [options[int(np.argmax(areas))]]
        keep = st.multiselect("Keep sharp", options, default=default, key="msel")
        selected = set([int(s.split('#')[1].split()[0]) for s in keep])
        st.session_state.selected = selected
    else:
        st.caption("No subjects detected.")

# Compose preview
if masks:
    out = composite_transition(curr, masks, selected, blur_len, angle,
                               blend_radius=blend_radius, curve=curve, gamma=gamma_val)
    overlay = paint_overlay(curr, masks, selected)

    with colB:
        st.write(
            f"Flow: **{angle:.1f}°** · Blur: **{blur_len}px** · "
            f"Edge softness: **{blend_radius}px** · Curve: **{curve}**"
            + (f" (γ={gamma_val:.1f})" if curve == "Gamma" else "")
            + f" · imgsz: **{imgsz}** · Selected: **{sorted(list(selected))}**"
        )
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                 caption=f"Frame {idx} – masks/selection", use_container_width=True)
        c2.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                 caption="Action Pan composite", use_container_width=True)
else:
    with colB:
        st.warning("No people detected in this frame.")

# Save current
if save_current and masks:
    save_path = os.path.join(st.session_state.run_dir, f"pan_{os.path.basename(frames[idx])}")
    ensure_dir(os.path.dirname(save_path))
    cv2.imwrite(save_path, out)
    st.success(f"Saved {save_path}")

# -------------------- Batch modes --------------------
def process_all(frames, choose_fn):
    prog = st.progress(0)
    count = 0
    for i, f in enumerate(frames):
        curr = cv2.imread(f)
        prev = cv2.imread(frames[i-1]) if i>0 else curr
        mks, _ = segment_people(model, curr, imgsz=imgsz)
        if not mks:
            prog.progress((i+1)/len(frames)); continue
        chosen = choose_fn(mks)
        ang = estimate_flow_angle(prev, curr)
        out_i = composite_transition(curr, mks, chosen, blur_len, ang,
                                     blend_radius=blend_radius, curve=curve, gamma=gamma_val)
        save_path = os.path.join(st.session_state.run_dir, f"pan_{os.path.basename(f)}")
        ensure_dir(os.path.dirname(save_path))
        cv2.imwrite(save_path, out_i)
        count += 1
        prog.progress((i+1)/len(frames))
    st.success(f"Batch complete: saved {count} frames to {st.session_state.run_dir}")

if run_batch:
    ensure_dir(st.session_state.run_dir)
    if batch_mode == "largest":
        process_all(frames, choose_fn=lambda mks: [int(np.argmax([m.sum() for m in mks]))])
    elif batch_mode == "first":
        if not selected:
            st.error("Select at least one subject (left panel) before running batch 'first'.")
        else:
            process_all(frames, choose_fn=lambda mks: list(selected))
    else:
        st.info("Interactive mode doesn’t batch. Choose 'largest' or 'first'.")
