# scripts/app.py
import os, glob, time, shlex, subprocess, tempfile
from datetime import datetime
import numpy as np
import cv2
import streamlit as st
from ultralytics import YOLO

# ======================= General utils =======================

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def list_frames(dirpath):
    pats = ["frame_*.jpg", "frame_*.jpeg", "frame_*.png", "frame_*.tif", "frame_*.tiff"]
    files = []
    for pat in pats:
        files += glob.glob(os.path.join(dirpath, pat))
    return sorted(files)

@st.cache_resource
def load_model(name):  # cache across reruns
    return YOLO(name)

def ffmpeg_extract(video_path, frames_dir, fps=30, tone_map=False, fmt="tiff16"):
    """
    Extract frames and stream logs to UI.
    fmt: 'jpg' (8-bit HQ), 'png' (lossless 8-bit), 'tiff16' (lossless 16-bit RGB).
    """
    ensure_dir(frames_dir)
    existing = list_frames(frames_dir)
    if existing:
        st.info(f"Frames already present in **{frames_dir}** ({len(existing)}).")
        return len(existing)

    vf = f"fps={fps}"
    if tone_map:
        # HDR/HLG → SDR tonemap (safe defaults)
        vf = f"fps={fps},zscale=t=linear:npl=100,tonemap=hable,zscale=t=bt709,format=gbrp16le"

    if fmt == "tiff16":
        out_pattern = os.path.join(frames_dir, "frame_%05d.tiff")
        cmd = f'ffmpeg -hide_banner -y -i "{video_path}" -map 0:v:0 -vf "{vf}" ' \
              f'-pix_fmt rgb48le -vcodec tiff -compression lzw "{out_pattern}"'
    elif fmt == "png":
        out_pattern = os.path.join(frames_dir, "frame_%05d.png")
        cmd = f'ffmpeg -hide_banner -y -i "{video_path}" -map 0:v:0 -vf "{vf}" ' \
              f'-pix_fmt rgb24 -vcodec png "{out_pattern}"'
    else:  # jpg high quality
        out_pattern = os.path.join(frames_dir, "frame_%05d.jpg")
        cmd = f'ffmpeg -hide_banner -y -i "{video_path}" -map 0:v:0 -vf "{vf}" -q:v 1 "{out_pattern}"'

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

# ======================= Color space helpers (linear light) =======================

def srgb_to_linear(img16):
    """uint16 BGR → float32 linear (0..1)."""
    x = (img16.astype(np.float32) / 65535.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb_f32(x):
    """float32 linear (0..1) → float32 sRGB (0..1)."""
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0/2.4)) - 0.055)

# ======================= Motion & flow =======================

def directional_gaussian_kernel(length_px, angle_deg, aspect=0.18):
    """Photographic motion PSF (elongated Gaussian along motion)."""
    sigma_par  = max(0.8, float(length_px) / 3.0)
    sigma_perp = max(0.8, sigma_par * aspect)
    kx = int(6 * sigma_par) | 1
    ky = int(6 * sigma_perp) | 1
    ksize = max(3, max(kx, ky) | 1)
    ax = np.arange(-(ksize//2), ksize//2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    a = np.deg2rad(angle_deg)
    cosA, sinA = np.cos(-a), np.sin(-a)
    xprime =  cosA*xx - sinA*yy
    yprime =  sinA*xx + cosA*yy
    k = np.exp(-0.5 * ((xprime/sigma_par)**2 + (yprime/sigma_perp)**2))
    k /= k.sum()
    return k.astype(np.float32)

def estimate_flow(prev_bgr8, curr_bgr8):
    prev_gray = cv2.cvtColor(prev_bgr8, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr8, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    flow = dis.calc(prev_gray, curr_gray, None)  # HxWx2
    mag = np.linalg.norm(flow, axis=2)
    angle = np.degrees(np.arctan2(flow[...,1], flow[...,0]))
    return flow, mag, angle

def estimate_flow_angle(prev_bgr8, curr_bgr8):
    flow, mag, _ = estimate_flow(prev_bgr8, curr_bgr8)
    thr = np.percentile(mag, 75)
    mask = mag > thr
    if mask.sum() == 0: return 0.0
    dx = flow[...,0][mask].mean()
    dy = flow[...,1][mask].mean()
    return np.degrees(np.arctan2(dy, dx))

# ======================= Segmentation (robust) =======================

def segment_people(model, img8, imgsz=1280):
    """
    YOLOv11-seg with retina masks, person-only.
    Returns masks aligned to original image size with defensive cleanup.
    """
    H, W = img8.shape[:2]
    r = model(img8, imgsz=imgsz, retina_masks=True, conf=0.25, classes=[0], verbose=False)[0]
    if r.masks is None:
        return [], []

    m = r.masks.data.float().cpu().numpy()  # (N, Hm, Wm), 0..1
    m = (m > 0.5).astype(np.uint8)
    Hm, Wm = m.shape[1], m.shape[2]
    if (Hm, Wm) != (H, W):
        m = np.stack([cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST) for mi in m], axis=0)

    boxes = r.boxes.xyxy.int().cpu().numpy() if r.boxes is not None else None
    masks = []
    for i, mi in enumerate(m):
        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = boxes[i]
            x1 = int(np.clip(x1, 0, W-1)); x2 = int(np.clip(x2, 0, W-1))
            y1 = int(np.clip(y1, 0, H-1)); y2 = int(np.clip(y2, 0, H-1))
            box_mask = np.zeros((H, W), dtype=np.uint8)
            box_mask[y1:y2+1, x1:x2+1] = 1
            mi = (mi & box_mask).astype(np.uint8)
        mi = cv2.morphologyEx(mi, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
        masks.append(mi)
    return masks, (boxes if boxes is not None else [])

def paint_overlay(img8, masks, selected):
    """Green fill + black contours; selected masks tinted orange."""
    overlay = img8.copy()
    base_color = (80, 255, 120)   # green
    sel_color  = (255, 140, 80)   # orange
    for i, m in enumerate(masks):
        color = sel_color if i in selected else base_color
        idx = m.astype(bool)
        overlay[idx] = (overlay[idx]*0.55 + np.array(color)*0.45).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0,0,0), 2)
    return overlay

# ======================= Signed-distance alpha (offset + asym widths) =======================

def apply_curve(alpha_lin, curve="Linear", gamma=2.2):
    a = np.clip(alpha_lin, 0.0, 1.0)
    if curve == "Linear":     return a
    if curve == "Smoothstep": return a*a*(3.0 - 2.0*a)
    if curve == "Cosine":     return 0.5 - 0.5*np.cos(np.pi*a)
    if curve == "Gamma":      return np.power(a, max(0.1, float(gamma)))
    return a

def alpha_signed_distance(mask_u8,
                          offset_px=0.0,      # <0 = center shifted inward, >0 outward
                          width_in_px=20.0,   # half-width of inside half
                          width_out_px=40.0,  # half-width of outside half
                          curve="Smoothstep",
                          gamma=2.2,
                          hard_core_px=0.0):  # clamp alpha=1 for s>=this
    mask = (mask_u8 > 0).astype(np.uint8)
    dist_in  = cv2.distanceTransform(mask,     cv2.DIST_L2, 3)  # grows inward
    dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)  # grows outward

    # signed distance: + inside, - outside
    s = dist_in - dist_out  # float32

    # normalized ramps on each side of the offset
    if width_in_px > 1e-6:
        t_in = np.clip((s - offset_px) / float(width_in_px), 0.0, 1.0)     # 0 at offset → 1 deep inside
    else:
        t_in = (s >= offset_px).astype(np.float32)

    if width_out_px > 1e-6:
        t_out = np.clip(1.0 - ((offset_px - s) / float(width_out_px)), 0.0, 1.0)  # 1 at offset → 0 outward
    else:
        t_out = (s >= offset_px).astype(np.float32)

    use_in  = (s >= offset_px).astype(np.float32)
    use_out = 1.0 - use_in
    t = use_in * t_in + use_out * t_out

    t = apply_curve(t, curve=curve, gamma=gamma)

    alpha = t.astype(np.float32)
    if hard_core_px > 0.0:
        alpha[s >= hard_core_px] = 1.0
    alpha[(s <= offset_px - width_out_px - 1.0)] = 0.0
    return np.clip(alpha, 0.0, 1.0)

def refine_alpha_with_guided(alpha, guide8, radius=10, eps=1e-3):
    """Edge-aware alpha refinement (guided filter if available, else bilateral)."""
    try:
        import cv2.ximgproc as xp
        g = cv2.cvtColor(guide8, cv2.COLOR_BGR2GRAY)
        a_ref = xp.guidedFilter(guide=g, src=alpha.astype(np.float32), radius=radius, eps=eps)
        return np.clip(a_ref, 0.0, 1.0)
    except Exception:
        a8 = (alpha*255).astype(np.uint8)
        a_b = cv2.bilateralFilter(a8, d=7, sigmaColor=50, sigmaSpace=7)
        return a_b.astype(np.float32)/255.0

# ======================= Compositors (linear-light, 16-bit) =======================

def composite_uniform_16u(img16, masks, chosen, blur_len, angle,
                          offset_px=0.0, width_in_px=8.0, width_out_px=40.0, hard_core_px=16.0,
                          curve="Smoothstep", gamma=2.2,
                          aspect=0.18, guide8=None):
    H, W = img16.shape[:2]
    keep = np.zeros((H, W), np.uint8)
    for i in chosen: keep |= masks[i]
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)

    alpha = alpha_signed_distance(keep, offset_px, width_in_px, width_out_px,
                                  curve=curve, gamma=gamma, hard_core_px=hard_core_px)
    if guide8 is not None:
        alpha = refine_alpha_with_guided(alpha, guide8, radius=10, eps=1e-3)

    lin = srgb_to_linear(img16)
    k = directional_gaussian_kernel(blur_len, angle, aspect=aspect)
    lin_blur = cv2.filter2D(lin, -1, k)

    a = alpha[..., None]
    lin_out = a * lin + (1.0 - a) * lin_blur

    srgb_out = np.clip(linear_to_srgb_f32(lin_out), 0.0, 1.0)
    out16 = (srgb_out * 65535.0 + 0.5).astype(np.uint16)
    return out16, alpha

# ---- Zonal helpers ----
def make_blur_stack_linear(lin_img, angle_deg, levels, aspect=0.18):
    outs = []
    for L in levels:
        if L <= 0:
            outs.append(lin_img.copy())
        else:
            k = directional_gaussian_kernel(int(L), angle_deg, aspect=aspect)
            outs.append(cv2.filter2D(lin_img, -1, k))
    return outs

def radial_weights(H, W, center=None, softness=0.35, power=1.0, n=8):
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = (W//2, H//2) if center is None else center
    r = np.hypot(xx - cx, yy - cy)
    r = (r / (r.max() + 1e-6)) ** power
    edges = np.linspace(0, 1, n+1)
    ws = []
    width = max(1e-6, softness * (edges[1]-edges[0]))
    for i in range(n):
        c = 0.5*(edges[i] + edges[i+1])
        w_i = np.maximum(0.0, 1.0 - np.abs(r - c) / width)
        ws.append(w_i)
    Wmap = np.stack(ws, axis=-1)
    Wmap /= (Wmap.sum(axis=-1, keepdims=True) + 1e-6)
    return Wmap

def flow_weights(prev8, curr8, n=8, power=1.0):
    _, mag, _ = estimate_flow(prev8, curr8)
    a = (mag / (mag.max() + 1e-6)) ** power
    edges = np.linspace(0, 1, n+1)
    ws = []
    width = max(1e-6, 0.5*(edges[1]-edges[0]))
    for i in range(n):
        c = 0.5*(edges[i] + edges[i+1])
        w_i = np.maximum(0.0, 1.0 - np.abs(a - c) / width)
        ws.append(w_i)
    Wmap = np.stack(ws, axis=-1)
    Wmap /= (Wmap.sum(axis=-1, keepdims=True) + 1e-6)
    return Wmap

def smooth_weight_map(W, sigma_px=7):
    if W.ndim == 2: W = W[..., None]
    out = np.empty_like(W, dtype=np.float32)
    for c in range(W.shape[-1]):
        out[..., c] = cv2.GaussianBlur(W[..., c].astype(np.float32), (0,0), sigma_px)
    s = out.sum(axis=-1, keepdims=True) + 1e-6
    return out / s

def composite_zonal_16u(img16, prev8, curr8, masks, chosen, angle_deg,
                        blur_len, offset_px=0.0, width_in_px=8.0, width_out_px=40.0, hard_core_px=16.0,
                        curve="Smoothstep", gamma=2.2,
                        profile="Radial", radial_soft=0.35, radial_power=1.0, flow_power=1.0,
                        aspect=0.18, guide8=None):
    H, W = img16.shape[:2]
    keep = np.zeros((H, W), np.uint8)
    for i in chosen: keep |= masks[i]
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)

    alpha = alpha_signed_distance(keep, offset_px, width_in_px, width_out_px,
                                  curve=curve, gamma=gamma, hard_core_px=hard_core_px)
    if guide8 is not None:
        alpha = refine_alpha_with_guided(alpha, guide8, radius=10, eps=1e-3)

    lin = srgb_to_linear(img16)
    levels = np.linspace(0, blur_len, 8).astype(int).tolist()
    stack = make_blur_stack_linear(lin, angle_deg, levels, aspect=aspect)

    if profile == "Flow":
        Wmap = flow_weights(prev8, curr8, n=len(stack), power=flow_power)
    else:
        Wmap = radial_weights(H, W, center=None, softness=radial_soft, power=radial_power, n=len(stack))
    Wmap = smooth_weight_map(Wmap, sigma_px=7)

    out_bg = np.zeros_like(stack[0])
    for i in range(len(stack)):
        out_bg += (Wmap[..., i:i+1] * stack[i])

    a = alpha[..., None]
    lin_out = a * srgb_to_linear(img16) + (1.0 - a) * out_bg

    srgb_out = np.clip(linear_to_srgb_f32(lin_out), 0.0, 1.0)
    out16 = (srgb_out * 65535.0 + 0.5).astype(np.uint16)
    return out16, alpha

# ======================= Streamlit UI =======================

st.set_page_config(page_title="Neuromotion – Action Pan", layout="wide")
st.title("Neuromotion – Action Pan (Signed-Distance Edge, 16-bit TIFF)")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Segmentation model",
        ["yolo11n-seg.pt","yolo11s-seg.pt","yolo11m-seg.pt","yolo11l-seg.pt"],
        index=1
    )
    imgsz = st.slider("Inference size (imgsz)", 640, 1920, 1280, 64)
    blur_len = st.slider("Blur length (px)", 3, 151, 45, 2)
    psf_aspect = st.slider("PSF aspect (width)", 0.08, 0.35, 0.18, 0.01)

    st.markdown("**Edge transition (signed distance)**")
    boundary_offset_px = st.slider("Boundary center offset (px)", -60, 60, -6, 1,
                                   help="Shift the 50% blend line: − = inward (into subject), + = outward")
    width_in_px  = st.slider("Inside half-width (px)", 0, 60, 8, 1,
                             help="Blend band width INSIDE the edge")
    width_out_px = st.slider("Outside half-width (px)", 5, 120, 50, 1,
                             help="Blend band width OUTSIDE the edge")
    hard_core_px = st.slider("Hard core clamp (px)", 0, 80, 16, 1,
                             help="Guarantee alpha=1 inside this distance (protects faces/hands)")
    curve = st.selectbox("Transition curve", ["Linear","Smoothstep","Cosine","Gamma"], index=1)
    gamma_val = 2.2
    if curve == "Gamma":
        gamma_val = st.slider("Gamma", 0.5, 4.0, 2.2, 0.1)
    show_alpha = st.checkbox("Debug: show alpha matte", value=False)

    st.markdown("**Blur profile**")
    profile = st.selectbox("Profile", ["Uniform","Radial","Flow"], index=0)
    radial_soft = st.slider("Radial softness", 0.1, 1.0, 0.35, 0.05) if profile == "Radial" else 0.35
    radial_power = st.slider("Radial power", 0.5, 2.5, 1.0, 0.1) if profile == "Radial" else 1.0
    flow_power = st.slider("Flow power", 0.5, 2.5, 1.0, 0.1) if profile == "Flow" else 1.0

    st.markdown("**Output**")
    save_format = st.selectbox("Save format", ["JPEG (8-bit)", "TIFF (16-bit)"], index=1)

    batch_mode = st.selectbox("Mode", ["interactive","largest","first"], index=0)

    st.markdown("---")
    st.caption("Video or frames:")
    uploaded = st.file_uploader("Upload video (MP4/MOV)", type=["mp4","mov","MP4","MOV"])
    frames_dir = st.text_input("…or existing frames dir", "data/extracted_frames/C0031")
    extract_fmt_label = st.selectbox("Extract as", ["TIFF 16-bit", "PNG (lossless 8-bit)", "JPEG 8-bit (HQ)"], index=0)
    fmt_key = "tiff16" if extract_fmt_label.startswith("TIFF") else ("png" if extract_fmt_label.startswith("PNG") else "jpg")
    tone_map = st.checkbox("Tone-map HDR/HLG → SDR (ffmpeg)", value=False)
    go_extract = st.button("Extract frames (or refresh)")

# Timestamped output dir: outputs/action_pan/<clip>/<timestamp>/
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
        ffmpeg_extract(tmp_video, frames_dir, fps=30, tone_map=tone_map, fmt=fmt_key)
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

# Load current/prev, preserve depth
fpath = frames[idx]
prev_path = frames[idx-1] if idx > 0 else frames[idx]
img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)        # uint8 or uint16
img_prev = cv2.imread(prev_path, cv2.IMREAD_UNCHANGED)
is_16bit = (img.dtype == np.uint16)
img16 = img if is_16bit else (img.astype(np.uint16) * 257)
prev16 = img_prev if img_prev.dtype == np.uint16 else (img_prev.astype(np.uint16) * 257)
img8 = (img16 / 257).astype(np.uint8)
prev8 = (prev16 / 257).astype(np.uint8)

# Inference
model = load_model(model_name)
masks, boxes = segment_people(model, img8, imgsz=imgsz)
angle = estimate_flow_angle(prev8, img8)

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
    guide8 = img8  # for alpha refinement
    if profile == "Uniform":
        out16, alpha_last = composite_uniform_16u(
            img16, masks, selected, blur_len, angle,
            offset_px=boundary_offset_px, width_in_px=width_in_px, width_out_px=width_out_px,
            hard_core_px=hard_core_px,
            curve=curve, gamma=gamma_val, aspect=psf_aspect, guide8=guide8
        )
    else:
        out16, alpha_last = composite_zonal_16u(
            img16, prev8, img8, masks, selected, angle,
            blur_len,
            offset_px=boundary_offset_px, width_in_px=width_in_px, width_out_px=width_out_px,
            hard_core_px=hard_core_px,
            curve=curve, gamma=gamma_val,
            profile=("Flow" if profile=="Flow" else "Radial"),
            radial_soft=radial_soft, radial_power=radial_power, flow_power=flow_power,
            aspect=psf_aspect, guide8=guide8
        )

    overlay = paint_overlay(img8, masks, selected)

    with colB:
        st.write(
            f"Flow: **{angle:.1f}°** · Blur: **{blur_len}px** · PSF aspect: **{psf_aspect:.2f}** · "
            f"Offset: **{boundary_offset_px}px** · Inside: **{width_in_px}px** · Outside: **{width_out_px}px** · "
            f"Core clamp: **{hard_core_px}px** · Curve: **{curve}** · imgsz: **{imgsz}** · "
            f"Selected: **{sorted(list(selected))}** · Profile: **{profile}**"
        )
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                 caption=f"Frame {idx} – masks/selection", use_container_width=True)
        c2.image(cv2.cvtColor((out16/257).astype(np.uint8), cv2.COLOR_BGR2RGB),
                 caption="Action Pan composite", use_container_width=True)
        if show_alpha:
            st.image((alpha_last*255).astype(np.uint8), clamp=True, caption="Alpha matte (255=sharp, 0=blur)")

else:
    with colB:
        st.warning("No people detected in this frame.")

# Save current
def save_image(out16, base_dir, name, fmt_choice):
    ensure_dir(base_dir)
    if fmt_choice.startswith("TIFF"):
        path = os.path.join(base_dir, f"{name}.tiff")
        cv2.imwrite(path, out16)   # uint16 TIFF
    else:
        out8 = (out16 / 257).astype(np.uint8)
        path = os.path.join(base_dir, f"{name}.jpg")
        cv2.imwrite(path, out8)
    return path

if save_current and masks:
    base = f"pan_{os.path.splitext(os.path.basename(fpath))[0]}"
    save_path = save_image(out16, st.session_state.run_dir, base, save_format)
    st.success(f"Saved {save_path}")

# -------------------- Batch processing --------------------

def choose_indices(mks, mode, selected_now):
    if mode == "largest":
        areas = [m.sum() for m in mks]
        return [int(np.argmax(areas))]
    if mode == "first":
        return list(selected_now) if selected_now else []
    return list(selected_now)

def process_all(frames, mode):
    prog = st.progress(0)
    count = 0
    for i, f in enumerate(frames):
        curr = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        prev = cv2.imread(frames[i-1], cv2.IMREAD_UNCHANGED) if i>0 else curr
        c16 = curr if curr.dtype == np.uint16 else (curr.astype(np.uint16) * 257)
        p16 = prev if prev.dtype == np.uint16 else (prev.astype(np.uint16) * 257)
        c8  = (c16 / 257).astype(np.uint8)
        p8  = (p16 / 257).astype(np.uint8)

        mks, _ = segment_people(model, c8, imgsz=imgsz)
        if not mks:
            prog.progress((i+1)/len(frames)); continue

        ang = estimate_flow_angle(p8, c8)
        chosen = choose_indices(mks, mode, selected)

        guide8 = c8
        if profile == "Uniform":
            out16_i, _ = composite_uniform_16u(
                c16, mks, chosen, blur_len, ang,
                offset_px=boundary_offset_px, width_in_px=width_in_px, width_out_px=width_out_px,
                hard_core_px=hard_core_px,
                curve=curve, gamma=gamma_val, aspect=psf_aspect, guide8=guide8
            )
        else:
            out16_i, _ = composite_zonal_16u(
                c16, p8, c8, mks, chosen, ang,
                blur_len,
                offset_px=boundary_offset_px, width_in_px=width_in_px, width_out_px=width_out_px,
                hard_core_px=hard_core_px,
                curve=curve, gamma=gamma_val,
                profile=("Flow" if profile=="Flow" else "Radial"),
                radial_soft=radial_soft, radial_power=radial_power, flow_power=flow_power,
                aspect=psf_aspect, guide8=guide8
            )

        base = f"pan_{os.path.splitext(os.path.basename(f))[0]}"
        save_image(out16_i, st.session_state.run_dir, base, save_format)
        count += 1
        prog.progress((i+1)/len(frames))
    st.success(f"Batch complete: saved {count} frames to {st.session_state.run_dir}")

with colA:
    if run_batch:
        if batch_mode == "interactive":
            st.info("Interactive mode doesn’t batch. Choose 'largest' or 'first'.")
        elif batch_mode == "first" and not selected:
            st.error("Select at least one subject (left panel) before running batch 'first'.")
        else:
            process_all(frames, batch_mode)
