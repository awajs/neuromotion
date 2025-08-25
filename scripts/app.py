import os, glob, time, shlex, subprocess, tempfile, gc
from datetime import datetime
import numpy as np
import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas

# ----------------------- Performance knobs -----------------------
cv2.setNumThreads(1)                  # avoid OpenCV spawning
MAX_PREVIEW_PIXELS = 2_000_000        # ~2MP auto preview target

# ----------------------- FS helpers -----------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def list_frames(dirpath):
    pats = ["frame_*.jpg", "frame_*.jpeg", "frame_*.png", "frame_*.tif", "frame_*.tiff"]
    files = []
    for pat in pats:
        files += glob.glob(os.path.join(dirpath, pat))
    return sorted(files)

# ----------------------- Model -----------------------
@st.cache_resource
def load_model(name):  # cache across reruns
    return YOLO(name)

# ----------------------- Extraction (ffmpeg) -----------------------
def ffmpeg_extract(video_path, frames_dir, fps=30, tone_map=False, fmt="tiff16"):
    """fmt: 'jpg' (HQ 8bit) | 'png' (lossless 8bit) | 'tiff16' (lossless 16bit)."""
    ensure_dir(frames_dir)
    if list_frames(frames_dir):
        st.info(f"Frames already present in **{frames_dir}**.")
        return

    vf = f"fps={fps}"
    if tone_map:
        vf = f"fps={fps},zscale=t=linear:npl=100,tonemap=hable,zscale=t=bt709,format=gbrp16le"

    if fmt == "tiff16":
        out = os.path.join(frames_dir, "frame_%05d.tiff")
        cmd = f'ffmpeg -hide_banner -y -i "{video_path}" -map 0:v:0 -vf "{vf}" ' \
              f'-pix_fmt rgb48le -vcodec tiff -compression lzw "{out}"'
    elif fmt == "png":
        out = os.path.join(frames_dir, "frame_%05d.png")
        cmd = f'ffmpeg -hide_banner -y -i "{video_path}" -map 0:v:0 -vf "{vf}" -pix_fmt rgb24 -vcodec png "{out}"'
    else:  # jpg HQ
        out = os.path.join(frames_dir, "frame_%05d.jpg")
        cmd = f'ffmpeg -hide_banner -y -i "{video_path}" -map 0:v:0 -vf "{vf}" -q:v 1 "{out}"'

    log = st.empty()
    status = st.status("Extracting frames…", expanded=True) if hasattr(st, "status") else None
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    lines = []
    for line in p.stderr:
        lines.append(line.rstrip())
        log.code("\n".join(lines[-40:]), language="bash")
    rc = p.wait()
    n = len(list_frames(frames_dir))
    if rc == 0 and n > 0:
        if status: status.update(label=f"Done — frames: {n}", state="complete")
        st.success(f"✅ Extracted **{n}** frames → **{frames_dir}**")
    else:
        if status: status.update(label="Extraction failed", state="error")
        st.error("❌ ffmpeg failed."); st.code("\n".join(lines[-80:]), language="bash")

# ----------------------- Color space (linear light) -----------------------
def srgb_to_linear(img16):
    x = (img16.astype(np.float32) / 65535.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb_f32(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0/2.4)) - 0.055)

# ----------------------- Motion & flow -----------------------
def directional_gaussian_kernel(length_px, angle_deg, aspect=0.18):
    sigma_par  = max(0.8, float(length_px) / 3.0)
    sigma_perp = max(0.8, sigma_par * aspect)
    kx = int(6 * sigma_par) | 1
    ky = int(6 * sigma_perp) | 1
    ksize = max(3, max(kx, ky) | 1)
    ax = np.arange(-(ksize//2), ksize//2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    a = np.deg2rad(angle_deg); ca, sa = np.cos(-a), np.sin(-a)
    xprime =  ca*xx - sa*yy
    yprime =  sa*xx + ca*yy
    k = np.exp(-0.5 * ((xprime/sigma_par)**2 + (yprime/sigma_perp)**2))
    k /= k.sum()
    return k.astype(np.float32)

def estimate_flow(prev8, curr8):
    pg = cv2.cvtColor(prev8, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(curr8, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    flow = dis.calc(pg, cg, None)
    mag = np.linalg.norm(flow, axis=2)
    ang = np.degrees(np.arctan2(flow[...,1], flow[...,0]))
    return flow, mag, ang

def estimate_flow_angle(prev8, curr8):
    flow, mag, _ = estimate_flow(prev8, curr8)
    thr = np.percentile(mag, 75); m = mag > thr
    if m.sum() == 0: return 0.0
    dx = flow[...,0][m].mean(); dy = flow[...,1][m].mean()
    return np.degrees(np.arctan2(dy, dx))

# ----------------------- Segmentation (robust + cached) -----------------------
@st.cache_data(show_spinner=False)
def yolo_segment_cached(model_name, imgsz, frame_bgr8_bytes, hw):
    model = load_model(model_name)
    H, W = hw
    img8 = cv2.imdecode(np.frombuffer(frame_bgr8_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    r = model(img8, imgsz=imgsz, retina_masks=True, conf=0.25, classes=[0], verbose=False)[0]
    masks = []
    if r.masks is not None:
        m = (r.masks.data.float().cpu().numpy() > 0.5).astype(np.uint8)  # (N,h,w)
        Hm, Wm = m.shape[1], m.shape[2]
        if (Hm, Wm) != (H, W):
            m = np.stack([cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST) for mi in m], axis=0)
        boxes = r.boxes.xyxy.int().cpu().numpy() if r.boxes is not None else None
        for i, mi in enumerate(m):
            if boxes is not None and i < len(boxes):
                x1,y1,x2,y2 = boxes[i]; x1=max(0,x1); y1=max(0,y1); x2=min(W-1,x2); y2=min(H-1,y2)
                box_mask = np.zeros((H,W), np.uint8); box_mask[y1:y2+1, x1:x2+1]=1
                mi = (mi & box_mask).astype(np.uint8)
            mi = cv2.morphologyEx(mi, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),1)
            masks.append(mi)
    return masks

def segment_people(model_name, imgsz, img8):
    _, buf = cv2.imencode(".jpg", img8, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return yolo_segment_cached(model_name, imgsz, buf.tobytes(), img8.shape[:2])

def paint_overlay(img8, masks, selected):
    overlay = img8.copy()
    base = (80,255,120); sel=(255,140,80)
    for i,m in enumerate(masks):
        col = sel if i in selected else base
        idx = m.astype(bool)
        overlay[idx] = (overlay[idx]*0.55 + np.array(col)*0.45).astype(np.uint8)
        cnts,_=cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0,0,0), 2)
    return overlay

# ----------------------- Signed-distance alpha -----------------------
def apply_curve(a, curve="Linear", gamma=2.2):
    a=np.clip(a,0,1)
    if curve=="Smoothstep": return a*a*(3.0-2.0*a)
    if curve=="Cosine":     return 0.5 - 0.5*np.cos(np.pi*a)
    if curve=="Gamma":      return np.power(a, max(0.1,float(gamma)))
    return a

def alpha_signed_distance(mask, offset_px=0.0, width_in_px=20.0, width_out_px=40.0,
                          curve="Smoothstep", gamma=2.2, hard_core_px=0.0):
    mask=(mask>0).astype(np.uint8)
    di=cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    do=cv2.distanceTransform(1-mask, cv2.DIST_L2, 3)
    s=di-do  # + inside
    t_in  = np.clip((s-offset_px)/max(1e-6,width_in_px), 0,1)
    t_out = np.clip(1-((offset_px-s)/max(1e-6,width_out_px)), 0,1)
    use_in=(s>=offset_px).astype(np.float32)
    t = use_in*t_in + (1-use_in)*t_out
    t = apply_curve(t, curve=curve, gamma=gamma)
    alpha=t.astype(np.float32)
    if hard_core_px>0: alpha[s>=hard_core_px]=1.0
    alpha[s <= offset_px - width_out_px - 1.0] = 0.0
    return np.clip(alpha,0,1)

def refine_alpha_with_guided(alpha, guide8, radius=12, eps=1e-3):
    try:
        import cv2.ximgproc as xp
        g=cv2.cvtColor(guide8,cv2.COLOR_BGR2GRAY)
        a=xp.guidedFilter(guide=g,src=alpha.astype(np.float32),radius=radius,eps=eps)
        return np.clip(a,0,1)
    except Exception:
        a8=(alpha*255).astype(np.uint8)
        a=cv2.bilateralFilter(a8,d=9,sigmaColor=60,sigmaSpace=9)/255.0
        return np.clip(a,0,1)

# ----------------------- Compositors (linear-light, 16-bit) -----------------------
def dir_blend(img16, lin_blur, alpha):
    lin=srgb_to_linear(img16)
    lin_out = alpha[...,None]*lin + (1-alpha[...,None])*lin_blur
    srgb_out=np.clip(linear_to_srgb_f32(lin_out),0,1)
    return (srgb_out*65535.0+0.5).astype(np.uint16)

def composite_uniform_16u(img16, masks, chosen, blur_len, angle,
                          offset_px=0.0, width_in_px=8.0, width_out_px=40.0, hard_core_px=16.0,
                          curve="Smoothstep", gamma=2.2,
                          aspect=0.18, guide8=None, extra_keep=None):
    H,W=img16.shape[:2]
    keep=np.zeros((H,W),np.uint8)
    for i in chosen: keep|=masks[i]
    if extra_keep is not None: keep|=(extra_keep>0).astype(np.uint8)
    keep=cv2.morphologyEx(keep,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),1)

    alpha=alpha_signed_distance(keep, -offset_px, width_in_px, width_out_px, curve=curve, gamma=gamma, hard_core_px=hard_core_px)
    alpha=cv2.GaussianBlur(alpha.astype(np.float32),(0,0),1.25)
    if guide8 is not None: alpha=refine_alpha_with_guided(alpha,guide8,radius=12,eps=1e-3)

    k=directional_gaussian_kernel(blur_len,angle,aspect=aspect)
    lin_blur=cv2.filter2D(srgb_to_linear(img16),-1,k)
    return dir_blend(img16,lin_blur,alpha), alpha

def make_blur_stack_linear(lin_img, angle_deg, levels, aspect=0.18):
    outs=[]; 
    for L in levels:
        if L<=0: outs.append(lin_img.copy())
        else:
            k=directional_gaussian_kernel(int(L),angle_deg,aspect=aspect)
            outs.append(cv2.filter2D(lin_img,-1,k))
    return outs

def radial_weights(H,W,center=None,softness=0.35,power=1.0,n=8):
    yy,xx=np.mgrid[0:H,0:W]
    cx,cy=(W//2,H//2) if center is None else center
    r=np.hypot(xx-cx,yy-cy); r=(r/(r.max()+1e-6))**power
    edges=np.linspace(0,1,n+1); ws=[]; width=max(1e-6,softness*(edges[1]-edges[0]))
    for i in range(n):
        c=0.5*(edges[i]+edges[i+1])
        w=np.maximum(0.0,1.0-np.abs(r-c)/width); ws.append(w)
    Wmap=np.stack(ws,axis=-1); Wmap/= (Wmap.sum(axis=-1,keepdims=True)+1e-6)
    return Wmap

def flow_weights(prev8,curr8,n=8,power=1.0):
    _,mag,_=estimate_flow(prev8,curr8)
    a=(mag/(mag.max()+1e-6))**power
    edges=np.linspace(0,1,n+1); ws=[]; width=max(1e-6,0.5*(edges[1]-edges[0]))
    for i in range(n):
        c=0.5*(edges[i]+edges[i+1])
        w=np.maximum(0.0,1.0-np.abs(a-c)/width); ws.append(w)
    Wmap=np.stack(ws,axis=-1); Wmap/= (Wmap.sum(axis=-1,keepdims=True)+1e-6)
    return Wmap

def smooth_weight_map(W,sigma_px=7):
    if W.ndim==2: W=W[...,None]
    out=np.empty_like(W,dtype=np.float32)
    for c in range(W.shape[-1]):
        out[...,c]=cv2.GaussianBlur(W[...,c].astype(np.float32),(0,0),sigma_px)
    s=out.sum(axis=-1,keepdims=True)+1e-6
    return out/s

def composite_zonal_16u(img16, prev8, curr8, masks, chosen, angle_deg,
                        blur_len, offset_px=0.0, width_in_px=8.0, width_out_px=40.0, hard_core_px=16.0,
                        curve="Smoothstep", gamma=2.2,
                        profile="Radial", radial_soft=0.35, radial_power=1.0, flow_power=1.0,
                        aspect=0.18, guide8=None, extra_keep=None):
    H,W=img16.shape[:2]
    keep=np.zeros((H,W),np.uint8)
    for i in chosen: keep|=masks[i]
    if extra_keep is not None: keep|=(extra_keep>0).astype(np.uint8)
    keep=cv2.morphologyEx(keep,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),1)

    alpha=alpha_signed_distance(keep, -offset_px, width_in_px, width_out_px, curve=curve, gamma=gamma, hard_core_px=hard_core_px)
    alpha=cv2.GaussianBlur(alpha.astype(np.float32),(0,0),1.25)
    if guide8 is not None: alpha=refine_alpha_with_guided(alpha,guide8,radius=12,eps=1e-3)

    lin=srgb_to_linear(img16)
    levels=np.linspace(0,blur_len,8).astype(int).tolist()
    stack=make_blur_stack_linear(lin,angle_deg,levels,aspect=aspect)

    if profile=="Flow": Wmap=flow_weights(prev8,curr8,n=len(stack),power=flow_power)
    else: Wmap=radial_weights(H,W,center=None,softness=radial_soft,power=radial_power,n=len(stack))
    Wmap=smooth_weight_map(Wmap,sigma_px=7)

    out_bg=np.zeros_like(stack[0])
    for i in range(len(stack)): out_bg += (Wmap[...,i:i+1]*stack[i])
    return dir_blend(img16,out_bg,alpha), alpha

# ----------------------- Preview scaling helpers -----------------------
def compute_preview_scale(h, w, auto=True, ui_scale=0.5):
    if not auto: return float(ui_scale)
    return float(min(1.0, (MAX_PREVIEW_PIXELS / max(1, h*w)) ** 0.5))

def rescale(img, s, interp=cv2.INTER_AREA):
    if s >= 0.999: return img
    nh, nw = max(1, int(img.shape[0]*s)), max(1, int(img.shape[1]*s))
    return cv2.resize(img, (nw, nh), interpolation=interp)

# ----------------------- UI -----------------------
st.set_page_config(page_title="Neuromotion – Action Pan", layout="wide")
st.title("Neuromotion – Action Pan (Signed-Distance Edge, Brush Keep, 16-bit TIFF)")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Segmentation model",
                              ["yolo11n-seg.pt","yolo11s-seg.pt","yolo11m-seg.pt","yolo11l-seg.pt"], 1)
    imgsz = st.slider("Inference size (imgsz)", 640, 1920, 1280, 64)
    blur_len = st.slider("Blur length (px)", 3, 151, 45, 2)
    psf_aspect = st.slider("PSF aspect (width)", 0.08, 0.35, 0.20, 0.01)

    st.markdown("**Edge transition (signed distance)**")
    boundary_offset_px = st.slider("Boundary center offset (px)", -60, 60, -6, 1,
                                   help="− = inward (shrinks sharp zone), + = outward (widens)")
    width_in_px  = st.slider("Inside half-width (px)", 0, 60, 10, 1)
    width_out_px = st.slider("Outside half-width (px)", 5, 120, 70, 1)
    hard_core_px = st.slider("Hard core clamp (px)", 0, 80, 12, 1)
    curve = st.selectbox("Transition curve", ["Linear","Smoothstep","Cosine","Gamma"], 1)
    gamma_val = st.slider("Gamma", 0.5, 4.0, 2.2, 0.1) if curve=="Gamma" else 2.2
    show_alpha = st.checkbox("Debug: show alpha matte", value=False)

    st.markdown("**Custom region (optional)**")
    use_custom_region = st.checkbox("Enable brush", value=False,
                                    help="Turn on to paint an extra keep-sharp area")
    apply_brush_now = st.button("Apply brush now", disabled=not use_custom_region,
                                help="Debounced apply — preview recomputes only when pressed")
    apply_brush_on_save = st.checkbox("Apply brush on Save/Batch", value=True, disabled=not use_custom_region)
    brush_size = st.slider("Brush size", 3, 80, 28, 1, disabled=not use_custom_region)
    clear_brush = st.button("Clear brush strokes", disabled=not use_custom_region)

    st.markdown("**Blur profile**")
    profile = st.selectbox("Profile", ["Uniform","Radial","Flow"], 0)
    radial_soft = st.slider("Radial softness", 0.1, 1.0, 0.35, 0.05) if profile=="Radial" else 0.35
    radial_power = st.slider("Radial power", 0.5, 2.5, 1.0, 0.1) if profile=="Radial" else 1.0
    flow_power = st.slider("Flow power", 0.5, 2.5, 1.0, 0.1) if profile=="Flow" else 1.0

    st.markdown("**Preview scaling**")
    auto_preview = st.checkbox("Auto preview scale (by megapixels)", True)
    preview_scale_ui = st.slider("Manual preview scale", 0.25, 1.0, 0.5, 0.05, disabled=auto_preview)

    st.markdown("**Output**")
    save_format = st.selectbox("Save format", ["JPEG (8-bit)","TIFF (16-bit)"], 1)
    batch_mode = st.selectbox("Mode", ["interactive","largest","first"], 0)

    st.markdown("---")
    uploaded = st.file_uploader("Upload video (MP4/MOV)", type=["mp4","mov","MP4","MOV"])
    frames_dir = st.text_input("…or frames dir", "data/extracted_frames/C0031")
    ext_label = st.selectbox("Extract as", ["TIFF 16-bit","PNG (lossless 8-bit)","JPEG 8-bit (HQ)"], 0)
    fmt_key = "tiff16" if ext_label.startswith("TIFF") else ("png" if ext_label.startswith("PNG") else "jpg")
    tone_map = st.checkbox("Tone-map HDR/HLG → SDR (ffmpeg)", False)
    go_extract = st.button("Extract frames (or refresh)")

# Output dir
clip_name = os.path.basename(os.path.normpath(frames_dir)) or "session"
run_dir = os.path.join("outputs","action_pan",clip_name, datetime.now().strftime("%Y%m%d_%H%M%S"))
if "run_dir" not in st.session_state: st.session_state.run_dir=run_dir
ensure_dir(st.session_state.run_dir)
st.write(f"**Output run:** `{st.session_state.run_dir}`")

# Extract
tmp_video=None
if uploaded is not None:
    tmp_video = os.path.join(tempfile.gettempdir(), uploaded.name)
    with open(tmp_video,"wb") as f: f.write(uploaded.read())
if go_extract:
    if tmp_video: ffmpeg_extract(tmp_video, frames_dir, fps=30, tone_map=tone_map, fmt=fmt_key)
    elif os.path.exists(frames_dir): st.info(f"Frames dir exists, found {len(list_frames(frames_dir))} frames.")
    else: st.error("Provide a video or a valid frames directory.")

frames = list_frames(frames_dir)
st.write(f"**Frames found:** {len(frames)}")
if len(frames)<2: st.info("Provide frames or extract them."); st.stop()

# Layout
colA, colB = st.columns([1,5], gap="large")
with colA:
    idx = st.slider("Frame index", 0, len(frames)-1, 0, 1)

# Load full-res
fpath = frames[idx]
prev_path = frames[idx-1] if idx>0 else frames[idx]
img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED);  img16 = img if img.dtype==np.uint16 else (img.astype(np.uint16)*257)
prv = cv2.imread(prev_path, cv2.IMREAD_UNCHANGED); prev16 = prv if prv.dtype==np.uint16 else (prv.astype(np.uint16)*257)
img8  = (img16/257).astype(np.uint8); prev8=(prev16/257).astype(np.uint8)
H,W = img8.shape[:2]

# Preview scaling
PREVIEW_S = min(1.0, compute_preview_scale(H,W, auto_preview, preview_scale_ui))
def RS(x, s, interp=cv2.INTER_AREA): return x if s>=0.999 else cv2.resize(x,(int(W*s),int(H*s)),interpolation=interp)
p_img16=RS(img16,PREVIEW_S); p_img8=RS(img8,PREVIEW_S); p_prev8=RS(prev8,PREVIEW_S)

# Masks (cached)
masks = segment_people(model_name, imgsz, img8)
p_masks = [RS(m.astype(np.uint8), PREVIEW_S, cv2.INTER_NEAREST) for m in masks]
angle = estimate_flow_angle(prev8, img8)

# Brush state containers
if "brush_json" not in st.session_state: st.session_state.brush_json={}
if "brush_mask_full" not in st.session_state: st.session_state.brush_mask_full={}
if "brush_mask_prev" not in st.session_state: st.session_state.brush_mask_prev={}

# ---- Brush area (debounced; no background image to avoid component bugs) ----
custom_keep_mask=None; p_extra=None
with colA:
    if use_custom_region:
        st.caption("Paint extra area to keep sharp (left click/drag).")

        # Make canvas size match preview size to avoid scaling confusion
        p_h,p_w = p_img8.shape[:2]
        cv_w, cv_h = p_w, p_h  # exact match

        # Restore JSON scene if exists
        init_json = st.session_state.brush_json.get(fpath, None)
        if st.button("Reset strokes", key=f"reset_{idx}", disabled=not use_custom_region):
            st.session_state.brush_json.pop(fpath, None)
            st.session_state.brush_mask_full.pop(fpath, None)
            st.session_state.brush_mask_prev.pop(fpath, None)
            init_json=None

        # Debounced canvas: update_streamlit=False -> app reruns only when Apply pressed
        canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=brush_size,   # ← use the slider’s value
            stroke_color="rgba(255, 0, 0, 0.85)",
            background_color="#00000000",
            update_streamlit=False,
            height=cv_h, width=cv_w,
            drawing_mode="freedraw",
            initial_drawing=init_json,
            key=f"canvas_{idx}",
        )


        # Reference below canvas (same size)
        st.image(cv2.cvtColor(p_img8, cv2.COLOR_BGR2RGB), caption="Reference", use_container_width=False)

        # Apply brush now -> build masks, persist JSON, and preview overlay
        if apply_brush_now and canvas is not None:
            st.session_state.brush_json[fpath] = canvas.json_data
            if canvas.image_data is not None:
                a_small = canvas.image_data[:, :, 3]
                draw_small = (a_small > 0).astype(np.uint8)
                st.session_state.brush_mask_prev[fpath] = draw_small.copy()
                st.session_state.brush_mask_full[fpath] = cv2.resize(draw_small, (W,H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # Retrieve persisted masks for this frame
        p_extra = st.session_state.brush_mask_prev.get(fpath, None)
        custom_keep_mask = st.session_state.brush_mask_full.get(fpath, None)

        # Show overlay so it's obvious what's applied
        if p_extra is not None:
            overlay_prev = p_img8.copy()
            tint = np.zeros_like(overlay_prev); tint[:] = (0,0,255)
            mask3 = np.repeat(p_extra[...,None],3,axis=2).astype(bool)
            overlay_prev[mask3] = (overlay_prev[mask3]*0.4 + tint[mask3]*0.6).astype(np.uint8)
            st.image(cv2.cvtColor(overlay_prev, cv2.COLOR_BGR2RGB),
                     caption=("Brush overlay (APPLIED)" if apply_brush_on_save else "Brush overlay (NOT applied on save)"),
                     use_container_width=False)

# Selection state
if "selected" not in st.session_state: st.session_state.selected=set()
selected = st.session_state.selected
with colA:
    if masks:
        areas=[int(m.sum()) for m in masks]
        opts=[f"subject #{i} (area={a})" for i,a in enumerate(areas)]
        default=[]
        if batch_mode!="interactive" and not selected and len(areas)>0:
            default=[opts[int(np.argmax(areas))]]
        keep = st.multiselect("Keep sharp (YOLO subjects)", opts, default=default, key="msel")
        selected=set([int(s.split('#')[1].split()[0]) for s in keep]); st.session_state.selected=selected
    else: st.caption("No subjects detected.")

# ---- PREVIEW compositing (small) ----
extra_preview = p_extra if (use_custom_region and p_extra is not None) else None
if masks:
    if profile=="Uniform":
        p_out16, p_alpha = composite_uniform_16u(
            p_img16, p_masks, selected, blur_len, angle,
            offset_px=boundary_offset_px, width_in_px=width_in_px, width_out_px=width_out_px,
            hard_core_px=hard_core_px, curve=curve, gamma=gamma_val,
            aspect=0.20, guide8=p_img8, extra_keep=extra_preview
        )
    else:
        p_out16, p_alpha = composite_zonal_16u(
            p_img16, p_prev8, p_img8, p_masks, selected, angle,
            blur_len, offset_px=boundary_offset_px, width_in_px=width_in_px, width_out_px=width_out_px,
            hard_core_px=hard_core_px, curve=curve, gamma=gamma_val,
            profile=("Flow" if profile=="Flow" else "Radial"),
            radial_soft=radial_soft, radial_power=radial_power, flow_power=flow_power,
            aspect=0.20, guide8=p_img8, extra_keep=extra_preview
        )

    with colB:
        st.image(cv2.cvtColor((p_out16/257).astype(np.uint8), cv2.COLOR_BGR2RGB),
                 use_container_width=True, caption="Action Pan composite – preview")
        with st.expander("Details (masks & alpha)"):
            c1,c2=st.columns(2)
            c1.image(cv2.cvtColor(paint_overlay(p_img8, p_masks, selected), cv2.COLOR_BGR2RGB),
                     use_container_width=True, caption="Masks / selection")
            if show_alpha:
                c2.image((p_alpha*255).astype(np.uint8), clamp=True, use_container_width=True,
                         caption="Alpha matte (255=sharp)")

# ---- SAVE (full-res recompute) ----
def save_image(out16, base_dir, name, fmt_choice):
    ensure_dir(base_dir)
    if fmt_choice.startswith("TIFF"):
        path=os.path.join(base_dir,f"{name}.tiff"); cv2.imwrite(path,out16)
    else:
        out8=(out16/257).astype(np.uint8)
        path=os.path.join(base_dir,f"{name}.jpg"); cv2.imwrite(path,out8)
    return path

with colA:
    if st.button("Save current"):
        extra_full = custom_keep_mask if (use_custom_region and apply_brush_on_save and custom_keep_mask is not None) else None
        if profile=="Uniform":
            out16,_=composite_uniform_16u(
                img16, masks, selected, blur_len, angle,
                offset_px=boundary_offset_px, width_in_px=width_in_px, width_out_px=width_out_px,
                hard_core_px=hard_core_px, curve=curve, gamma=gamma_val,
                aspect=0.20, guide8=img8, extra_keep=extra_full
            )
        else:
            out16,_=composite_zonal_16u(
                img16, prev8, img8, masks, selected, angle,
                blur_len, offset_px=boundary_offset_px, width_in_px=width_in_px, width_out_px=width_out_px,
                hard_core_px=hard_core_px, curve=curve, gamma=gamma_val,
                profile=("Flow" if profile=="Flow" else "Radial"),
                radial_soft=radial_soft, radial_power=radial_power, flow_power=flow_power,
                aspect=0.20, guide8=img8, extra_keep=extra_full
            )
        base=f"pan_{os.path.splitext(os.path.basename(frames[idx]))[0]}"
        path=save_image(out16, st.session_state.run_dir, base, save_format)
        del out16; gc.collect()
        st.success(f"Saved {path}")

# -------------------- BATCH (full-res) --------------------
def choose_indices(mks, mode, selected_now):
    if mode=="largest": return [int(np.argmax([m.sum() for m in mks]))]
    if mode=="first":   return list(selected_now) if selected_now else []
    return list(selected_now)

with colA:
    if st.button("Run batch"):
        if batch_mode=="interactive":
            st.info("Interactive mode doesn’t batch. Choose 'largest' or 'first'.")
        elif batch_mode=="first" and not selected:
            st.error("Select at least one subject before batch 'first'.")
        else:
            prog=st.progress(0); count=0
            for i,f in enumerate(frames):
                curr=cv2.imread(f,cv2.IMREAD_UNCHANGED)
                prev=cv2.imread(frames[i-1],cv2.IMREAD_UNCHANGED) if i>0 else curr
                c16=curr if curr.dtype==np.uint16 else (curr.astype(np.uint16)*257)
                p16=prev if prev.dtype==np.uint16 else (prev.astype(np.uint16)*257)
                c8=(c16/257).astype(np.uint8); p8=(p16/257).astype(np.uint8)

                mks = segment_people(model_name, imgsz, c8)
                if not mks: prog.progress((i+1)/len(frames)); continue
                ang = estimate_flow_angle(p8, c8)
                chosen = choose_indices(mks, batch_mode, selected)

                extra_full=None
                if use_custom_region and apply_brush_on_save:
                    # optional: reuse same brush for all frames? For now, ignore in batch
                    extra_full=None

                if profile=="Uniform":
                    out16,_=composite_uniform_16u(
                        c16, mks, chosen, blur_len, ang,
                        offset_px=boundary_offset_px, width_in_px=width_in_px, width_out_px=width_out_px,
                        hard_core_px=hard_core_px, curve=curve, gamma=gamma_val,
                        aspect=0.20, guide8=c8, extra_keep=extra_full
                    )
                else:
                    out16,_=composite_zonal_16u(
                        c16, p8, c8, mks, chosen, ang,
                        blur_len, offset_px=boundary_offset_px, width_in_px=width_in_px, width_out_px=width_out_px,
                        hard_core_px=hard_core_px, curve=curve, gamma=gamma_val,
                        profile=("Flow" if profile=="Flow" else "Radial"),
                        radial_soft=radial_soft, radial_power=radial_power, flow_power=flow_power,
                        aspect=0.20, guide8=c8, extra_keep=extra_full
                    )
                base=f"pan_{os.path.splitext(os.path.basename(f))[0]}"
                save_image(out16, st.session_state.run_dir, base, save_format)
                del out16; gc.collect()
                count+=1; prog.progress((i+1)/len(frames))
            st.success(f"Batch complete: saved {count} frames to {st.session_state.run_dir}")
