import os, glob, time, shlex, subprocess, tempfile, gc, base64
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

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def list_frames(dirpath):
    """Find all supported image files in a directory, sorted by name."""
    if not os.path.isdir(dirpath):
        return []
    files = []
    for f in os.listdir(dirpath):
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
            files.append(os.path.join(dirpath, f))
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
    status = st.status("Extracting frames...", expanded=True) if hasattr(st, "status") else None
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    lines = []
    for line in p.stderr:
        lines.append(line.rstrip())
        log.code("\n".join(lines[-40:]), language="bash")
    rc = p.wait()
    n = len(list_frames(frames_dir))
    if rc == 0 and n > 0:
        if status: status.update(label=f"Done - frames: {n}", state="complete")
        st.success(f"Extracted **{n}** frames to **{frames_dir}**")
    else:
        if status: status.update(label="Extraction failed", state="error")
        st.error("ffmpeg failed."); st.code("\n".join(lines[-80:]), language="bash")

# ----------------------- Color space (linear light) -----------------------
def srgb_to_linear(img16):
    x = (img16.astype(np.float32) / 65535.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb_f32(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0/2.4)) - 0.055)

# ----------------------- Highlight management -----------------------
def soft_clamp_highlights(lin, shoulder=0.8, max_val=4.0):
    """Soft-compress highlights above shoulder to prevent halo bleed during blur.
    Uses a filmic shoulder curve that preserves relative brightness ordering."""
    out = lin.copy()
    mask = lin > shoulder
    # Soft roll-off: shoulder + (1-shoulder) * tanh((x-shoulder) / (max_val-shoulder))
    scale = max_val - shoulder
    out[mask] = shoulder + (1.0 - shoulder) * np.tanh((lin[mask] - shoulder) / max(scale, 1e-6))
    return out

def restore_highlights(lin_original, lin_clamped_blurred, alpha):
    """In sharp regions (alpha~1), restore original highlights. In blurred regions, keep clamped."""
    a3 = alpha[..., None] if alpha.ndim == 2 else alpha
    return a3 * lin_original + (1.0 - a3) * lin_clamped_blurred

# ----------------------- Motion & flow -----------------------
def directional_gaussian_kernel(length_px, angle_deg, aspect=0.18):
    length_px = max(3, int(length_px))
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

def disc_kernel(radius):
    """Circular disc kernel for bokeh/depth-of-field simulation."""
    radius = max(1, int(radius))
    ksize = 2 * radius + 1
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = (xx**2 + yy**2 <= radius**2).astype(np.float32)
    k /= k.sum()
    return k

def estimate_flow(prev8, curr8, quality="fast"):
    pg = cv2.cvtColor(prev8, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(curr8, cv2.COLOR_BGR2GRAY)
    preset = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM if quality == "medium" else cv2.DISOPTICAL_FLOW_PRESET_FAST
    dis = cv2.DISOpticalFlow_create(preset)
    flow = dis.calc(pg, cg, None)
    mag = np.linalg.norm(flow, axis=2)
    ang = np.degrees(np.arctan2(flow[...,1], flow[...,0]))
    return flow, mag, ang

def estimate_flow_angle(prev8, curr8, subject_mask=None, quality="fast"):
    """Estimate dominant background flow angle, optionally excluding subject pixels."""
    flow, mag, _ = estimate_flow(prev8, curr8, quality=quality)
    thr = np.percentile(mag, 75)
    m = mag > thr
    if subject_mask is not None:
        # Exclude subject pixels from flow estimation
        bg = (subject_mask == 0)
        m = m & bg
    if m.sum() == 0: return 0.0
    dx = flow[...,0][m].mean(); dy = flow[...,1][m].mean()
    return np.degrees(np.arctan2(dy, dx))

def estimate_flow_full(prev8, curr8, subject_mask=None, quality="fast"):
    """Return flow field, magnitude, angle, and masked background angle."""
    flow, mag, ang = estimate_flow(prev8, curr8, quality=quality)
    thr = np.percentile(mag, 75)
    m = mag > thr
    if subject_mask is not None:
        m = m & (subject_mask == 0)
    if m.sum() == 0:
        bg_angle = 0.0
    else:
        dx = flow[...,0][m].mean(); dy = flow[...,1][m].mean()
        bg_angle = np.degrees(np.arctan2(dy, dx))
    return flow, mag, ang, bg_angle

def multi_frame_flow(frames_list, idx, img8, window=2, subject_mask=None, quality="fast"):
    """Average optical flow across multiple neighboring frame pairs for stability.
    Useful for low-fps stills where single-pair flow can be noisy."""
    H, W = img8.shape[:2]
    accum_flow = np.zeros((H, W, 2), dtype=np.float64)
    count = 0
    start = max(0, idx - window)
    end = min(len(frames_list) - 1, idx + window)
    for i in range(start, end):
        f_prev = cv2.imread(frames_list[i], cv2.IMREAD_UNCHANGED)
        f_curr = cv2.imread(frames_list[i+1], cv2.IMREAD_UNCHANGED)
        if f_prev is None or f_curr is None:
            continue
        p8 = (f_prev / 257).astype(np.uint8) if f_prev.dtype == np.uint16 else f_prev
        c8 = (f_curr / 257).astype(np.uint8) if f_curr.dtype == np.uint16 else f_curr
        if p8.shape[:2] != (H, W):
            p8 = cv2.resize(p8, (W, H))
            c8 = cv2.resize(c8, (W, H))
        flow_i, _, _ = estimate_flow(p8, c8, quality=quality)
        accum_flow += flow_i.astype(np.float64)
        count += 1
    if count == 0:
        return np.zeros((H, W, 2), dtype=np.float32), 0.0
    avg_flow = (accum_flow / count).astype(np.float32)
    mag = np.linalg.norm(avg_flow, axis=2)
    thr = np.percentile(mag, 75)
    m = mag > thr
    if subject_mask is not None:
        m = m & (subject_mask == 0)
    if m.sum() == 0:
        bg_angle = 0.0
    else:
        dx = avg_flow[...,0][m].mean(); dy = avg_flow[...,1][m].mean()
        bg_angle = np.degrees(np.arctan2(dy, dx))
    return avg_flow, bg_angle

# ----------------------- Subject inpainting -----------------------
def inpaint_subject(img, subject_mask, radius=5):
    """Fill subject region with surrounding background before blurring.
    Prevents subject ghost bleeding into the blurred background."""
    mask8 = (subject_mask > 0).astype(np.uint8) * 255
    # Dilate mask slightly to cover edge pixels
    mask8 = cv2.dilate(mask8, np.ones((3,3), np.uint8), iterations=2)
    if img.dtype in (np.float32, np.float64):
        # cv2.inpaint supports 8-bit 3ch or 32-bit float 1ch only;
        # convert to 8-bit, inpaint, then restore to float
        img_8u = np.clip(img * 255, 0, 255).astype(np.uint8)
        result_8u = cv2.inpaint(img_8u, mask8, radius, cv2.INPAINT_TELEA)
        return result_8u.astype(np.float32) / 255.0
    else:
        if img.dtype == np.uint16:
            img_8u = (img / 257).astype(np.uint8)
            result_8u = cv2.inpaint(img_8u, mask8, radius, cv2.INPAINT_TELEA)
            return result_8u.astype(np.uint16) * 257
        return cv2.inpaint(img, mask8, radius, cv2.INPAINT_TELEA)

# ----------------------- Per-patch flow-directed blur -----------------------
def angle_binned_blur(lin_img, flow, blur_len, aspect=0.18, n_bins=12, mag_scale=False):
    """Spatially-varying directional blur using angle bins from the flow field.
    Instead of one global blur direction, quantizes per-pixel flow angles into bins
    and applies a directional kernel per bin, blending with per-pixel weights."""
    mag = np.linalg.norm(flow, axis=2)
    ang = np.degrees(np.arctan2(flow[..., 1], flow[..., 0]))
    mag_max = mag.max() + 1e-6

    bin_width = 360.0 / n_bins
    bin_centers = np.linspace(-180 + bin_width/2, 180 - bin_width/2, n_bins)

    result = np.zeros_like(lin_img, dtype=np.float64)
    total_w = np.zeros(lin_img.shape[:2], dtype=np.float64)

    for center in bin_centers:
        # Triangular weight: 1 at center, 0 at +/- bin_width
        diff = np.abs(((ang - center + 180) % 360) - 180)
        w = np.maximum(0.0, 1.0 - diff / bin_width).astype(np.float64)

        if w.max() < 1e-6:
            continue

        if mag_scale:
            # Scale blur length by average magnitude in this bin's active region
            active = w > 0.1
            if active.any():
                local_mag = np.median(mag[active]) / mag_max
                local_len = max(3, int(blur_len * np.clip(local_mag, 0.1, 1.0)))
            else:
                local_len = blur_len
        else:
            local_len = blur_len

        k = directional_gaussian_kernel(local_len, center, aspect=aspect)
        blurred = cv2.filter2D(lin_img, -1, k)

        result += w[..., None] * blurred
        total_w += w

    total_w = np.maximum(total_w, 1e-6)
    return (result / total_w[..., None]).astype(np.float32)

# ----------------------- Segmentation (robust + cached) -----------------------
@st.cache_data(show_spinner=False)
def yolo_segment_cached(model_name, imgsz, frame_bgr8_bytes, hw):
    model = load_model(model_name)
    H, W = hw
    img8 = cv2.imdecode(np.frombuffer(frame_bgr8_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    r = model(img8, imgsz=imgsz, retina_masks=False, conf=0.25, classes=[0], verbose=False)[0]
    masks = []
    if r.masks is not None:
        # Get masks at proto resolution and upscale ourselves with bilinear
        # interpolation. retina_masks=True can introduce letterbox padding
        # artifacts (hard horizontal/vertical lines) at high imgsz values.
        m_float = r.masks.data.float().cpu().numpy()  # (N, proto_h, proto_w)
        for i in range(m_float.shape[0]):
            mi = m_float[i]
            # Always bilinear upscale soft probabilities, then threshold
            mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_LINEAR)
            mi = (mi > 0.5).astype(np.uint8)
            mi = cv2.morphologyEx(mi, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), 1)
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
    # Draw subject numbers
    for i,m in enumerate(masks):
        ys, xs = np.where(m > 0)
        if len(ys) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            label = str(i + 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.6, min(2.0, m.sum() / 50000))
            thick = max(1, int(scale * 2))
            (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
            cv2.putText(overlay, label, (cx - tw//2, cy + th//2), font, scale, (255,255,255), thick+2, cv2.LINE_AA)
            cv2.putText(overlay, label, (cx - tw//2, cy + th//2), font, scale, (0,0,0), thick, cv2.LINE_AA)
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
def dir_blend(img16, lin_blur, alpha, highlight_protect=False):
    """Blend sharp original with blurred background in linear light."""
    lin = srgb_to_linear(img16)
    if highlight_protect:
        lin_blur = restore_highlights(lin, lin_blur, alpha)
    lin_out = alpha[..., None] * lin + (1 - alpha[..., None]) * lin_blur
    srgb_out = np.clip(linear_to_srgb_f32(lin_out), 0, 1)
    return (srgb_out * 65535.0 + 0.5).astype(np.uint16)

def build_blur_image(lin_img, angle_deg, blur_len, aspect, flow=None,
                     bokeh_mix=0.0, use_flow_dir=False, mag_scale=False,
                     subject_mask=None, inpaint=False):
    """Central blur pipeline with all improvements applied.

    Args:
        lin_img: linear-light float32 image
        angle_deg: global dominant motion angle
        blur_len: blur length in pixels
        aspect: PSF aspect ratio
        flow: optical flow field (H,W,2) - needed for flow-directed blur
        bokeh_mix: 0-1 blend of disc blur mixed with directional blur
        use_flow_dir: use per-pixel flow angles instead of global angle
        mag_scale: scale blur length by local flow magnitude
        subject_mask: combined subject mask for inpainting
        inpaint: whether to inpaint subjects before blurring
    """
    # Step 1: Inpaint subject region to prevent ghost bleeding
    src = lin_img
    if inpaint and subject_mask is not None and subject_mask.any():
        src = inpaint_subject(src, subject_mask, radius=7)

    # Step 2: Soft-clamp highlights to prevent halo artifacts
    src = soft_clamp_highlights(src, shoulder=0.85)

    # Step 3: Apply directional blur (per-pixel or global)
    if use_flow_dir and flow is not None:
        dir_blurred = angle_binned_blur(src, flow, blur_len, aspect=aspect,
                                        n_bins=12, mag_scale=mag_scale)
    else:
        k = directional_gaussian_kernel(blur_len, angle_deg, aspect=aspect)
        dir_blurred = cv2.filter2D(src, -1, k)

    # Step 4: Optional bokeh disc blur blend
    if bokeh_mix > 0.01:
        bokeh_radius = max(1, int(blur_len * 0.3))
        dk = disc_kernel(bokeh_radius)
        disc_blurred = cv2.filter2D(src, -1, dk)
        blurred = (1.0 - bokeh_mix) * dir_blurred + bokeh_mix * disc_blurred
    else:
        blurred = dir_blurred

    return blurred.astype(np.float32)

def composite_uniform_16u(img16, masks, chosen, blur_len, angle,
                          offset_px=0.0, width_in_px=8.0, width_out_px=40.0, hard_core_px=16.0,
                          curve="Smoothstep", gamma=2.2,
                          aspect=0.18, guide8=None, extra_keep=None,
                          flow=None, bokeh_mix=0.0, use_flow_dir=False,
                          mag_scale=False, inpaint_bg=False, highlight_protect=False):
    H, W = img16.shape[:2]
    keep = np.zeros((H, W), np.uint8)
    for i in chosen: keep |= masks[i]
    if extra_keep is not None: keep |= (extra_keep > 0).astype(np.uint8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)

    alpha = alpha_signed_distance(keep, -offset_px, width_in_px, width_out_px,
                                  curve=curve, gamma=gamma, hard_core_px=hard_core_px)
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0,0), 1.25)
    if guide8 is not None: alpha = refine_alpha_with_guided(alpha, guide8, radius=12, eps=1e-3)

    lin = srgb_to_linear(img16)
    lin_blur = build_blur_image(lin, angle, blur_len, aspect,
                                flow=flow, bokeh_mix=bokeh_mix,
                                use_flow_dir=use_flow_dir, mag_scale=mag_scale,
                                subject_mask=keep if inpaint_bg else None,
                                inpaint=inpaint_bg)
    return dir_blend(img16, lin_blur, alpha, highlight_protect=highlight_protect), alpha

def make_blur_stack_linear(lin_img, angle_deg, levels, aspect=0.18,
                           flow=None, bokeh_mix=0.0, use_flow_dir=False,
                           mag_scale=False, subject_mask=None, inpaint=False):
    outs = []
    # Pre-process source once: inpaint + highlight clamp
    src = lin_img
    if inpaint and subject_mask is not None and subject_mask.any():
        src = inpaint_subject(src, subject_mask, radius=7)
    src = soft_clamp_highlights(src, shoulder=0.85)

    for L in levels:
        if L <= 0:
            outs.append(src.copy())
        else:
            if use_flow_dir and flow is not None:
                blurred = angle_binned_blur(src, flow, int(L), aspect=aspect,
                                            n_bins=12, mag_scale=mag_scale)
            else:
                k = directional_gaussian_kernel(int(L), angle_deg, aspect=aspect)
                blurred = cv2.filter2D(src, -1, k)
            if bokeh_mix > 0.01:
                bokeh_r = max(1, int(L * 0.3))
                dk = disc_kernel(bokeh_r)
                disc_b = cv2.filter2D(src, -1, dk)
                blurred = (1.0 - bokeh_mix) * blurred + bokeh_mix * disc_b
            outs.append(blurred.astype(np.float32))
    return outs

def radial_weights(H, W, center=None, softness=0.35, power=1.0, n=8):
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = (W//2, H//2) if center is None else center
    r = np.hypot(xx - cx, yy - cy); r = (r / (r.max() + 1e-6)) ** power
    edges = np.linspace(0, 1, n+1); ws = []; width = max(1e-6, softness * (edges[1] - edges[0]))
    for i in range(n):
        c = 0.5 * (edges[i] + edges[i+1])
        w = np.maximum(0.0, 1.0 - np.abs(r - c) / width); ws.append(w)
    Wmap = np.stack(ws, axis=-1); Wmap /= (Wmap.sum(axis=-1, keepdims=True) + 1e-6)
    return Wmap

def flow_weights(prev8, curr8, n=8, power=1.0, subject_mask=None, quality="fast"):
    _, mag, _ = estimate_flow(prev8, curr8, quality=quality)
    a = (mag / (mag.max() + 1e-6)) ** power
    edges = np.linspace(0, 1, n+1); ws = []; width = max(1e-6, 0.5 * (edges[1] - edges[0]))
    for i in range(n):
        c = 0.5 * (edges[i] + edges[i+1])
        w = np.maximum(0.0, 1.0 - np.abs(a - c) / width); ws.append(w)
    Wmap = np.stack(ws, axis=-1); Wmap /= (Wmap.sum(axis=-1, keepdims=True) + 1e-6)
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
                        aspect=0.18, guide8=None, extra_keep=None,
                        flow=None, bokeh_mix=0.0, use_flow_dir=False,
                        mag_scale=False, inpaint_bg=False, highlight_protect=False,
                        flow_quality="fast"):
    H, W = img16.shape[:2]
    keep = np.zeros((H, W), np.uint8)
    for i in chosen: keep |= masks[i]
    if extra_keep is not None: keep |= (extra_keep > 0).astype(np.uint8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)

    alpha = alpha_signed_distance(keep, -offset_px, width_in_px, width_out_px,
                                  curve=curve, gamma=gamma, hard_core_px=hard_core_px)
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0,0), 1.25)
    if guide8 is not None: alpha = refine_alpha_with_guided(alpha, guide8, radius=12, eps=1e-3)

    lin = srgb_to_linear(img16)
    levels = np.linspace(0, blur_len, 8).astype(int).tolist()
    stack = make_blur_stack_linear(lin, angle_deg, levels, aspect=aspect,
                                   flow=flow, bokeh_mix=bokeh_mix,
                                   use_flow_dir=use_flow_dir, mag_scale=mag_scale,
                                   subject_mask=keep if inpaint_bg else None,
                                   inpaint=inpaint_bg)

    if profile == "Flow":
        Wmap = flow_weights(prev8, curr8, n=len(stack), power=flow_power, quality=flow_quality)
    else:
        Wmap = radial_weights(H, W, center=None, softness=radial_soft, power=radial_power, n=len(stack))
    Wmap = smooth_weight_map(Wmap, sigma_px=7)

    out_bg = np.zeros_like(stack[0])
    for i in range(len(stack)): out_bg += (Wmap[..., i:i+1] * stack[i])
    return dir_blend(img16, out_bg, alpha, highlight_protect=highlight_protect), alpha

# ----------------------- Preview scaling helpers -----------------------
def compute_preview_scale(h, w, auto=True, ui_scale=0.5):
    if not auto: return float(ui_scale)
    return float(min(1.0, (MAX_PREVIEW_PIXELS / max(1, h*w)) ** 0.5))

def rescale(img, s, interp=cv2.INTER_AREA):
    if s >= 0.999: return img
    nh, nw = max(1, int(img.shape[0]*s)), max(1, int(img.shape[1]*s))
    return cv2.resize(img, (nw, nh), interpolation=interp)

# ----------------------- Presets -----------------------
PRESETS = {
    "Subtle pan":   dict(blur_len=25,  psf_aspect=0.18, boundary_offset_px=-4,  width_in_px=8,  width_out_px=40,  hard_core_px=8,  curve="Smoothstep"),
    "Medium motion":dict(blur_len=45,  psf_aspect=0.20, boundary_offset_px=-6,  width_in_px=10, width_out_px=70,  hard_core_px=12, curve="Smoothstep"),
    "Heavy blur":   dict(blur_len=100, psf_aspect=0.22, boundary_offset_px=-8,  width_in_px=14, width_out_px=90,  hard_core_px=20, curve="Smoothstep"),
    "Soft edges":   dict(blur_len=50,  psf_aspect=0.20, boundary_offset_px=0,   width_in_px=20, width_out_px=100, hard_core_px=6,  curve="Cosine"),
    "Custom":       None,
}

# ----------------------- UI -----------------------
st.set_page_config(page_title="Neuromotion - Action Pan", layout="wide")
st.title("Neuromotion - Action Pan")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    # Preset selector
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=1,
                               help="Choose a starting point, then fine-tune below. Select 'Custom' for full manual control.")
    preset = PRESETS[preset_name]

    # --- Segmentation ---
    with st.expander("Segmentation", expanded=False):
        model_name = st.selectbox("Model",
                                  ["yolo11n-seg.pt","yolo11s-seg.pt","yolo11m-seg.pt","yolo11l-seg.pt"], 1,
                                  help="Larger models are more accurate but slower")
        imgsz = st.slider("Detection quality", 640, 1920, 1280, 64,
                          help="Higher values detect better but run slower")

    # --- Blur ---
    with st.expander("Blur", expanded=True):
        _bl_default = preset["blur_len"] if preset else 45
        _pa_default = preset["psf_aspect"] if preset else 0.20
        blur_len = st.slider("Blur strength (px)", 3, 151, _bl_default, 2,
                             help="How much motion blur to apply to the background")
        psf_aspect = st.slider("Blur shape (thin / wide)", 0.08, 0.35, _pa_default, 0.01,
                               help="Low = thin streak, high = wider smear")
        bokeh_mix = st.slider("Bokeh mix", 0.0, 1.0, 0.0, 0.05,
                              help="Blend disc/bokeh blur with directional blur. "
                                   "Adds depth-of-field look to the motion blur.")

    # --- Flow & Motion ---
    with st.expander("Flow & motion", expanded=False):
        use_flow_dir = st.checkbox("Per-pixel flow direction", value=False,
                                   help="Use optical flow to vary blur direction across the frame. "
                                        "More realistic but slower. Without this, one global angle is used.")
        mag_scale = st.checkbox("Flow-scaled blur magnitude", value=False,
                                help="Scale blur strength by local motion amount. "
                                     "Areas with less motion get less blur.")
        flow_quality_label = st.selectbox("Flow quality", ["Fast", "Medium"], 0,
                                          help="Medium is better for low-fps stills with large motion between frames")
        flow_quality = "medium" if flow_quality_label == "Medium" else "fast"
        use_multi_frame = st.checkbox("Multi-frame flow averaging", value=False,
                                      help="Average optical flow across neighboring frames for stability. "
                                           "Recommended for low-fps stills (e.g. 12fps burst photos).")
        multi_frame_window = st.slider("Averaging window (frames)", 1, 5, 2, 1,
                                       disabled=not use_multi_frame,
                                       help="Number of frame pairs to average in each direction")
        inpaint_bg = st.checkbox("Inpaint subjects before blur", value=False,
                                 help="Fill subject areas with background before blurring. "
                                      "Prevents ghost/halo artifacts at subject edges.")
        highlight_protect = st.checkbox("Highlight protection", value=True,
                                        help="Soft-clamp bright highlights before blurring to prevent "
                                             "glowing halos. Restores original highlights in sharp regions.")

    # --- Edge Transition ---
    with st.expander("Edge transition", expanded=False):
        _bo = preset["boundary_offset_px"] if preset else -6
        _wi = preset["width_in_px"] if preset else 10
        _wo = preset["width_out_px"] if preset else 70
        _hc = preset["hard_core_px"] if preset else 12
        _cu = preset["curve"] if preset else "Smoothstep"

        boundary_offset_px = st.slider("Edge position (shrink / expand)", -60, 60, _bo, 1,
                                       help="Negative shrinks the sharp zone inward, positive expands it outward")
        width_in_px  = st.slider("Inner edge softness (px)", 0, 60, _wi, 1,
                                 help="How gradually the sharp zone fades on the inside edge")
        width_out_px = st.slider("Outer edge softness (px)", 5, 120, _wo, 1,
                                 help="How gradually the blur fades in from outside the subject")
        hard_core_px = st.slider("Guaranteed sharp zone (px)", 0, 80, _hc, 1,
                                 help="Pixels this far inside the subject edge are always 100% sharp")
        curve_options = ["Linear","Smoothstep","Cosine","Gamma"]
        curve = st.selectbox("Transition curve", curve_options, curve_options.index(_cu),
                             help="Shape of the blend between sharp and blurred regions")
        gamma_val = st.slider("Gamma", 0.5, 4.0, 2.2, 0.1,
                              help="Controls the transition curve shape") if curve=="Gamma" else 2.2
        show_alpha = st.checkbox("Show alpha matte (debug)", value=False,
                                 help="Visualize the sharp/blur boundary mask")

    # --- Brush ---
    with st.expander("Brush (keep-sharp region)", expanded=False):
        use_custom_region = st.checkbox("Enable brush", value=False,
                                        help="Paint extra areas to keep sharp beyond auto-detection")
        apply_brush_on_save = st.checkbox("Apply brush on Save/Batch", value=True, disabled=not use_custom_region,
                                          help="Include brush strokes when saving full-resolution output")
        brush_size = st.slider("Brush size", 3, 80, 28, 1, disabled=not use_custom_region,
                               help="Size of the painting brush in pixels")

    # --- Blur Profile ---
    with st.expander("Blur profile", expanded=False):
        profile = st.selectbox("Profile type", ["Uniform","Radial","Flow"], 0,
                               help="Uniform: same blur everywhere. Radial: varies by distance from center. Flow: varies by motion amount.")
        radial_soft = st.slider("Radial softness", 0.1, 1.0, 0.35, 0.05,
                                help="How smoothly the blur zones blend together") if profile=="Radial" else 0.35
        radial_power = st.slider("Radial power", 0.5, 2.5, 1.0, 0.1,
                                 help="Higher = more aggressive radial falloff") if profile=="Radial" else 1.0
        flow_power = st.slider("Flow power", 0.5, 2.5, 1.0, 0.1,
                               help="Higher = blur follows motion more aggressively") if profile=="Flow" else 1.0

    # --- Preview ---
    with st.expander("Preview scaling", expanded=False):
        auto_preview = st.checkbox("Auto scale (target ~2MP)", True,
                                   help="Automatically scale preview for responsive editing")
        preview_scale_ui = st.slider("Manual scale", 0.25, 1.0, 0.5, 0.05,
                                     disabled=auto_preview,
                                     help="Manual preview resolution (1.0 = full size)")

    # --- Output ---
    with st.expander("Output", expanded=False):
        save_format = st.selectbox("Save format", ["JPEG (8-bit)","TIFF (16-bit)"], 1,
                                   help="TIFF preserves full quality, JPEG is smaller")
        batch_mode = st.selectbox("Batch subject selection", ["interactive","largest","first"], 0,
                                  help="Interactive: use current selection. Largest: auto-pick biggest subject. First: use your selection for all frames.")

    # --- Image Source (at the bottom, collapsed) ---
    with st.expander("Image source", expanded=False):
        source_mode = st.radio("Source type", ["Video extraction", "Stills folder"],
                               help="Use 'Video extraction' to pull frames from a video file. "
                                    "Use 'Stills folder' to load a sequence of photos (e.g. burst/continuous shooting).")

        if source_mode == "Video extraction":
            uploaded = st.file_uploader("Upload video (MP4/MOV)", type=["mp4","mov","MP4","MOV"])
            frames_dir = st.text_input("Output frames directory", "data/extracted_frames/C0031",
                                       help="Where to save extracted frames")
            ext_label = st.selectbox("Extract as", ["TIFF 16-bit","PNG (lossless 8-bit)","JPEG 8-bit (HQ)"], 0)
            fmt_key = "tiff16" if ext_label.startswith("TIFF") else ("png" if ext_label.startswith("PNG") else "jpg")
            tone_map = st.checkbox("Tone-map HDR/HLG to SDR", False,
                                   help="Enable if your source video is HDR and needs conversion")
            go_extract = st.button("Extract frames")
        else:
            uploaded = None
            go_extract = False
            frames_dir = st.text_input("Stills folder path", "data/stills",
                                       help="Path to a folder of sequential photos (JPG, PNG, TIFF). "
                                            "Files are sorted alphabetically, so name them in order "
                                            "(e.g. DSC_0001.jpg, DSC_0002.jpg, ...).")
            n_found = len(list_frames(frames_dir))
            if n_found > 0:
                st.success(f"Found **{n_found}** images in `{frames_dir}`")
            elif os.path.isdir(frames_dir):
                st.warning(f"No supported images found in `{frames_dir}` (looking for JPG, PNG, TIFF)")
            fmt_key = None
            tone_map = False

# Output dir
clip_name = os.path.basename(os.path.normpath(frames_dir)) or "session"
run_dir = os.path.join("outputs","action_pan",clip_name, datetime.now().strftime("%Y%m%d_%H%M%S"))
if "run_dir" not in st.session_state: st.session_state.run_dir=run_dir
ensure_dir(st.session_state.run_dir)

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
if len(frames)<2: st.info(f"**{len(frames)}** images found. Use the **Image source** section in the sidebar to extract frames from a video or point to a folder of stills."); st.stop()

# --- Frame navigation (full width, above columns) ---
st.markdown(f"**{len(frames)} frames** from `{frames_dir}` | Output: `{st.session_state.run_dir}`")

if "frame_idx" not in st.session_state: st.session_state.frame_idx = 0

nav_prev, nav_slider, nav_next = st.columns([1, 10, 1])
with nav_prev:
    if st.button("Prev", use_container_width=True):
        st.session_state.frame_idx = max(0, st.session_state.frame_idx - 1)
with nav_next:
    if st.button("Next", use_container_width=True):
        st.session_state.frame_idx = min(len(frames)-1, st.session_state.frame_idx + 1)
with nav_slider:
    idx = st.slider("Frame", 0, len(frames)-1, st.session_state.frame_idx, 1,
                    label_visibility="collapsed")
    st.session_state.frame_idx = idx

idx = st.session_state.frame_idx

# Load full-res
fpath = frames[idx]
prev_path = frames[idx-1] if idx>0 else frames[idx]

with st.spinner("Loading frame..."):
    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED);  img16 = img if img.dtype==np.uint16 else (img.astype(np.uint16)*257)
    prv = cv2.imread(prev_path, cv2.IMREAD_UNCHANGED); prev16 = prv if prv.dtype==np.uint16 else (prv.astype(np.uint16)*257)
    img8  = (img16/257).astype(np.uint8); prev8=(prev16/257).astype(np.uint8)
    H,W = img8.shape[:2]

# Preview scaling
PREVIEW_S = min(1.0, compute_preview_scale(H,W, auto_preview, preview_scale_ui))
def RS(x, s, interp=cv2.INTER_AREA): return x if s>=0.999 else cv2.resize(x,(int(W*s),int(H*s)),interpolation=interp)
p_img16=RS(img16,PREVIEW_S); p_img8=RS(img8,PREVIEW_S); p_prev8=RS(prev8,PREVIEW_S)

# Masks (cached)
with st.spinner("Detecting subjects..."):
    masks = segment_people(model_name, imgsz, img8)
p_masks = [RS(m.astype(np.uint8), PREVIEW_S, cv2.INTER_NEAREST) for m in masks]

# Build combined subject mask for flow estimation
subject_mask_full = np.zeros((H,W), np.uint8)
for m in masks: subject_mask_full |= m
p_subject_mask = RS(subject_mask_full, PREVIEW_S, cv2.INTER_NEAREST)

# Flow estimation (subject-aware, optionally multi-frame)
with st.spinner("Estimating motion..."):
    if use_multi_frame:
        p_flow, angle = multi_frame_flow(frames, idx, prev8, window=multi_frame_window,
                                         subject_mask=subject_mask_full, quality=flow_quality)
        # Scale flow to preview size
        if PREVIEW_S < 0.999:
            pH, pW = p_img8.shape[:2]
            p_flow = cv2.resize(p_flow, (pW, pH)) * PREVIEW_S
    else:
        p_flow_full, _, _, angle = estimate_flow_full(prev8, img8,
                                                       subject_mask=subject_mask_full,
                                                       quality=flow_quality)
        if PREVIEW_S < 0.999:
            pH, pW = p_img8.shape[:2]
            p_flow = cv2.resize(p_flow_full, (pW, pH)) * PREVIEW_S
        else:
            p_flow = p_flow_full

# --- Main layout: left panel (controls) + right panel (preview) ---
colA, colB = st.columns([2, 5], gap="large")

# Brush state containers
if "brush_json" not in st.session_state: st.session_state.brush_json={}
if "brush_mask_full" not in st.session_state: st.session_state.brush_mask_full={}
if "brush_mask_prev" not in st.session_state: st.session_state.brush_mask_prev={}

# ---- Brush area ----
custom_keep_mask=None; p_extra=None
with colA:
    # Subject selection with numbered labels
    if masks:
        areas=[int(m.sum()) for m in masks]
        opts=[f"Subject {i+1} (area {a:,})" for i,a in enumerate(areas)]
        default=[]
        if batch_mode!="interactive" and not st.session_state.get("selected", set()) and len(areas)>0:
            default=[opts[int(np.argmax(areas))]]
        keep = st.multiselect("Keep sharp", opts, default=default, key="msel",
                              help="Select which detected people should remain sharp. Numbers match the overlay.")
        selected=set([int(s.split()[1])-1 for s in keep]); st.session_state.selected=selected
    else:
        st.info("No subjects detected in this frame.")
        selected = set(); st.session_state.selected = selected

    # Masks overlay with numbered subjects
    if masks:
        overlay_img = paint_overlay(p_img8, p_masks, selected)
        st.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB),
                 caption="Subject detection", use_container_width=True)

    if use_custom_region:
        st.markdown("**Brush: paint areas to keep sharp**")

        p_h, p_w = p_img8.shape[:2]

        # Encode reference frame as base64 for CSS background
        bg_rgb = cv2.cvtColor(p_img8, cv2.COLOR_BGR2RGB)
        _, png_buf = cv2.imencode(".png", cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR))
        b64_bg = base64.b64encode(png_buf.tobytes()).decode()

        # CSS overlay: reference image behind the canvas
        st.markdown(
            f"""<style>
            div[data-testid="stVerticalBlock"] div[data-testid="element-container"]:has(canvas) {{
                background-image: url("data:image/png;base64,{b64_bg}");
                background-size: {p_w}px {p_h}px;
                background-repeat: no-repeat;
            }}
            </style>""",
            unsafe_allow_html=True,
        )

        # Restore JSON scene if exists
        init_json = st.session_state.brush_json.get(fpath, None)

        canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=brush_size,
            stroke_color="rgba(255, 80, 80, 0.7)",
            background_color="rgba(0, 0, 0, 0)",
            update_streamlit=False,
            height=p_h, width=p_w,
            drawing_mode="freedraw",
            initial_drawing=init_json,
            key=f"canvas_{idx}",
        )

        btn_apply, btn_clear = st.columns(2)
        with btn_apply:
            apply_brush_now = st.button("Apply brush", disabled=not use_custom_region,
                                        use_container_width=True,
                                        help="Apply current brush strokes to the preview")
        with btn_clear:
            clear_brush = st.button("Clear strokes", disabled=not use_custom_region,
                                    use_container_width=True)

        if clear_brush:
            st.session_state.brush_json.pop(fpath, None)
            st.session_state.brush_mask_full.pop(fpath, None)
            st.session_state.brush_mask_prev.pop(fpath, None)

        # Apply brush -> build masks, persist JSON
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

        # Show brush overlay status
        if p_extra is not None:
            brush_px = int(p_extra.sum())
            status_label = "Applied" if apply_brush_on_save else "Preview only (not applied on save)"
            st.caption(f"Brush: {brush_px:,} px painted - {status_label}")

# Common compositor kwargs
_comp_kwargs = dict(
    offset_px=boundary_offset_px, width_in_px=width_in_px, width_out_px=width_out_px,
    hard_core_px=hard_core_px, curve=curve, gamma=gamma_val,
    aspect=psf_aspect, bokeh_mix=bokeh_mix,
    use_flow_dir=use_flow_dir, mag_scale=mag_scale,
    inpaint_bg=inpaint_bg, highlight_protect=highlight_protect,
)

# ---- PREVIEW compositing (small) ----
extra_preview = p_extra if (use_custom_region and p_extra is not None) else None
if masks:
    with st.spinner("Computing preview..."):
        t0 = time.time()
        if profile=="Uniform":
            p_out16, p_alpha = composite_uniform_16u(
                p_img16, p_masks, selected, blur_len, angle,
                guide8=p_img8, extra_keep=extra_preview, flow=p_flow,
                **_comp_kwargs
            )
        else:
            p_out16, p_alpha = composite_zonal_16u(
                p_img16, p_prev8, p_img8, p_masks, selected, angle,
                blur_len, guide8=p_img8, extra_keep=extra_preview, flow=p_flow,
                profile=("Flow" if profile=="Flow" else "Radial"),
                radial_soft=radial_soft, radial_power=radial_power, flow_power=flow_power,
                flow_quality=flow_quality,
                **_comp_kwargs
            )
        elapsed = time.time() - t0

    with colB:
        # Before/After tabs
        tab_result, tab_compare, tab_details = st.tabs(["Result", "Before / After", "Details"])

        with tab_result:
            st.image(cv2.cvtColor((p_out16/257).astype(np.uint8), cv2.COLOR_BGR2RGB),
                     use_container_width=True, caption="Action Pan composite")
            features = []
            if use_flow_dir: features.append("flow-dir")
            if mag_scale: features.append("mag-scale")
            if inpaint_bg: features.append("inpaint")
            if highlight_protect: features.append("hi-protect")
            if bokeh_mix > 0.01: features.append(f"bokeh:{bokeh_mix:.0%}")
            if use_multi_frame: features.append(f"multi-frame:{multi_frame_window}")
            feat_str = " | " + ", ".join(features) if features else ""
            st.caption(f"Frame {idx+1}/{len(frames)} | {os.path.basename(fpath)} | "
                       f"Flow: {angle:.1f} deg | Scale: {PREVIEW_S:.0%} | "
                       f"Render: {elapsed:.2f}s{feat_str}")

        with tab_compare:
            cmp1, cmp2 = st.columns(2)
            with cmp1:
                st.image(cv2.cvtColor(p_img8, cv2.COLOR_BGR2RGB),
                         use_container_width=True, caption="Original")
            with cmp2:
                st.image(cv2.cvtColor((p_out16/257).astype(np.uint8), cv2.COLOR_BGR2RGB),
                         use_container_width=True, caption="Action Pan")

        with tab_details:
            d1, d2 = st.columns(2)
            with d1:
                st.image(cv2.cvtColor(paint_overlay(p_img8, p_masks, selected), cv2.COLOR_BGR2RGB),
                         use_container_width=True, caption="Subject detection overlay")
            with d2:
                if show_alpha:
                    st.image((p_alpha*255).astype(np.uint8), clamp=True, use_container_width=True,
                             caption="Alpha matte (white = sharp)")
                else:
                    st.info("Enable 'Show alpha matte' in Edge transition settings to see the alpha map here.")

# ---- SAVE (full-res recompute) ----
def save_image(out16, base_dir, name, fmt_choice):
    ensure_dir(base_dir)
    if fmt_choice.startswith("TIFF"):
        path=os.path.join(base_dir,f"{name}.tiff"); cv2.imwrite(path,out16)
    else:
        out8=(out16/257).astype(np.uint8)
        path=os.path.join(base_dir,f"{name}.jpg"); cv2.imwrite(path,out8)
    return path

def compute_full_flow(frames_list, i, c8, p8, sub_mask, use_mf, mf_window, fq):
    """Compute flow field and angle at full resolution for save/batch."""
    if use_mf:
        flow_full, ang = multi_frame_flow(frames_list, i, p8, window=mf_window,
                                          subject_mask=sub_mask, quality=fq)
    else:
        flow_full, _, _, ang = estimate_flow_full(p8, c8, subject_mask=sub_mask, quality=fq)
    return flow_full, ang

with colA:
    save_col, batch_col = st.columns(2)
    with save_col:
        do_save = st.button("Save frame", use_container_width=True,
                            help="Save the current frame at full resolution")
    with batch_col:
        do_batch = st.button("Run batch", use_container_width=True,
                             help="Process all frames with current settings")

    if do_save:
        with st.spinner("Saving full-resolution frame..."):
            extra_full = custom_keep_mask if (use_custom_region and apply_brush_on_save and custom_keep_mask is not None) else None
            flow_full, angle_full = compute_full_flow(
                frames, idx, img8, prev8, subject_mask_full,
                use_multi_frame, multi_frame_window, flow_quality)
            if profile=="Uniform":
                out16,_=composite_uniform_16u(
                    img16, masks, selected, blur_len, angle_full,
                    guide8=img8, extra_keep=extra_full, flow=flow_full,
                    **_comp_kwargs
                )
            else:
                out16,_=composite_zonal_16u(
                    img16, prev8, img8, masks, selected, angle_full,
                    blur_len, guide8=img8, extra_keep=extra_full, flow=flow_full,
                    profile=("Flow" if profile=="Flow" else "Radial"),
                    radial_soft=radial_soft, radial_power=radial_power, flow_power=flow_power,
                    flow_quality=flow_quality,
                    **_comp_kwargs
                )
            base=f"pan_{os.path.splitext(os.path.basename(frames[idx]))[0]}"
            path=save_image(out16, st.session_state.run_dir, base, save_format)
            del out16; gc.collect()
            st.success(f"Saved: {path}")

# -------------------- BATCH (full-res) --------------------
def choose_indices(mks, mode, selected_now):
    if mode=="largest": return [int(np.argmax([m.sum() for m in mks]))]
    if mode=="first":   return list(selected_now) if selected_now else []
    return list(selected_now)

with colA:
    if do_batch:
        if batch_mode=="interactive":
            st.info("Switch batch mode to 'largest' or 'first' to enable batch processing.")
        elif batch_mode=="first" and not selected:
            st.error("Select at least one subject before running batch with 'first' mode.")
        else:
            prog=st.progress(0, text="Starting batch..."); count=0
            for i,f in enumerate(frames):
                prog.progress((i)/len(frames), text=f"Processing frame {i+1}/{len(frames)}...")
                curr=cv2.imread(f,cv2.IMREAD_UNCHANGED)
                prev=cv2.imread(frames[i-1],cv2.IMREAD_UNCHANGED) if i>0 else curr
                c16=curr if curr.dtype==np.uint16 else (curr.astype(np.uint16)*257)
                p16=prev if prev.dtype==np.uint16 else (prev.astype(np.uint16)*257)
                c8=(c16/257).astype(np.uint8); p8=(p16/257).astype(np.uint8)

                mks = segment_people(model_name, imgsz, c8)
                if not mks: prog.progress((i+1)/len(frames)); continue

                # Build subject mask for this frame
                sub_mask = np.zeros(c8.shape[:2], np.uint8)
                for mk in mks: sub_mask |= mk

                flow_full, ang = compute_full_flow(
                    frames, i, c8, p8, sub_mask,
                    use_multi_frame, multi_frame_window, flow_quality)
                chosen = choose_indices(mks, batch_mode, selected)

                extra_full=None
                if use_custom_region and apply_brush_on_save:
                    extra_full=None

                if profile=="Uniform":
                    out16,_=composite_uniform_16u(
                        c16, mks, chosen, blur_len, ang,
                        guide8=c8, extra_keep=extra_full, flow=flow_full,
                        **_comp_kwargs
                    )
                else:
                    out16,_=composite_zonal_16u(
                        c16, p8, c8, mks, chosen, ang,
                        blur_len, guide8=c8, extra_keep=extra_full, flow=flow_full,
                        profile=("Flow" if profile=="Flow" else "Radial"),
                        radial_soft=radial_soft, radial_power=radial_power, flow_power=flow_power,
                        flow_quality=flow_quality,
                        **_comp_kwargs
                    )
                base=f"pan_{os.path.splitext(os.path.basename(f))[0]}"
                save_image(out16, st.session_state.run_dir, base, save_format)
                del out16; gc.collect()
                count+=1; prog.progress((i+1)/len(frames))
            st.success(f"Batch complete: saved {count} frames to `{st.session_state.run_dir}`")
