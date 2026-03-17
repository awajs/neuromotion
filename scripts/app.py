import os, glob, time, shlex, subprocess, tempfile, gc
from datetime import datetime
import numpy as np
import cv2
import streamlit as st
import torch
import torch.nn.functional as F
from ultralytics import YOLO

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

RAW_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".orf", ".raf", ".rw2", ".pef", ".srw"}

from PIL import Image as PILImage

from streamlit_drawable_canvas import st_canvas

# ----------------------- Performance knobs -----------------------
cv2.setNumThreads(1)                  # avoid OpenCV spawning
MAX_PREVIEW_PIXELS = 2_000_000        # ~2MP auto preview target
THUMB_SIZE = 120                      # thumbnail pixel size for browser

# ----------------------- GPU helpers (torch CUDA) -----------------------
_GPU_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_GPU = _GPU_DEV.type == "cuda"

def _to_t(arr, dtype=torch.float32):
    """NumPy HWC → torch NCHW on GPU."""
    if arr.ndim == 2:
        return torch.from_numpy(arr).to(device=_GPU_DEV, dtype=dtype).unsqueeze(0).unsqueeze(0)
    return torch.from_numpy(arr).to(device=_GPU_DEV, dtype=dtype).permute(2, 0, 1).unsqueeze(0)

def _from_t(t):
    """Torch NCHW → NumPy HWC float32."""
    t = t.squeeze(0)
    if t.ndim == 2:
        return t.cpu().numpy().astype(np.float32)
    return t.permute(1, 2, 0).cpu().numpy().astype(np.float32)

def gpu_srgb_to_linear(img16):
    """sRGB uint16 → linear float32 on GPU."""
    x = torch.from_numpy(img16.astype(np.float32) / 65535.0).to(_GPU_DEV)
    out = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return out.cpu().numpy()

def gpu_linear_to_srgb(x_np):
    """Linear float32 → sRGB float32 on GPU."""
    x = torch.from_numpy(x_np).to(_GPU_DEV)
    out = torch.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0/2.4)) - 0.055)
    return out.cpu().numpy()

def gpu_filter2d(img_np, kernel_np):
    """cv2.filter2D equivalent using torch conv2d on GPU."""
    img_t = _to_t(img_np)  # (1, C, H, W)
    k = torch.from_numpy(kernel_np).to(device=_GPU_DEV, dtype=torch.float32)
    k = k.unsqueeze(0).unsqueeze(0)  # (1, 1, kH, kW)
    C = img_t.shape[1]
    k = k.expand(C, 1, -1, -1)  # (C, 1, kH, kW) — depthwise
    pad_h, pad_w = kernel_np.shape[0] // 2, kernel_np.shape[1] // 2
    out = F.conv2d(img_t, k, padding=(pad_h, pad_w), groups=C)
    return _from_t(out)

def gpu_blend(lin_np, lin_blur_np, alpha_np):
    """Alpha blend on GPU: a*sharp + (1-a)*blur."""
    lin = torch.from_numpy(lin_np).to(_GPU_DEV)
    blur = torch.from_numpy(lin_blur_np).to(_GPU_DEV)
    a = torch.from_numpy(alpha_np).to(_GPU_DEV)
    if a.ndim == 2:
        a = a.unsqueeze(-1)
    out = a * lin + (1.0 - a) * blur
    return out.cpu().numpy()

def gpu_soft_clamp(lin_np, shoulder=0.8, max_val=4.0):
    """Soft-clamp highlights on GPU."""
    t = torch.from_numpy(lin_np).to(_GPU_DEV)
    scale = max_val - shoulder
    clamped = shoulder + (1.0 - shoulder) * torch.tanh((t - shoulder) / max(scale, 1e-6))
    out = torch.where(t > shoulder, clamped, t)
    return out.cpu().numpy()

# ----------------------- FS helpers -----------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"} | RAW_EXTS


def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    """Read an image file, with RAW format support via rawpy."""
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTS:
        if not HAS_RAWPY:
            raise ImportError("rawpy is required to read RAW files. Install with: pip install rawpy")
        with rawpy.imread(path) as raw:
            rgb16 = raw.postprocess(
                use_camera_wb=True,
                output_bps=16,
                no_auto_bright=True,
            )
        # rawpy returns RGB uint16, convert to BGR for OpenCV consistency
        return cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR)
    return cv2.imread(path, flags)

def list_frames(dirpath):
    """Find all supported image files in a directory, sorted by name."""
    if not os.path.isdir(dirpath):
        return []
    files = []
    try:
        entries = os.listdir(dirpath)
    except OSError:
        return []
    for f in entries:
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
            files.append(os.path.join(dirpath, f))
    return sorted(files)

# ----------------------- Thumbnail helpers -----------------------
@st.cache_data(show_spinner=False)
def make_thumbnail(path, size=THUMB_SIZE):
    """Return a small RGB thumbnail for the contact sheet."""
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTS:
        if not HAS_RAWPY:
            return np.zeros((size, size, 3), dtype=np.uint8)
        try:
            with rawpy.imread(path) as raw:
                thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                arr = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            else:
                arr = thumb.data  # already RGB
        except Exception:
            # Fallback: full postprocess then downscale
            with rawpy.imread(path) as raw:
                arr = raw.postprocess(use_camera_wb=True, half_size=True, output_bps=8)
    else:
        arr = cv2.imread(path, cv2.IMREAD_COLOR)
        if arr is None:
            return np.zeros((size, size, 3), dtype=np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    h, w = arr.shape[:2]
    scale = size / max(h, w)
    arr = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))),
                     interpolation=cv2.INTER_AREA)
    return arr

def list_subdirs(dirpath):
    """List immediate subdirectories sorted by name."""
    if not os.path.isdir(dirpath):
        return []
    return sorted([d for d in os.listdir(dirpath)
                   if os.path.isdir(os.path.join(dirpath, d)) and not d.startswith(".")])

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
    if HAS_GPU:
        return gpu_srgb_to_linear(img16)
    x = (img16.astype(np.float32) / 65535.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb_f32(x):
    if HAS_GPU:
        return gpu_linear_to_srgb(x)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0/2.4)) - 0.055)

# ----------------------- Highlight management -----------------------
def soft_clamp_highlights(lin, shoulder=0.8, max_val=4.0, inplace=False):
    """Soft-compress highlights above shoulder to prevent halo bleed during blur.
    Uses a filmic shoulder curve that preserves relative brightness ordering.
    If inplace=True, modifies lin directly to avoid a full-image copy."""
    if HAS_GPU and not inplace:
        return gpu_soft_clamp(lin, shoulder, max_val)
    out = lin if inplace else lin.copy()
    mask = out > shoulder
    scale = max_val - shoulder
    out[mask] = shoulder + (1.0 - shoulder) * np.tanh((out[mask] - shoulder) / max(scale, 1e-6))
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
        f_prev = read_image(frames_list[i])
        f_curr = read_image(frames_list[i+1])
        if f_prev is None or f_curr is None:
            continue
        p8 = (f_prev / 257).astype(np.uint8) if f_prev.dtype == np.uint16 else f_prev
        c8 = (f_curr / 257).astype(np.uint8) if f_curr.dtype == np.uint16 else f_curr
        del f_prev, f_curr  # free full-res frames immediately
        if p8.shape[:2] != (H, W):
            p8 = cv2.resize(p8, (W, H))
            c8 = cv2.resize(c8, (W, H))
        flow_i, _, _ = estimate_flow(p8, c8, quality=quality)
        del p8, c8
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

def multi_frame_bg_average(frames_list, idx, subject_mask, window=2):
    """Average pixel values from neighboring frames to capture real background motion.

    Reads frames within [idx-window, idx+window], converts to linear light,
    and averages only the background (non-subject) region. The subject region
    is filled from the current frame so compositing can overlay it cleanly.

    Returns a linear-light float32 image (H,W,3) and the count of frames averaged.
    """
    curr16 = read_image(frames_list[idx])
    if curr16 is None:
        return None, 0
    if curr16.dtype != np.uint16:
        curr16 = (curr16.astype(np.uint16) * 257)
    H, W = curr16.shape[:2]
    lin_curr = srgb_to_linear(curr16)

    accum = np.zeros((H, W, 3), dtype=np.float64)
    count = 0
    start = max(0, idx - window)
    end = min(len(frames_list) - 1, idx + window)
    for i in range(start, end + 1):
        f16 = read_image(frames_list[i])
        if f16 is None:
            continue
        if f16.dtype != np.uint16:
            f16 = (f16.astype(np.uint16) * 257)
        if f16.shape[:2] != (H, W):
            f16 = cv2.resize(f16, (W, H))
        lin_f = srgb_to_linear(f16)
        del f16
        accum += lin_f.astype(np.float64)
        count += 1
        del lin_f

    if count == 0:
        return lin_curr, 1

    avg_lin = (accum / count).astype(np.float32)
    del accum

    # Keep subject region from the current frame so the composite overlay is clean
    if subject_mask is not None and subject_mask.any():
        sm3 = subject_mask[..., None].astype(np.float32)
        avg_lin = avg_lin * (1.0 - sm3) + lin_curr * sm3
        del sm3

    return avg_lin, count

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

    result = np.zeros_like(lin_img, dtype=np.float32)
    total_w = np.zeros(lin_img.shape[:2], dtype=np.float32)

    for center in bin_centers:
        # Triangular weight: 1 at center, 0 at +/- bin_width
        diff = np.abs(((ang - center + 180) % 360) - 180)
        w = np.maximum(0.0, 1.0 - diff / bin_width).astype(np.float32)

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
    result /= total_w[..., None]
    return result

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

# ----------------------- Face detection -----------------------
@st.cache_resource
def _load_face_detector():
    """Load OpenCV's DNN face detector (ships with opencv-python)."""
    proto = cv2.data.haarcascades  # just for path reference
    # Use the more accurate DNN model if available, fall back to Haar
    try:
        net = cv2.FaceDetectorYN.create(
            cv2.samples.findFile("face_detection_yunet_2023mar.onnx", required=False) or "",
            "", (0, 0))
    except Exception:
        net = None
    return net

def _detect_faces_haar(img8, scale_factor=1.15, min_neighbors=5):
    """Haar cascade face detection (always available in OpenCV)."""
    gray = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                      minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces  # Nx4 array of (x, y, w, h) or empty tuple

@st.cache_data(show_spinner=False)
def detect_faces_cached(frame_bgr8_bytes, hw, padding=0.35):
    """Detect faces and return elliptical masks with padding around each face."""
    H, W = hw
    img8 = cv2.imdecode(np.frombuffer(frame_bgr8_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    faces = _detect_faces_haar(img8)
    masks = []
    if len(faces) == 0:
        return masks
    for (x, y, w, h) in faces:
        # Expand bounding box by padding factor
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        cx = x + w // 2
        cy = y + h // 2
        rx = w // 2 + pad_x
        ry = h // 2 + pad_y
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
        masks.append(mask)
    return masks

def segment_faces(img8, padding=0.35):
    _, buf = cv2.imencode(".jpg", img8, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return detect_faces_cached(buf.tobytes(), img8.shape[:2], padding=padding)

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
def dir_blend(img16, lin_blur, alpha, highlight_protect=False, lin=None):
    """Blend sharp original with blurred background in linear light.
    If lin is provided, reuses it instead of recomputing srgb_to_linear(img16)."""
    if lin is None:
        lin = srgb_to_linear(img16)
    if highlight_protect:
        lin_blur = restore_highlights(lin, lin_blur, alpha)
    if HAS_GPU:
        lin_out = gpu_blend(lin, lin_blur, alpha)
    else:
        a3 = alpha[..., None]
        lin_out = a3 * lin + (1.0 - a3) * lin_blur
    del lin, lin_blur
    gc.collect()
    srgb_out = np.clip(linear_to_srgb_f32(lin_out), 0, 1)
    del lin_out
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
    if inpaint and subject_mask is not None and subject_mask.any():
        src = inpaint_subject(lin_img, subject_mask, radius=7)
        # inpaint already returned a new array, clamp in-place
        soft_clamp_highlights(src, shoulder=0.85, inplace=True)
    else:
        # Need a copy since we'll modify it
        src = soft_clamp_highlights(lin_img, shoulder=0.85, inplace=False)

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
                          mag_scale=False, inpaint_bg=False, highlight_protect=False,
                          bg_lin=None):
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
    bg_src = bg_lin if bg_lin is not None else lin
    lin_blur = build_blur_image(bg_src, angle, blur_len, aspect,
                                flow=flow, bokeh_mix=bokeh_mix,
                                use_flow_dir=use_flow_dir, mag_scale=mag_scale,
                                subject_mask=keep if inpaint_bg else None,
                                inpaint=inpaint_bg)
    return dir_blend(img16, lin_blur, alpha, highlight_protect=highlight_protect, lin=lin), alpha

def _prepare_blur_src(lin_img, subject_mask=None, inpaint=False):
    """Pre-process source for blur: inpaint + highlight clamp. Returns new array."""
    if inpaint and subject_mask is not None and subject_mask.any():
        src = inpaint_subject(lin_img, subject_mask, radius=7)
        soft_clamp_highlights(src, shoulder=0.85, inplace=True)
    else:
        src = soft_clamp_highlights(lin_img, shoulder=0.85, inplace=False)
    return src

def make_blur_stack_linear(lin_img, angle_deg, levels, aspect=0.18,
                           flow=None, bokeh_mix=0.0, use_flow_dir=False,
                           mag_scale=False, subject_mask=None, inpaint=False):
    outs = []
    src = _prepare_blur_src(lin_img, subject_mask, inpaint)

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

def _radial_basis(H, W, center=None, power=1.0):
    """Return (H,W) float32 normalized radial distance field."""
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = (W//2, H//2) if center is None else center
    r = np.hypot((xx - cx).astype(np.float32), (yy - cy).astype(np.float32))
    del xx, yy
    return (r / (r.max() + 1e-6)) ** power

def _flow_basis(prev8, curr8, power=1.0, quality="fast"):
    """Return (H,W) float32 normalized flow magnitude field."""
    _, mag, _ = estimate_flow(prev8, curr8, quality=quality)
    return (mag / (mag.max() + 1e-6)) ** power

def _zone_weight(basis, i, edges, width):
    """Compute smoothed weight for zone *i* from a basis field. Returns (H,W) float32."""
    c = 0.5 * (edges[i] + edges[i + 1])
    w = np.maximum(0.0, 1.0 - np.abs(basis - c) / width)
    return cv2.GaussianBlur(w, (0, 0), 7)

def composite_zonal_16u(img16, prev8, curr8, masks, chosen, angle_deg,
                        blur_len, offset_px=0.0, width_in_px=8.0, width_out_px=40.0, hard_core_px=16.0,
                        curve="Smoothstep", gamma=2.2,
                        profile="Radial", radial_soft=0.35, radial_power=1.0, flow_power=1.0,
                        aspect=0.18, guide8=None, extra_keep=None,
                        flow=None, bokeh_mix=0.0, use_flow_dir=False,
                        mag_scale=False, inpaint_bg=False, highlight_protect=False,
                        flow_quality="fast", bg_lin=None):
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
    bg_src = bg_lin if bg_lin is not None else lin
    n_levels = 8
    levels = np.linspace(0, blur_len, n_levels).astype(int).tolist()

    # Compute lightweight (H,W) basis field instead of full (H,W,n) weight map
    if profile == "Flow":
        basis = _flow_basis(prev8, curr8, power=flow_power, quality=flow_quality)
        basis_softness = 0.5
    else:
        basis = _radial_basis(H, W, center=None, power=radial_power)
        basis_softness = radial_soft
    edges = np.linspace(0, 1, n_levels + 1, dtype=np.float32)
    zone_width = max(1e-6, basis_softness * (edges[1] - edges[0]))

    # Blend incrementally: compute one weight + blur level at a time
    src = _prepare_blur_src(bg_src, subject_mask=keep if inpaint_bg else None, inpaint=inpaint_bg)
    out_bg = np.zeros((H, W, 3), dtype=np.float32)
    sum_w = np.zeros((H, W), dtype=np.float32)
    for i, L in enumerate(levels):
        w_i = _zone_weight(basis, i, edges, zone_width)
        sum_w += w_i
        if L <= 0:
            blurred = src
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
        out_bg += w_i[..., None] * blurred
    del src, basis
    # Normalize by smoothed weight sum
    out_bg /= (sum_w[..., None] + 1e-6)
    del sum_w
    return dir_blend(img16, out_bg, alpha, highlight_protect=highlight_protect, lin=lin), alpha

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
        seg_mode = st.selectbox("Detection mode", ["Person (full body)", "Face"],
                                help="Person: full-body segmentation via YOLO. "
                                     "Face: detect faces and keep them sharp with an elliptical mask.")
        model_name = st.selectbox("Model",
                                  ["yolo11n-seg.pt","yolo11s-seg.pt","yolo11m-seg.pt","yolo11l-seg.pt"], 1,
                                  help="Larger models are more accurate but slower",
                                  disabled=seg_mode == "Face")
        imgsz = st.slider("Detection quality", 640, 1920, 1280, 64,
                          help="Higher values detect better but run slower",
                          disabled=seg_mode == "Face")
        face_padding = st.slider("Face padding", 0.0, 1.0, 0.35, 0.05,
                                 help="How much extra area around each face to keep sharp (0 = tight crop, 1 = very generous)",
                                 disabled=seg_mode != "Face")

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
        use_multi_frame = st.checkbox("Multi-frame averaging", value=False,
                                      help="Average neighboring frames to capture real background motion, "
                                           "then use motion blur to smooth edges.")
        multi_frame_window = st.slider("Averaging window (frames)", 1, 5, 2, 1,
                                       disabled=not use_multi_frame,
                                       help="Number of frames to average in each direction")
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
        source_mode = st.radio("Source type", ["Stills folder", "Video extraction"],
                               help="Use 'Stills folder' to browse and select photos visually. "
                                    "Use 'Video extraction' to pull frames from a video file.")

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
            frames_dir = None  # will be set by the visual browser
            fmt_key = None
            tone_map = False

# ----------------------- Video extraction (if applicable) -----------------------
tmp_video = None
if source_mode == "Video extraction":
    if uploaded is not None:
        tmp_video = os.path.join(tempfile.gettempdir(), uploaded.name)
        with open(tmp_video, "wb") as f: f.write(uploaded.read())
    if go_extract:
        if tmp_video: ffmpeg_extract(tmp_video, frames_dir, fps=30, tone_map=tone_map, fmt=fmt_key)
        elif frames_dir and os.path.exists(frames_dir):
            st.info(f"Frames dir exists, found {len(list_frames(frames_dir))} frames.")
        else: st.error("Provide a video or a valid frames directory.")

# ----------------------- Session state init -----------------------
if "browse_dir" not in st.session_state:
    _default_browse = "."
    if frames_dir and os.path.isdir(frames_dir):
        _default_browse = frames_dir
    elif os.path.isdir("data"):
        _default_browse = "data"
    st.session_state.browse_dir = os.path.abspath(_default_browse)
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "selected_frames" not in st.session_state:
    st.session_state.selected_frames = set()

# In video mode, frames come from frames_dir; in stills mode, from the browser
if source_mode == "Video extraction" and frames_dir:
    frames = list_frames(frames_dir)
else:
    frames_dir = st.session_state.browse_dir
    frames = list_frames(frames_dir)

# ----------------------- Visual browser & frame selection -----------------------
with st.expander("Browse & select frames", expanded=True):
    _bdir = st.session_state.browse_dir

    # --- Quick-access drives / bookmarks ---
    _quick = []
    _home = os.path.expanduser("~")
    _quick.append(("Home", _home))
    _quick.append(("Project", os.path.abspath(".")))
    # Detect mounted drives (WSL /mnt/X, Linux /media/$USER/*)
    for _mnt_letter in sorted(os.listdir("/mnt")) if os.path.isdir("/mnt") else []:
        _mnt_path = os.path.join("/mnt", _mnt_letter)
        if (len(_mnt_letter) == 1 and _mnt_letter.isalpha()
                and os.path.isdir(_mnt_path)):
            # Verify the mount is actually accessible (WSL mounts can be stale)
            try:
                os.listdir(_mnt_path)
                _quick.append((f"{_mnt_letter.upper()}:", _mnt_path))
            except OSError:
                pass
    _media_user = os.path.join("/media", os.environ.get("USER", ""))
    if os.path.isdir(_media_user):
        for _vol in sorted(os.listdir(_media_user)):
            _vol_path = os.path.join(_media_user, _vol)
            if os.path.isdir(_vol_path):
                _quick.append((_vol, _vol_path))

    _qcols = st.columns(len(_quick))
    for _qi, (_qlabel, _qpath) in enumerate(_quick):
        with _qcols[_qi]:
            if st.button(_qlabel, key=f"qnav_{_qi}", use_container_width=True,
                         type="primary" if os.path.abspath(_qpath) == os.path.abspath(_bdir) else "secondary"):
                st.session_state.browse_dir = os.path.abspath(_qpath)
                st.session_state.selected_frames = set()
                st.session_state.pop("frame_multiselect", None)
                st.rerun()

    # --- Directory path input + up button ---
    _path_col, _up_col = st.columns([8, 1])
    with _path_col:
        _new_path = st.text_input("Directory", _bdir, key="browse_path_input",
                                  label_visibility="collapsed")
        if os.path.isdir(_new_path) and os.path.abspath(_new_path) != _bdir:
            st.session_state.browse_dir = os.path.abspath(_new_path)
            st.session_state.selected_frames = set()
            st.session_state.pop("frame_multiselect", None)
            st.rerun()
    with _up_col:
        _parent = os.path.dirname(_bdir)
        if st.button("\u2191 Up", use_container_width=True, disabled=(_parent == _bdir)):
            st.session_state.browse_dir = _parent
            st.session_state.selected_frames = set()
            st.session_state.pop("frame_multiselect", None)
            st.rerun()

    # Breadcrumb
    _parts = os.path.normpath(_bdir).split(os.sep)
    st.caption(" / ".join(f"`{p}`" for p in _parts[-4:]))

    # --- Subdirectories ---
    _subdirs = list_subdirs(_bdir)
    if _subdirs:
        st.markdown("**Folders:**")
        _sd_cols = st.columns(min(len(_subdirs), 8))
        for _si, _sd in enumerate(_subdirs[:16]):
            with _sd_cols[_si % len(_sd_cols)]:
                if st.button(f"\U0001F4C1 {_sd}", key=f"sd_{_si}", use_container_width=True):
                    st.session_state.browse_dir = os.path.join(_bdir, _sd)
                    st.session_state.selected_frames = set()
                    st.rerun()

    # --- Images in this directory ---
    _browse_frames = list_frames(_bdir)
    if _browse_frames:
        # Selection toolbar
        _tb1, _tb2, _tb3 = st.columns([2, 2, 8])
        with _tb1:
            if st.button("Select all", use_container_width=True):
                st.session_state.selected_frames = set(range(len(_browse_frames)))
                st.rerun()
        with _tb2:
            if st.button("Clear", use_container_width=True):
                st.session_state.selected_frames = set()
                st.session_state.pop("frame_multiselect", None)
                st.rerun()
        with _tb3:
            _n_sel = len(st.session_state.selected_frames)
            if _n_sel:
                st.markdown(f"**{len(_browse_frames)} images** \u2014 **{_n_sel} selected** for session")
            else:
                st.markdown(f"**{len(_browse_frames)} images** \u2014 select frames to add to session, or use all")

        # Multiselect for keyboard/search-based selection
        _frame_names = [os.path.basename(f) for f in _browse_frames]
        _default_sel = [_frame_names[i] for i in sorted(st.session_state.selected_frames)
                        if i < len(_frame_names)]
        _picked = st.multiselect(
            "Selected frames",
            options=_frame_names,
            default=_default_sel,
            key="frame_multiselect",
            placeholder="Type to search and select frames...",
            label_visibility="collapsed",
        )
        _picked_set = set(_frame_names.index(n) for n in _picked if n in _frame_names)
        if _picked_set != st.session_state.selected_frames:
            st.session_state.selected_frames = _picked_set

        # Thumbnail grid - paginated
        THUMBS_PER_PAGE = 24
        _n_pages = max(1, (len(_browse_frames) + THUMBS_PER_PAGE - 1) // THUMBS_PER_PAGE)
        _page = 0
        if _n_pages > 1:
            _page = st.slider("Page", 0, _n_pages - 1, 0, 1,
                              format=f"Page %d of {_n_pages}",
                              key="thumb_page")
        _page_start = _page * THUMBS_PER_PAGE
        _page_frames = _browse_frames[_page_start:_page_start + THUMBS_PER_PAGE]

        _COLS = 8
        _rows = (len(_page_frames) + _COLS - 1) // _COLS
        for _r in range(_rows):
            _tcols = st.columns(_COLS)
            for _c in range(_COLS):
                _ti = _r * _COLS + _c
                if _ti >= len(_page_frames):
                    break
                _tf = _page_frames[_ti]
                _global_idx = _page_start + _ti
                with _tcols[_c]:
                    _thumb = make_thumbnail(_tf, size=THUMB_SIZE)
                    _in_sel = _global_idx in st.session_state.selected_frames
                    # Check if this thumbnail is the currently-editing frame
                    _is_active = False
                    if (os.path.abspath(_bdir) == os.path.abspath(frames_dir) and
                            st.session_state.frame_idx < len(frames)):
                        _is_active = (os.path.abspath(_tf) ==
                                      os.path.abspath(frames[st.session_state.frame_idx]))
                    # Caption with status markers
                    _cap = os.path.splitext(os.path.basename(_tf))[0]
                    if _is_active and _in_sel:
                        _cap = f"\u25B6\u2713 {_cap}"
                    elif _is_active:
                        _cap = f"\u25B6 {_cap}"
                    elif _in_sel:
                        _cap = f"\u2713 {_cap}"
                    st.image(_thumb, use_column_width=True, caption=_cap)
                    # Checkbox + edit button
                    _ck_col, _go_col = st.columns([1, 1])
                    with _ck_col:
                        _checked = st.checkbox(
                            "sel", value=_in_sel,
                            key=f"sel_{_global_idx}",
                            label_visibility="collapsed")
                        if _checked and not _in_sel:
                            st.session_state.selected_frames.add(_global_idx)
                        elif not _checked and _in_sel:
                            st.session_state.selected_frames.discard(_global_idx)
                    with _go_col:
                        if st.button("\u25B6", key=f"go_{_global_idx}",
                                     use_container_width=True,
                                     help="Edit this frame"):
                            # Store the file path so we can find it in the
                            # working frames list regardless of selection filtering
                            st.session_state.active_frame_path = os.path.abspath(_tf)
                            # Also add to selection so it appears in the working set
                            st.session_state.selected_frames.add(_global_idx)
                            if source_mode != "Video extraction":
                                st.session_state.browse_dir = os.path.abspath(_bdir)
                            st.rerun()
    elif not _subdirs:
        st.info("No supported images or subdirectories found here.")

# ----------------------- Resolve working frames list -----------------------
# In stills mode the browser directory is always the source
if source_mode != "Video extraction":
    frames_dir = st.session_state.browse_dir
    _all_dir_frames = list_frames(frames_dir)
    # If user selected specific frames, use only those (in order)
    if st.session_state.selected_frames and _all_dir_frames:
        frames = [_all_dir_frames[i] for i in sorted(st.session_state.selected_frames)
                  if i < len(_all_dir_frames)]
    else:
        frames = _all_dir_frames

# Output dir (based on resolved frames_dir)
clip_name = (os.path.basename(os.path.normpath(frames_dir)) or "session") if frames_dir else "session"
run_dir = os.path.join("outputs", "action_pan", clip_name, datetime.now().strftime("%Y%m%d_%H%M%S"))
if "run_dir" not in st.session_state:
    st.session_state.run_dir = run_dir
ensure_dir(st.session_state.run_dir)

if len(frames) < 1:
    st.info("No images found. Use the **Browse** panel above to navigate to a folder of images "
            "and select frames to work on.")
    st.stop()

# --- Frame navigation (full width, above columns) ---
_sel_label = f" ({len(st.session_state.selected_frames)} selected)" if st.session_state.selected_frames else ""
st.markdown(f"**{len(frames)} frames{_sel_label}** from `{frames_dir}` | Output: `{st.session_state.run_dir}`")

# Resolve active frame path → index in the working frames list
if "active_frame_path" in st.session_state and st.session_state.active_frame_path:
    _target = st.session_state.active_frame_path
    _abs_frames = [os.path.abspath(f) for f in frames]
    if _target in _abs_frames:
        st.session_state.frame_idx = _abs_frames.index(_target)
    st.session_state.active_frame_path = None  # consumed

# Clamp frame_idx to working frames list
if st.session_state.frame_idx >= len(frames):
    st.session_state.frame_idx = len(frames) - 1

if len(frames) > 1:
    nav_prev, nav_slider, nav_next = st.columns([1, 10, 1])
    with nav_prev:
        if st.button("Prev", use_container_width=True):
            st.session_state.frame_idx = max(0, st.session_state.frame_idx - 1)
    with nav_next:
        if st.button("Next", use_container_width=True):
            st.session_state.frame_idx = min(len(frames) - 1, st.session_state.frame_idx + 1)
    with nav_slider:
        idx = st.slider("Frame", 0, len(frames) - 1, st.session_state.frame_idx, 1,
                        label_visibility="collapsed")
        st.session_state.frame_idx = idx
else:
    st.session_state.frame_idx = 0

idx = st.session_state.frame_idx

# Load full-res
fpath = frames[idx]
prev_path = frames[idx-1] if idx>0 else frames[idx]

with st.spinner("Loading frame..."):
    img = read_image(fpath);  img16 = img if img.dtype==np.uint16 else (img.astype(np.uint16)*257)
    if img is not img16: del img  # free original if we made a copy
    prv = read_image(prev_path); prev16 = prv if prv.dtype==np.uint16 else (prv.astype(np.uint16)*257)
    if prv is not prev16: del prv
    img8  = (img16/257).astype(np.uint8); prev8=(prev16/257).astype(np.uint8)
    del prev16  # only need prev8 from here; img16 is needed for compositing
    H,W = img8.shape[:2]

# Preview scaling
PREVIEW_S = min(1.0, compute_preview_scale(H,W, auto_preview, preview_scale_ui))
def RS(x, s, interp=cv2.INTER_AREA): return x if s>=0.999 else cv2.resize(x,(int(W*s),int(H*s)),interpolation=interp)
p_img16=RS(img16,PREVIEW_S); p_img8=RS(img8,PREVIEW_S); p_prev8=RS(prev8,PREVIEW_S)

# Masks (cached)
with st.spinner("Detecting subjects..."):
    if seg_mode == "Face":
        masks = segment_faces(img8, padding=face_padding)
    else:
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

# Multi-frame background averaging (real motion capture)
p_bg_lin = None
if use_multi_frame:
    with st.spinner("Averaging frames for background..."):
        bg_lin_full, _bg_count = multi_frame_bg_average(
            frames, idx, subject_mask_full, window=multi_frame_window)
        if bg_lin_full is not None:
            if PREVIEW_S < 0.999:
                pH, pW = p_img8.shape[:2]
                p_bg_lin = cv2.resize(bg_lin_full, (pW, pH))
            else:
                p_bg_lin = bg_lin_full
            del bg_lin_full

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
    _label = "Face" if seg_mode == "Face" else "Subject"
    if masks:
        areas=[int(m.sum()) for m in masks]
        opts=[f"{_label} {i+1} (area {a:,})" for i,a in enumerate(areas)]
        default=[]
        if batch_mode!="interactive" and not st.session_state.get("selected", set()) and len(areas)>0:
            default=[opts[int(np.argmax(areas))]]
        keep = st.multiselect("Keep sharp", opts, default=default, key="msel",
                              help=f"Select which detected {_label.lower()}s should remain sharp. Numbers match the overlay.")
        selected=set([int(s.split()[1])-1 for s in keep]); st.session_state.selected=selected
    else:
        st.info(f"No {_label.lower()}s detected in this frame.")
        selected = set(); st.session_state.selected = selected

    # Masks overlay with numbered subjects
    if masks:
        overlay_img = paint_overlay(p_img8, p_masks, selected)
        st.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB),
                 caption=f"{_label} detection", use_column_width=True)

    if use_custom_region:
        # Retrieve persisted masks for this frame (needed before preview)
        p_extra = st.session_state.brush_mask_prev.get(fpath, None)
        custom_keep_mask = st.session_state.brush_mask_full.get(fpath, None)

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
p_blur_len = max(1, int(blur_len * PREVIEW_S))
if masks:
    with st.spinner("Computing preview..."):
        t0 = time.time()
        if profile=="Uniform":
            p_out16, p_alpha = composite_uniform_16u(
                p_img16, p_masks, selected, p_blur_len, angle,
                guide8=p_img8, extra_keep=extra_preview, flow=p_flow,
                bg_lin=p_bg_lin,
                **_comp_kwargs
            )
        else:
            p_out16, p_alpha = composite_zonal_16u(
                p_img16, p_prev8, p_img8, p_masks, selected, angle,
                p_blur_len, guide8=p_img8, extra_keep=extra_preview, flow=p_flow,
                profile=("Flow" if profile=="Flow" else "Radial"),
                radial_soft=radial_soft, radial_power=radial_power, flow_power=flow_power,
                flow_quality=flow_quality,
                bg_lin=p_bg_lin,
                **_comp_kwargs
            )
        elapsed = time.time() - t0

    with colB:
        # Before/After tabs
        tab_result, tab_compare, tab_details = st.tabs(["Result", "Before / After", "Details"])

        with tab_result:
            p_out8_rgb = cv2.cvtColor((p_out16/257).astype(np.uint8), cv2.COLOR_BGR2RGB)
            if use_custom_region:
                # Canvas overlaid on the output image for brush painting
                _prev_h, _prev_w = p_out8_rgb.shape[:2]
                # Scale canvas to fit column width (Streamlit columns are ~700px max)
                _canvas_max_w = 700
                if _prev_w > _canvas_max_w:
                    _canvas_scale = _canvas_max_w / _prev_w
                    _canvas_w = _canvas_max_w
                    _canvas_h = int(_prev_h * _canvas_scale)
                    _canvas_img = cv2.resize(p_out8_rgb, (_canvas_w, _canvas_h), interpolation=cv2.INTER_AREA)
                else:
                    _canvas_w, _canvas_h = _prev_w, _prev_h
                    _canvas_img = p_out8_rgb
                bg_pil = PILImage.fromarray(_canvas_img)
                init_json = st.session_state.brush_json.get(fpath, None)
                canvas = st_canvas(
                    fill_color="rgba(255, 255, 255, 0.0)",
                    stroke_width=brush_size,
                    stroke_color="rgba(255, 80, 80, 0.7)",
                    background_image=bg_pil,
                    update_streamlit=True,
                    height=_canvas_h, width=_canvas_w,
                    drawing_mode="freedraw",
                    initial_drawing=init_json,
                    key=f"canvas_{idx}",
                )
                btn_apply, btn_clear = st.columns(2)
                with btn_apply:
                    apply_brush_now = st.button("Apply brush",
                                                use_container_width=True,
                                                help="Apply current brush strokes to the preview")
                with btn_clear:
                    clear_brush = st.button("Clear strokes",
                                            use_container_width=True)
                if clear_brush:
                    st.session_state.brush_json.pop(fpath, None)
                    st.session_state.brush_mask_full.pop(fpath, None)
                    st.session_state.brush_mask_prev.pop(fpath, None)
                    st.rerun()
                if apply_brush_now and canvas is not None:
                    st.session_state.brush_json[fpath] = canvas.json_data
                    if canvas.image_data is not None:
                        a_small = canvas.image_data[:, :, 3]
                        draw_canvas = (a_small > 0).astype(np.uint8)
                        # Resize canvas mask to preview resolution
                        draw_small = cv2.resize(draw_canvas, (_prev_w, _prev_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                        st.session_state.brush_mask_prev[fpath] = draw_small.copy()
                        st.session_state.brush_mask_full[fpath] = cv2.resize(draw_small, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    st.rerun()
                p_extra = st.session_state.brush_mask_prev.get(fpath, None)
                custom_keep_mask = st.session_state.brush_mask_full.get(fpath, None)
                if p_extra is not None:
                    brush_px = int(p_extra.sum())
                    status_label = "Applied" if apply_brush_on_save else "Preview only (not applied on save)"
                    st.caption(f"Brush: {brush_px:,} px painted - {status_label}")
            else:
                st.image(p_out8_rgb,
                         use_column_width=True, caption="Action Pan composite")
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
                         use_column_width=True, caption="Original")
            with cmp2:
                st.image(cv2.cvtColor((p_out16/257).astype(np.uint8), cv2.COLOR_BGR2RGB),
                         use_column_width=True, caption="Action Pan")

        with tab_details:
            d1, d2 = st.columns(2)
            with d1:
                st.image(cv2.cvtColor(paint_overlay(p_img8, p_masks, selected), cv2.COLOR_BGR2RGB),
                         use_column_width=True, caption="Subject detection overlay")
            with d2:
                if show_alpha:
                    st.image((p_alpha*255).astype(np.uint8), clamp=True, use_column_width=True,
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
            # Compute full-res averaged background if multi-frame is enabled
            save_bg_lin = None
            if use_multi_frame:
                save_bg_lin, _ = multi_frame_bg_average(
                    frames, idx, subject_mask_full, window=multi_frame_window)
            if profile=="Uniform":
                out16,_=composite_uniform_16u(
                    img16, masks, selected, blur_len, angle_full,
                    guide8=img8, extra_keep=extra_full, flow=flow_full,
                    bg_lin=save_bg_lin,
                    **_comp_kwargs
                )
            else:
                out16,_=composite_zonal_16u(
                    img16, prev8, img8, masks, selected, angle_full,
                    blur_len, guide8=img8, extra_keep=extra_full, flow=flow_full,
                    profile=("Flow" if profile=="Flow" else "Radial"),
                    radial_soft=radial_soft, radial_power=radial_power, flow_power=flow_power,
                    flow_quality=flow_quality,
                    bg_lin=save_bg_lin,
                    **_comp_kwargs
                )
            del save_bg_lin
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
            for i, f in enumerate(frames):
                prog.progress(i/len(frames), text=f"Processing frame {i+1}/{len(frames)}...")
                curr=read_image(f)
                prev=read_image(frames[i-1]) if i>0 else curr
                c16=curr if curr.dtype==np.uint16 else (curr.astype(np.uint16)*257)
                p16=prev if prev.dtype==np.uint16 else (prev.astype(np.uint16)*257)
                del curr, prev  # free raw reads
                c8=(c16/257).astype(np.uint8); p8=(p16/257).astype(np.uint8)
                del p16  # keep c16 (needed for compositing), free p16

                mks = segment_faces(c8, padding=face_padding) if seg_mode == "Face" else segment_people(model_name, imgsz, c8)
                if not mks:
                    del c16, c8, p8
                    prog.progress((i+1)/len(frames)); continue

                # Build subject mask for this frame
                sub_mask = np.zeros(c8.shape[:2], np.uint8)
                for mk in mks: sub_mask |= mk

                flow_full, ang = compute_full_flow(
                    frames, i, c8, p8, sub_mask,
                    use_multi_frame, multi_frame_window, flow_quality)
                chosen = choose_indices(mks, batch_mode, selected)

                # Multi-frame background averaging for batch
                batch_bg_lin = None
                if use_multi_frame:
                    batch_bg_lin, _ = multi_frame_bg_average(
                        frames, i, sub_mask, window=multi_frame_window)

                extra_full=None
                if use_custom_region and apply_brush_on_save:
                    extra_full=None

                if profile=="Uniform":
                    out16,_=composite_uniform_16u(
                        c16, mks, chosen, blur_len, ang,
                        guide8=c8, extra_keep=extra_full, flow=flow_full,
                        bg_lin=batch_bg_lin,
                        **_comp_kwargs
                    )
                else:
                    out16,_=composite_zonal_16u(
                        c16, p8, c8, mks, chosen, ang,
                        blur_len, guide8=c8, extra_keep=extra_full, flow=flow_full,
                        profile=("Flow" if profile=="Flow" else "Radial"),
                        radial_soft=radial_soft, radial_power=radial_power, flow_power=flow_power,
                        flow_quality=flow_quality,
                        bg_lin=batch_bg_lin,
                        **_comp_kwargs
                    )
                del c16, c8, p8, flow_full, mks, sub_mask, batch_bg_lin
                base=f"pan_{os.path.splitext(os.path.basename(f))[0]}"
                save_image(out16, st.session_state.run_dir, base, save_format)
                del out16; gc.collect()
                count+=1; prog.progress((i+1)/len(frames))
            st.success(f"Batch complete: saved {count} frames to `{st.session_state.run_dir}`")
