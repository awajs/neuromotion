# Neuromotion - Action Pan

A video motion processing tool that creates cinematic "action pan" effects — selectively blurring backgrounds while keeping subjects sharp. Built with Streamlit, YOLO segmentation, and optical flow analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

## What it does

Action panning is a photography/cinematography technique where the camera follows a moving subject, producing a sharp subject against a motion-blurred background. Neuromotion automates this effect on video frames by:

1. Detecting people using YOLO segmentation
2. Estimating motion direction via optical flow
3. Applying directional motion blur to the background
4. Blending with edge-aware alpha matting for natural transitions

## Features

- **Video or stills input** — Extract frames from video or use a folder of sequential photos (burst/continuous shooting)
- **Automatic subject detection** — YOLOv11 segmentation with multiple model sizes (nano to large)
- **Directional motion blur** — Physically accurate blur aligned to camera motion via optical flow
- **Edge-aware compositing** — Signed-distance alpha matting with configurable transition curves
- **Linear-light blending** — sRGB to linear conversion for correct color math
- **16-bit workflow** — Full 16-bit TIFF support for maximum quality
- **Brush tool** — Paint custom keep-sharp regions beyond auto-detection
- **Presets** — Quick-start presets (Subtle pan, Medium motion, Heavy blur, Soft edges)
- **Batch processing** — Process all frames with consistent settings
- **Before/After comparison** — Side-by-side preview tabs
- **Multiple blur profiles** — Uniform, Radial (distance-based), and Flow (motion-based)

## Prerequisites

- Python 3.10+
- ffmpeg (for video frame extraction)
- GPU recommended for YOLO inference (CUDA via PyTorch)

## Installation

```bash
git clone https://github.com/your-username/neuromotion.git
cd neuromotion
pip install -r requirements.txt
```

YOLO model weights are downloaded automatically on first run, or you can place them in the project root:
- `yolo11n-seg.pt` (nano, fastest)
- `yolo11s-seg.pt` (small, good balance)
- `yolo11m-seg.pt` (medium)
- `yolo11l-seg.pt` (large, most accurate)

## Usage

```bash
streamlit run scripts/app.py
```

This opens a browser UI (default: `http://localhost:8501`).

### Workflow

1. **Load images** — Expand "Image source" in the sidebar. Either extract frames from a video (MP4/MOV) or point to a folder of sequential stills (e.g. burst photos at 12fps). Any JPG, PNG, or TIFF files in the folder are picked up automatically.
2. **Navigate** — Use Prev/Next buttons or the frame slider to select a frame.
3. **Choose a preset** — Pick a starting point from the preset dropdown at the top of the sidebar.
4. **Select subjects** — Use the "Keep sharp" dropdown in the left panel. Numbered overlays on the image show which subject is which.
5. **Fine-tune** — Expand sidebar sections to adjust blur strength, edge transitions, and blur profile.
6. **Compare** — Switch between Result, Before/After, and Details tabs in the main area.
7. **Save** — "Save frame" exports the current frame at full resolution. "Run batch" processes all frames.

### Key settings

| Setting | What it does |
|---|---|
| **Blur strength** | Amount of motion blur applied to the background |
| **Blur shape** | Aspect ratio of the blur kernel (thin streak vs wide smear) |
| **Edge position** | Shifts the sharp/blur boundary inward or outward |
| **Inner/Outer edge softness** | Controls how gradually the transition blends |
| **Guaranteed sharp zone** | Pixels this far inside the subject are always fully sharp |
| **Transition curve** | Shape of the blend (Linear, Smoothstep, Cosine, Gamma) |
| **Blur profile** | Uniform, Radial (by distance from center), or Flow (by motion magnitude) |

### Brush tool

Enable the brush in the sidebar to manually paint areas that should stay sharp (useful for objects the detector misses). Paint directly on the canvas overlay, then click "Apply brush" to see the result.

## Project structure

```
neuromotion/
  scripts/
    app.py              # Main Streamlit application
  data/
    raw/                # Source video files
    extracted_frames/   # Extracted frame sequences
  outputs/
    action_pan/         # Processed results (timestamped runs)
  requirements.txt
```

## Output formats

- **JPEG 8-bit** — Smaller files, suitable for web/preview
- **TIFF 16-bit** — Lossless, full dynamic range, suitable for further grading

## Technical details

- Optical flow estimation uses OpenCV DIS (Dense Inverse Search) algorithm
- Blur kernels are anisotropic Gaussians oriented along the flow direction
- Compositing is done in linear light (sRGB gamma removed before blending, reapplied after)
- Alpha mattes are generated from signed distance fields with configurable transition functions
- Edge refinement uses guided filtering (or bilateral filter as fallback)
- Zonal compositing blends 8 blur levels using radial or flow-magnitude weight maps

## License

All rights reserved.
