
# core/crowd_processor.py
# Fix: per-zone P95 + adaptive global floor + faster warmup

import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
from torchvision import transforms
from models.csrnet_model import CSRNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load CSRNet ───────────────────────────────────────────────
csrnet = CSRNet().to(device)
checkpoint = torch.load("models/best_csrnet_finetuned.pth", map_location=device)
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]
clean_state = {
    (k.replace("module.", "") if k.startswith("module.") else k)
    .replace("output_layer", "output"): v
    for k, v in checkpoint.items()
}
csrnet.load_state_dict(clean_state)
csrnet.eval()

# ── Load YOLO ─────────────────────────────────────────────────
yolo = YOLO("best.pt")
yolo.to(device)
YOLO_CONF = 0.25

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Constants ─────────────────────────────────────────────────
WARMUP_FRAMES   = 5       # reduced: stable after 5 frames
GLOBAL_BUF_SIZE = 5000
GLOBAL_WARMUP   = 10      # global P95 kicks in fast
THRESH_SAFE     = 40.0
THRESH_MODERATE = 70.0
VISUAL_BLEND    = 0.20    # 20% visual; CSRNet is more reliable

# ── Global state ──────────────────────────────────────────────
_global_raw_buf: deque = deque(maxlen=GLOBAL_BUF_SIZE)
_per_zone_buf:   dict  = {}   # zone_id -> deque of raw values
_global_max:     float = 1.0
_frame_count:    dict  = {}


def reset_processor():
    global _global_max
    _global_raw_buf.clear()
    _per_zone_buf.clear()
    _global_max = 1.0
    _frame_count.clear()


def _raw_to_pct(raw: float, zone_id: int) -> float:
    """
    Normalise raw density using BOTH global P95 and per-zone P95.
    Global P95 anchors zones to an absolute scale.
    Per-zone P95 ensures relative changes within a zone are captured.
    Final = max(global_pct, zone_pct) — take the more alarming signal.
    """
    global _global_max

    # Per-zone buffer
    if zone_id not in _per_zone_buf:
        _per_zone_buf[zone_id] = deque(maxlen=500)
    _per_zone_buf[zone_id].append(raw)

    if raw > _global_max:
        _global_max = raw

    # Global normalisation
    if len(_global_raw_buf) >= GLOBAL_WARMUP:
        g_p95 = float(np.percentile(list(_global_raw_buf), 95))
        global_pct = min(100.0, 100.0 * raw / max(g_p95, 1.0))
    else:
        global_pct = min(100.0, 100.0 * raw / max(_global_max, 1.0))

    # Per-zone normalisation (captures relative spikes within a zone)
    z_buf = list(_per_zone_buf[zone_id])
    if len(z_buf) >= 10:
        z_p95 = float(np.percentile(z_buf, 95))
        zone_pct = min(100.0, 100.0 * raw / max(z_p95, 1.0))
    else:
        zone_pct = global_pct

    # Use global as primary; only boost from zone_pct if it's significantly higher
    # This prevents sparse zones from falsely inflating due to their own tiny P95
    return float(np.clip(0.8 * global_pct + 0.2 * zone_pct, 0.0, 100.0))


def _visual_density_pct(heatmap_bgr: np.ndarray) -> float:
    """Red+yellow pixel fraction in COLORMAP_JET BGR, scaled to 0-100."""
    if heatmap_bgr is None or heatmap_bgr.size == 0:
        return 0.0
    b = heatmap_bgr[:, :, 0].astype(np.float32)
    g = heatmap_bgr[:, :, 1].astype(np.float32)
    r = heatmap_bgr[:, :, 2].astype(np.float32)
    red_mask    = (r > 160) & (g < 100) & (b < 100)
    yellow_mask = (r > 150) & (g > 150) & (b < 100)
    total    = float(heatmap_bgr.shape[0] * heatmap_bgr.shape[1])
    weighted = float(red_mask.sum()) + 0.5 * float(yellow_mask.sum())
    # Scale factor 250 (was 300): less aggressive amplification
    return float(np.clip(weighted / max(total, 1.0) * 250.0, 0.0, 100.0))


def _blend(csrnet_pct: float, visual_pct: float) -> float:
    """80% CSRNet (reliable density model) + 20% visual heatmap."""
    return float(np.clip(
        (1.0 - VISUAL_BLEND) * csrnet_pct + VISUAL_BLEND * visual_pct,
        0.0, 100.0
    ))


def _get_status(zone_id: int, pct: float) -> str:
    if _frame_count.get(zone_id, 0) < WARMUP_FRAMES:
        return "WARMUP"
    if pct < THRESH_SAFE:
        return "SAFE"
    elif pct < THRESH_MODERATE:
        return "MODERATE"
    return "RISK"


def _skin_pixel_ratio(crop_bgr: np.ndarray) -> float:
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    mask  = cv2.inRange(ycrcb,
                        np.array([0, 133, 77],   dtype=np.uint8),
                        np.array([255, 173, 127], dtype=np.uint8))
    return float(mask.sum()) / (mask.size * 255)


def _pose_detected(crop_bgr: np.ndarray) -> bool:
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    h, w = crop_bgr.shape[:2]
    return h >= 40 and w >= 15 and h / (w + 1e-6) >= 1.4 and _skin_pixel_ratio(crop_bgr) >= 0.03


def process_video_frame(frame: np.ndarray, zone_id: int = 0, video_fps: float = 30.0):
    """
    Returns: (proc_frame, final_pct, vulnerable_score, status, counts)
    Pipeline: CSRNet density → global+zone P95 → blend 80/20 with visual heatmap
    """
    frame = cv2.resize(frame, (640, 480))
    H, W  = frame.shape[:2]

    # CSRNet inference
    csr_input  = cv2.resize(frame, (448, 256))
    csr_tensor = transform(csr_input).unsqueeze(0).to(device)
    with torch.no_grad():
        density_map = torch.relu(csrnet(csr_tensor))
    density_map = density_map.squeeze().cpu().numpy()
    dH, dW      = density_map.shape
    density_map = density_map * ((256 * 448) / (dH * dW))
    raw_density = float(density_map.sum())

    _global_raw_buf.append(raw_density)
    _frame_count[zone_id] = _frame_count.get(zone_id, 0) + 1

    # Heatmap overlay
    vis         = cv2.resize(density_map, (W, H))
    norm        = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_bgr = cv2.applyColorMap(norm.astype("uint8"), cv2.COLORMAP_JET)
    frame       = cv2.addWeighted(frame, 0.6, heatmap_bgr, 0.4, 0)

    # YOLO detection
    results       = yolo(frame, conf=YOLO_CONF, verbose=False)[0]
    child_count   = 0
    elderly_count = 0
    adult_count   = 0

    for box in results.boxes:
        if int(box.cls[0]) != 0:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_h        = y2 - y1
        height_ratio = box_h / H
        crop         = frame[y1:y2, x1:x2]
        score = 0
        if height_ratio < 0.33:
            score += 2
        if crop.size > 0 and not _pose_detected(crop):
            score += 1
        if score >= 2:
            child_count += 1
            label, color = "Child",   (0, 0, 255)
        elif height_ratio > 0.6:
            elderly_count += 1
            label, color = "Elderly", (255, 0, 0)
        else:
            adult_count += 1
            label, color = "Adult",   (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    vulnerable_score = 1.5 * child_count + 1.2 * elderly_count

    csrnet_pct = _raw_to_pct(raw_density, zone_id)
    visual_pct = _visual_density_pct(heatmap_bgr)
    final_pct  = _blend(csrnet_pct, visual_pct)
    status     = _get_status(zone_id, final_pct)

    counts = {"Child": child_count, "Adult": adult_count, "Elderly": elderly_count}
    return frame, final_pct, vulnerable_score, status, counts