# ============================================================
#  SentinelAI – Human Anomaly Detector Module
#  YOLOv8 Architecture
#  YOLOv8 · Custom 5-class Anomaly Model
#               Classes: Accident, Explosion, Fighting, Shooting, Vandalism
# ============================================================

import os
import cv2
import numpy as np
import threading
import time
import logging
from datetime import datetime
from collections import deque

import sys
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)
from model_utils import resolve_model_path

logger = logging.getLogger("AnomalyDetector")

# ─── Monkey-patch torch.load for safe deserialization ────────
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO


# ─── YOLO Anomaly Class Configuration ───────────────────────
ANOMALY_CLASSES = {
    0: "Accident",
    1: "Explosion",
    2: "Fighting",
    3: "Shooting",
    4: "Vandalism",
}

ANOMALY_META = {
    "Accident":  {"level": "Critical", "color": (0,   0,  255), "emoji": "💥"},
    "Explosion": {"level": "Critical", "color": (0,  80,  255), "emoji": "🔥"},
    "Fighting":  {"level": "High",     "color": (0, 140,  255), "emoji": "👊"},
    "Shooting":  {"level": "Critical", "color": (0,   0,  200), "emoji": "🔫"},
    "Vandalism": {"level": "High",     "color": (0, 200,  255), "emoji": "🔨"},
}

BOX_COLORS = {
    "Accident":  (0,   0,  255),  # Red
    "Explosion": (0,  80,  255),  # Orange-Red
    "Fighting":  (0, 140,  255),  # Orange
    "Shooting":  (0,   0,  200),  # Dark Red
    "Vandalism": (0, 200,  255),  # Yellow
}


# ─── YOLO Confidence ────────────────────────────────────────
DEFAULT_YOLO_CONF = 0.25   # Lower than wildlife (0.40) due to mAP ≈ 0.342

# ─── Minimum bounding box area fraction ─────────────────────
MIN_BOX_AREA_FRACTION = 0.003  # 0.3% of frame area


class AnomalyDetector:
    """
    YOLOv8 anomaly detector.

    Classes:
    - Accident
    - Explosion
    - Fighting
    - Shooting
    - Vandalism
    """

    def __init__(self, yolo_model_path: str,
                 camera_index: int = 0, history_window: int = 60):
        self.yolo_model_path = resolve_model_path(yolo_model_path)
        self.camera_index    = camera_index
        self.history_window  = history_window

        self.yolo_model      = None
        self.cap             = None
        self._lock           = threading.Lock()
        self._running        = False
        self._thread         = None

        # YOLO confidence threshold
        self.conf_threshold  = DEFAULT_YOLO_CONF

        # Rolling error history for chart (autoencoder)
        self._error_history  = deque(maxlen=history_window)
        self._frame_count    = 0


        # Latest result (shared state for live polling)
        self._latest_frame   = None
        self._latest_result  = {
            "status":        "Normal",
            "score":         0.0,
            "anomaly_type":  "Normal",
            "confidence":    0,
            "timestamp":     "",
            "camera":        "CAM-02",
            "active":        False,
            "bbox":          None,
        }

        self._load_models()

    # ─── Model Loading ───────────────────────────────────────
    def _load_models(self):
        logger.info(f"Loading Anomaly YOLO model from: {self.yolo_model_path}")
        try:
            import ultralytics.nn.tasks as u_tasks
            torch.serialization.add_safe_globals([u_tasks.DetectionModel])
            self.yolo_model = YOLO(self.yolo_model_path)
            # Warm-up
            dummy = np.zeros((128, 128, 3), dtype=np.uint8)
            self.yolo_model.predict(dummy, verbose=False, conf=self.conf_threshold)
            logger.info(f"✅ Anomaly YOLO loaded. Classes: {self.yolo_model.names}")
        except Exception as e:
            logger.error(f"❌ Failed to load Anomaly YOLO model: {e}")
            self.yolo_model = None

        

    # ─── Camera Start ────────────────────────────────────────
    def start(self, shared_cap=None):
        if self._running:
            return

        # Reset state for fresh start
        self._frame_count = 0
        logger.debug("AnomalyDetector: state reset for fresh start.")

        if shared_cap is not None:
            self.cap = shared_cap
            self._shared_cap = True
        else:
            self._shared_cap = False
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.warning("⚠️  Webcam not found for anomaly. Using demo mode.")
                self.cap = None
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                logger.info("✅ Anomaly webcam opened.")

        self._running = True
        self._thread  = threading.Thread(target=self._inference_loop,
                                          daemon=True, name="AnomalyThread")
        self._thread.start()
        logger.info("✅ Anomaly detection thread started.")

    def stop(self):
        """Signal the inference thread to stop, release webcam, and reset references."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        if self.cap is not None and not getattr(self, "_shared_cap", False):
            self.cap.release()
            self.cap = None
        logger.info("⏹ Anomaly detector stopped and camera released.")

    # ─── Main Inference Loop ─────────────────────────────────
    def _inference_loop(self):
        while self._running:
            frame = self._grab_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            self._frame_count += 1
            annotated, result_data = self._run_inference(frame)

            ok, jpeg = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 82]
            )
            if not ok:
                logger.warning(
                    "AnomalyDetector: cv2.imencode failed on frame #%d – skipping.",
                    self._frame_count,
                )
                time.sleep(0.04)
                continue

            with self._lock:
                self._latest_frame  = jpeg.tobytes()
                self._latest_result = result_data

            time.sleep(0.04)   # ~25 FPS

    # ─── Frame Acquisition ───────────────────────────────────
    def _grab_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        # Demo mode
        demo = np.zeros((480, 640, 3), dtype=np.uint8)
        demo[:] = (15, 20, 32)
        cv2.putText(demo, "DEMO MODE - Anomaly Detection",
                    (120, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 200, 120), 2, cv2.LINE_AA)
        cv2.putText(demo, "Connect webcam for live detection",
                    (110, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (80, 130, 160), 1, cv2.LINE_AA)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(demo, ts, (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (60, 90, 120), 1, cv2.LINE_AA)
        return demo

    # ─── Dual-Model Inference ────────────────────────────────
    def _run_inference(self, frame: np.ndarray):
        """
        Run YOLO on a single frame.
        Returns (annotated_frame, result_dict).
        """
        h, w = frame.shape[:2]
        ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        annotated = frame.copy()

        result_data = {
            "status":        "Normal",
            "score":         0.0,
            "anomaly_type":  "Normal",
            "confidence":    0,
            "timestamp":     ts,
            "camera":        "CAM-02",
            "active":        False,
            "bbox":          None,
        }

        # ── PRIMARY: YOLO object detection ──────────────────
        best_conf = 0.0
        best_box  = None
        best_cls  = None

        if self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(
                    frame,
                    conf=self.conf_threshold,
                    verbose=False,
                    imgsz=640,
                    stream=False,
                )
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        conf_val = float(box.conf[0])
                        cls_id   = int(box.cls[0])
                        if conf_val >= self.conf_threshold and conf_val > best_conf:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            box_area = (x2 - x1) * (y2 - y1)
                            if box_area >= (MIN_BOX_AREA_FRACTION * w * h):
                                best_conf = conf_val
                                best_cls  = cls_id
                                best_box  = [int(x1), int(y1), int(x2), int(y2)]

                if best_box is not None:
                    anomaly_name = str(self.yolo_model.names[best_cls]).title()
                    confidence   = int(round(best_conf * 100))
                    meta         = ANOMALY_META.get(anomaly_name, {"level": "High", "emoji": "⚠️"})
                    color        = BOX_COLORS.get(anomaly_name, (0, 0, 255))

                    # Draw bounding box
                    x1, y1, x2, y2 = best_box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    # Label background
                    label = f"{meta.get('emoji', '')} {anomaly_name.upper()} {confidence}%"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
                    cv2.putText(annotated, label, (x1 + 4, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                    # Status bar at bottom
                    overlay = annotated.copy()
                    cv2.rectangle(overlay, (30, h - 70), (w - 30, h - 40), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
                    cv2.putText(annotated,
                                f"DETECTED: {anomaly_name.upper()}  |  CONF: {confidence}%  |  THREAT: {meta['level'].upper()}",
                                (36, h - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                    # Compute score as normalized confidence
                    score = round(best_conf, 4)
                    anomaly_pct = min(confidence, 100)

                    result_data.update({
                        "status":       meta["level"],
                        "score":        score,
                        "anomaly_type": anomaly_name,
                        "confidence":   confidence,
                        "active":       True,
                        "bbox":         best_box,
                    })

                    logger.info(
                        "AnomalyDetector: 🚨 %s detected – confidence=%d%% threat=%s",
                        anomaly_name, confidence, meta["level"],
                    )

            except Exception as e:
                logger.error("AnomalyDetector: YOLO inference error – %s", e, exc_info=True)

        self._draw_hud(annotated, result_data)
        return annotated, result_data

    # ─── Autoencoder Thumbnail Overlay ──────────────────────
    def _draw_hud(self, frame: np.ndarray, result: dict):
        h, w = frame.shape[:2]
        ts   = result.get("timestamp", "")

        # Corner brackets
        bl = 22
        c  = (0, 200, 120)
        cv2.line(frame, (5, 5),  (5 + bl, 5),  c, 2)
        cv2.line(frame, (5, 5),  (5, 5 + bl),  c, 2)
        cv2.line(frame, (w - 5, 5),  (w - 5 - bl, 5),  c, 2)
        cv2.line(frame, (w - 5, 5),  (w - 5, 5 + bl),  c, 2)
        cv2.line(frame, (5, h - 5), (5 + bl, h - 5), c, 2)
        cv2.line(frame, (5, h - 5), (5, h - 5 - bl), c, 2)
        cv2.line(frame, (w - 5, h - 5), (w - 5 - bl, h - 5), c, 2)
        cv2.line(frame, (w - 5, h - 5), (w - 5, h - 5 - bl), c, 2)

        # Camera label top-left
        cv2.putText(frame, "CAM-02 | HUMAN ANOMALY DETECTION",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 200, 120), 1, cv2.LINE_AA)

        # Status badge top-right
        status = result.get("status", "Normal")
        badge_color = (0, 0, 200) if result.get("active") else (0, 150, 80)
        cv2.rectangle(frame, (w - 90, 8), (w - 8, 28), badge_color, -1)
        cv2.putText(frame, f" {status.upper()}", (w - 85, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Timestamp bottom-left
        cv2.putText(frame, ts, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 130, 160), 1, cv2.LINE_AA)

        # Model info bottom-right
        cv2.putText(frame, f"YOLOv8 | Frame #{self._frame_count}",
                    (w - 200, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 130, 160), 1, cv2.LINE_AA)

    # ─── Public Accessors ────────────────────────────────────
    def get_latest_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_frame

    def get_latest_result(self) -> dict:
        with self._lock:
            return dict(self._latest_result)

    def is_running(self) -> bool:
        return self._running

    # ─── One-Shot Inference (for uploads) ────────────────────
    def run_on_image(self, frame: np.ndarray) -> dict:
        """
        Run YOLO on a single BGR numpy frame.
        Returns result dict + annotated JPEG (base64).
        For single images, YOLO detections are immediately valid
        (no streak/smoothing requirement).
        """
        import base64
        import time as _time
        t0 = _time.perf_counter()


        print("Running YOLO")
        annotated, result = self._run_inference(frame)
        print("Prediction result:", result)
        elapsed_ms = int((_time.perf_counter() - t0) * 1000)


        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return {
            **result,
            "annotated_b64":     base64.b64encode(jpeg.tobytes()).decode("utf-8"),
            "detection_time_ms": elapsed_ms,
        }

    def run_on_video(self, video_path: str, output_path: str,
                     progress_callback=None) -> dict:
        """
        Process every frame of video_path with YOLO.
        Writes annotated output to output_path.
        Returns { total_frames, anomaly_frames, output_path }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file"}

        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))


        anomaly_frames = []
        frame_idx      = 0
        last_pct       = -1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            print("Running YOLO")
            annotated, result = self._run_inference(frame)
            print("Prediction result:", result)

            out.write(annotated)


            if result["active"]:
                anomaly_frames.append({
                    "frame":        frame_idx,
                    "anomaly_type": result["anomaly_type"],
                    "confidence":   result.get("confidence", 0),
                    "score":        result["score"],
                    "status":       result["status"],
                    "timestamp":    result["timestamp"],
                })

            if progress_callback and total > 0:
                pct = int(frame_idx / total * 100)
                if pct != last_pct:
                    last_pct = pct
                    progress_callback(pct)

        cap.release()
        out.release()


        return {
            "total_frames":   frame_idx,
            "anomaly_frames": anomaly_frames,
            "output_path":    output_path,
        }


