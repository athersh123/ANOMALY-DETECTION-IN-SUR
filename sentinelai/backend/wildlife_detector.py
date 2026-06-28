# ============================================================
#  SentinelAI – Wildlife Detector Module
#  YOLOv8 · Custom 6-class Wildlife Model
#  Model: D:\Anomaly detection\weights\best.pt
#  Classes: Tiger, Leopard, Lion, Bear, Elephant, Cheetah
# ============================================================

import os
import cv2
from matplotlib.pyplot import box
import numpy as np
import threading
import time
import logging
from datetime import datetime
from collections import deque
import torch

import sys
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)
from model_utils import resolve_model_path

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
from ultralytics import YOLO

logger = logging.getLogger("WildlifeDetector")

# ─── Class Configuration ────────────────────────────────────
CLASS_NAMES = {
    0: "Tiger",
    1: "Leopard",
    2: "Lion",
    3: "Bear",
    4: "Elephant",
    5: "Cheetah",
}

THREAT_LEVELS = {
    "Tiger":    {"level": "Critical",  "color": (0,   0,   255), "emoji": "🐅"},
    "Leopard":  {"level": "Critical",  "color": (0,   69,  255), "emoji": "🐆"},
    "Lion":     {"level": "Critical",  "color": (0,   0,   200), "emoji": "🦁"},
    "Bear":     {"level": "High",      "color": (0,  140,  255), "emoji": "🐻"},
    "Elephant": {"level": "High",      "color": (255, 144,  30), "emoji": "🐘"},
    "Cheetah":  {"level": "High",      "color": (0,  100,  255), "emoji": "🐾"},
}

# ─── Bounding Box Colors (BGR) ───────────────────────────────
BOX_COLORS = {
    "Tiger":    (0,   0,   255),   # Red
    "Leopard":  (0,  60,   255),   # Orange-Red
    "Lion":     (0,  215,  255),   # Gold
    "Bear":     (42,  42,  165),   # Brown
    "Elephant": (205, 133,  63),   # Blue-Gray
    "Cheetah":  (0,  165,  255),   # Orange
}

# ─── Bounding Box Filters ────────────────────────────────────
MIN_BOX_AREA_FRACTION = 0.003  # 0.3% of the frame area (Issue 2)


class WildlifeDetector:
    """
    Thread-safe YOLOv8 wildlife detector.
    Reads from webcam, runs inference, stores latest result.
    """

    def __init__(self, model_path: str, camera_index: int = 0,
                 confidence_threshold: float = 0.40):
        self.model_path          = resolve_model_path(model_path)
        self.camera_index        = camera_index
        self.conf_threshold      = max(0.05, min(0.95, float(confidence_threshold)))

        self.model               = None
        self.cap                 = None
        self._lock               = threading.Lock()
        self._running            = False
        self._thread             = None

        # Confidence Smoothing (Issue 3)
        self._conf_buffer        = deque(maxlen=3)
        self._last_confirmed_animal = "None"
        self._stable_streak      = 0

        # FPS Tracking (Issue 4)
        self._fps_start_time     = time.time()
        self._fps_window_count   = 0
        self._current_fps        = 0.0

        # Latest inference results (shared state)
        self._latest_frame       = None   # raw JPEG bytes (annotated)
        self._latest_result      = {
            "animal":     "None",
            "confidence": 0,
            "threat":     "None",
            "camera":     "CAM-01",
            "timestamp":  "",
            "bbox":       None,
            "active":     False,
        }
        self._frame_count        = 0
        self._detection_history  = []   # rolling 50-item list

        self._load_model()

    # ─── Model Loading ───────────────────────────────────────
    def _load_model(self):
        logger.info(f"Loading YOLO model from: {self.model_path}")
        try:
            import ultralytics.nn.tasks as u_tasks
            torch.serialization.add_safe_globals([u_tasks.DetectionModel])
            self.model = YOLO(self.model_path)
            
            print("Wildlife model loaded:", self.model_path)
            print("Classes:", self.model.names)
            
            expected_classes = {"tiger", "leopard", "lion", "bear", "elephant", "cheetah"}
            loaded_classes = {str(name).lower() for name in self.model.names.values()}
            if loaded_classes != expected_classes:
                raise ValueError(f"Loaded classes do not match expected wildlife classes. Expected: {expected_classes}, Got: {loaded_classes}")

            # Warm-up pass
            dummy = np.zeros((128, 128, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, conf=self.conf_threshold)
            logger.info("✅ YOLO model loaded and warmed up.")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            self.model = None
            raise  # Stop startup and raise an error

    # ─── Camera Start ────────────────────────────────────────
    def start(self):
        if self._running:
            return

        # Reset confidence smoothing states
        self._conf_buffer.clear()
        self._last_confirmed_animal = "None"
        self._stable_streak = 0

        # Reset FPS tracking
        self._fps_start_time = time.time()
        self._fps_window_count = 0
        self._current_fps = 0.0

        self.cap = cv2.VideoCapture(self.camera_index)
        if self.cap is None or not self.cap.isOpened():
            logger.warning("⚠️  Webcam not found. Using demo mode.")
            self.cap = None
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            logger.info("✅ Webcam opened.")

        self._running = True
        self._thread  = threading.Thread(target=self._inference_loop,
                                          daemon=True, name="WildlifeThread")
        self._thread.start()
        logger.info("✅ Wildlife detection thread started.")

    def stop(self):
        """Signal the inference thread to stop, release webcam, and reset references."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None          # prevent stale reference on next start()
        if self.cap is not None:
            self.cap.release()
            self.cap = None              # ensure next start() opens a fresh capture
        logger.info("⏹ Wildlife detector stopped and camera released.")

    # ─── Main Inference Loop ──────────────────────────────────
    def _inference_loop(self):
        while self._running:
            frame = self._grab_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            self._frame_count += 1

            # FPS calculation
            now = time.time()
            self._fps_window_count += 1
            if now - self._fps_start_time >= 1.0:
                elapsed = now - self._fps_start_time
                self._current_fps = round(self._fps_window_count / elapsed, 1)
                self._fps_window_count = 0
                self._fps_start_time = now

            annotated, result_data = self._run_inference(frame)

            # Encode frame to JPEG – check success before writing shared state
            ok, jpeg = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if not ok:
                logger.warning(
                    "WildlifeDetector: cv2.imencode failed on frame #%d – skipping frame.",
                    self._frame_count,
                )
                time.sleep(0.03)
                continue

            with self._lock:
                self._latest_frame  = jpeg.tobytes()
                self._latest_result = result_data
                # Rolling detection history kept inside the lock to prevent
                # concurrent read/write on the list from different threads.
                if result_data["active"]:
                    self._detection_history.append(dict(result_data))
                    if len(self._detection_history) > 50:
                        self._detection_history.pop(0)

            time.sleep(0.03)   # ~30 FPS target

    # ─── Frame Acquisition ───────────────────────────────────
    def _grab_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        # Demo mode: generate a dark frame with overlay text
        demo = np.zeros((480, 640, 3), dtype=np.uint8)
        demo[:] = (18, 24, 38)
        cv2.putText(demo, "DEMO MODE - No Camera",
                    (140, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 212, 255), 2, cv2.LINE_AA)
        cv2.putText(demo, "Connect webcam to enable live detection",
                    (70, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100, 150, 200), 1, cv2.LINE_AA)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(demo, ts, (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (60, 100, 140), 1, cv2.LINE_AA)
        return demo

    # ─── YOLO Inference ─────────────────────────────────────
    def _run_inference(self, frame: np.ndarray, bypass_smoothing: bool = False):
        """Run YOLO on frame, draw annotations, return (annotated_frame, result_dict)."""
        h, w = frame.shape[:2]
        annotated = frame.copy()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result_data = {
            "animal":     "None",
            "confidence": 0,
            "threat":     "None",
            "camera":     "CAM-01",
            "timestamp":  ts,
            "bbox":       None,
            "active":     False,
        }

        if self.model is None:
            self._draw_hud(annotated, result_data)
            return annotated, result_data

        try:
            print("RUNNING YOLO...")
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                verbose=False,
                imgsz=640,
                stream=False,
            )
            best_conf = 0.0
            best_box = None
            best_cls = None
            
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    print("BOX FOUND")
                    print("CLASS:", int(box.cls[0]))
                    print("CONF:", float(box.conf[0]))
                    conf_val = float(box.conf[0])
                    cls_id   = int(box.cls[0])
                    if conf_val >= self.conf_threshold and conf_val > best_conf:
                        # Ignore very small bounding boxes (Issue 2)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        box_area = (x2 - x1) * (y2 - y1)
                        if box_area >= (MIN_BOX_AREA_FRACTION * w * h):
                            best_conf = conf_val
                            best_cls  = cls_id
                            best_box  = [x1, y1, x2, y2]
      
            # Confidence smoothing (Issue 3)
            detected_animal = "None"
            if best_box is not None:
                detected_animal = str(self.model.names[best_cls]).title()
                print("DETECTED:", detected_animal)

            if bypass_smoothing:
                confirmed_animal = detected_animal
            else:
                self._conf_buffer.append(detected_animal)
                
                # Determine confirmed animal from buffer (requires >= 2 of last 3 frames)
                counts = {}
                for item in self._conf_buffer:
                    counts[item] = counts.get(item, 0) + 1
                    
                confirmed_animal = "None"
                for animal, count in counts.items():
                    if animal != "None" and count >= 2:
                        confirmed_animal = animal
                        break
                
                # Update streak and logs
                if confirmed_animal != "None":
                    if confirmed_animal == self._last_confirmed_animal:
                        self._stable_streak += 1
                    else:
                        self._last_confirmed_animal = confirmed_animal
                        self._stable_streak = 1
                else:
                    self._last_confirmed_animal = "None"
                    self._stable_streak = 0

                logger.debug(
                    "WildlifeDetector: raw_detected=%s, confirmed=%s, streak=%d, buffer=%s",
                    detected_animal, confirmed_animal, self._stable_streak, list(self._conf_buffer)
                )

            # Fire detection if we have a confirmed animal
            if confirmed_animal != "None":
                animal_name = confirmed_animal
                threat_info = THREAT_LEVELS.get(animal_name, {"level": "Medium", "color": (0, 200, 255)})
                color       = BOX_COLORS.get(animal_name, (0, 0, 255))
                
                # We can only draw the bounding box if the current frame actually contains a detection for this animal
                is_active = True
                has_bbox = (detected_animal == confirmed_animal) and (best_box is not None)
                
                if has_bbox:
                    x1, y1, x2, y2 = best_box
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    # Corner decorators
                    corner_len = 15
                    cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, 3)
                    cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, 3)
                    cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, 3)
                    cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, 3)
                    cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, 3)
                    cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, 3)
                    cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, 3)
                    cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, 3)

                    # Label background
                    label      = f"{animal_name}  {best_conf * 100:.1f}%"
                    font       = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.55
                    thickness  = 1
                    (lw, lh), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    lx, ly = x1, y1 - 8
                    cv2.rectangle(annotated,
                                  (lx - 2, ly - lh - 6),
                                  (lx + lw + 4, ly + 2),
                                  color, -1)
                    cv2.putText(annotated, label, (lx, ly),
                                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                    # Threat badge
                    threat_lbl = f"  {threat_info['level'].upper()}  "
                    cv2.rectangle(annotated, (x1, y2 + 2), (x1 + 110, y2 + 22), color, -1)
                    cv2.putText(annotated, threat_lbl, (x1 + 2, y2 + 17),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                print("FINAL DETECTION:", detected_animal)
                print("BEST CONF:", best_conf)
                print("BEST BOX:", best_box)
                result_data = {
                    
                    "animal":     animal_name,
                    "confidence": round(best_conf * 100, 1) if has_bbox else 0.0,
                    "threat":     threat_info["level"],
                    "camera":     "CAM-01",
                    "timestamp":  ts,
                    "bbox":       [int(x1), int(y1), int(x2), int(y2)] if has_bbox else None,
                    "active":     is_active,
                }

        except Exception as e:
            logger.error(f"YOLO inference error: {e}")

        self._draw_hud(annotated, result_data)
        return annotated, result_data

    # ─── HUD Overlay ────────────────────────────────────────
    def _draw_hud(self, frame: np.ndarray, result: dict):
        h, w = frame.shape[:2]

        # Scanline effect (every 4th row slight darken)
        overlay = frame.copy()
        for y in range(0, h, 4):
            cv2.line(overlay, (0, y), (w, y), (0, 0, 0), 1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        # Corner brackets
        bl = 22
        c  = (0, 212, 255)
        cv2.line(frame, (5, 5),  (5 + bl, 5),  c, 2)
        cv2.line(frame, (5, 5),  (5, 5 + bl),  c, 2)
        cv2.line(frame, (w - 5, 5),  (w - 5 - bl, 5),  c, 2)
        cv2.line(frame, (w - 5, 5),  (w - 5, 5 + bl),  c, 2)
        cv2.line(frame, (5, h - 5), (5 + bl, h - 5), c, 2)
        cv2.line(frame, (5, h - 5), (5, h - 5 - bl), c, 2)
        cv2.line(frame, (w - 5, h - 5), (w - 5 - bl, h - 5), c, 2)
        cv2.line(frame, (w - 5, h - 5), (w - 5, h - 5 - bl), c, 2)

        # Camera ID top-left
        cam_lbl = "CAM-01 | WILDLIFE SURVEILLANCE"
        cv2.putText(frame, cam_lbl, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 212, 255), 1, cv2.LINE_AA)

        # LIVE badge top-right
        cv2.rectangle(frame, (w - 75, 8), (w - 8, 28), (0, 0, 200), -1)
        cv2.putText(frame, " LIVE", (w - 70, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Timestamp bottom-left
        ts = result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        cv2.putText(frame, ts, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 160, 200), 1, cv2.LINE_AA)

        # Model info bottom-right (Issue 7)
        model_name = os.path.basename(self.model_path) if self.model_path else "best.pt"
        fps_val = self.fps
        hud_text = f"YOLOv8 · {model_name} | FPS: {fps_val} | Frame #{self._frame_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1
        (tw, th), _ = cv2.getTextSize(hud_text, font, font_scale, thickness)
        cv2.putText(frame, hud_text, (w - tw - 10, h - 10),
                    font, font_scale, (120, 160, 200), thickness, cv2.LINE_AA)

        # Alert bar if detection active
        if result.get("active"):
            bar_h = 36
            bar_overlay = frame.copy()
            cv2.rectangle(bar_overlay, (0, h - bar_h - 30), (w, h - 30), (0, 0, 180), -1)
            cv2.addWeighted(bar_overlay, 0.55, frame, 0.45, 0, frame)
            alert_txt = (f"  ALERT: {result['animal'].upper()} DETECTED  |"
                         f"  CONF: {result['confidence']}%  |"
                         f"  THREAT: {result['threat'].upper()}")
            cv2.putText(frame, alert_txt, (6, h - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # ─── Public Accessors ────────────────────────────────────
    def get_latest_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_frame

    def get_latest_result(self) -> dict:
        with self._lock:
            return dict(self._latest_result)

    def get_detection_history(self) -> list:
        with self._lock:
            return list(self._detection_history[-20:])

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        with self._lock:
            return self._current_fps

    # ─── One-Shot Inference (for uploads) ────────────────────
    def run_on_image(self, frame: np.ndarray) -> dict:
        """
        Run YOLO on a single BGR numpy frame (no threading).
        Returns result dict + annotated JPEG bytes (base64 encoded).
        """
        import base64
        import time as _time
        t0 = _time.perf_counter()
        annotated, result = self._run_inference(frame, bypass_smoothing=True)
        elapsed_ms = int((_time.perf_counter() - t0) * 1000)
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return {
            **result,
            "annotated_b64":   base64.b64encode(jpeg.tobytes()).decode("utf-8"),
            "detection_time_ms": elapsed_ms,
        }

    def run_on_video(self, video_path: str, output_path: str,
                     progress_callback=None) -> dict:
        """
        Process every frame of video_path with YOLO.
        Writes annotated output to output_path (mp4).
        progress_callback(pct:int) called periodically.
        Returns { total_frames, detections, output_path }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file"}

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        detections   = []
        frame_idx    = 0
        last_pct     = -1

        # Reset confidence smoothing state for video processing
        self._conf_buffer.clear()
        self._last_confirmed_animal = "None"
        self._stable_streak = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            annotated, result = self._run_inference(frame)
            out.write(annotated)

            if result["active"]:
                detections.append({
                    "frame":      frame_idx,
                    "animal":     result["animal"],
                    "confidence": result["confidence"],
                    "threat":     result["threat"],
                    "timestamp":  result["timestamp"],
                })

            if progress_callback and total > 0:
                pct = int(frame_idx / total * 100)
                if pct != last_pct:
                    last_pct = pct
                    progress_callback(pct)

        cap.release()
        out.release()

        return {
            "total_frames": frame_idx,
            "detections":   detections,
            "output_path":  output_path,
        }
