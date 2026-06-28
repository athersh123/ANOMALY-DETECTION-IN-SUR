# ============================================================
#  SentinelAI – Flask Backend API  (Production Edition)
#  app.py
# ============================================================
#
#  Endpoints (original):
#    GET  /                          Health check
#    GET  /api/status                System status
#    GET  /api/wildlife              Latest wildlife result
#    GET  /api/anomaly               Latest anomaly result
#    GET  /api/wildlife/history      Detection history list
#    GET  /api/anomaly/history       Error history (chart)
#    GET  /api/alerts                Combined alert feed
#    GET  /video_feed/wildlife       MJPEG stream
#    GET  /video_feed/anomaly        MJPEG stream
#    POST /api/settings              Update thresholds
#
#  NEW endpoints:
#    POST /api/wildlife/upload-image   One-shot YOLO on uploaded JPG/PNG
#    POST /api/anomaly/upload-image    One-shot autoencoder on uploaded image
#    POST /api/wildlife/upload-video   YOLO on every frame of uploaded video
#    POST /api/anomaly/upload-video    Autoencoder on every frame of uploaded video
#    GET  /api/job/<job_id>            Poll background video-processing job
#    GET  /api/history                 Query SQLite detection history
#    GET  /api/history/export-csv      Download CSV of detections
#    GET  /api/history/stats           Aggregate counts
#    GET  /api/history/filters         Distinct label/camera lists
#    GET  /api/report/download         Download PDF report
#    GET  /snapshots/<filename>        Serve saved snapshots
# ============================================================

import os
import sys
from unittest import result
import uuid
import time
import base64
import logging
import threading
import io
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename

# sys is already imported above; do not duplicate it here.
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)
from model_utils import resolve_model_path

# ─── Path setup ─────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR       = os.path.dirname(BASE_DIR)
SNAPSHOTS_DIR  = os.path.join(BASE_DIR, "snapshots")
UPLOADS_DIR    = os.path.join(BASE_DIR, "uploads")

os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR,   exist_ok=True)

# Wildlife YOLO – 6-class (Tiger, Leopard, Lion, Bear, Elephant, Cheetah)
WILDLIFE_YOLO_PATH = resolve_model_path(r"C:\Users\Athersh JR\runs\detect\train-12\weights\best.pt", r"D:\Anomaly detection\weights\best.pt", "weights/best.pt")
# Anomaly YOLO – 5-class (Accident, Explosion, Fighting, Shooting, Vandalism)
ANOMALY_YOLO_PATH  = r"C:\Users\Athersh JR\runs\detect\train-17\weights\best.pt"
# Autoencoder removed

ALLOWED_IMAGES     = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEOS     = {"mp4", "avi", "mov", "mkv", "wmv"}
MAX_UPLOAD_MB      = 500
MAX_IMAGE_DIMENSION = 8192   # reject images wider or taller than this (px)

# ─── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(BASE_DIR, "sentinelai.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("SentinelAI")

# ─── Import modules ──────────────────────────────────────────
from wildlife_detector import WildlifeDetector
from anomaly_detector  import AnomalyDetector
import database as db

# ─── Flask App ───────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app, origins="*")

# ─── Globals ────────────────────────────────────────────────
wildlife_detector: WildlifeDetector = None
anomaly_detector:  AnomalyDetector  = None
startup_time       = datetime.now()
alert_log          = []
_critical_alert_count = 0
_alert_lock        = threading.Lock()

# Background video-processing jobs: job_id → { status, progress, result }
_jobs      = {}
_jobs_lock = threading.Lock()


# ════════════════════════════════════════════════════════════
#  STARTUP
# ════════════════════════════════════════════════════════════

def init_detectors():
    global wildlife_detector, anomaly_detector

    logger.info("=" * 60)
    logger.info("  SentinelAI Backend Starting (Production Edition)")
    logger.info(f"  Wildlife YOLO : {WILDLIFE_YOLO_PATH}")
    logger.info(f"  Anomaly YOLO  : {ANOMALY_YOLO_PATH}")
    logger.info(f"  Snapshots     : {SNAPSHOTS_DIR}")
    logger.info("=" * 60)

    # ── Init DB ──────────────────────────────────────────────
    db.init_db()

    # ── Wildlife detector – LOAD ONLY, do NOT start camera ───
    logger.info("Loading Wildlife Detector (model only)…")
    wildlife_detector = WildlifeDetector(
        model_path           = WILDLIFE_YOLO_PATH,
        camera_index         = 0,
        confidence_threshold = 0.40,
    )
    # NOTE: wildlife_detector.start() is NOT called here.
    # The camera starts only when POST /api/start-wildlife is called.

    # ── Anomaly detector – LOAD ONLY, do NOT start camera ────
    logger.info("Loading Anomaly Detector (model only)…")
    anomaly_detector = AnomalyDetector(
        yolo_model_path       = ANOMALY_YOLO_PATH,
        camera_index          = 0,
        history_window        = 60,
    )
    # NOTE: anomaly_detector.start() is NOT called here.
    # The camera starts only when POST /api/start-anomaly is called.

    # ── Alert watcher ────────────────────────────────────────
    watcher = threading.Thread(target=_alert_watcher, daemon=True, name="AlertWatcher")
    watcher.start()

    logger.info("✅ Models loaded. Cameras are OFF – waiting for user to start monitoring.")


# ─── Alert watcher ──────────────────────────────────────────
def _alert_watcher():
    last_wl_ts = ""
    last_an_ts = ""

    while True:
        try:
            if wildlife_detector:
                wr = wildlife_detector.get_latest_result()
                ts = wr.get("timestamp", "")
                if wr.get("active") and ts != last_wl_ts:
                    last_wl_ts = ts
                    entry = {
                        "id":         int(time.time() * 1000),
                        "type":       "wildlife",
                        "title":      f"{wr['animal']} Detected",
                        "animal":     wr["animal"],
                        "confidence": wr["confidence"],
                        "threat":     wr["threat"],
                        "camera":     wr["camera"],
                        "timestamp":  ts,
                        "level":      "critical" if wr["threat"] in ("Critical",) else "high",
                    }
                    _push_alert(entry)
                    # Save snapshot from live frame
                    snap = wildlife_detector.get_latest_frame()
                    snap_path = ""
                    if snap:
                        snap_name = f"wl_live_{int(time.time())}.jpg"
                        snap_path = os.path.join(SNAPSHOTS_DIR, snap_name)
                        with open(snap_path, "wb") as f:
                            f.write(snap)
                        snap_path = f"snapshots/{snap_name}"
                    # Persist to DB
                    db.insert_detection(
                        category   = "wildlife",
                        label      = wr["animal"],
                        confidence = wr["confidence"],
                        score      = 0.0,
                        status     = wr["threat"],
                        camera_id  = wr["camera"],
                        image_path = snap_path,
                        source     = "live",
                        timestamp  = ts,
                    )

            if anomaly_detector:
                ar = anomaly_detector.get_latest_result()
                ts = ar.get("timestamp", "")
                if ar.get("active") and ts != last_an_ts:
                    last_an_ts = ts
                    entry = {
                        "id":           int(time.time() * 1000) + 1,
                        "type":         "human",
                        "title":        f"{ar['anomaly_type']} Detected",
                        "anomaly_type": ar["anomaly_type"],
                        "score":        ar["score"],
                        "status":       ar["status"],
                        "camera":       ar["camera"],
                        "timestamp":    ts,
                        "level":        "critical" if ar["status"] == "Anomaly" else "high",
                    }
                    _push_alert(entry)
                    snap = anomaly_detector.get_latest_frame()
                    snap_path = ""
                    if snap:
                        snap_name = f"an_live_{int(time.time())}.jpg"
                        snap_path = os.path.join(SNAPSHOTS_DIR, snap_name)
                        with open(snap_path, "wb") as f:
                            f.write(snap)
                        snap_path = f"snapshots/{snap_name}"
                    db.insert_detection(
                        category   = "human",
                        label      = ar["anomaly_type"],
                        confidence = round(ar["score"] * 100, 1),
                        score      = ar["error"],
                        status     = ar["status"],
                        camera_id  = ar["camera"],
                        image_path = snap_path,
                        source     = "live",
                        timestamp  = ts,
                    )
        except Exception as e:
            logger.error(f"Alert watcher error: {e}")

        time.sleep(1)


def _push_alert(entry: dict):
    global _critical_alert_count
    with _alert_lock:
        alert_log.insert(0, entry)
        if entry.get("level") == "critical":
            _critical_alert_count += 1
        if len(alert_log) > 100:
            popped = alert_log.pop()
            if popped.get("level") == "critical":
                _critical_alert_count -= 1
    logger.info(f"🚨 Alert: {entry['title']}")
    # Telegram (if configured)
    _send_telegram(entry.get("title", ""), entry.get("camera", ""))


def _send_telegram(title: str, camera: str):
    """Non-blocking Telegram alert. Silently skips if not configured."""
    token   = db.get_setting("telegram_token", "")
    chat_id = db.get_setting("telegram_chat_id", "")
    if not token or not chat_id:
        return
    def _send():
        try:
            import requests as req
            ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = f"🚨 SentinelAI Alert\n{title}\n📷 {camera}\n🕐 {ts}"
            req.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": msg},
                timeout=5,
            )
        except Exception as e:
            logger.debug(f"Telegram send failed: {e}")
    threading.Thread(target=_send, daemon=True).start()


def ensure_wildlife_detector_started():
    global wildlife_detector
    if wildlife_detector is not None:
        return wildlife_detector
    logger.info("Initializing wildlife detector lazily for upload route")
    wildlife_detector = WildlifeDetector(
        model_path           = WILDLIFE_YOLO_PATH,
        camera_index         = 0,
        confidence_threshold = 0.40,
    )
    # NOTE: We do not call wildlife_detector.start() here to avoid opening the webcam
    # when doing one-shot image/video processing (Issue 11).
    return wildlife_detector


# ─── MJPEG Generator ────────────────────────────────────────
def _mjpeg_stream(detector, label: str):
    blank = _blank_jpeg(label)
    while True:
        frame_bytes = detector.get_latest_frame() if detector else None
        data = frame_bytes if frame_bytes else blank
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + data + b"\r\n"
        )
        time.sleep(0.033)


def _blank_jpeg(label="LOADING…") -> bytes:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (18, 24, 38)
    cv2.putText(img, label, (200, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 212, 255), 2)
    cv2.putText(img, "Initializing AI models…", (160, 265),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 130, 180), 1)
    _, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpg.tobytes()


# ─── File Upload Helpers ─────────────────────────────────────
def _allowed(filename: str, types: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in types


def _read_uploaded_image(file_storage) -> np.ndarray | None:
    """Read werkzeug FileStorage → BGR numpy array."""
    data = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _save_snapshot(jpeg_bytes: bytes, prefix: str) -> str:
    """Save JPEG bytes to snapshots dir, return relative path."""
    name = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(SNAPSHOTS_DIR, name)
    with open(path, "wb") as f:
        f.write(jpeg_bytes)
    return f"snapshots/{name}"


# ════════════════════════════════════════════════════════════
#  ORIGINAL ROUTES
# ════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service":  "SentinelAI Backend",
        "version":  "2.0.0",
        "status":   "running",
        "uptime_s": int((datetime.now() - startup_time).total_seconds()),
    })


@app.route("/api/status", methods=["GET"])
def api_status():
    wd_running = wildlife_detector is not None and getattr(wildlife_detector, '_running', False)
    ad_running = anomaly_detector  is not None and getattr(anomaly_detector,  '_running', False)
    cam_active = wd_running or ad_running
    stats = db.get_stats()
    return jsonify({
        # Legacy fields (kept for backward compatibility)
        "wildlife_detector": "online" if wd_running else "offline",
        "anomaly_detector":  "online" if ad_running else "offline",
        # New explicit fields for on-demand monitoring UI
        "wildlife_running":  wd_running,
        "anomaly_running":   ad_running,
        "camera_active":     cam_active,
        "total_cameras":     24,
        "active_alerts":     _critical_alert_count,
        "wildlife_alerts":   stats["today"]["wildlife"],
        "human_anomalies":   stats["today"]["human"],
        "uptime_s":          int((datetime.now() - startup_time).total_seconds()),
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/wildlife", methods=["GET"])
def api_wildlife():
    if not wildlife_detector:
        return jsonify({"error": "Wildlife detector not initialized"}), 503
    r = wildlife_detector.get_latest_result()
    return jsonify({
        "animal":      r.get("animal",     "None"),
        "confidence":  r.get("confidence", 0),
        "threat":      r.get("threat",     "None"),
        "camera":      r.get("camera",     "CAM-01"),
        "timestamp":   r.get("timestamp",  ""),
        "bbox":        r.get("bbox",       None),
        "active":      r.get("active",     False),
        "frame_count": wildlife_detector.frame_count,
    })


@app.route("/api/wildlife/history", methods=["GET"])
def api_wildlife_history():
    if not wildlife_detector:
        return jsonify([]), 503
    return jsonify(wildlife_detector.get_detection_history())


@app.route("/api/anomaly", methods=["GET"])

def api_anomaly():
    if not anomaly_detector:
        return jsonify({"error": "Anomaly detector not initialized"}), 503
    r = anomaly_detector.get_latest_result()
    print("API anomaly response:", anomaly_detector._latest_result)
    return jsonify({
        "status":       r.get("status",       "Normal"),
        "score":        r.get("score",        0.0),
        "normal":       r.get("normal_pct",   100),
        "anomaly":      r.get("anomaly_pct",  0),
        "error":        r.get("error",        0.0),
        "threshold":    r.get("threshold",    0.00250),
        "anomaly_type": r.get("anomaly_type", "Normal"),
        "camera":       r.get("camera",       "CAM-02"),
        "timestamp":    r.get("timestamp",    ""),
        "confidence":   r.get("confidence",   0),
        "bbox":         r.get("bbox",         None),
        "active":       r.get("active",       False),
    })


@app.route("/api/anomaly/history", methods=["GET"])
def api_anomaly_history():
    if not anomaly_detector:
        return jsonify([]), 503
    history = anomaly_detector.get_error_history()
    return jsonify([
        {
            "timestamp":  h["timestamp"],
            "error":      round(h["error"], 6),
            "status":     h["status"],
            "is_anomaly": h["status"] in ("Anomaly", "Suspicious"),
        }
        for h in history
    ])


@app.route("/api/alerts", methods=["GET"])
def api_alerts():
    limit  = request.args.get("limit", 50, type=int)
    type_f = request.args.get("type",  "")
    with _alert_lock:
        data = list(alert_log)
    if type_f:
        data = [a for a in data if a.get("type") == type_f]
    return jsonify(data[:limit])


@app.route("/api/settings", methods=["POST", "GET"])
def api_settings():
    if request.method == "GET":
        return jsonify({
            "yolo_confidence":           getattr(wildlife_detector, "conf_threshold", 0.40)
                                         if wildlife_detector else 0.40,
            "anomaly_normal_threshold":  getattr(anomaly_detector, "normal_threshold", 0.00167)
                                         if anomaly_detector else 0.00167,
            "anomaly_anomaly_threshold": getattr(anomaly_detector, "anomaly_threshold", 0.00350)
                                         if anomaly_detector else 0.00350,
            "telegram_token":            db.get_setting("telegram_token", ""),
            "telegram_chat_id":          db.get_setting("telegram_chat_id", ""),
        })
    data = request.get_json(silent=True) or {}
    if "yolo_confidence" in data and wildlife_detector:
        val = max(0.05, min(0.99, float(data["yolo_confidence"])))
        wildlife_detector.conf_threshold = val
    if "anomaly_normal_threshold" in data and anomaly_detector:
        val_norm = float(data["anomaly_normal_threshold"])
        anomaly_detector.normal_threshold = val_norm
    if "anomaly_anomaly_threshold" in data and anomaly_detector:
        val_anom = float(data["anomaly_anomaly_threshold"])
        anomaly_detector.anomaly_threshold = val_anom
        if "anomaly_normal_threshold" not in data:
            anomaly_detector.normal_threshold = val_anom * 0.477
        # update suspicious threshold proportionally
        anomaly_detector.suspicious_threshold = anomaly_detector.normal_threshold + (anomaly_detector.anomaly_threshold - anomaly_detector.normal_threshold) * 0.5
    if "telegram_token" in data:
        db.set_setting("telegram_token", str(data["telegram_token"]))
    if "telegram_chat_id" in data:
        db.set_setting("telegram_chat_id", str(data["telegram_chat_id"]))
    return jsonify({"success": True, "message": "Settings applied"})


@app.route("/video_feed/wildlife")
def video_feed_wildlife():
    return Response(
        _mjpeg_stream(wildlife_detector, "WILDLIFE CAM"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video_feed/anomaly")
def video_feed_anomaly():
    return Response(
        _mjpeg_stream(anomaly_detector, "ANOMALY CAM"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ════════════════════════════════════════════════════════════
#  ON-DEMAND MONITORING CONTROL  (Start / Stop)
# ════════════════════════════════════════════════════════════

@app.route("/api/start-wildlife", methods=["POST"])
def api_start_wildlife():
    """Start wildlife detector + camera on user demand."""
    if not wildlife_detector:
        return jsonify({"error": "Wildlife detector not initialized"}), 503
    if getattr(wildlife_detector, '_running', False):
        return jsonify({"status": "already_running", "message": "Wildlife detector is already running"})
    try:
        wildlife_detector.start()
        logger.info("▶ Wildlife detector STARTED by user request")
        return jsonify({"status": "started", "message": "Wildlife detector started"})
    except Exception as e:
        logger.error(f"Failed to start wildlife detector: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stop-wildlife", methods=["POST"])
def api_stop_wildlife():
    """Stop wildlife detector + release camera on user demand."""
    if not wildlife_detector:
        return jsonify({"error": "Wildlife detector not initialized"}), 503
    if not getattr(wildlife_detector, '_running', False):
        return jsonify({"status": "already_stopped", "message": "Wildlife detector is not running"})
    try:
        wildlife_detector.stop()
        logger.info("⏹ Wildlife detector STOPPED by user request")
        return jsonify({"status": "stopped", "message": "Wildlife detector stopped"})
    except Exception as e:
        logger.error(f"Failed to stop wildlife detector: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/start-anomaly", methods=["POST"])
def api_start_anomaly():
    """Start anomaly detector + camera on user demand."""
    if not anomaly_detector:
        return jsonify({"error": "Anomaly detector not initialized"}), 503
    if getattr(anomaly_detector, '_running', False):
        return jsonify({"status": "already_running", "message": "Anomaly detector is already running"})
    try:
        # Share the wildlife camera if it is already open to save hardware resources
        shared = getattr(wildlife_detector, 'cap', None) if (
            wildlife_detector and getattr(wildlife_detector, '_running', False)
        ) else None
        anomaly_detector.start(shared_cap=shared)
        logger.info("▶ Anomaly detector STARTED by user request")
        return jsonify({"status": "started", "message": "Anomaly detector started"})
    except Exception as e:
        logger.error(f"Failed to start anomaly detector: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stop-anomaly", methods=["POST"])
def api_stop_anomaly():
    """Stop anomaly detector + release camera on user demand."""
    if not anomaly_detector:
        return jsonify({"error": "Anomaly detector not initialized"}), 503
    if not getattr(anomaly_detector, '_running', False):
        return jsonify({"status": "already_stopped", "message": "Anomaly detector is not running"})
    try:
        anomaly_detector.stop()
        logger.info("⏹ Anomaly detector STOPPED by user request")
        return jsonify({"status": "stopped", "message": "Anomaly detector stopped"})
    except Exception as e:
        logger.error(f"Failed to stop anomaly detector: {e}")
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
#  NEW: UPLOAD IMAGE – WILDLIFE
# ════════════════════════════════════════════════════════════

@app.route("/api/wildlife/upload-image", methods=["POST"])
def upload_wildlife_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename or not _allowed(f.filename, ALLOWED_IMAGES):
        return jsonify({"error": "Invalid file type. Accepted: JPG, PNG, BMP, WEBP."}), 400

    logger.info(f"Upload received: wildlife image {f.filename}")
    detector = ensure_wildlife_detector_started()
    if detector is None:
        return jsonify({"error": "Wildlife detector not ready"}), 503

    frame = _read_uploaded_image(f)
    if frame is None:
        return jsonify({"error": "Cannot decode image. File may be corrupt."}), 422

    # Reject excessively large images that could exhaust server RAM.
    h, w = frame.shape[:2]
    if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
        return jsonify({
            "error": (
                f"Image too large ({w}×{h} px). "
                f"Maximum allowed size is {MAX_IMAGE_DIMENSION}×{MAX_IMAGE_DIMENSION} px."
            )
        }), 413

    logger.info("Model inference started for wildlife image")
    result = detector.run_on_image(frame)

    # SAVE RESULT FOR FRONTEND POLLING
    try:
        with wildlife_detector._lock:
            wildlife_detector._latest_result = result
    except:
        wildlife_detector._latest_result = result

    logger.info(f"Prediction result: {result['animal']} ({result['confidence']}% confidence)")
    # Save annotated snapshot
    jpeg_bytes  = base64.b64decode(result["annotated_b64"])
    snap_path   = _save_snapshot(jpeg_bytes, "wl_upload")

    # Persist to DB
    db.insert_detection(
        category   = "wildlife",
        label      = result["animal"],
        confidence = result["confidence"],
        score      = 0.0,
        status     = result["threat"],
        camera_id  = "UPLOAD",
        image_path = snap_path,
        source     = "upload_image",
        timestamp  = result["timestamp"],
    )

    return jsonify({
        "animal":            result["animal"],
        "confidence":        result["confidence"],
        "threat":            result["threat"],
        "active":            result["active"],
        "bbox":              result.get("bbox"),
        "timestamp":         result["timestamp"],
        "detection_time_ms": result["detection_time_ms"],
        "annotated_b64":     result["annotated_b64"],
        "snapshot_url":      f"/{snap_path}",
    })


# ════════════════════════════════════════════════════════════
#  NEW: UPLOAD IMAGE – ANOMALY
# ════════════════════════════════════════════════════════════

@app.route("/api/anomaly/upload-image", methods=["POST"])
def upload_anomaly_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename or not _allowed(f.filename, ALLOWED_IMAGES):
        return jsonify({"error": "Invalid file type. Accepted: JPG, PNG, BMP, WEBP."}), 400
    if not anomaly_detector:
        return jsonify({"error": "Anomaly detector not ready"}), 503

    logger.info(f"Upload received: anomaly image {f.filename}")
    print("Anomaly upload received")
    print("Model path:", anomaly_detector.yolo_model_path)
    frame = _read_uploaded_image(f)
    if frame is None:
        return jsonify({"error": "Cannot decode image. File may be corrupt."}), 422

    # Reject excessively large images that could exhaust server RAM.
    h, w = frame.shape[:2]
    if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
        return jsonify({
            "error": (
                f"Image too large ({w}×{h} px). "
                f"Maximum allowed size is {MAX_IMAGE_DIMENSION}×{MAX_IMAGE_DIMENSION} px."
            )
        }), 413

    logger.info("Model inference started for anomaly image")
    result = anomaly_detector.run_on_image(frame)
    with anomaly_detector._lock:
        anomaly_detector._latest_result = result
    print("Saved latest result:", anomaly_detector._latest_result)   

    print("Saved latest result:", anomaly_detector._latest_result)
    logger.info(f"Prediction result: {result['anomaly_type']} (confidence={result.get('confidence', 0)}%, score={result['score']})")

    jpeg_bytes = base64.b64decode(result["annotated_b64"])
    snap_path  = _save_snapshot(jpeg_bytes, "an_upload")

    db.insert_detection(
        category   = "human",
        label      = result["anomaly_type"],
        confidence = result.get("confidence", round(result["score"] * 100, 1)),
        score      = result["score"],
        status     = result["status"],
        camera_id  = "UPLOAD",
        image_path = snap_path,
        source     = "upload_image",
        timestamp  = result["timestamp"],
    )

    return jsonify({
    "status": result["status"],
    "score": result["score"],
    "anomaly_type": result["anomaly_type"],
    "confidence": result.get("confidence", 0),
    "active": result["active"],
    "bbox": result.get("bbox"),
    "timestamp": result["timestamp"],
    "detection_time_ms": result["detection_time_ms"],
    "annotated_b64": result["annotated_b64"],
    "snapshot_url": f"/{snap_path}"
})


# ════════════════════════════════════════════════════════════
#  NEW: UPLOAD VIDEO – WILDLIFE  (background job)
# ════════════════════════════════════════════════════════════

@app.route("/api/wildlife/upload-video", methods=["POST"])
def upload_wildlife_video():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename or not _allowed(f.filename, ALLOWED_VIDEOS):
        return jsonify({"error": "Invalid file type. Use MP4/AVI/MOV."}), 400

    logger.info(f"Upload received: wildlife video {f.filename}")
    detector = ensure_wildlife_detector_started()
    if detector is None:
        return jsonify({"error": "Wildlife detector not ready"}), 503

    job_id = str(uuid.uuid4())
    # Save uploaded file
    tmp_name = secure_filename(f"wl_{job_id[:8]}_{f.filename}")
    tmp_path = os.path.join(UPLOADS_DIR, tmp_name)
    f.save(tmp_path)

    out_name = f"wl_processed_{job_id[:8]}.mp4"
    out_path = os.path.join(SNAPSHOTS_DIR, out_name)

    with _jobs_lock:
        _jobs[job_id] = {"status": "processing", "progress": 0, "result": None, "type": "wildlife"}

    def _worker():
        try:
            def _prog(pct):
                with _jobs_lock:
                    _jobs[job_id]["progress"] = pct

            logger.info("Model inference started for wildlife video")
            result = detector.run_on_video(tmp_path, out_path, _prog)
            logger.info(f"Prediction result: processed wildlife video, found {len(result.get('detections', []))} detections")
            # Persist unique detections to DB
            for det in result.get("detections", []):
                db.insert_detection(
                    category   = "wildlife",
                    label      = det["animal"],
                    confidence = det["confidence"],
                    score      = 0.0,
                    status     = det["threat"],
                    camera_id  = "UPLOAD",
                    image_path = f"snapshots/{out_name}",
                    source     = "upload_video",
                    timestamp  = det["timestamp"],
                )
            result["processed_video_url"] = f"/snapshots/{out_name}"
            with _jobs_lock:
                _jobs[job_id]["status"]   = "done"
                _jobs[job_id]["progress"] = 100
                _jobs[job_id]["result"]   = result
        except Exception as e:
            logger.error(f"Video job error: {e}")
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["result"] = {"error": str(e)}
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            
            # Schedule memory cleanup of completed/errored job after 10 minutes (Issue 13)
            def _cleanup():
                with _jobs_lock:
                    if job_id in _jobs:
                        del _jobs[job_id]
                        logger.info(f"Video job {job_id} cleaned up from memory.")
            threading.Timer(600.0, _cleanup).start()

    threading.Thread(target=_worker, daemon=True, name=f"VideoJob-{job_id[:8]}").start()

    return jsonify({"job_id": job_id, "status": "processing"}), 202


# ════════════════════════════════════════════════════════════
#  NEW: UPLOAD VIDEO – ANOMALY  (background job)
# ════════════════════════════════════════════════════════════

@app.route("/api/anomaly/upload-video", methods=["POST"])
def upload_anomaly_video():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename or not _allowed(f.filename, ALLOWED_VIDEOS):
        return jsonify({"error": "Invalid file type. Use MP4/AVI/MOV."}), 400
    if not anomaly_detector:
        return jsonify({"error": "Anomaly detector not ready"}), 503

    logger.info(f"Upload received: anomaly video {f.filename}")
    print("Anomaly upload received")
    print("Model path:", anomaly_detector.yolo_model_path)
    job_id = str(uuid.uuid4())
    tmp_name = secure_filename(f"an_{job_id[:8]}_{f.filename}")
    tmp_path = os.path.join(UPLOADS_DIR, tmp_name)
    f.save(tmp_path)

    out_name = f"an_processed_{job_id[:8]}.mp4"
    out_path = os.path.join(SNAPSHOTS_DIR, out_name)

    with _jobs_lock:
        _jobs[job_id] = {"status": "processing", "progress": 0, "result": None, "type": "anomaly"}

    def _worker():
        try:
            def _prog(pct):
                with _jobs_lock:
                    _jobs[job_id]["progress"] = pct

            logger.info("Model inference started for anomaly video")
            result = anomaly_detector.run_on_video(tmp_path, out_path, _prog)
            logger.info(f"Prediction result: processed anomaly video, found {len(result.get('anomaly_frames', []))} anomaly events")
            for af in result.get("anomaly_frames", []):
                db.insert_detection(
                    category   = "human",
                    label      = af["anomaly_type"],
                    confidence = af.get("confidence", round(af["score"] * 100, 1)),
                    score      = af.get("score", 0.0),
                    status     = af["status"],
                    camera_id  = "UPLOAD",
                    image_path = f"snapshots/{out_name}",
                    source     = "upload_video",
                    timestamp  = af["timestamp"],
                )
            result["processed_video_url"] = f"/snapshots/{out_name}"
            with _jobs_lock:
                _jobs[job_id]["status"]   = "done"
                _jobs[job_id]["progress"] = 100
                _jobs[job_id]["result"]   = result
        except Exception as e:
            logger.error(f"Anomaly video job error: {e}")
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["result"] = {"error": str(e)}
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            # Schedule memory cleanup of completed/errored job after 10 minutes (Issue 13)
            def _cleanup():
                with _jobs_lock:
                    if job_id in _jobs:
                        del _jobs[job_id]
                        logger.info(f"Anomaly video job {job_id} cleaned up from memory.")
            threading.Timer(600.0, _cleanup).start()

    threading.Thread(target=_worker, daemon=True, name=f"AnomJob-{job_id[:8]}").start()

    return jsonify({"job_id": job_id, "status": "processing"}), 202


# ════════════════════════════════════════════════════════════
#  NEW: JOB STATUS POLLING
# ════════════════════════════════════════════════════════════

@app.route("/api/job/<job_id>", methods=["GET"])
def api_job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "job_id":   job_id,
        "status":   job["status"],
        "progress": job["progress"],
        "type":     job["type"],
        "result":   job["result"],
    })


# ════════════════════════════════════════════════════════════
#  NEW: DETECTION HISTORY
# ════════════════════════════════════════════════════════════

@app.route("/api/history", methods=["GET"])
def api_history():
    days     = request.args.get("days",     7,    type=int)
    category = request.args.get("category", "")
    label    = request.args.get("label",    "")
    camera   = request.args.get("camera",   "")
    source   = request.args.get("source",   "")
    limit    = request.args.get("limit",    100,  type=int)
    offset   = request.args.get("offset",   0,    type=int)
    rows     = db.query_detections(days=days, category=category,
                                   label=label, camera=camera,
                                   source=source, limit=limit, offset=offset)
    total    = db.count_detections(days=days, category=category)
    return jsonify({"total": total, "rows": rows})


@app.route("/api/history/stats", methods=["GET"])
def api_history_stats():
    return jsonify(db.get_stats())


@app.route("/api/history/filters", methods=["GET"])
def api_history_filters():
    category = request.args.get("category", "")
    return jsonify({
        "labels":  db.get_labels(category),
        "cameras": db.get_cameras(),
    })


# ════════════════════════════════════════════════════════════
#  NEW: CSV EXPORT
# ════════════════════════════════════════════════════════════

@app.route("/api/history/export-csv", methods=["GET"])
def api_export_csv():
    days     = request.args.get("days",     7,   type=int)
    category = request.args.get("category", "")
    csv_str  = db.export_csv(days=days, category=category)
    buf = io.BytesIO(csv_str.encode("utf-8"))
    buf.seek(0)
    fname = f"sentinelai_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True, download_name=fname)


# ════════════════════════════════════════════════════════════
#  NEW: PDF REPORT
# ════════════════════════════════════════════════════════════

@app.route("/api/report/download", methods=["GET"])
def api_report_download():
    try:
        from fpdf import FPDF
    except ImportError:
        return jsonify({"error": "fpdf2 not installed. Run: pip install fpdf2"}), 500

    now    = datetime.now()
    stats  = db.get_stats()
    recent = db.query_detections(days=1, limit=20)
    wl_det = db.query_detections(days=1, category="wildlife", limit=5)
    an_det = db.query_detections(days=1, category="human",    limit=5)

    # ── Build PDF ────────────────────────────────────────────
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header banner
    pdf.set_fill_color(10, 20, 40)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(0, 212, 255)
    pdf.set_xy(10, 8)
    pdf.cell(0, 12, "SentinelAI Detection Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(140, 180, 220)
    pdf.set_xy(10, 20)
    pdf.cell(0, 6, f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}  |  Cyber Surveillance Operations Center")

    pdf.ln(20)

    # ── Report details ───────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Report Details", ln=True)
    pdf.set_draw_color(0, 212, 255)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    fields = [
        ("Date",          now.strftime("%A, %B %d, %Y")),
        ("Time",          now.strftime("%H:%M:%S")),
        ("Report Period", "Last 24 Hours"),
        ("System",        "SentinelAI v2.0"),
    ]
    for label, val in fields:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(50, 7, label + ":")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, val, ln=True)

    pdf.ln(4)

    # ── Statistics ───────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Statistics Summary", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    pdf.set_fill_color(240, 248, 255)
    headers = ["Period", "Total Alerts", "Wildlife", "Human Anomalies"]
    col_w   = [45, 45, 45, 55]
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(10, 20, 40)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 8, h, border=1, fill=True)
    pdf.ln()
    pdf.set_text_color(30, 30, 30)
    for period_key, period_name in [("today", "Today"), ("week", "Last 7 Days"), ("month", "Last 30 Days")]:
        s = stats[period_key]
        pdf.set_font("Helvetica", "", 10)
        pdf.set_fill_color(245, 250, 255)
        pdf.cell(col_w[0], 7, period_name,   border=1, fill=True)
        pdf.cell(col_w[1], 7, str(s["total"]),    border=1)
        pdf.cell(col_w[2], 7, str(s["wildlife"]), border=1)
        pdf.cell(col_w[3], 7, str(s["human"]),    border=1)
        pdf.ln()

    pdf.ln(6)

    # ── Wildlife Section ─────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(180, 0, 0)
    pdf.cell(0, 8, "Wildlife Detections (Last 24h)", ln=True)
    pdf.set_text_color(30, 30, 30)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    if wl_det:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(10, 20, 40)
        pdf.set_text_color(255, 255, 255)
        for h, w in [("Timestamp", 50), ("Camera", 30), ("Animal", 35), ("Confidence", 35), ("Threat", 40)]:
            pdf.cell(w, 8, h, border=1, fill=True)
        pdf.ln()
        pdf.set_text_color(30, 30, 30)
        for r in wl_det:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_fill_color(255, 248, 248)
            pdf.cell(50, 6, r["timestamp"][:19], border=1, fill=True)
            pdf.cell(30, 6, r["camera_id"],       border=1)
            pdf.cell(35, 6, r["label"],            border=1)
            pdf.cell(35, 6, f"{r['confidence']:.1f}%", border=1)
            pdf.cell(40, 6, r["status"],           border=1)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 7, "No wildlife detections in the last 24 hours.", ln=True)

    pdf.ln(6)

    # ── Anomaly Section ──────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 100, 200)
    pdf.cell(0, 8, "Human Anomaly Detections (Last 24h)", ln=True)
    pdf.set_text_color(30, 30, 30)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    if an_det:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(10, 20, 40)
        pdf.set_text_color(255, 255, 255)
        for h, w in [("Timestamp", 50), ("Camera", 30), ("Anomaly Type", 50), ("Score", 30), ("Status", 30)]:
            pdf.cell(w, 8, h, border=1, fill=True)
        pdf.ln()
        pdf.set_text_color(30, 30, 30)
        for r in an_det:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_fill_color(248, 248, 255)
            pdf.cell(50, 6, r["timestamp"][:19], border=1, fill=True)
            pdf.cell(30, 6, r["camera_id"],       border=1)
            pdf.cell(50, 6, r["label"],            border=1)
            pdf.cell(30, 6, f"{r['confidence']:.1f}%", border=1)
            pdf.cell(30, 6, r["status"],           border=1)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 7, "No anomaly detections in the last 24 hours.", ln=True)

    pdf.ln(8)

    # ── Footer ───────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, "SentinelAI – AI-Powered Surveillance System  |  Confidential", ln=True, align="C")

    # ── Output ───────────────────────────────────────────────
    pdf_bytes = pdf.output()
    buf = io.BytesIO(pdf_bytes if isinstance(pdf_bytes, bytes) else bytes(pdf_bytes))
    buf.seek(0)
    fname = f"SentinelAI_Report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True, download_name=fname)


# ════════════════════════════════════════════════════════════
#  NEW: SERVE SNAPSHOTS
# ════════════════════════════════════════════════════════════

@app.route("/snapshots/<path:filename>")
def serve_snapshot(filename: str):
    # secure_filename strips directory delimiters, but we resolve realpath to be completely safe (Issue 16)
    safe = secure_filename(filename)
    path = os.path.realpath(os.path.join(SNAPSHOTS_DIR, safe))
    real_snapshots_dir = os.path.realpath(SNAPSHOTS_DIR)
    if not path.startswith(real_snapshots_dir):
        abort(403)
    if not os.path.exists(path):
        abort(404)
    return send_file(path)


# ──────────────────────────────────────────────────────────────────────────
# Debug endpoints (helpful for UI troubleshooting)
# ──────────────────────────────────────────────────────────────────────────
@app.route('/api/debug/jobs', methods=['GET'])
def api_debug_jobs():
    """Return the current in-memory job table for debugging."""
    with _jobs_lock:
        # Return shallow copy to avoid race conditions
        data = {k: v.copy() for k, v in _jobs.items()}
    return jsonify({'jobs': data, 'count': len(data)})


def _tail_file(path: str, lines: int = 200):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            block_size = 1024
            data = b''
            while end > 0 and data.count(b'\n') <= lines:
                read_size = min(block_size, end)
                end -= read_size
                f.seek(end)
                data = f.read(read_size) + data
            text = data.decode('utf-8', errors='replace')
            return '\n'.join(text.splitlines()[-lines:])
    except Exception:
        return ''


@app.route('/api/debug/logs', methods=['GET'])
def api_debug_logs():
    """Return the tail of the backend log file for the UI to display."""
    lines = request.args.get('lines', 200, type=int)
    log_path = os.path.join(BASE_DIR, 'sentinelai.log')
    if not os.path.exists(log_path):
        return jsonify({'error': 'log file not found', 'path': log_path}), 404
    txt = _tail_file(log_path, lines)
    return jsonify({'path': log_path, 'lines': lines, 'content': txt})


# ─── CORS preflight ──────────────────────────────────────────
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    init_detectors()
    logger.info("🚀 SentinelAI v2.0 starting on http://0.0.0.0:5000")
    app.run(
        host        = "0.0.0.0",
        port        = 5000,
        debug       = False,
        threaded    = True,
        use_reloader= False,
    )
