# ============================================================
#  SentinelAI – SQLite Database Layer
#  database.py
#  Tables:
#    detections  – every detection event (live + uploads)
#  Thread-safe via sqlite3 check_same_thread=False + lock
# ============================================================

import os
import sqlite3
import threading
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("SentinelDB")

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentinelai.db")
_lock    = threading.Lock()


# ─── Schema ─────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,
    camera_id    TEXT    NOT NULL DEFAULT 'CAM-01',
    category     TEXT    NOT NULL,          -- 'wildlife' | 'human'
    label        TEXT    NOT NULL,          -- animal name OR anomaly type
    confidence   REAL    DEFAULT 0.0,       -- 0-100 for wildlife; 0-100 score% for human
    score        REAL    DEFAULT 0.0,       -- raw anomaly error (null for wildlife)
    status       TEXT    DEFAULT 'Active',  -- threat level / status string
    image_path   TEXT    DEFAULT '',        -- relative path to snapshot JPEG
    source       TEXT    DEFAULT 'live'     -- 'live' | 'upload_image' | 'upload_video'
);

CREATE INDEX IF NOT EXISTS idx_ts       ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_category ON detections(category);
CREATE INDEX IF NOT EXISTS idx_label    ON detections(label);
CREATE INDEX IF NOT EXISTS idx_camera   ON detections(camera_id);
"""

_SETTINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# ─── Init ────────────────────────────────────────────────────
def init_db():
    """Create tables if they don't exist. Call once at startup."""
    with _lock:
        conn = _get_conn()
        conn.executescript(_SCHEMA)
        conn.executescript(_SETTINGS_SCHEMA)
        conn.commit()
        conn.close()
    logger.info(f"✅ SQLite DB ready at: {_DB_PATH}")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # Enable Write-Ahead Log for concurrent readers (Issue 17)
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn


# ─── Insert ──────────────────────────────────────────────────
def insert_detection(
    category:    str,
    label:       str,
    confidence:  float  = 0.0,
    score:       float  = 0.0,
    status:      str    = "Active",
    camera_id:   str    = "CAM-01",
    image_path:  str    = "",
    source:      str    = "live",
    timestamp:   str    = "",
) -> int:
    """Insert one detection row. Returns new row id."""
    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sql = """
        INSERT INTO detections
            (timestamp, camera_id, category, label, confidence, score, status, image_path, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _lock:
        conn = _get_conn()
        cur  = conn.execute(sql, (timestamp, camera_id, category, label,
                                  confidence, score, status, image_path, source))
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
    return row_id


# ─── Query ───────────────────────────────────────────────────
def query_detections(
    days:     int  = 7,
    category: str  = "",     # '' = all
    label:    str  = "",     # '' = all
    camera:   str  = "",     # '' = all
    source:   str  = "",     # '' = all
    limit:    int  = 200,
    offset:   int  = 0,
) -> list[dict]:
    """Return detections matching filters as a list of dicts."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    conditions = ["timestamp >= ?"]
    params     = [since]

    if category:
        conditions.append("category = ?")
        params.append(category)
    if label:
        conditions.append("label = ?")
        params.append(label)
    if camera:
        conditions.append("camera_id = ?")
        params.append(camera)
    if source:
        conditions.append("source = ?")
        params.append(source)

    where = " AND ".join(conditions)
    sql   = f"""
        SELECT * FROM detections
        WHERE {where}
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """
    params += [limit, offset]

    with _lock:
        conn = _get_conn()
        rows = conn.execute(sql, params).fetchall()
        conn.close()

    return [dict(r) for r in rows]


def count_detections(days: int = 7, category: str = "") -> int:
    """Return total count matching the filter (for pagination)."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    conditions = ["timestamp >= ?"]
    params     = [since]
    if category:
        conditions.append("category = ?")
        params.append(category)
    where = " AND ".join(conditions)
    sql   = f"SELECT COUNT(*) FROM detections WHERE {where}"
    with _lock:
        conn = _get_conn()
        cnt  = conn.execute(sql, params).fetchone()[0]
        conn.close()
    return cnt


# ─── Stats ───────────────────────────────────────────────────
def get_stats() -> dict:
    """Return aggregate counts for today, 7 days, 30 days using conditional aggregation (Issue 19)."""
    t1 = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    t7 = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    t30 = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")

    sql = """
        SELECT
            COUNT(CASE WHEN timestamp >= :t1 THEN 1 END) AS today_total,
            COUNT(CASE WHEN timestamp >= :t1 AND category = 'wildlife' THEN 1 END) AS today_wildlife,
            COUNT(CASE WHEN timestamp >= :t1 AND category = 'human' THEN 1 END) AS today_human,

            COUNT(CASE WHEN timestamp >= :t7 THEN 1 END) AS week_total,
            COUNT(CASE WHEN timestamp >= :t7 AND category = 'wildlife' THEN 1 END) AS week_wildlife,
            COUNT(CASE WHEN timestamp >= :t7 AND category = 'human' THEN 1 END) AS week_human,

            COUNT(CASE WHEN timestamp >= :t30 THEN 1 END) AS month_total,
            COUNT(CASE WHEN timestamp >= :t30 AND category = 'wildlife' THEN 1 END) AS month_wildlife,
            COUNT(CASE WHEN timestamp >= :t30 AND category = 'human' THEN 1 END) AS month_human
        FROM detections
        WHERE timestamp >= :t30
    """

    with _lock:
        conn = _get_conn()
        res = conn.execute(sql, {"t1": t1, "t7": t7, "t30": t30}).fetchone()
        conn.close()

    res = dict(res) if res else {}
    return {
        "today": {
            "total":    res.get("today_total", 0),
            "wildlife": res.get("today_wildlife", 0),
            "human":    res.get("today_human", 0),
        },
        "week": {
            "total":    res.get("week_total", 0),
            "wildlife": res.get("week_wildlife", 0),
            "human":    res.get("week_human", 0),
        },
        "month": {
            "total":    res.get("month_total", 0),
            "wildlife": res.get("month_wildlife", 0),
            "human":    res.get("month_human", 0),
        },
    }


# ─── Distinct label/camera lists (for filter dropdowns) ──────
def get_labels(category: str = "") -> list[str]:
    conditions = []
    params     = []
    if category:
        conditions.append("category = ?")
        params.append(category)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql   = f"SELECT DISTINCT label FROM detections {where} ORDER BY label"
    with _lock:
        conn = _get_conn()
        rows = conn.execute(sql, params).fetchall()
        conn.close()
    return [r[0] for r in rows]


def get_cameras() -> list[str]:
    sql = "SELECT DISTINCT camera_id FROM detections ORDER BY camera_id"
    with _lock:
        conn = _get_conn()
        rows = conn.execute(sql).fetchall()
        conn.close()
    return [r[0] for r in rows]


# ─── Settings helpers ─────────────────────────────────────────
def set_setting(key: str, value: str):
    sql = "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)"
    with _lock:
        conn = _get_conn()
        conn.execute(sql, (key, value))
        conn.commit()
        conn.close()


def get_setting(key: str, default: str = "") -> str:
    sql = "SELECT value FROM settings WHERE key = ?"
    with _lock:
        conn = _get_conn()
        row  = conn.execute(sql, (key,)).fetchone()
        conn.close()
    return row[0] if row else default


# ─── Export CSV helper ────────────────────────────────────────
def export_csv(days: int = 7, category: str = "") -> str:
    """Return CSV string of detections using standard csv module for proper quoting (Issue 18)."""
    import csv
    import io
    rows = query_detections(days=days, category=category, limit=10000)
    
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    
    # Header
    writer.writerow(["id", "timestamp", "camera_id", "category", "label", "confidence", "score", "status", "image_path", "source"])
    
    for r in rows:
        writer.writerow([
            r.get("id", ""),
            r.get("timestamp", ""),
            r.get("camera_id", ""),
            r.get("category", ""),
            r.get("label", ""),
            r.get("confidence", 0.0),
            r.get("score", 0.0),
            r.get("status", ""),
            r.get("image_path", ""),
            r.get("source", "")
        ])
    return output.getvalue()
