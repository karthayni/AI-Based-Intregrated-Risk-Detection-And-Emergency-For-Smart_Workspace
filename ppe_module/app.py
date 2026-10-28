"""Unified dashboard server (Modules 1+2 workplace alerts and Module 3 transport tracking).
Receives GPS readings, runs tracker.py, reads the PPE/stress log, and serves the dashboard.

Start:           python app.py
Open:            http://localhost:5000
Demo vehicles:   python simulator.py        (in a second terminal)

Real GPS sources can send readings to  /api/location  (see README.md):
  * any device or app that can send an HTTP POST with JSON, or
  * the Android app "GPSLogger" with a custom URL (GET request).
"""
import argparse
import atexit
import csv
import io
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, send_from_directory

import tracker as T

HERE = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static", template_folder=str(HERE / "templates"))
TRACKER = None
PPE_LOG = None
API_KEY = None
WINDOW_MIN = 5
CAM = {"proc": None, "user_stop": False, "dir": None, "python": None, "port": 5001, "log": None, "source": None, "stress": False}


# ----------------------------------------------------------------------------- alert hook
def send_alert(alert):
    """Called for every new alert. Replace the print with an SMS / email / Telegram call when you add notifications."""
    print(f"[TRANSPORT ALERT] {alert['severity'].upper():6} {alert['type']:16} {alert['message']}", flush=True)


# ----------------------------------------------------------------------------- helpers
def parse_time(value):
    """Accepts epoch seconds, epoch milliseconds or an ISO-8601 text. Returns epoch seconds, or None if unusable."""
    if value in (None, ""):
        return None
    try:
        x = float(value)
        return x / 1000 if x > 1e11 else x
    except (TypeError, ValueError):
        pass
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def get_param(data, *names):
    for n in names:
        if data.get(n) not in (None, ""):
            return data.get(n)
    return None


def authorised():
    return not API_KEY or request.headers.get("X-API-Key") == API_KEY or request.args.get("key") == API_KEY


# ----------------------------------------------------------------------------- pages and API
@app.route("/")
def home():
    return render_template("home.html", section="home", page="home")


@app.route("/dashboard")
def overview():
    return render_template("overview.html", section="dashboard", page="overview")


@app.route("/dashboard/workplace")
def workplace():
    return render_template("workplace.html", section="dashboard", page="workplace")


@app.route("/dashboard/transport")
def transport():
    return render_template("transport.html", section="dashboard", page="transport")


@app.route("/dashboard/camera")
def camera():
    return render_template("camera.html", section="dashboard", page="camera")


@app.route("/about")
def about():
    return render_template("about.html", section="about", page="about")


@app.route("/workplace")
@app.route("/transport")
@app.route("/camera")
def old_links():
    """Old addresses still work: they go to the dashboard page of the same name."""
    return redirect("/dashboard" + request.path)


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


# ----------------------------------------------------------------------------- live camera (Modules 1 and 2)
def is_local():
    return request.remote_addr in ("127.0.0.1", "::1")


def cam_running():
    return CAM["proc"] is not None and CAM["proc"].poll() is None


def cam_files():
    d = CAM["dir"]
    if not d:
        return dict(ready=False, problem="The PPE project folder is not set. Start this server with --ppe-dir C:\\path\\to\\ppe_project.")
    d = Path(d)
    if not (d / "ppe_detect.py").exists():
        return dict(ready=False, problem=f"ppe_detect.py was not found in {d}.")
    weights = next((w for w in (d / "best.pt", d / "runs" / "ppe" / "weights" / "best.pt") if w.exists()), None)
    if weights is None:
        return dict(ready=False, problem=f"best.pt (your trained PPE model) was not found in {d}.")
    stress_ok = (d / "model_unquant.tflite").exists() and (d / "labels.txt").exists() and (d / "stress.py").exists()
    return dict(ready=True, weights=weights, stress_ok=stress_ok, zone=(d / "zone.json").exists())


@app.route("/api/camera/status")
def camera_status():
    f = cam_files()
    tail = []
    if CAM["log"] and Path(CAM["log"]).exists():
        try:
            tail = Path(CAM["log"]).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-6:]
        except OSError:
            pass
    code = CAM["proc"].poll() if CAM["proc"] is not None else None
    return jsonify(running=cam_running(), exit_code=code, ready=f["ready"], problem=f.get("problem"),
                   stress_ok=f.get("stress_ok", False), zone=f.get("zone", False), port=CAM["port"],
                   source=CAM["source"], stress=CAM["stress"], stopped_by_user=CAM["user_stop"], log=tail)


@app.route("/api/camera/start", methods=["POST"])
def camera_start():
    if not is_local():
        return jsonify(error="The camera can only be started from the computer that runs the server."), 403
    if cam_running():
        return jsonify(error="The camera is already running."), 409
    f = cam_files()
    if not f["ready"]:
        return jsonify(error=f["problem"]), 400
    data = request.get_json(silent=True) or {}
    source = str(data.get("source", "0")).strip() or "0"
    if not source.isdigit():
        path = Path(source)
        path = path if path.is_absolute() else Path(CAM["dir"]) / path
        if not path.exists():
            return jsonify(error=f"Video file not found: {source}"), 400
        source = str(path)
    want_stress = bool(data.get("stress"))
    if want_stress and not f["stress_ok"]:
        return jsonify(error="Stress files are missing in the PPE folder: model_unquant.tflite, labels.txt and stress.py."), 400
    cmd = [CAM["python"], "ppe_detect.py", "--weights", str(f["weights"]), "--source", source,
           "--stream", str(CAM["port"]), "--no-window"]
    if want_stress:
        cmd += ["--stress-model", "model_unquant.tflite", "--stress-labels", "labels.txt"]
    log_dir = Path(CAM["dir"]) / "logs"
    log_dir.mkdir(exist_ok=True)
    CAM["log"] = str(log_dir / "detect_console.txt")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    try:
        with open(CAM["log"], "w", encoding="utf-8") as lf:
            CAM["proc"] = subprocess.Popen(cmd, cwd=CAM["dir"], stdout=lf, stderr=subprocess.STDOUT, env=env)
    except OSError as e:
        return jsonify(error=f"Could not start Python at {CAM['python']}: {e}"), 500
    CAM["source"], CAM["stress"], CAM["user_stop"] = source, want_stress, False
    return jsonify(ok=True)


@app.route("/api/camera/stop", methods=["POST"])
def camera_stop():
    if not is_local():
        return jsonify(error="The camera can only be stopped from the computer that runs the server."), 403
    stop_camera()
    return jsonify(ok=True)


def stop_camera():
    proc = CAM["proc"]
    if proc is not None and proc.poll() is None:
        CAM["user_stop"] = True
        proc.terminate()
        try:
            proc.wait(timeout=6)
        except subprocess.TimeoutExpired:
            proc.kill()


@app.route("/api/state")
def state():
    return jsonify(TRACKER.snapshot())


@app.route("/api/location", methods=["POST", "GET"])
def location():
    """One GPS reading. JSON body or URL parameters:  vehicle, lat, lon, [speed (m/s)], [time or ts]."""
    if not authorised():
        return jsonify(error="bad or missing API key"), 401
    data = request.get_json(silent=True) or {}
    data = {**request.args.to_dict(), **data}
    vid = get_param(data, "vehicle", "vehicle_id", "id")
    lat = get_param(data, "lat", "latitude")
    lon = get_param(data, "lon", "lng", "longitude")
    if vid is None or lat is None or lon is None:
        return jsonify(error="need vehicle, lat and lon"), 400
    ts = parse_time(get_param(data, "ts", "timestamp", "time"))
    if ts is not None and abs(ts - time.time()) > 86400 * 2 and not data.get("simulated"):
        ts = None                                     # device clock is clearly wrong: use the server clock instead
    speed = get_param(data, "speed", "speed_ms")
    try:
        result = TRACKER.ingest(str(vid), lat, lon, ts=ts, speed=None if speed is None else float(speed))
    except KeyError as e:
        return jsonify(error=str(e.args[0])), 404
    except (ValueError, TypeError) as e:
        return jsonify(error=f"bad value: {e}"), 400
    return jsonify(result)


@app.route("/api/reset/<vehicle_id>", methods=["POST"])
def reset(vehicle_id):
    if not authorised():
        return jsonify(error="bad or missing API key"), 401
    try:
        TRACKER.reset(vehicle_id)
    except KeyError:
        return jsonify(error="unknown vehicle"), 404
    return jsonify(ok=True)


RANK = {"SAFE": 0, "CAUTION": 1, "VIOLATION": 2, "HIGH_RISK": 3}


def snapshot_dir():
    return Path(PPE_LOG).parent / "snapshots" if PPE_LOG else None


@app.route("/api/ppe")
def ppe_events():
    """Module 1/2 log (logs/events.csv): newest rows, counts, and the current workplace risk level."""
    empty = dict(available=False, rows=[], counts={}, today={}, stress_high=0, stress_today=0, hourly=[], last_hour=0,
                 total=0, current="SAFE", window_min=WINDOW_MIN)
    try:
        limit = max(1, min(int(request.args.get("limit", 40)), 500))
    except ValueError:
        limit = 40
    if not PPE_LOG:
        return jsonify(empty)
    if Path(PPE_LOG).exists():
        try:
            with open(PPE_LOG, newline="", encoding="utf-8", errors="replace") as f:
                all_rows = list(csv.DictReader(f))
        except OSError:
            return jsonify(empty)
    elif Path(PPE_LOG).parent.is_dir():
        all_rows = []                                 # the logs folder exists but is empty (for example after clearing)
    else:
        return jsonify(empty)
    now, today = datetime.now(), datetime.now().date()
    counts, today_counts, stress_high, stress_today, current, last_hour = {}, {}, 0, 0, "SAFE", 0
    start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=11)
    hourly = [dict(hour=(start + timedelta(hours=i)).strftime("%H:00"), CAUTION=0, VIOLATION=0, HIGH_RISK=0) for i in range(12)]
    snaps = snapshot_dir()
    out = []
    for r in all_rows:
        status = (r.get("status") or "").strip()
        details = r.get("details") or ""
        try:
            when = datetime.fromisoformat((r.get("time") or "").strip())
        except ValueError:
            when = None
        counts[status] = counts.get(status, 0) + 1
        if when and when.date() == today:
            today_counts[status] = today_counts.get(status, 0) + 1
        if "stress=HIGH" in details:
            stress_high += 1
            if when and when.date() == today:
                stress_today += 1
        if when and 0 <= (now - when).total_seconds() <= 3600:
            last_hour += 1
        if when and when >= start and status in ("CAUTION", "VIOLATION", "HIGH_RISK"):
            idx = int((when - start).total_seconds() // 3600)
            if 0 <= idx < 12:
                hourly[idx][status] += 1
        if when and (now - when).total_seconds() <= WINDOW_MIN * 60 and RANK.get(status, 0) > RANK.get(current, 0):
            current = status
        out.append((when, status, details, r.get("snapshot") or ""))
    rows = []
    for when, status, details, snap in reversed(out[-limit:]):
        name = snap.replace("\\", "/").split("/")[-1] if snap else ""
        has_pic = bool(name and snaps and (snaps / name).exists())
        rows.append(dict(time=when.isoformat(timespec="seconds") if when else "", status=status, details=details,
                         stress_high="stress=HIGH" in details, snapshot=f"/snapshot/{name}" if has_pic else ""))
    return jsonify(available=True, rows=rows, counts=counts, today=today_counts, stress_high=stress_high,
                   stress_today=stress_today, hourly=hourly, last_hour=last_hour, total=len(all_rows), current=current, window_min=WINDOW_MIN)


@app.route("/api/workplace/clear", methods=["POST"])
def clear_workplace():
    """Move the workplace alert log and its pictures into logs/archive/<time>/ so the website starts empty."""
    if not is_local():
        return jsonify(error="Alerts can only be cleared from the computer that runs the server."), 403
    if not PPE_LOG or not Path(PPE_LOG).exists():
        return jsonify(error="There is no workplace log to clear."), 404
    log = Path(PPE_LOG)
    snaps = log.parent / "snapshots"
    dest = log.parent / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    pictures = len(list(snaps.glob("*"))) if snaps.is_dir() else 0
    try:
        dest.mkdir(parents=True)
        shutil.move(str(log), str(dest / "events.csv"))
        if snaps.is_dir():
            shutil.move(str(snaps), str(dest / "snapshots"))
    except OSError as e:
        return jsonify(error=f"Could not clear the alerts. Close any program that has events.csv open (such as Excel) and try again. ({e})"), 500
    return jsonify(ok=True, pictures=pictures, archive=str(dest))


@app.route("/export/events.csv")
def export_events():
    """The Module 1/2 alert log as a download (for your report)."""
    if not PPE_LOG or not Path(PPE_LOG).exists():
        return jsonify(error="workplace log not found"), 404
    return send_file(PPE_LOG, mimetype="text/csv", as_attachment=True, download_name="events.csv")


@app.route("/snapshot/<path:name>")
def snapshot(name):
    """Pictures saved by Module 1/2 when an alert was logged (only from the snapshots folder next to events.csv)."""
    folder = snapshot_dir()
    if not folder or not folder.is_dir():
        return ("", 404)
    return send_from_directory(folder, name)


@app.route("/export/<table>.csv")
def export(table):
    """Download a table as CSV (for your report): gps_log, transport_alert, stop_event."""
    if table not in ("gps_log", "transport_alert", "stop_event", "vehicle"):
        return jsonify(error="unknown table"), 404
    with TRACKER.lock:
        cur = TRACKER.db.execute(f"SELECT * FROM {table}")
        header = [c[0] for c in cur.description]
        rows = cur.fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={table}.csv"})


# ----------------------------------------------------------------------------- start
def create_tracker(config_path, db_path, notify=send_alert):
    return T.Tracker(T.load_config(config_path), db_path=db_path, notify=notify)


def signal_watcher(interval=5):
    while True:
        time.sleep(interval)
        try:
            TRACKER.check_signal()
        except Exception as e:                        # never let the watcher thread die
            print("signal check error:", e, flush=True)


def main():
    global TRACKER, PPE_LOG, API_KEY, WINDOW_MIN
    ap = argparse.ArgumentParser(description="Module 3: employee transport tracking dashboard")
    ap.add_argument("--config", default=str(HERE / "routes.json"))
    ap.add_argument("--db", default=str(HERE / "transport.db"))
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to let a phone on the same Wi-Fi send GPS readings")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--ppe-log", default=None, help="path to Module 1/2 logs/events.csv (default: looks for logs/events.csv here, then in ../ppe_project and ../ppe_module)")
    ap.add_argument("--ppe-dir", default=None, help="your ppe_project folder (lets the website start and stop the camera detection)")
    ap.add_argument("--ppe-python", default=None, help="python.exe of the ppe_project venv (default: venv inside --ppe-dir)")
    ap.add_argument("--stream-port", type=int, default=5001, help="port for the live camera picture")
    ap.add_argument("--window", type=int, default=5, help="minutes of workplace alerts that count as 'current' risk")
    ap.add_argument("--key", default=os.environ.get("TRANSPORT_API_KEY"), help="optional API key required for /api/location")
    ap.add_argument("--fresh", action="store_true", help="delete the old database before starting")
    a = ap.parse_args()
    if a.fresh and Path(a.db).exists():
        Path(a.db).unlink()
    if not a.ppe_log:
        for guess in (HERE / "logs" / "events.csv", HERE.parent / "ppe_project" / "logs" / "events.csv",
                      HERE.parent / "ppe_module" / "logs" / "events.csv"):
            if guess.exists():
                a.ppe_log = str(guess)
                break
    WINDOW_MIN = a.window
    if a.ppe_dir:
        CAM["dir"] = str(Path(a.ppe_dir).resolve())
        if not a.ppe_log:
            a.ppe_log = str(Path(CAM["dir"]) / "logs" / "events.csv")
    elif a.ppe_log:
        CAM["dir"] = str(Path(a.ppe_log).resolve().parent.parent)
    if CAM["dir"]:
        d = Path(CAM["dir"])
        guess = [d / "venv" / "Scripts" / "python.exe", d / "venv" / "bin" / "python"]
        CAM["python"] = a.ppe_python or next((str(g) for g in guess if g.exists()), sys.executable)
    CAM["port"] = a.stream_port
    atexit.register(stop_camera)
    TRACKER, PPE_LOG, API_KEY = create_tracker(a.config, a.db), a.ppe_log, a.key
    print("Workplace log:", PPE_LOG or "not found (start with --ppe-log path\\to\\events.csv)")
    threading.Thread(target=signal_watcher, daemon=True).start()
    print(f"Dashboard: http://localhost:{a.port}   |   vehicles: {', '.join(TRACKER.vehicles)}   |   key required: {bool(API_KEY)}")
    if a.host == "0.0.0.0":
        print("Listening on your network. Anyone on the same Wi-Fi can reach it; use --key to protect /api/location.")
    app.run(host=a.host, port=a.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
